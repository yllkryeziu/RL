# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimPO (Simple Preference Optimization) Training Script.

This script runs SimPO training, which is a reference-free variant of DPO
that uses average log probability as the implicit reward.

Key differences from DPO:
- No reference model needed (saves memory)
- Uses length-normalized log probabilities
- Includes a margin term (gamma) for better preference separation
"""

import argparse
import os
import pprint
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.simpo import SimPOLossFn
from nemo_rl.algorithms.utils import get_tokenizer, maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import preference_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_preference_dataset
from nemo_rl.data.datasets.preference_datasets import PreferenceDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import Logger, LoggerConfig, get_next_experiment_dir
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class SimPOSaveState(TypedDict):
    epoch: int
    step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int


class SimPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int
    beta: float
    gamma: float
    preference_loss_weight: float
    sft_loss_weight: float


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    simpo: SimPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


class SimPOValMetrics(TypedDict):
    loss: float
    sft_loss: float
    preference_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float
    global_valid_seqs: float
    global_valid_toks: float


def _default_simpo_save_state() -> SimPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SimPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def simpo_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for SimPO training."""
    assert len(datum_dict["completions"]) == 2, (
        "SimPO training supports only two completions"
    )

    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError("Ties are not supported.")

    messages_chosen = datum_dict["context"] + chosen_completion["completion"]
    messages_rejected = datum_dict["context"] + rejected_completion["completion"]

    message_log_chosen = get_formatted_message_log(
        messages_chosen, tokenizer, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, tokenizer, task_data_spec
    )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        warnings.warn(
            f"Sequence length {max(length_chosen, length_rejected)} exceeds max_seq_length {max_seq_length}. Ignoring example."
        )
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log_chosen": message_log_chosen,
        "length_chosen": length_chosen,
        "message_log_rejected": message_log_rejected,
        "length_rejected": length_rejected,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    """Setup training and validation datasets."""
    print("\n▶ Setting up data...")

    data = load_preference_dataset(data_config)
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]

    print(f"  ✓ Training dataset loaded with {len(train_dataset)} samples.")
    if val_dataset:
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        task_spec,
        simpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    if "val_data_paths" in data_config and data_config["val_data_paths"]:
        val_dataset = {}
        assert isinstance(data_config["val_data_paths"], dict)
        val_data_paths = data_config["val_data_paths"]

        for val_dataset_name, val_dataset_path in val_data_paths.items():
            assert val_dataset_name not in val_dataset
            val_data = PreferenceDataset(val_dataset_path)
            print(
                f"  ✓ Validation dataset '{val_dataset_name}' loaded with {len(val_data.formatted_ds['train'])} samples."
            )
            val_dataset[val_dataset_name] = AllTaskProcessedDataset(
                val_data.formatted_ds["train"],
                tokenizer,
                val_data.task_spec,
                simpo_preprocessor,
                max_seq_length=data_config["max_input_seq_length"],
            )
    else:
        val_dataset = (
            {
                "default": AllTaskProcessedDataset(
                    val_dataset,
                    tokenizer,
                    task_spec,
                    simpo_preprocessor,
                    max_seq_length=data_config["max_input_seq_length"],
                )
            }
            if val_dataset
            else {}
        )

    return train_dataset, val_dataset, task_spec


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple:
    """Setup SimPO training components."""
    assert not master_config["policy"]["dynamic_batching"]["enabled"], (
        "Dynamic batching is not supported with SimPO."
    )
    assert not master_config["policy"]["sequence_packing"]["enabled"], (
        "Sequence packing is not supported with SimPO."
    )

    set_seed(master_config["simpo"]["seed"])

    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    simpo_config = master_config["simpo"]

    # Logger
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # Checkpointing
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    simpo_save_state: Optional[SimPOSaveState] = cast(
        Optional[SimPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    # Data
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=partial(
            preference_collate_fn,
            tokenizer=tokenizer,
            make_sequence_length_divisible_by=policy_config[
                "make_sequence_length_divisible_by"
            ],
            add_loss_mask=True,
        ),
        drop_last=True,
        num_workers=data_config["num_workers"],
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = {
        k: StatefulDataLoader(
            v,
            batch_size=simpo_config["val_global_batch_size"],
            shuffle=False,
            collate_fn=partial(
                preference_collate_fn,
                tokenizer=tokenizer,
                make_sequence_length_divisible_by=policy_config[
                    "make_sequence_length_divisible_by"
                ],
                add_loss_mask=True,
            ),
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
        for k, v in val_dataset.items()
    }

    # Cluster
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="simpo_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # Model (NOTE: SimPO doesn't need reference model!)
    print("\n▶ Setting up model...")
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            simpo_config["max_num_steps"],
            simpo_config["max_num_epochs"] * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters * 2

        if "scheduler" in policy_config["megatron_cfg"]:
            for k in policy_config["megatron_cfg"]["scheduler"]:
                if "iters" in k:
                    policy_config["megatron_cfg"]["scheduler"][k] *= 2

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=False,  # SimPO doesn't need reference model!
    )
    policy.print_node_ip_and_gpu_id()

    # SimPO Loss Function
    loss_fn = SimPOLossFn(master_config["simpo"])
    print("  ✓ Model and SimPO loss initialized (no reference model needed!)")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        simpo_save_state,
        master_config,
    )


def simpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    simpo_save_state: SimPOSaveState,
) -> None:
    """Run SimPO training loop."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if simpo_save_state is None:
        simpo_save_state = _default_simpo_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
        total_valid_tokens = 0
    else:
        current_epoch = simpo_save_state["epoch"]
        current_step = simpo_save_state["step"]
        total_steps = simpo_save_state["total_steps"]
        total_valid_tokens = simpo_save_state.get("total_valid_tokens", 0)

    simpo_config = master_config["simpo"]
    val_period = simpo_config["val_period"]
    val_at_start = simpo_config["val_at_start"]
    max_num_epochs = simpo_config["max_num_epochs"]

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["simpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in iter(train_dataloader):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['simpo']['max_num_steps'])} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)

            with timer.time("total_step_time"):
                print("▶ Taking a training step (SimPO - no reference model!)...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        batch,
                        loss_fn,
                        eval_mode=False,
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                    )

                is_last_step = total_steps + 1 >= master_config["simpo"][
                    "max_num_steps"
                ] or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                # Checkpointing
                simpo_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    simpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    simpo_save_state["total_steps"] = total_steps + 1
                    simpo_save_state["epoch"] = current_epoch
                    simpo_save_state["total_valid_tokens"] = total_valid_tokens

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, simpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results (SimPO):")
            print(f"  • loss: {float(metrics['loss']):.4f}")
            print(f"  • accuracy: {float(metrics.get('accuracy', 0)):.4f}")
            print(f"  • margin: {float(metrics.get('margin', 0)):.4f}")
            print(f"  • rewards_chosen: {float(metrics.get('rewards_chosen_mean', 0)):.4f}")
            print(f"  • rewards_rejected: {float(metrics.get('rewards_rejected_mean', 0)):.4f}")

            print("\n⏱️  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics.get("global_valid_toks", 0) / max(total_time, 1) / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout reached, stopping training", flush=True)
                return
            if total_steps >= master_config["simpo"]["max_num_steps"]:
                print("Max steps reached, stopping training", flush=True)
                return

        current_epoch += 1
        current_step = 0


def main():
    """Main entry point for SimPO training."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            "nemo_simpo_7b_test.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("\n" + "=" * 60)
    print(" " * 15 + "SimPO TRAINING")
    print(" " * 10 + "(Reference-Free Preference Optimization)")
    print("=" * 60)
    print("\nFinal config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    train_dataset, val_dataset, task_spec = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        simpo_save_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    simpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        simpo_save_state,
    )


if __name__ == "__main__":
    main()
