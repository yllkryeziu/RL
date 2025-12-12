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

"""SimPO (Simple Preference Optimization) loss function.

SimPO is a reference-free preference optimization algorithm that uses
average log probability as the implicit reward, eliminating the need
for a reference model.

Paper: https://arxiv.org/abs/2405.14734
"""

from typing import Any, Optional, TypedDict, TypeVar

import torch
import torch.distributed

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.loss_functions import NLLLoss, PreferenceLoss, masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_logprobs,
    get_logprobs_from_vocab_parallel_logits,
)

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class SimPOLossConfig(TypedDict):
    """Configuration for SimPO loss function."""

    beta: float  # Scaling factor for reward difference (typically 2.0)
    gamma: float  # Margin term (typically 0.3-1.0)
    preference_loss_weight: float  # Weight for preference loss
    sft_loss_weight: float  # Weight for optional SFT loss on chosen responses


class SimPOLossDataDict(TypedDict):
    """Required keys for the SimPO loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class SimPOLossFn(PreferenceLoss):
    """SimPO (Simple Preference Optimization) loss function.

    SimPO is a reference-free variant of DPO that uses the average log probability
    of the policy as the implicit reward, rather than the log probability ratio
    with a reference model. This eliminates the need for a reference model,
    reducing memory usage and simplifying training.

    The SimPO loss is computed as:
    L(θ) = -E[log(σ(β * (r_chosen - r_rejected - γ)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (typically 2.0)
    - γ is a margin term (typically 0.3-1.0)
    - r = (1/n) * Σ_t log π_θ(a_t|s_t) is the average log probability

    The key differences from DPO:
    1. No reference model: rewards are computed directly from policy log probs
    2. Length normalization: log probs are averaged over tokens (not summed)
    3. Margin term γ: encourages larger reward gaps between chosen/rejected

    Args:
        cfg (SimPOLossConfig): Configuration dictionary containing:
            - beta (float): Scaling factor for reward difference (default: 2.0)
            - gamma (float): Margin term (default: 0.3)
            - preference_loss_weight (float): Weight for preference loss (default: 1.0)
            - sft_loss_weight (float): Weight for SFT loss on chosen (default: 0.0)
    """

    def __init__(self, cfg: SimPOLossConfig):
        self.beta = cfg.get("beta", 2.0)
        self.gamma = cfg.get("gamma", 0.3)
        self.preference_loss_weight = cfg.get("preference_loss_weight", 1.0)
        self.sft_loss_weight = cfg.get("sft_loss_weight", 0.0)
        self.sft_loss = NLLLoss()

        self.loss_type = LossType.SEQUENCE_LEVEL

    def _compute_rewards(
        self,
        token_logprobs: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        """Compute SimPO rewards (average log probability per sequence).

        Args:
            token_logprobs: Log probabilities for each token [batch_size, seq_len]
            token_mask: Mask for valid tokens [batch_size, seq_len]

        Returns:
            rewards: Average log probability per sequence [batch_size]
        """
        # Mask the log probs
        masked_logprobs = token_logprobs * token_mask

        # Sum log probs per sequence
        summed_logprobs = masked_logprobs.sum(dim=-1)

        # Count valid tokens per sequence
        num_tokens = token_mask.sum(dim=-1).clamp(min=1)

        # Average log probability (length normalization is key for SimPO)
        rewards = summed_logprobs / num_tokens

        return rewards

    def _simpo_loss(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[SimPOLossDataDict],
        global_valid_seqs: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute SimPO preference loss.

        Returns:
            tuple of (loss, accuracy, rewards_chosen_mean, rewards_rejected_mean)
        """
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        seq_index = data.get("seq_index", None)

        next_token_logits = next_token_logits.to(torch.float32)

        # Compute log probabilities
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Compute rewards (average log probability per sequence)
        rewards = self._compute_rewards(token_logprobs, token_mask)

        # Split into chosen and rejected (interleaved: chosen, rejected, chosen, rejected, ...)
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)

        # Compute SimPO loss with margin
        # L = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))
        rewards_delta = rewards_chosen - rewards_rejected - self.gamma

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(self.beta * rewards_delta)
            * sample_mask[::2]
        )

        # Average over valid samples
        preference_loss = masked_mean(
            per_sample_loss,
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        )

        # Compute accuracy
        accuracy = masked_mean(
            (rewards_chosen > rewards_rejected).float(),
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        )

        # Compute mean rewards for logging
        rewards_chosen_mean = masked_mean(
            rewards_chosen,
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        )
        rewards_rejected_mean = masked_mean(
            rewards_rejected,
            sample_mask[1::2],
            global_normalization_factor=global_valid_seqs / 2,
        )

        return preference_loss, accuracy, rewards_chosen_mean, rewards_rejected_mean

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[SimPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute SimPO loss.

        Args:
            next_token_logits: Logits from the model
            data: Batch data containing input_ids, token_mask, sample_mask
            global_valid_seqs: Number of valid sequences for normalization
            global_valid_toks: Number of valid tokens for normalization
            vocab_parallel_rank: Vocab parallel rank (for tensor parallelism)
            vocab_parallel_group: Vocab parallel process group
            context_parallel_group: Context parallel process group

        Returns:
            tuple of (loss, metrics_dict)
        """
        # Optional SFT loss on chosen responses
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logits,
                data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                dpo_loss=True,
                dpo_average_log_probs=True,  # Always average for SimPO
            )
            sft_loss_chosen, _ = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(
                sft_loss_chosen,
                data["sample_mask"][::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        # Compute SimPO preference loss
        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._simpo_loss(
            next_token_logits,
            data,
            global_valid_seqs,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )

        # Combine losses
        total_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        # Compute margin for logging
        margin = rewards_chosen_mean - rewards_rejected_mean

        num_valid_samples = data["sample_mask"].sum() / 2

        return total_loss, {
            "loss": total_loss.item(),
            "sft_loss": sft_loss_chosen.item() if torch.is_tensor(sft_loss_chosen) else sft_loss_chosen,
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "margin": margin.item(),
            "num_valid_samples": num_valid_samples.item(),
        }
