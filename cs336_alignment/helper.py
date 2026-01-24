from typing import Any, Dict, List, Optional, Tuple, Callable, Literal
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F

# test by `uv run pytest -k test_tokenize_prompt_and_output`
def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    batch_size = len(prompt_strs)

    # Tokenize prompts and outputs without adding special tokens
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_ids_list = prompt_tokens["input_ids"]
    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    output_ids_list = output_tokens["input_ids"]
    full_ids_list = [prompt + output for prompt, output in zip(prompt_ids_list, output_ids_list)]

    # Pad sequences
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_full_len = max(len(ids) for ids in full_ids_list)
    full_padded_list = torch.full(
        (batch_size, max_full_len),
        pad_id,
        dtype=torch.long
    )
    for i, ids in enumerate(full_ids_list):
        full_padded_list[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Shift to create labels
    input_ids = full_padded_list[:, :-1].contiguous()
    labels = full_padded_list[:, 1:].contiguous()

    # create response_mask
    response_mask = torch.zeros_like(labels, dtype=torch.long)
    for i in range(batch_size):
        prompt_len = len(prompt_ids_list[i])
        output_len = len(output_ids_list[i])
        if output_len == 0:
            continue
        
        start = max(prompt_len-1,0)
        end = min(prompt_len + output_len - 1, labels.size(1))
        if end > start:
            response_mask[i, start:end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

# test by `uv run pytest -k test_compute_entropy`
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Computes the entropy of the categorical distribution defined by the logits.
    Args:
        logits: A tensor of shape (batch_size, seq_length, vocab_size) representing the logits.
    Returns:
        A tensor of shape (batch_size, seq_length) representing the entropy at each position.
    """
    # log Z = log sum exp logits
    log_z = torch.logsumexp(logits, dim=-1)  # (batch_size, seq_length)

    # sum p(x) log p(x) = sum exp(logits - log_z) * (logits - log_z)
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_length, vocab_size)

    # expected_logit = sum p(x) * log p(x)
    expected_logit = torch.sum(probs * logits, dim=-1)  # (batch_size, seq_length)

    # entropy = log Z - expected_logit
    entropy = log_z - expected_logit  # (batch_size, seq_length)

    return entropy

# test by `uv run pytest -k test_get_response_log_probs`
def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    # Forward to get logits (Batch, Seq Len, Vocab Size)
    logits = model(input_ids).logits

    logits_softmax = F.log_softmax(logits, dim=-1)  # (Batch, Seq Len, Vocab Size)

    log_probs = torch.gather(
        logits_softmax,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (Batch, Seq Len)

    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result

# test by `uv run pytest -k test_masked_normalize`
def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    Args:  
        tensor: torch.Tensor The tensor to sum and normalize.  
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.  
        normalize_constant: float the constant to divide by for normalization.  
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.  
    Returns:  torch.Tensor
    """
    masked_tensor = tensor * mask
    
    if dim is None:
        summed = torch.sum(masked_tensor)
    else:
        summed = torch.sum(masked_tensor, dim=dim)

    normalized = summed / normalize_constant
    return normalized

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    One SFT microbatch step: masked NLL, batch-mean, normalize, grad-acc scaling, backward.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:  
        tuple[torch.Tensor, dict[str, torch.Tensor]].
    """
    per_example_nll = masked_normalize(policy_log_probs, response_mask, normalize_constant, 1) # (batch_size, Sequence_length) -> (batch_size,)

    # batch mean
    microbatch_loss = -1 * per_example_nll.mean()  # scalar

    # scale for gradient accumulation
    loss = microbatch_loss / float(gradient_accumulation_steps)

    # backward
    loss.backward()

    metadata = {
        "microbatch_loss": microbatch_loss.detach(),
    }

    return loss, metadata

################ REINFORCEMENT LEARNING HELPERS ################

# test by `uv run pytest -k test_compute_group_normalized_rewards`
def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], Dict[str, float]],
        rollout_responses: List[str],
        repeated_ground_truths: List[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute per-response rewards and group-normalized advantages.

    Args:
        reward_fn: Callable[[response, ground_truth], dict] returning keys:
                  "reward", "format_reward", "answer_reward"
        rollout_responses: list[str], length = rollout_batch_size
        repeated_ground_truths: list[str], length = rollout_batch_size
        group_size: int, number of responses per question
        advantage_eps: float, small constant to avoid division by zero
        normalize_by_std: bool, if True use (r-mean)/(std+eps) else (r-mean)

    Returns:
        advantages: torch.Tensor, shape (rollout_batch_size,)
        raw_rewards: torch.Tensor, shape (rollout_batch_size,)
        metadata: dict[str, float] with useful stats
    """
    assert len(rollout_responses) == len(repeated_ground_truths), "Inconsistent rollout responses and ground truths"

    # Compute raw rewards
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(response, ground_truth)
        raw_rewards.append(reward["reward"])
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    # Compute group-normalized advantages
    group_num = len(rollout_responses) // group_size
    reward_by_group = raw_rewards.view(group_num, group_size)
    reward_mean = reward_by_group.mean(dim=1, keepdim=True)
    if normalize_by_std:
        reward_std = reward_by_group.std(dim=1, keepdim=True)
        advantages = (reward_by_group - reward_mean) / (reward_std + advantage_eps)
    else:
        advantages = reward_by_group - reward_mean

    advantages = advantages.view(-1)

    return advantages, raw_rewards, {}

# test by `uv run pytest -k test_compute_naive_policy_gradient_loss`
def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token naive policy gradient loss:
        loss_{b,t} = -A_b * log_prob_{b,t}

    Args:
        raw_rewards_or_advantages: Tensor of shape (batch_size, 1)
        policy_log_probs: Tensor of shape (batch_size, sequence_length)

    Returns:
        Tensor of shape (batch_size, sequence_length)
    """
    assert raw_rewards_or_advantages.dim() == 1 or raw_rewards_or_advantages.dim() == 2, "raw_rewards_or_advantages must be 1D or 2D"
    assert policy_log_probs.dim() == 2, "policy_log_probs must be 2D"
    assert raw_rewards_or_advantages.size(0) == policy_log_probs.size(0), "Batch size mismatch"

    advantages = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)

    return -1 * advantages * policy_log_probs

# test by `uv run pytest -k test_compute_grpo_clip_loss`
def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        loss_type: Literal["grpo_clip", "grpo_no_clip"] = "grpo_clip",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Per-token GRPO-Clip loss:
        loss_{b,t} = - min( r_{b,t} * A_b, clip(r_{b,t}, 1-eps, 1+eps) * A_b )

    where r_{b,t} = exp(policy_log_probs - old_log_probs).

    Args:
        advantages: (B, 1)
        policy_log_probs: (B, T)
        old_log_probs: (B, T)
        cliprange: eps

    Returns:
        loss: (B, T)
        metadata: dict of tensors (e.g., is_clipped mask)
    """
    advantages = advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)

    ratios = torch.exp(policy_log_probs - old_log_probs)  # (B, T)
    if loss_type == "grpo_clip":
        unclipped_loss = ratios * advantages  # (B, T)
        clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)  # (B, T)
        clipped_loss = clipped_ratios * advantages  # (B, T)

        loss = -1 * torch.min(unclipped_loss, clipped_loss)  # (B, T)

        is_clipped = (clipped_loss < unclipped_loss).float()  # (B, T)

        metadata = {
            "is_clipped": is_clipped,
        }
        return loss, metadata
    elif loss_type == "grpo_no_clip":
        loss = -1 * ratios * advantages  # (B, T)
        metadata = {}
        return loss, metadata
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convenience wrapper to compute different policy-gradient losses.

    Args:
        policy_log_probs: (B, T) log-probabilities from current policy.
        loss_type: "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
        raw_rewards: required if loss_type == "no_baseline", shape (B, 1)
        advantages: required if loss_type in {"reinforce_with_baseline", "grpo_clip"}, shape (B, 1)
        old_log_probs: required if loss_type == "grpo_clip", shape (B, T)
        cliprange: required if loss_type == "grpo_clip", scalar eps

    Returns:
        loss: (B, T) per-token loss
        metadata: dict[str, Tensor] auxiliary stats
    """
    assert loss_type in {"no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"}, f"Unknown loss_type: {loss_type}"
    assert policy_log_probs.dim() == 2, "policy_log_probs must be 2D"

    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip" or loss_type == "grpo_no_clip":
        loss, pg_metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
            loss_type=loss_type,
        )
        metadata.update(pg_metadata)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss, metadata

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Compute mean of `tensor` considering only elements where mask == 1.

    Args:
        tensor: torch.Tensor, data to average
        mask: torch.Tensor, same shape as tensor; 1/True positions included
        dim: int or None. If None, mean over all masked elements.

    Returns:
        torch.Tensor, masked mean with semantics like tensor.mean(dim)
    """
    assert tensor.shape == mask.shape, "tensor and mask must have the same shape"
    
    if dim is None:
        masked_sum = torch.sum(tensor * mask)
        length = torch.sum(mask)
        return masked_sum / (length)
    else:
        masked_sum = torch.sum(tensor * mask, dim=dim)
        length = torch.sum(mask, dim=dim)
        return masked_sum / (length)

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        length_norm: Literal["masked_mean", "masked_normalize"] = "masked_mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute one GRPO microbatch forward+backward step.

    Steps:
      1) compute per-token PG loss (B,T)
      2) masked_mean over response tokens -> per-example loss (B,)
      3) batch mean -> microbatch_loss (scalar)
      4) scale by gradient_accumulation_steps
      5) backward
    """
    # 1) compute per-token PG loss
    pg_loss, pg_metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )  # (B, T)

    # 2) masked_mean over response tokens -> per-example loss (B,)
    if length_norm == "masked_mean":
        per_example_loss = masked_mean(pg_loss, response_mask, dim=1)  # (B,)
    elif length_norm == "masked_normalize":
        per_example_loss = masked_normalize(pg_loss, response_mask, normalize_constant=1024, dim=1)  # (B,)
    else:
        raise ValueError(f"Unknown length_norm: {length_norm}")
    
    # 3) batch mean -> microbatch_loss (scalar)
    microbatch_loss = per_example_loss.mean()  # scalar

    # 4) scale by gradient_accumulation_steps
    loss = microbatch_loss / float(gradient_accumulation_steps)

    # 5) backward
    loss.backward()

    metadata = {
        "microbatch_loss": microbatch_loss.detach(),
        "per_example_loss_mean": per_example_loss.detach().mean(),
        "per_example_loss_std": per_example_loss.detach().std(unbiased=False) if per_example_loss.numel() > 1 else torch.zeros((), device=per_example_loss.device)
    }
    metadata.update(pg_metadata)

    return loss, metadata
