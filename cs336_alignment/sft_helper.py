from typing import Any, Dict, List, Optional, Tuple
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
