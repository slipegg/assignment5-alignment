import os
import re
import json
import argparse
from typing import Callable, Dict, List, Optional, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn  # type: ignore
from collections import Counter

def extract_answer_tag(response: str) -> str:
    """
    For analysis only: pull content inside <answer>...</answer> if present.
    """
    if not response:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], Dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams,
        out_jsonl_path: str,
        fast: bool = True,
) -> Dict[str, Any]:
    assert len(prompts) == len(ground_truths)

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)
    
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    n = len(prompts)
    sum_format = 0.0
    sum_answer = 0.0
    sum_reward = 0.0
    sum_correct_avg_length = 0.0
    sum_incorrect_avg_length = 0.0
    combo_counts = Counter()

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text if out.outputs else ""
            gt = ground_truths[i] if ground_truths[i] is not None else ""

            scores = reward_fn(gen_text, gt, fast=fast)  # type: ignore

            fr = int(scores.get("format_reward", 0.0) >= 0.5)
            ar = int(scores.get("answer_reward", 0.0) >= 0.5)

            combo_counts[(fr, ar)] += 1

            sum_format += float(scores.get("format_reward", 0.0))
            sum_answer += float(scores.get("answer_reward", 0.0))
            sum_reward += float(scores.get("reward", 0.0))
            if fr == 1 and ar == 1:
                sum_correct_avg_length += len(gen_text)
            else:
                sum_incorrect_avg_length += len(gen_text)

            rec = {
                "idx": i,
                "prompt": prompts[i],
                "generation": gen_text,
                "pred_answer": extract_answer_tag(gen_text),
                "gold_final": gt,
                "scores": scores,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    combo_table = {
        "format=1 answer=1": combo_counts[(1, 1)],
        "format=1 answer=0": combo_counts[(1, 0)],
        "format=0 answer=0": combo_counts[(0, 0)],
        "format=0 answer=1": combo_counts[(0, 1)],  # 理论上应为 0
    }

    metrics = {
        "n": n,
        "format_rate": (sum_format / n) if n else 0.0,
        "answer_accuracy": (sum_answer / n) if n else 0.0,
        "reward_mean": (sum_reward / n) if n else 0.0,
        "counts": combo_table,
        "correct_avg_length": (sum_correct_avg_length / combo_counts[(1, 1)]) if combo_counts[(1, 1)] else 0.0,
        "incorrect_avg_length": (sum_incorrect_avg_length / (n - combo_counts[(1, 1)])) if (n - combo_counts[(1, 1)]) else 0.0,
    }
    return metrics