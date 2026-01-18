import os
import json
import argparse
from unittest.mock import patch
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_vllm import evaluate_vllm

# --- Helper: Prompt Formatting ---
def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        # Fallback if file missing
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "User: {question}\n"
            "Assistant: <think>"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template: str, question: str) -> str:
    """Applies the R1-Zero template to the raw math question."""
    return template.format(question=question.strip())

# --- vLLM Helper Functions ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.5):
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
        )

def load_data(data_path: str):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        # Check first char to see if it's a list '[' or a dict '{'
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Case A: It is a JSON List (Your format)
            print("Detected JSON List format.")
            data = json.load(f)
        else:
            # Case B: It is JSONL (Line-by-line)
            print("Detected JSONL format.")
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="huggingface/Qwen2.5-Math-1.5B")
    parser.add_argument("--val_data_path", type=str, default="huggingface/sft-cs336-assign5-datasets/sft-reason/val.jsonl")
    parser.add_argument("--res_json_path", type=str, default="result/math_baseline/results.jsonl")
    # Path to your prompt file
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
        
    # Hyperparameters
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    # 1. Load Prompt Template
    template = load_prompt_template(args.prompt_template_path)
    print("Loaded Prompt Template.")

    # 2. Load Eval Data
    eval_data = load_data(args.val_data_path)

    # 3. Prepare Prompts and GTs
    data_prompts = []
    data_gts = []
    for data in eval_data:
        # Use raw 'problem' and format it
        q = data.get('problem', data.get('query', ''))
        gt = data.get('expected_answer', data.get('answer', data.get('solution', '')))

        prompt = format_prompt(template, q)
        data_prompts.append(prompt)
        data_gts.append(gt)

    print(f"Loaded {len(data_prompts)} data samples.")
    print("Sample Prompt:", data_prompts[0])
    print("Sample GT:", data_gts[0])


    # 4. Init vLLM (GPU 1)
    print("Initializing vLLM on cuda:1...")
    device_eval = "cuda" # Use all available GPUs
    llm = init_vllm(args.base_model, device_eval, args.seed)

    # 5. Evaluate
    data_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=data_prompts,
        ground_truths=data_gts,
        eval_sampling_params=data_sampling_params,
        out_jsonl_path=args.res_json_path,
        fast=True
    )
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()