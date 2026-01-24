import os
import json
import argparse
import random
import torch
import wandb
from pathlib import Path
from unittest.mock import patch

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.helper import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_vllm import evaluate_vllm


def _checkpoint_dir(base_dir: str) -> Path:
    ckpt_dir = Path(base_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_checkpoint(
    ckpt_base_dir: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    max_to_keep: int = 3,
) -> Path:
    if max_to_keep < 1:
        return
    """Save a training checkpoint and keep only the latest `max_to_keep` ones."""
    ckpt_dir = _checkpoint_dir(ckpt_base_dir)
    ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch}_step{global_step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Model + tokenizer (HF format)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)

    # Optimizer + metadata
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        ckpt_path / "trainer_state.pt",
    )

    # Rotate old checkpoints
    all_ckpts = sorted(
        [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")],
        key=lambda p: p.stat().st_mtime,
    )
    if len(all_ckpts) > max_to_keep:
        for p in all_ckpts[: len(all_ckpts) - max_to_keep]:
            for child in p.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            # remove empty dirs bottom-up
            for child in sorted([c for c in p.rglob("*") if c.is_dir()], key=lambda x: len(str(x)), reverse=True):
                try:
                    child.rmdir()
                except OSError:
                    pass
            try:
                p.rmdir()
            except OSError:
                pass

    return ckpt_path

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

def sample_formatted_eval_data(val_data, val_num, template):
    problems = []
    expected_answers = []
    for data in val_data:
        problem = data.get('problem', data.get('query', ''))
        formatted_problem = format_prompt(template, problem)
        answer = data.get('expected_answer', data.get('answer', data.get('solution', '')))
        problems.append(formatted_problem)
        expected_answers.append(answer)
        if len(problems) >= val_num:
            break
    return problems, expected_answers

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    print("Loading new policy weights into vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
    print("Weights loaded.")

def main():
    parser = argparse.ArgumentParser()
    # Model & Data
    parser.add_argument("--base_model", type=str, default="./huggingface/Qwen2.5-Math-1.5B")
    parser.add_argument("--sft_data_path", type=str, default="./huggingface/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl")
    parser.add_argument("--val_data_path", type=str, default="./huggingface/sft-cs336-assign5-datasets/sft-reason/val.jsonl")
    parser.add_argument("--val_output_dir", type=str, default="./result/sft/")
    # Path to your prompt file
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
    
    # Experiment Settings
    parser.add_argument("--max_examples", type=int, default=-1, help="If > 0, subsample the dataset")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--normalize_constant", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    parser.add_argument("--eval_batch_steps", type=int, default=10)
    parser.add_argument("--eval_max_examples", type=int, default=64)

    # Checkpointing
    parser.add_argument("--checkpoint_base_dir", type=str, default="./result/checkpoints/sft/")
    parser.add_argument("--checkpoint_max_to_keep", type=int, default=3)

    # Other
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging")
    group.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")

    
    args = parser.parse_args()
    args.enable_wandb = (not args.disable_wandb)

    # 0. Init
    if args.enable_wandb:
        wandb.init(
            project="cs336-assignment5-alignment",
            name=f"sft-vllm-r1-zero-{os.path.basename(args.base_model)}",
            config=vars(args)
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Prompt Template
    print("Loading Prompt Template.")
    template = load_prompt_template(args.prompt_template_path)

    # 2. Load SFT Data
    print("Loading SFT Data...")
    sft_data = load_data(args.sft_data_path)
    if args.max_examples > 0 and len(sft_data) > args.max_examples:
        sft_data = random.sample(sft_data, args.max_examples)
        print(f"Subsampled SFT data to {args.max_examples} examples.")

    # 3. Load Eval Data
    print("Loading Eval Data...")
    val_data = load_data(args.val_data_path)

    # 3. Init Policy Model (GPU 0) and Optimizer
    print("Initializing Policy Model on cuda:0...")
    policy_device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2"
    ).to(policy_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Init Rollout by vLLM (GPU 1)
    print("Initializing Rollout vLLM on cuda:1...")
    vllm = init_vllm(args.base_model, "cuda:1", args.seed)
    vllm_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # 5. Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch+1}/{args.epochs}...")
        random.shuffle(sft_data)

        batch_loss = 0.0
        batch_entropy = 0.0
        for i in range(0, len(sft_data), args.micro_batch_size):
            # ==== One Micro Batch ====
            batch = sft_data[i:i + args.micro_batch_size]
            if not batch: 
                continue
            
            problem_batch = []
            response_batch = []
            for data in batch:
                problem = data.get('problem', '')
                formatted_problem = format_prompt(template, problem)
                response = data.get('reasoning_trace', '')
                if not problem or not response:
                    continue
                problem_batch.append(formatted_problem)
                response_batch.append(response)
            
            if not problem_batch:
                continue

            # Tokenize & Move to Device
            tokenize_data = tokenize_prompt_and_output(problem_batch, response_batch, tokenizer)
            input_ids = tokenize_data['input_ids'].to(policy_device)
            labels = tokenize_data['labels'].to(policy_device)
            response_mask = tokenize_data['response_mask'].to(policy_device)

            # Forward Pass
            forward_res = get_response_log_probs(model, input_ids, labels, True)
            batch_entropy += forward_res['token_entropy'].mean().item()/float(args.grad_accum_steps)

            # Loss & Backward
            loss, _ = sft_microbatch_train_step(
                forward_res['log_probs'],
                response_mask,
                args.grad_accum_steps,
                args.normalize_constant,
            )
            batch_loss += loss.item()

            global_step += 1

            # One Batch Complete, Optimizer Step
            if (global_step) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                
                print(f"Epoch {epoch+1}, Step {global_step}: Loss = {batch_loss:.4f}, Token Entropy = {batch_entropy:.4f}")
                if args.enable_wandb:
                    wandb.log({
                            "train/loss": batch_loss,
                            "train/token_entropy": batch_entropy,
                        }, 
                        step = global_step // args.grad_accum_steps
                    )

                optimizer.zero_grad()
                batch_loss = 0.0
                batch_entropy = 0.0

                # Evaluation Step
                if (global_step // args.grad_accum_steps) % args.eval_batch_steps == 0:
                    print(f"Evaluating at step {global_step}...")

                    # Save checkpoint before evaluation (so eval corresponds to a saved model)
                    ckpt_path = save_checkpoint(
                        ckpt_base_dir=args.checkpoint_base_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                        max_to_keep=args.checkpoint_max_to_keep,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")

                    load_policy_into_vllm_instance(model, vllm)
                    problems, expected_answers = sample_formatted_eval_data(val_data, args.eval_max_examples, template)
                    
                    metrics = evaluate_vllm(
                        vllm_model=vllm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=problems,
                        ground_truths=expected_answers,
                        eval_sampling_params=vllm_sampling_params,
                        out_jsonl_path = os.path.join(args.val_output_dir, f"eval_step_{(global_step // args.grad_accum_steps) // args.eval_batch_steps}_results.jsonl"),
                        fast=True
                    )

                    if args.enable_wandb:
                        wandb.log({
                            "eval/answer_accuracy": metrics["answer_accuracy"],
                            "eval/format_rate": metrics["format_rate"],
                            "eval/reward_mean": metrics["reward_mean"],
                            "eval/correct_avg_length": metrics["correct_avg_length"],
                            "eval/incorrect_avg_length": metrics["incorrect_avg_length"],
                            "eval/step": (global_step // args.grad_accum_steps) // args.eval_batch_steps,
                        }, step=global_step // args.grad_accum_steps)

    print("Training Complete.")

    # Final checkpoint at end of training
    ckpt_path = save_checkpoint(
        ckpt_base_dir=args.checkpoint_base_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=args.epochs,
        global_step=global_step,
        max_to_keep=args.checkpoint_max_to_keep,
    )
    print(f"Saved final checkpoint to {ckpt_path}")

# uv run scripts/sft.py > result/sft/sft_training.log 2>&1 &
if __name__ == "__main__":
    main()
