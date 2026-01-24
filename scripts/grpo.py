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

from cs336_alignment.helper import tokenize_prompt_and_output, get_response_log_probs, compute_group_normalized_rewards, grpo_microbatch_train_step
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
    parser.add_argument("--train_data_path", type=str, default="./huggingface/sft-cs336-assign5-datasets/sft-reason/train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="./huggingface/sft-cs336-assign5-datasets/sft-reason/val.jsonl")
    parser.add_argument("--val_output_dir", type=str, default="./result/grpo/")

    # Path to your prompt file
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
    
    # GRPO Hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--use_std_normalization", action="store_true", default=False)
    parser.add_argument("--length_norm", type=str, default="masked_mean") # "masked_mean" or "masked_normalize"

    # Loss Type: "no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"
    parser.add_argument("--loss_type", type=str, default="grpo_clip")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--eval_batch_steps", type=int, default=10)
    parser.add_argument("--eval_max_examples", type=int, default=64)

    # Checkpointing
    parser.add_argument("--checkpoint_base_dir", type=str, default="./result/checkpoints/grpo/")
    parser.add_argument("--checkpoint_max_to_keep", type=int, default=3)

    # Other
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging")
    group.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")

    
    args = parser.parse_args()
    args.enable_wandb = (not args.disable_wandb)
    
    # Sanity Checks (from assignment PDF)
    assert args.train_batch_size % args.grad_accum_steps == 0
    micro_batch_size = args.train_batch_size // args.grad_accum_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size

    # 0. Init
    if args.enable_wandb:
        wandb.init(
            project="cs336-assignment5-alignment",
            name=f"grpo-vllm-r1-zero-{os.path.basename(args.base_model)}",
            config=vars(args)
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Prompt Template
    print("Loading Prompt Template.")
    template = load_prompt_template(args.prompt_template_path)

    # 2. Load Training Question Data
    print("Loading Training Question Data...")
    all_questions = load_data(args.train_data_path)

    # 3. Load Eval Data
    print("Loading Eval Data...")
    val_data = load_data(args.val_data_path)

    # 4. Init Policy Model (GPU 0) and Optimizer
    print("Initializing Policy Model on cuda:0...")
    policy_device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2"
    ).to(policy_device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # 5. Init Rollout by vLLM (GPU 1)
    print("Initializing Rollout vLLM on cuda:1...")
    vllm = init_vllm(args.base_model, "cuda:1", args.seed)
    rollout_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True,
        n=args.group_size
    )
    eval_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # 6. Training Loop
    ''' Default: 
        n_grpo_steps: 200
            rollout_batch_size = 256, group_size = 8 -> n_prompts_per_rollout = 32
            rollout_outputs = 256, 8 response/group
            advatages:(256,)
            
            epochs_per_rollout_batch: 1
                micro_batch_size = 1
                grad_accum_steps = rollout_outputs = 256
                Therefore, once all rollout_outputs have been processed, which constitutes one batch, parameter updates are performed.

            eval_batch_steps = 10, every 10 GRPO steps do evaluation
    '''
    print("Starting GRPO Training Loop...")
    batch_step = 0
    for step in range(1, args.n_grpo_steps + 1):
        print(f"======GRPO Step {step}/{args.n_grpo_steps}======")

        ## 6.1 Sample questions
        one_batch_questions = random.sample(all_questions, n_prompts_per_rollout)
        one_batch_prompted_questions = [format_prompt(template, q.get('problem')) for q in one_batch_questions]
        ground_truths = [q.get('expected_answer') for q in one_batch_questions]

        ## 6.2 Rollout answer
        load_policy_into_vllm_instance(policy, vllm)
        print(f"Rollouting {args.rollout_batch_size} questions...")
        rollout_outputs = vllm.generate(one_batch_prompted_questions, rollout_sampling_params)

        ## 6.3 Process rollout result to get advantages
        rollout_responses = []
        repeated_questions = []
        repeated_ground_truths = []
        for i, one_question_responses in enumerate(rollout_outputs):
            for response in one_question_responses.outputs:
                rollout_responses.append(response.text)
                repeated_questions.append(one_batch_prompted_questions[i])
                repeated_ground_truths.append(ground_truths[i])

        advantages, raw_rewards, meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        if args.enable_wandb:
            wandb.log({
                "rollout/raw_reward_mean": meta["raw_reward_mean"],
                "rollout/raw_reward_std": meta["raw_reward_std"],
                "rollout/raw_format_reward_mean": meta["raw_format_reward_mean"],
                "rollout/raw_answer_reward_mean": meta["raw_answer_reward_mean"],
                "step": step,
            }, step=batch_step)

        advantages = advantages.unsqueeze(-1)
        raw_rewards = raw_rewards.unsqueeze(-1)

        ## 6.4 Compute old logprobs
        old_log_probs = None
        if args.loss_type == "grpo_clip" or args.loss_type == "grpo_no_clip":
            print("Computing old log probabilities...")
            policy.eval()
            with torch.no_grad():
                old_log_probs_list = []
                for i in range(0, len(repeated_questions), micro_batch_size):
                    micro_batch_questions = repeated_questions[i:i + micro_batch_size]
                    micro_batch_responses = rollout_responses[i:i + micro_batch_size]

                    tokenized_result = tokenize_prompt_and_output(micro_batch_questions, micro_batch_responses, tokenizer)
                    input_ids = tokenized_result['input_ids'].to(policy_device)
                    labels = tokenized_result['labels'].to(policy_device)

                    policy_response = get_response_log_probs(policy, input_ids, labels)
                    old_log_probs_list.append(policy_response['log_probs'].cpu())

                max_seq_len = max([lp.size(1) for lp in old_log_probs_list])
                padded = []
                for log_probs in old_log_probs_list:
                    if log_probs.size(1) < max_seq_len:
                        log_probs = torch.nn.functional.pad(log_probs, (0, max_seq_len - log_probs.size(1)), value = 0.0)
                    padded.append(log_probs)
                old_log_probs = torch.cat(padded, dim=0)

        ## 6.5 Training
        policy.train()
        dataset_indices = list(range(len(rollout_responses)))
        # On-policy: epochs_per_rollout_batch = 1, Off-polocy: epochs_per_rollout_batch > 1
        for epoch in range(1, args.epochs_per_rollout_batch+1):
            random.shuffle(dataset_indices)

            acc_loss = 0.0
            acc_entropy = 0.0
            finished_micro_batch_num = 0
            for i in range(0, len(dataset_indices), micro_batch_size):
                ### One Micro Batch Training ###

                # Pre process Micro Batch Data
                micro_train_data_indices = dataset_indices[i: i + micro_batch_size]
                micro_batch_questions = [repeated_questions[idx] for idx in micro_train_data_indices]
                micro_batch_responses = [rollout_responses[idx] for idx in micro_train_data_indices]
                micro_batch_advantages = advantages[micro_train_data_indices].to(policy_device)
                micro_batch_raw_rewards = raw_rewards[micro_train_data_indices].to(policy_device)

                tokenized_result = tokenize_prompt_and_output(micro_batch_questions, micro_batch_responses, tokenizer)
                input_ids = tokenized_result['input_ids'].to(policy_device)
                labels = tokenized_result['labels'].to(policy_device)
                response_mask = tokenized_result['response_mask'].to(policy_device)

                # Forward Pass by Latest Policy
                forward_res = get_response_log_probs(policy, input_ids, labels, True)
                policy_log_probs = forward_res["log_probs"]
                token_entropy = forward_res["token_entropy"]

                micro_batch_old_log_probs = None
                if old_log_probs is not None:
                    micro_batch_old_log_probs = old_log_probs[micro_train_data_indices]
                    # 下面这两行有什么用
                    T = policy_log_probs.size(1)
                    micro_batch_old_log_probs = micro_batch_old_log_probs[:, :T].to(policy_device)

                # Loss
                loss, loss_meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.grad_accum_steps,
                    loss_type=args.loss_type,
                    raw_rewards=micro_batch_raw_rewards,
                    advantages=micro_batch_advantages,
                    old_log_probs=micro_batch_old_log_probs,
                    cliprange=args.cliprange,
                    length_norm=args.length_norm
                )
                acc_loss += loss.item()
                finished_micro_batch_num += 1

                mask_float = response_mask.float()
                num_tokens = mask_float.sum().item()
                if num_tokens > 0:
                    acc_entropy += (token_entropy * mask_float).sum().item() / num_tokens / args.grad_accum_steps

                if finished_micro_batch_num >= args.grad_accum_steps or (i + micro_batch_size) >= len(dataset_indices):
                    # One Batch Complete, Optimizer Step
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    
                    optimizer.step()
                    batch_step += 1

                    print(f"GRPO Step {step}/{args.n_grpo_steps}, Epoch {epoch}/{args.epochs_per_rollout_batch}, Batch Step {batch_step}: Loss = {acc_loss:.4f}, Token Entropy = {acc_entropy:.4f}")
                    if args.enable_wandb:
                        wandb.log({
                                "train/loss": acc_loss,
                                "train/token_entropy": acc_entropy,
                                "train/grad_norm": grad_norm.item(),
                                "step": step,
                            }, 
                            step = batch_step
                        )

                    optimizer.zero_grad()
                    acc_loss = 0.0
                    acc_entropy = 0.0
                    finished_micro_batch_num = 0

                    del input_ids, labels, response_mask
                    del policy_log_probs, token_entropy
                    del forward_res, loss_meta

        if step % args.eval_batch_steps == 0:
            print(f"Evaluating at GRPO Step {step}/{args.n_grpo_steps}...")

            # Save checkpoint before evaluation (so eval corresponds to a saved model)
            ckpt_path = save_checkpoint(
                ckpt_base_dir=args.checkpoint_base_dir,
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=step,
                global_step=step,
                max_to_keep=args.checkpoint_max_to_keep,
            )
            print(f"Saved checkpoint to {ckpt_path}")

            load_policy_into_vllm_instance(policy, vllm)
            problems, expected_answers = sample_formatted_eval_data(val_data, args.eval_max_examples, template)
            
            metrics = evaluate_vllm(
                vllm_model=vllm,
                reward_fn=r1_zero_reward_fn,
                prompts=problems,
                ground_truths=expected_answers,
                eval_sampling_params=eval_sampling_params,
                out_jsonl_path = os.path.join(args.val_output_dir, f"eval_grpo_step_{step}_results.jsonl"),
                fast=True
            )
            print(f"Eval Metrics at Step {step}: format_rate: {metrics['format_rate']}, answer_accuracy: {metrics['answer_accuracy']}, reward_mean: {metrics['reward_mean']}, correct_avg_length: {metrics['correct_avg_length']}, incorrect_avg_length: {metrics['incorrect_avg_length']}")

            if args.enable_wandb:
                wandb.log({
                    "eval/answer_accuracy": metrics["answer_accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/reward_mean": metrics["reward_mean"],
                    "eval/correct_avg_length": metrics["correct_avg_length"],
                    "eval/incorrect_avg_length": metrics["incorrect_avg_length"],
                    "step": step,
                }, step=batch_step)

    print("GRPO Training Complete.")

    # Final checkpoint at end of training
    ckpt_path = save_checkpoint(
        ckpt_base_dir=args.checkpoint_base_dir,
        model=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=args.epochs,
        global_step=args.n_grpo_steps+1,
        max_to_keep=args.checkpoint_max_to_keep,
    )
    print(f"Saved final checkpoint to {ckpt_path}")

# uv run scripts/grpo.py --eval_batch_steps 1 --disable_wandb > result/grpo/grpo_training.log 2>&1 &
# uv run scripts/grpo.py > result/grpo/grpo_training.log 2>&1 &
if __name__ == "__main__":
    main()
