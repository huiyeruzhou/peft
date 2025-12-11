from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
# from math_utils import last_boxed_only_string, remove_boxed, is_equiv
from pathlib import Path
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import argparse
# import wandb
import time
import json
import math

# --- 1. Standard LoRA Implementation (For non-SVD layers) ---
class StandardLoraLinear(nn.Module):
    def __init__(self, original_layer, rank=1, alpha=32):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # Keep original weight frozen
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        
        dtype = original_layer.weight.dtype
        device = original_layer.weight.device
        d_out, d_in = self.weight.shape
        
        # Standard LoRA Matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype, device=device))
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x: (batch, seq, in)
        base_out = F.linear(x, self.weight, self.bias)
        
        # LoRA path: x @ A.T @ B.T * scale
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def get_effective_lora_weights(self):
        """Returns A and B directly."""
        return self.lora_A, self.lora_B

# --- 2. SVD LoRA Implementation (For specific hypothesis testing) ---
class SVDLoraLinear(nn.Module):
    def __init__(self, original_layer, rank=1, alpha=32):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # Perform SVD
        with torch.no_grad():
            W = original_layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            dtype = original_layer.weight.dtype
            self.register_buffer('U', U.to(dtype))   # Frozen Output Basis
            self.register_buffer('Vh', Vh.to(dtype)) # Frozen Input Basis
            
            self.weight = original_layer.weight
            self.bias = original_layer.bias

        # Learnable Coefficients
        k = self.U.shape[1]
        # Coefficients for B (Output side)
        self.coeffs_B = nn.Parameter(torch.zeros(k, self.rank, dtype=dtype))
        # Coefficients for A (Input side)
        self.coeffs_A = nn.Parameter(torch.zeros(self.rank, k, dtype=dtype))

        # Init
        nn.init.kaiming_uniform_(self.coeffs_A, a=5**0.5)
        nn.init.zeros_(self.coeffs_B)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        breakpoint()
        # SVD Path: x -> Vh.T -> coeffs_A.T -> coeffs_B.T -> U.T
        x_k = x @ self.Vh.T 
        x_r = x_k @ self.coeffs_A.T 
        x_k_out = x_r @ self.coeffs_B.T
        lora_out = x_k_out @ self.U.T
        
        return base_out + lora_out * self.scaling

    def get_effective_lora_weights(self):
        """reconstructs effective A and B to look like standard LoRA."""
        lora_A = self.coeffs_A @ self.Vh
        lora_B = self.U @ self.coeffs_B
        return lora_A, lora_B

# --- 3. Hybrid Injection Logic ---
def inject_hybrid_lora(model, rank=1, target_modules=[], svd_modules=[]):
    """
    Injects SVDLoraLinear for modules in 'svd_modules', 
    and StandardLoraLinear for other 'target_modules'.
    """
    print(f"Injecting Hybrid LoRA (r={rank})")
    print(f"  - SVD Modules: {svd_modules}")
    print(f"  - Standard Modules: {set(target_modules) - set(svd_modules)}")
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    modules_to_replace = []
    for name, module in model.named_modules():
        # Check if this module matches ANY target
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            modules_to_replace.append(name)

    for name in modules_to_replace:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name)
        original_module = getattr(parent, child_name)
        
        # DECISION LOGIC: Is this specific module an SVD target?
        # Check if any string in svd_modules is part of the current name (e.g. "k_proj")
        is_svd = any(s in name for s in svd_modules)
        
        if is_svd:
            new_layer = SVDLoraLinear(original_module, rank=rank).to(original_module.weight.device)
            layer_type = "SVD"
        else:
            new_layer = StandardLoraLinear(original_module, rank=rank).to(original_module.weight.device)
            layer_type = "Standard"
            
        setattr(parent, child_name, new_layer)
        print(f"  Replaced {name} with {layer_type} LoRA")
    
    return model

def save_hybrid_lora_compatible(model, save_path, rank, target_modules):
    """Saves both types of layers as standard LoRA for vLLM."""
    os.makedirs(save_path, exist_ok=True)
    peft_state_dict = {}
    
    for name, module in model.named_modules():
        # Check for our custom classes
        if isinstance(module, (SVDLoraLinear, StandardLoraLinear)):
            eff_A, eff_B = module.get_effective_lora_weights()
            peft_state_dict[f"{name}.lora_A.weight"] = eff_A.cpu()
            peft_state_dict[f"{name}.lora_B.weight"] = eff_B.cpu()
            
    torch.save(peft_state_dict, os.path.join(save_path, "adapter_model.bin"))
    
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
        json.dump(config, f)

# --- Main Script ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid SVD/Standard LoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_r", type=int, default=1)
    
    # All modules to train
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"])
    
    # Subset of modules to use SVD decomposition on
    parser.add_argument("--lora_svd_modules", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"], 
                        help="Modules to use SVD decomposition on (subset of target_modules)")

    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--n_grpo_steps", type=int, default=50)
    parser.add_argument("--n_prompts_per_step", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--base_dir", type=str, default="runs")
    parser.add_argument("--prompt_template", type=str, default="boxed.prompt")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="math-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.base_dir, exist_ok=True)
    i = 1
    while os.path.exists(f"{args.base_dir}/{i}"): i += 1
    run_name = f"{args.base_dir}/{i}"
    os.makedirs(run_name)
    
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, config=vars(args), dir=run_name)

    # Load Data
    train_dataset = load_dataset("qwedsacf/competition_math", split=f"train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split=f"train[-5000:]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read().strip()

    def process_data(example):
        with_template = template.replace("{question}", example["problem"])
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": with_template}], tokenize=False, add_generation_prompt=True)
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process_data)
    val_dataset = val_dataset.map(process_data)

    def generate(prompts, vllm_model_id, temperature=0, responses_per_prompt=1):
        api_url = f"{args.vllm_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": vllm_model_id, "prompt": prompts, "max_tokens": 1024, "temperature": temperature, "n": responses_per_prompt}
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return [choice["text"] for choice in response.json()["choices"]]

    loaded_loras = []
    def load_lora(lora_name):
        if lora_name in loaded_loras: return
        api_url = f"{args.vllm_url}/v1/load_lora_adapter"
        payload = {"lora_name": lora_name, "lora_path": str(Path.cwd() / lora_name)}
        requests.post(api_url, json=payload).raise_for_status()
        loaded_loras.append(lora_name)

    def save_lora(model, step):
        lora_name = f"{run_name}/step={step}"
        if not os.path.exists(lora_name):
            # Use HYBRID saver
            save_hybrid_lora_compatible(model, lora_name, args.lora_r, args.lora_target_modules)
        return lora_name

    def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
        prompt_t = [tokenizer.encode(p) for p in prompt_strs]
        output_t = [tokenizer.encode(o) for o in output_strs]
        full = []
        max_len = 0
        for i in range(len(prompt_t)): max_len = max(max_len, len(prompt_t[i]) + len(output_t[i]))
        for i in range(len(prompt_t)):
            padding = [tokenizer.pad_token_id] * (max_len - len(prompt_t[i]) - len(output_t[i]))
            full.append(torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long).unsqueeze(0))
        f2 = torch.cat(full)
        input_ids = f2[:, :-1]
        labels = f2[:, 1:]
        response_mask = torch.zeros(len(prompt_strs), max_len - 1)
        for i in range(len(prompt_t)):
            response_mask[i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1] = 1
        return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask.bool()}

    def get_response_log_probs(model, input_ids, labels):
        logits = model(input_ids).logits
        logprobs = logits.log_softmax(dim=-1)
        return torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def eval_model(model, step):
        lora_name = save_lora(model, step)
        load_lora(lora_name)
        val_prompts = val_dataset[:1000]["prompt"]
        eval_start = time.time()
        outputs = generate(val_prompts, lora_name, temperature=0)
        correct = sum([1 for i, o in enumerate(outputs) if is_equiv(remove_boxed(last_boxed_only_string(o)), val_dataset[i]["answer"])])
        accuracy = correct / len(outputs)
        print(f"step={step}, correct: {correct} / {len(outputs)} ({accuracy:.2%})")
        if not args.disable_wandb: wandb.log({"eval/accuracy": accuracy}, step=step)
        return correct

    device = "cuda:0"
    model_kwargs = dict(attn_implementation="sdpa", torch_dtype=torch.bfloat16, use_cache=False, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # --- HYBRID INJECTION ---
    model = inject_hybrid_lora(
        model, 
        rank=args.lora_r, 
        target_modules=args.lora_target_modules,
        svd_modules=args.lora_svd_modules
    )
    
    # Enable gradients for all LoRA parameters (both coeffs and standard A/B)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_" in name or "coeffs_" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Total trainable parameters: {len(trainable_params)} tensors")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    print("Starting initial evaluation...")
    eval_model(model, 0)
    model.train()

    for i in range(args.n_grpo_steps):
        step_start_time = time.time()
        sample_indices = random.sample(range(len(train_dataset)), args.n_prompts_per_step)
        batch = train_dataset[sample_indices]

        vllm_model_id = save_lora(model, step=i)
        load_lora(vllm_model_id)

        outputs = generate(batch["prompt"], vllm_model_id=vllm_model_id, temperature=1, responses_per_prompt=args.group_size)
        
        generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]
        raw_reward = [1.0 if is_equiv(a, batch["answer"][idx // args.group_size]) else 0.0 for idx, a in enumerate(generated_answers)]
        raw_reward_tensor = torch.tensor(raw_reward, dtype=torch.float).reshape(args.n_prompts_per_step, args.group_size)
        advantages = (raw_reward_tensor - raw_reward_tensor.mean(dim=-1, keepdim=True)).reshape(-1)

        prompts_expanded = [p for p in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)

        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(len(input_ids) // args.micro_batch_size):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                old_logprobs_all.append(get_response_log_probs(model, input_ids[idx:end], labels[idx:end]).detach())
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)

        for epoch in range(args.epochs_per_step):
            for b in tqdm(range(len(input_ids) // args.micro_batch_size), desc=f"Step {i+1}"):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                
                policy_logprobs = get_response_log_probs(model, input_ids[idx:end], labels[idx:end])
                ratio = torch.exp(policy_logprobs - old_logprobs_all[idx:end])
                loss = -(ratio * advantages[idx:end].to(device).unsqueeze(-1) * response_mask[idx:end]).sum() / response_mask[idx:end].sum()
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        print(f"Step {i+1} finished. Reward: {raw_reward_tensor.mean().item():.2%}")
        if (i + 1) % 5 == 0:
            eval_model(model, i + 1)

    if not args.disable_wandb: wandb.finish()

if __name__ == "__main__":
    main()