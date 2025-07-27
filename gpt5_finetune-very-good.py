#!/usr/bin/env python
import os, re, math, warnings, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class Args:
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b")
    reward_model_id: str = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_sft: bool = True

    # paths
    merged_path: str = "./gemma3_cot_sft_merged"
    lora_ckpt_dir: str = "./gemma3_cot_sft_lora"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # SFT
    max_seq_len: int = 2048
    sft_epochs: int = 1
    sft_lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    load_in_4bit: bool = True
    seed: int = 42

    # PPO
    ppo_batch_size: int = 4
    ppo_mini_bs: int = 1
    ppo_epochs: int = 4
    ppo_lr: float = 5e-6
    ppo_target_kl: float = 6.0
    ppo_max_new_tokens: int = 512

    # Data sampling caps
    longtalk_n: int = 0  # 0 -> all
    gsm8k_n: int = 0
    mmlu_pro_n: int = 1000


def load_and_prepare_longtalk() -> Dataset:
    ds = load_dataset("kenhktsui/longtalk-cot-v0.1", split="train")
    def _convert(ex):
        messages = ex["messages"]
        user_msg      = next((m["content"] for m in messages if m["role"]=="user"), "").strip()
        assistant_msg = next((m["content"] for m in messages if m["role"]=="assistant"), "").strip()
        m = re.search(r"(?:Answer:|####)(.*)", assistant_msg)
        if m:
            chain = assistant_msg[:m.start()].strip()
            final = m.group(1).strip()
        else:
            parts = [p for p in assistant_msg.split("\n") if p.strip()]
            chain = "\n".join(parts[:-1]) if len(parts) > 1 else ""
            final = parts[-1] if parts else ""
        return {"prompt": user_msg, "response": f"<think>{chain}</think>\n\n{final}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def load_and_prepare_gsm8k() -> Dataset:
    ds = load_dataset("thesven/gsm8k-reasoning", split="train")
    def _convert(ex):
        q = ex["question"].strip()
        gen = ex.get("generation", "") or ""
        cot_parts = []
        for tag in ["thinking","reasoning","reflection","adjustment"]:
            m = re.search(fr"<{tag}>(.*?)</{tag}>", gen, re.DOTALL)
            if m: cot_parts.append(m.group(1).strip())
        chain = "\n\n".join(cot_parts)
        out_m = re.search(r"<output>(.*?)</output>", gen, re.DOTALL)
        final = out_m.group(1).strip() if out_m else ex["answer"].strip()
        return {"prompt": q, "response": f"<think>{chain}</think>\n\n{final}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def load_and_prepare_mmlu_pro() -> Dataset:
    try:
        ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train[:1000]")
    except Exception:
        return Dataset.from_dict({"prompt":[], "response":[]})
    def _convert(ex):
        q, chain = ex["question"].strip(), "\n".join(s.strip() for s in ex.get("cot",[]))
        return {"prompt": q, "response": f"<think>{chain}</think>\n\n{ex['answer']}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def build_sft_dataset(args: Args) -> Dataset:
    sources: List[Dataset] = []
    for fn in (load_and_prepare_longtalk, load_and_prepare_gsm8k, load_and_prepare_mmlu_pro):
        try:
            d = fn()
            if len(d): sources.append(d)
        except Exception as e:
            print(f"[WARN] loading dataset failed: {e}")
    if not sources:
        raise RuntimeError("No data available for SFT.")
    ds = concatenate_datasets(sources).shuffle(seed=args.seed)
    total = len(ds)
    cap = 0
    cap += args.longtalk_n if args.longtalk_n > 0 else 0
    cap += args.gsm8k_n if args.gsm8k_n > 0 else 0
    cap += args.mmlu_pro_n if args.mmlu_pro_n > 0 else 0
    if cap > 0 and total > cap:
        ds = ds.select(range(cap))
    print(f"[DATA] total samples: {len(ds)}")
    return ds


def make_sft_collator(tokenizer: AutoTokenizer, max_len: int):
    def collate(batch):
        prompts = [b["prompt"] for b in batch]
        responses = [b["response"] for b in batch]
        # Encode prompt and full to mask prompt tokens only
        enc_prompt = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc_full = tokenizer([p + "\n\n" + r for p, r in zip(prompts, responses)], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        labels = input_ids.clone()
        # Mask prompt tokens
        for i in range(len(batch)):
            plen = (enc_prompt["attention_mask"][i] == 1).sum().item()
            labels[i, :plen] = -100
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


def prepare_tokenizer(path_or_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def sft_train(args: Args):
    from unsloth import FastLanguageModel
    set_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.target_modules),
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=True,
    )

    ds = build_sft_dataset(args)
    collator = make_sft_collator(tokenizer, args.max_seq_len)

    train_args = TrainingArguments(
        output_dir="./gemma3_cot_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.sft_epochs,
        learning_rate=args.sft_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.lora_ckpt_dir)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_path, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_path)
    del model, merged_model
    torch.cuda.empty_cache()
    return args.merged_path


def build_reward(args: Args):
    tok = AutoTokenizer.from_pretrained(args.reward_model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.reward_model_id, torch_dtype=torch.bfloat16, device_map="auto")
    inc = tok.convert_tokens_to_ids("<INCORRECT>")
    cor = tok.convert_tokens_to_ids("<CORRECT>")
    step = tok.convert_tokens_to_ids("<STEP>")
    if inc == tok.unk_token_id or cor == tok.unk_token_id:
        raise ValueError("Reward model lacks <CORRECT>/<INCORRECT> tokens")
    if step == tok.unk_token_id:
        step = tok.eos_token_id
    return tok, mdl, inc, cor, step


def compute_step_scores(text: str, tok: AutoTokenizer, mdl: AutoModelForCausalLM, inc_id: int, cor_id: int, step_id: int, device: str) -> float:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    content = m.group(1) if m else text
    formatted = (" \n\n\n\n".join(ln.strip() for ln in content.split("\n")) + " \n\n\n\n")
    batch = tok([formatted], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = mdl(**batch).logits[..., [inc_id, cor_id]]
        probs = logits.softmax(-1)[..., 1]
    mask = (batch["input_ids"] == step_id).float()
    denom = mask.sum().item()
    if denom < 1:
        # Fallback: average over all tokens in the <think> block
        return probs.mean().item()
    return (probs * mask).sum().item() / denom


def ppo_train(args: Args, merged_path: str, sft_prompts: List[str]):
    tok = prepare_tokenizer(merged_path)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_path, torch_dtype=torch.bfloat16, device_map="auto")
    ppo_model.resize_token_embeddings(len(tok))
    ref_model = create_reference_model(ppo_model)

    r_tok, r_mdl, inc_id, cor_id, step_id = build_reward(args)

    def score_fn(texts: List[str]) -> List[float]:
        return [compute_step_scores(t, r_tok, r_mdl, inc_id, cor_id, step_id, args.device) for t in texts]

    cfg = PPOConfig(
        model_name="gemma3_cot_sft",
        learning_rate=args.ppo_lr,
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_bs,
        adaptive_kl_ctrl=True,
        target_kl=args.ppo_target_kl,
        ppo_epochs=args.ppo_epochs,
    )

    trainer = PPOTrainer(config=cfg, model=ppo_model, ref_model=ref_model, tokenizer=tok)

    # Build prompts from SFT data (or any dataset)
    prompts = sft_prompts
    bs = args.ppo_batch_size
    for epoch in range(args.ppo_epochs):
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i : i + bs]
            batch_inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                gen = ppo_model.generate(**batch_inputs, max_new_tokens=args.ppo_max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=tok.eos_token_id)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            rewards = score_fn(texts)
            # PPO expects lists of tensors
            query_tensors = [batch_inputs["input_ids"][j] for j in range(len(batch_prompts))]
            response_tensors = [gen[j] for j in range(len(batch_prompts))]
            trainer.step(query_tensors, response_tensors, rewards)
        print(f"[PPO] finished epoch {epoch+1}/{args.ppo_epochs}")


def main():
    args = Args()
    set_seed(args.seed)
    merged_path = args.merged_path

    if args.run_sft:
        merged_path = sft_train(args)
    else:
        print("RUN_SFT=False – skipping SFT and loading existing merged model.")

    # build a small set of prompts for PPO
    ds = build_sft_dataset(args)
    n = min(128, len(ds))
    rl_prompts = ds.shuffle(seed=args.seed).select(range(n))["prompt"]
    ppo_train(args, merged_path, rl_prompts)

if __name__ == "__main__":
    main()