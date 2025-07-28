#!/usr/bin/env python
"""
Fine-tune Gemma 3 or other large models on chain-of-thought (CoT) data with
supervised fine-tuning (SFT) and reinforcement learning (RL).  The code
targets NVIDIA H200 class GPUs.

This script performs two stages:
1. **Supervised fine-tuning (SFT)** using LoRA.
2. **Reinforcement learning (RL)** with TRL's PPO implementation.

Dependencies:
    pip install "transformers>=4.50.0" peft datasets "trl>=0.19" -U accelerate
"""

import os
import re
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from huggingface_hub import HfApi

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BASE_MODEL_ID   = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b")
REWARD_MODEL_ID = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Flip to True when you want to run SFT+merge; leave False to skip straight to PPO on an existing merge
RUN_SFT = False

# Where to save the merged SFT model
MERGED_PATH = "./gemma3_cot_sft_merged"
LORA_CHECKPOINT_PATH = "./gemma3_cot_sft_lora"

HF_REPO        = os.environ.get("HF_REPO")
HF_TOKEN       = os.environ.get("HF_TOKEN")
MAX_SHARD_SIZE = "10GB"

# ------------------------------------------------------------------
# TOKENIZER & MODEL LOADING
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})

# Only load and wrap in LoRA if doing SFT
if RUN_SFT:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

# ------------------------------------------------------------------
# DATASET PREPARATION (unchanged)
# ------------------------------------------------------------------
def load_and_prepare_longtalk() -> Dataset:
    ds = load_dataset("kenhktsui/longtalk-cot-v0.1", split="train")
    def _convert(example):
        messages = example["messages"]
        user_msg = next((m["content"] for m in messages if m["role"]=="user"), "").strip()
        assistant_msg = next((m["content"] for m in messages if m["role"]=="assistant"), "").strip()
        answer_match = re.search(r"(?:Answer:|####)(.*)", assistant_msg)
        if answer_match:
            chain = assistant_msg[:answer_match.start()].strip()
            final_answer = answer_match.group(1).strip()
        else:
            parts = [p for p in assistant_msg.split("\n") if p.strip()]
            chain = "\n".join(parts[:-1]) if len(parts)>1 else ""
            final_answer = parts[-1] if parts else ""
        return {"prompt": user_msg, "response": f"<think>{chain}</think>\n\n{final_answer}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def load_and_prepare_gsm8k() -> Dataset:
    ds = load_dataset("thesven/gsm8k-reasoning", split="train")
    def _convert(example):
        question = example["question"].strip()
        gen = example.get("generation","") or ""
        cot_parts = []
        for tag in ["thinking","reasoning","reflection","adjustment"]:
            m = re.search(fr"<{tag}>(.*?)</{tag}>", gen, re.DOTALL)
            if m: cot_parts.append(m.group(1).strip())
        chain = "\n\n".join(cot_parts)
        out_m = re.search(r"<output>(.*?)</output>", gen, re.DOTALL)
        final_answer = out_m.group(1).strip() if out_m else example["answer"].strip()
        return {"prompt": question, "response": f"<think>{chain}</think>\n\n{final_answer}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def load_and_prepare_mmlu_pro() -> Dataset:
    try:
        ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train[:1000]")
    except:
        return Dataset.from_dict({"prompt":[], "response":[]})
    def _convert(example):
        question = example["question"].strip()
        chain = "\n".join(s.strip() for s in example.get("cot",[]))
        return {"prompt": question, "response": f"<think>{chain}</think>\n\n{example['answer']}"}
    return ds.map(_convert, remove_columns=ds.column_names)


def build_sft_dataset() -> Dataset:
    ds_list = []
    for fn in (load_and_prepare_longtalk, load_and_prepare_gsm8k, load_and_prepare_mmlu_pro):
        try:
            d = fn()
            if len(d)>0: ds_list.append(d)
        except Exception as e:
            print(f"Warning loading data: {e}")
    if not ds_list:
        raise RuntimeError("No data for SFT.")
    return concatenate_datasets(ds_list).shuffle(seed=42)


# ------------------------------------------------------------------
# SUPERVISED FINE-TUNING (SFT)
# ------------------------------------------------------------------
if RUN_SFT:
    sft_dataset = build_sft_dataset()

    def sft_data_collator(features):
        texts = [f["prompt"] + "\n\n" + f["response"] for f in features]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"]==0] = -100
        return {"input_ids": enc["input_ids"], "labels": labels, "attention_mask": enc["attention_mask"]}

    training_args = TrainingArguments(
        output_dir="./gemma3_cot_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        tokenizer=tokenizer,
        data_collator=sft_data_collator,
    )

    # Run SFT
    trainer.train()
    # Save the LoRA adapter checkpoint for resuming if needed
    trainer.save_model(LORA_CHECKPOINT_PATH)

    # Merge and unload LoRA into base, then save merged model
    merged_model = model.merge_and_unload()
    os.makedirs(MERGED_PATH, exist_ok=True)
    tokenizer.save_pretrained(MERGED_PATH)
    merged_model.save_pretrained(
        MERGED_PATH,
        max_shard_size=MAX_SHARD_SIZE,
    )
    if HF_REPO:
        merged_model.push_to_hub(
            HF_REPO,
            token=HF_TOKEN,
            max_shard_size=MAX_SHARD_SIZE,
        )
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)

    # Clean up to free VRAM
    del base_model, model
    torch.cuda.empty_cache()

else:
    print("RUN_SFT=False → skipping SFT. Loading existing merged model for PPO.")

# ------------------------------------------------------------------
# REINFORCEMENT LEARNING WITH PPO
# ------------------------------------------------------------------
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MERGED_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
ppo_model.resize_token_embeddings(len(tokenizer))
ref_model = create_reference_model(ppo_model)

# Load reward model
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
reward_model     = AutoModelForCausalLM.from_pretrained(REWARD_MODEL_ID).to(DEVICE).eval()

incorrect_id = reward_tokenizer.convert_tokens_to_ids("<INCORRECT>")
correct_id   = reward_tokenizer.convert_tokens_to_ids("<CORRECT>")
step_id      = reward_tokenizer.convert_tokens_to_ids("<STEP>") \
                 if reward_tokenizer.convert_tokens_to_ids("<STEP>")!=reward_tokenizer.unk_token_id \
                 else reward_tokenizer.eos_token_id

def compute_step_scores(text: str) -> float:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    formatted = (" \n\n\n\n".join([ln.strip() for ln in m.group(1).split("\n")]) + " \n\n\n\n") if m else text
    input_ids = reward_tokenizer.encode(formatted, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = reward_model(input_ids).logits[..., [incorrect_id, correct_id]]
        probs  = logits.softmax(-1)[..., 1]
    mask = (input_ids == step_id).float()
    return (probs * mask).sum().item() / max(mask.sum().item(), 1)

def build_rl_queries(dataset: Dataset, num_samples: int=100) -> list:
    num = min(len(dataset), num_samples)
    return dataset.shuffle(seed=0).select(range(num))["prompt"]

# Reuse sft_dataset if available, else reload minimal data for prompts
if RUN_SFT:
    rl_prompts = build_rl_queries(sft_dataset, num_samples=128)
else:
    rl_prompts = build_rl_queries(build_sft_dataset(), num_samples=128)

ppo_config = PPOConfig(
    model_name="gemma3_cot_sft",
    learning_rate=5e-6,
    batch_size=4,
    mini_batch_size=1,
    adaptive_kl_ctrl=True,
    target_kl=6.0,
    ppo_epochs=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

def rl_finetune(num_epochs: int = 1):
    ppo_trainer.model.train()
    for epoch in range(num_epochs):
        for prompt in rl_prompts:
            query_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = ppo_model.generate(
                input_ids=query_ids["input_ids"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            rewards = [compute_step_scores(tokenizer.decode(out[0], skip_special_tokens=True))]
            ppo_trainer.step([query_ids["input_ids"][0]], [out[0]], rewards)
        print(f"Completed RL epoch {epoch+1}/{num_epochs}")

    final_dir = os.path.join(MERGED_PATH, "ppo_final")
    os.makedirs(final_dir, exist_ok=True)
    ppo_model.save_pretrained(
        final_dir,
        max_shard_size=MAX_SHARD_SIZE,
    )
    tokenizer.save_pretrained(final_dir)
    if HF_REPO:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO, exist_ok=True)
        api.upload_folder(folder_path=final_dir, repo_id=HF_REPO)

# Uncomment to run RL:
# rl_finetune(num_epochs=1)
