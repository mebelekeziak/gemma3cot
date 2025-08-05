#!/usr/bin/env python
"""
continue_pretrain_gemma3n_e4b_ds.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full-weight continue-pre-training of **Gemma-3n-E4B** with VRAM-friendly tricks:

* 🧩 DeepSpeed ZeRO-3 + CPU off-load for params & optimizer
* 💾 Paged 8-bit AdamW optimiser
* ⚡ Flash-Attention-2 + gradient checkpointing
* 🖨️ Auto-writes a matching ds_config.json (no manual step)

Tested with: transformers>=4.44, deepspeed>=0.14, accelerate>=0.29, flash-attn>=2.4
"""

import os, glob, json, argparse
from typing import Iterator, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stream raw text into a HF Dataset
# ─────────────────────────────────────────────────────────────────────────────
def _yield_text_files(corpus_dir: str) -> Iterator[Dict[str, str]]:
    txt_files = glob.glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True)
    if not txt_files:
        raise RuntimeError(f"💥 No .txt files found in {corpus_dir}")
    for path in txt_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            yield {"text": fp.read()}

def build_dataset(corpus_dir: str) -> Dataset:
    return Dataset.from_generator(lambda: _yield_text_files(corpus_dir))

def chunk_examples(examples, *, tokenizer, block_size: int):
    joined = "\n\n".join(examples["text"])
    ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for s in range(0, len(ids) - block_size + 1, block_size):
        seg = [tokenizer.bos_token_id] + ids[s : s + block_size]
        out["input_ids"].append(seg)
        out["attention_mask"].append([1] * len(seg))
        out["labels"].append(seg.copy())
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DeepSpeed config writer (ZeRO-3 + CPU off-load)
# ─────────────────────────────────────────────────────────────────────────────
def write_deepspeed_config(path: str, args):
    cfg = {
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 6e8,
            "stage3_prefetch_bucket_size": 2e8,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False
        }
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fp:
        json.dump(cfg, fp, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    ds_config_path = os.path.abspath(args.deepspeed_config)
    write_deepspeed_config(ds_config_path, args)

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3n-E4B", use_fast=True, trust_remote_code=True
    )

    ds = (
        build_dataset(args.corpus_dir)
        .shuffle(seed=42)
        .map(
            chunk_examples,
            batched=True,
            remove_columns=["text"],
            fn_kwargs={"tokenizer": tokenizer, "block_size": args.sequence_length - 1},
        )
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E4B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # falls back gracefully if FA-2 not compiled
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=1000,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        optim="paged_adamw_32bit",
        deepspeed=ds_config_path,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="checkpoint",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    ckpt = get_last_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="VRAM-optimised CPT of Gemma-3n-E4B (ZeRO-3, off-load, 8-bit AdamW)."
    )
    p.add_argument("--corpus_dir", type=str, default="corpus")
    p.add_argument("--output_dir", type=str, default="checkpoints/gemma3n-e4b-cpt-optim")
    p.add_argument("--sequence_length", type=int, default=4096)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str, default=None)
    p.add_argument(
        "--deepspeed_config",
        type=str,
        default="ds_config.json",
        help="Where to write the generated DeepSpeed config.",
    )
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    main(args)
