#!/usr/bin/env python
"""
continue_pretrain_gemma3n_e4b.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Continues *pre‑training* of Google **Gemma‑3n‑E4B** on your own raw corpus.

**Key features**
----------------
* 📚 Streams all `.txt` files recursively under `./corpus/` (create and drop your text there).
* 🔒 Keeps Gemma’s tokenizer fixed, so the vocabulary is *not* expanded; this is standard for CPT.
* 🧪 Packs text into 4 096‑token sequences (adjustable up to Gemma’s 32 k context).
* ⚙️ Resumes the *base* checkpoint `google/gemma-3n-e4b` and trains **all** parameters (no LoRA).
* ⛽ Uses BF16 + gradient checkpointing → fits in ≈36 GB GPU memory (8 × A100‑80 GB or TPU v5e slice).
* 🛠️ Fully compatible with 🤗 Transformers v4.43+, Datasets v2.20+, and Accelerate v0.29+.
* ☁️ Optional: `--push_to_hub` pushes checkpoints to the HF Hub when the env var `HF_TOKEN` is set.

> ⚠️ **Licence notice**
> Continuing pre‑training is allowed under the Gemma licence, provided you respect the
> [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/docs/gemma-3n) and attribute Google.
> You are responsible for vetting your corpus for disallowed content.

Run example (single A100‑80 GB, 1 epoch):
```bash
pip install "transformers>=4.44" "datasets>=2.20" accelerate bitsandbytes sentencepiece --upgrade
python continue_pretrain_gemma3n_e4b.py \
  --corpus_dir corpus \
  --output_dir checkpoints/gemma3n-e4b-cpt \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --epochs 1
```

Colocate your text in `corpus/`, accept the Gemma weights on Hugging Face once, and you’re set.
"""

import os, glob, argparse, math
from typing import Iterator, Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Raw‑text streaming → Hugging Face Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _yield_text_files(corpus_dir: str) -> Iterator[Dict[str, str]]:
    """Yield dicts of raw text for every *.txt file under *corpus_dir*."""
    txt_files: List[str] = glob.glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True)
    if not txt_files:
        raise RuntimeError(
            f"💥 No .txt files found under {corpus_dir}. Populate it with raw text before running."
        )
    for path in txt_files:
        with open(path, "r", encoding="utf‑8", errors="ignore") as fp:
            yield {"text": fp.read()}


def build_dataset(corpus_dir: str) -> Dataset:
    """Construct a `datasets.Dataset` from local text files."""
    return Dataset.from_generator(lambda: _yield_text_files(corpus_dir))

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Tokenisation & sequence packing
# ──────────────────────────────────────────────────────────────────────────────

def chunk_examples(examples, *, tokenizer, block_size: int):
    joined: str = "\n\n".join(examples["text"])
    ids: List[int] = tokenizer(joined, add_special_tokens=False)["input_ids"]
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for start in range(0, len(ids) - block_size + 1, block_size):
        segment = ids[start : start + block_size]
        seg = [tokenizer.bos_token_id] + segment  # prepend <bos>
        result["input_ids"].append(seg)
        result["attention_mask"].append([1] * len(seg))
        result["labels"].append(seg.copy())
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Main training entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-e4b", use_fast=True, trust_remote_code=True)
    ds = build_dataset(args.corpus_dir).shuffle(seed=42)
    block = args.sequence_length - 1  # reserve 1 token for <bos>
    ds = ds.map(chunk_examples, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer, "block_size": block})

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-e4b", torch_dtype=torch.bfloat16, revision=None, trust_remote_code=True
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


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue pre‑training Gemma‑3n‑E4B on a raw text corpus.")
    parser.add_argument("--corpus_dir", type=str, default="corpus", help="Path containing .txt files.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gemma3n-e4b-cpt", help="Where to write checkpoints.")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Total sequence length (≤ 32768).")
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Micro‑batch size per GPU/TPU core.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Effective batch = per_device * grad_acc.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak LR for cosine scheduler.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of full passes over the dataset.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload checkpoints to Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HF repo ID (e.g. username/gemma3n‑e4b‑my‑cpt).")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # speed‑up on Ampere+
    main(args)

