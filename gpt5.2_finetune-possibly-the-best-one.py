#!/usr/bin/env python
import os, re, math, json, time, signal, warnings, random, subprocess, sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

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

# ----------------------- Utilities -----------------------

def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class Args:
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b-int")
    reward_model_id: str = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_sft: bool = True

    # paths
    merged_path: str = "./gemma3_cot_sft_merged"
    lora_ckpt_dir: str = "./gemma3_cot_sft_lora"
    output_dir: str = "./runs/gemma3_cot"

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
    pack_sequences: bool = False  # Optional greedy packing to reduce padding

    # PPO
    ppo_batch_size: int = 4
    ppo_mini_bs: int = 1
    ppo_epochs: int = 4
    ppo_lr: float = 5e-6
    ppo_target_kl: float = 6.0
    ppo_max_new_tokens: int = 512
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_top_k: int = 0
    repetition_penalty: float = 1.05
    min_new_tokens: int = 16
    length_penalty: float = 1.0

    # Reward shaping
    reward_samples: int = 1          # average N generations per prompt for reward (costly)
    reward_temperature: float = 0.7

    # Data sampling caps (0 -> all)
    longtalk_n: int = 0
    gsm8k_n: int = 0
    mmlu_pro_n: int = 1000

    # Distributed hints (for accelerate/FSDP/DeepSpeed)
    fsdp: bool = False
    deepspeed_config: Optional[str] = None

    # Manifest & logging
    manifest_name: str = "manifest.json"
    jsonl_log: str = "metrics.jsonl"


def parse_args(argv: List[str]) -> Args:
    import argparse
    p = argparse.ArgumentParser(description="Gemma-3 CoT SFT + PPO trainer")

    # High-level toggles
    p.add_argument("--run-sft", action="store_true", default=None, help="Run SFT stage before PPO (default: True)")
    p.add_argument("--no-run-sft", dest="run_sft", action="store_false")

    # Model IDs / paths
    p.add_argument("--base-model-id", type=str, default=None)
    p.add_argument("--reward-model-id", type=str, default=None)
    p.add_argument("--merged-path", type=str, default=None)
    p.add_argument("--lora-ckpt-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)

    # LoRA
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--target-modules", type=str, nargs="*", default=None)

    # SFT
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--sft-epochs", type=int, default=None)
    p.add_argument("--sft-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--load-in-4bit", action="store_true", default=None)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--pack-sequences", action="store_true", default=None)

    # PPO
    p.add_argument("--ppo-batch-size", type=int, default=None)
    p.add_argument("--ppo-mini-bs", type=int, default=None)
    p.add_argument("--ppo-epochs", type=int, default=None)
    p.add_argument("--ppo-lr", type=float, default=None)
    p.add_argument("--ppo-target-kl", type=float, default=None)
    p.add_argument("--ppo-max-new-tokens", type=int, default=None)

    # Generation params
    p.add_argument("--gen-temperature", type=float, default=None)
    p.add_argument("--gen-top-p", type=float, default=None)
    p.add_argument("--gen-top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--min-new-tokens", type=int, default=None)
    p.add_argument("--length-penalty", type=float, default=None)

    # Reward shaping
    p.add_argument("--reward-samples", type=int, default=None)
    p.add_argument("--reward-temperature", type=float, default=None)

    # Data caps
    p.add_argument("--longtalk-n", type=int, default=None)
    p.add_argument("--gsm8k-n", type=int, default=None)
    p.add_argument("--mmlu-pro-n", type=int, default=None)

    # Distributed hints
    p.add_argument("--fsdp", action="store_true", default=None)
    p.add_argument("--deepspeed-config", type=str, default=None)

    args_ns = p.parse_args(argv)
    args = Args()  # defaults
    for k, v in vars(args_ns).items():
        if v is not None:
            setattr(args, k, v)
    return args

# ----------------------- Data -----------------------

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
        ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
    except Exception:
        return Dataset.from_dict({"prompt":[], "response":[]})
    def _convert(ex):
        q, chain = ex["question"].strip(), "\n".join(s.strip() for s in ex.get("cot",[]))
        return {"prompt": q, "response": f"<think>{chain}</think>\n\n{ex['answer']}"}
    return ds.map(_convert, remove_columns=ds.column_names)

def load_and_prepare_r1_distill(subset: str = "v0") -> Dataset:
    """
    ServiceNow‑AI/R1‑Distill‑SFT

    Maps the dataset to:
        prompt   = `problem`
        response = "<think>{chain‑of‑thought}</think>\\n\\n{solution}"
    where `chain‑of‑thought` comes from `reannotated_assistant_content`
    with any existing <think> tags stripped to avoid nesting.
    """
    ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", subset, split="train")

    def _convert(ex):
        # Extract and clean chain‑of‑thought
        chain = re.sub(r"</?think>", "", ex["reannotated_assistant_content"]).strip()
        return {
            "prompt": ex["problem"].strip(),
            "response": f"<think>{chain}</think>\n\n{ex['solution'].strip()}",
        }

    return ds.map(_convert, remove_columns=ds.column_names)

def concatenate_and_cap(dsets: List[Dataset], cap: int, seed: int) -> Dataset:
    ds = concatenate_datasets(dsets).shuffle(seed=seed)
    if cap > 0 and len(ds) > cap:
        ds = ds.select(range(cap))
    return ds


def build_sft_dataset(args: Args) -> Dataset:
    sources: List[Dataset] = []
    for fn in (load_and_prepare_longtalk, load_and_prepare_gsm8k, load_and_prepare_mmlu_pro, load_and_prepare_r1_distill,):
        try:
            d = fn()
            if len(d): sources.append(d)
        except Exception as e:
            print(f"[WARN] loading dataset failed: {e}")
    if not sources:
        raise RuntimeError("No data available for SFT.")
    cap = 0
    cap += args.longtalk_n if args.longtalk_n > 0 else 0
    cap += args.gsm8k_n if args.gsm8k_n > 0 else 0
    cap += args.mmlu_pro_n if args.mmlu_pro_n > 0 else 0
    ds = concatenate_and_cap(sources, cap, seed=args.seed)
    print(f"[DATA] total samples: {len(ds)}")
    return ds

# ----------------------- Tokenizer -----------------------

def prepare_tokenizer(path_or_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    # BOS/EOS/PAD checks
    if tok.eos_token is None:
        # fallbacks: prefer sep or pad
        if tok.sep_token is not None:
            tok.eos_token = tok.sep_token
        elif tok.pad_token is not None:
            tok.eos_token = tok.pad_token
        else:
            tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ----------------------- Collators & packing -----------------------

def make_sft_collator(tokenizer: AutoTokenizer, max_len: int):
    def collate(batch):
        prompts = [b["prompt"] for b in batch]
        responses = [b["response"] for b in batch]
        enc_prompt = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        full_texts = [p + "\n\n" + r for p, r in zip(prompts, responses)]
        enc_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        labels = input_ids.clone()
        for i in range(len(batch)):
            plen = int((enc_prompt["attention_mask"][i] == 1).sum().item())
            labels[i, :plen] = -100
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


def greedy_pack_examples(texts: List[str], tokenizer: AutoTokenizer, max_len: int) -> List[str]:
    """Pack multiple short samples into longer sequences up to max_len tokens.
    Simple greedy strategy; for production, prefer specialized packers.
    """
    packed: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for t in texts:
        l = len(tokenizer.encode(t, add_special_tokens=False))
        if l > max_len:
            # truncate overly long example as a standalone
            trunc = tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)[:max_len])
            if cur:
                packed.append("\n\n".join(cur))
                cur, cur_len = [], 0
            packed.append(trunc)
            continue
        if cur_len + l <= max_len:
            cur.append(t)
            cur_len += l
        else:
            packed.append("\n\n".join(cur))
            cur, cur_len = [t], l
    if cur:
        packed.append("\n\n".join(cur))
    return packed

# ----------------------- SFT -----------------------

def sft_train(args: Args) -> str:
    from unsloth import FastLanguageModel
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
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

    if args.pack_sequences:
        texts = [(ex["prompt"] + "\n\n" + ex["response"]) for ex in ds]
        packed_texts = greedy_pack_examples(texts, tokenizer, args.max_seq_len)
        ds = Dataset.from_dict({"prompt": ["" for _ in packed_texts], "response": packed_texts})
        print(f"[PACK] packed sequences: {len(packed_texts)}")
        # Collator will still work since prompt is empty (all labels are response)

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "sft"),
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

    # Guarded merge
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.merged_path, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_path)
    except Exception as e:
        print(f"[ERROR] merge failed: {e}")
        raise
    finally:
        del model
        if "merged_model" in locals():
            del merged_model
        torch.cuda.empty_cache()

    return args.merged_path

# ----------------------- Reward -----------------------

def build_reward(args: Args):
    tok = AutoTokenizer.from_pretrained(args.reward_model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.reward_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
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
        return probs.mean().item()
    return (probs * mask).sum().item() / denom

# ----------------------- PPO -----------------------

def ppo_train(args: Args, merged_path: str, sft_prompts: List[str]):
    tok = prepare_tokenizer(merged_path)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        merged_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ppo_model.resize_token_embeddings(len(tok))
    ref_model = create_reference_model(ppo_model)

    r_tok, r_mdl, inc_id, cor_id, step_id = build_reward(args)

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

    # JSONL logging
    jsonl_path = os.path.join(args.output_dir, args.jsonl_log)
    def log_jsonl(rec: Dict[str, Any]):
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    stop_flag = {"stop": False}
    def handle_sigint(sig, frame):
        print("\n[INFO] Caught signal, saving PPO model and exiting...")
        trainer.save_pretrained(os.path.join(args.output_dir, "ppo"))
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    prompts = sft_prompts
    bs = args.ppo_batch_size
    global_step = 0
    for epoch in range(args.ppo_epochs):
        if stop_flag["stop"]: break
        for i in range(0, len(prompts), bs):
            if stop_flag["stop"]: break
            batch_prompts = prompts[i : i + bs]
            batch_inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(args.device)

            # Generate once, or multiple times for reward smoothing
            texts_all: List[str] = []
            rewards_all: List[float] = []
            samples = max(1, args.reward_samples)
            for s in range(samples):
                with torch.no_grad():
                    gen = ppo_model.generate(
                        **batch_inputs,
                        max_new_tokens=args.ppo_max_new_tokens,
                        do_sample=True,
                        temperature=args.gen_temperature if samples == 1 else args.reward_temperature,
                        top_p=args.gen_top_p,
                        top_k=args.gen_top_k,
                        repetition_penalty=args.repetition_penalty,
                        length_penalty=args.length_penalty,
                        min_new_tokens=args.min_new_tokens,
                        eos_token_id=tok.eos_token_id,
                    )
                texts = tok.batch_decode(gen, skip_special_tokens=True)
                rewards = [
                    compute_step_scores(t, r_tok, r_mdl, inc_id, cor_id, step_id, args.device) for t in texts
                ]
                texts_all.append("\n".join(texts))  # for logging (collapsed)
                rewards_all.append(float(np.mean(rewards)))

            reward_value = float(np.mean(rewards_all))
            # One query/response per prompt for PPO step; we use last generation
            query_tensors = [batch_inputs["input_ids"][j] for j in range(len(batch_prompts))]
            response_tensors = [gen[j] for j in range(len(batch_prompts))]
            stats = trainer.step(query_tensors, response_tensors, [reward_value] * len(batch_prompts))
            global_step += 1

            log_jsonl({
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": i // bs,
                "reward": reward_value,
                "stats": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()},
            })
        print(f"[PPO] finished epoch {epoch+1}/{args.ppo_epochs}")

    trainer.save_pretrained(os.path.join(args.output_dir, "ppo_final"))

# ----------------------- Manifest -----------------------

def library_versions() -> Dict[str, Any]:
    import transformers, datasets as hf_datasets, trl as trl_lib
    try:
        import unsloth as unsloth_lib
        unsloth_version = getattr(unsloth_lib, "__version__", "unknown")
    except Exception:
        unsloth_version = "missing"
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": hf_datasets.__version__,
        "trl": trl_lib.__version__,
        "unsloth": unsloth_version,
    }


def gpu_info() -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            infos.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}",
            })
    return infos


def write_manifest(args: Args, stage: str, extra: Dict[str, Any]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "git_commit": get_git_commit(),
        "stage": stage,
        "args": asdict(args),
        "versions": library_versions(),
        "gpu": gpu_info(),
    }
    manifest.update(extra)
    path = os.path.join(args.output_dir, args.manifest_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] wrote {path}")

# ----------------------- Main -----------------------

def main(argv: List[str]):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    write_manifest(args, stage="start", extra={})

    merged_path = args.merged_path
    if args.run_sft:
        merged_path = sft_train(args)
        write_manifest(args, stage="post_sft", extra={"merged_path": merged_path})
    else:
        print("RUN_SFT=False – skipping SFT and loading existing merged model.")

    ds = build_sft_dataset(args)
    n = min(128000, len(ds))
    rl_prompts = ds.shuffle(seed=args.seed).select(range(n))["prompt"]

    ppo_train(args, merged_path, rl_prompts)
    write_manifest(args, stage="done", extra={"merged_path": merged_path})

if __name__ == "__main__":
    main(sys.argv[1:])