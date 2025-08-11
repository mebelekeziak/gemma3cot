#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, time, signal, warnings, random, subprocess, sys
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
    BitsAndBytesConfig,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from huggingface_hub import HfApi
from peft import PeftModel

warnings.filterwarnings("ignore", category=UserWarning)

# ======================= Utilities =======================

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

def _gpu_capability() -> Tuple[int, int]:
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)

def _bf16_hw_supported() -> bool:
    # Only Ampere+ (SM >= 80) truly supports bf16
    if not torch.cuda.is_available():
        return False
    major, _ = _gpu_capability()
    return major >= 8

def _is_bf16_supported() -> bool:
    # Keep a single source of truth
    return _bf16_hw_supported()

def _is_gemma3(model_id: str) -> bool:
    return "gemma-3" in (model_id or "").lower()

@dataclass
class PrecisionCfg:
    mode: str                 # "auto", "bf16", "fp16", "fp32"
    model_dtype: torch.dtype  # dtype to load policy/models with
    bnb_compute_dtype: torch.dtype  # dtype for 4-bit compute
    trainer_bf16: bool
    trainer_fp16: bool
    note: str

def choose_precision(mode: Optional[str], base_model_id: str) -> PrecisionCfg:
    req = (mode or os.environ.get("PRECISION") or "auto").lower()
    cuda = torch.cuda.is_available()
    major, minor = _gpu_capability()
    bf16_ok = _bf16_hw_supported()

    def cfg(model_dtype, bnb_dtype, bf16_flag, fp16_flag, note):
        return PrecisionCfg(req, model_dtype, bnb_dtype, bf16_flag, fp16_flag, note)

    if req not in {"auto","bf16","fp16","fp32"}:
        req = "auto"

    if req == "bf16":
        if not bf16_ok:
            # fallback: for Gemma-3 we must not use fp16 -> use fp32
            if _is_gemma3(base_model_id):
                return cfg(torch.float32, torch.float16, False, False, "forced bf16 but unsupported; Gemma-3 -> fp32")
            return cfg(torch.float16, torch.float16, False, True, "forced bf16 but unsupported; fallback fp16")
        return cfg(torch.bfloat16, torch.bfloat16, True, False, "forced bf16")

    if req == "fp16":
        # Gemma-3 doesn't run correctly in fp16 => force fp32
        if _is_gemma3(base_model_id):
            return cfg(torch.float32, torch.float16, False, False, "Gemma-3 forbids fp16 -> using fp32")
        return cfg(torch.float16, torch.float16, False, True, "forced fp16")

    if req == "fp32":
        return cfg(torch.float32, torch.float16, False, False, "forced fp32")

    # AUTO
    if cuda and bf16_ok:
        # H100/H200/Ampere etc.
        return cfg(torch.bfloat16, torch.bfloat16, True, False, "auto->bf16 (Ampere/Hopper)")
    if cuda:
        # Pre-Ampere like T4. Gemma-3 cannot use fp16 -> fp32
        if _is_gemma3(base_model_id):
            return cfg(torch.float32, torch.float16, False, False, "auto on Turing + Gemma-3 -> fp32")
        return cfg(torch.float16, torch.float16, False, True, "auto on pre-Ampere -> fp16")
    # CPU
    return cfg(torch.float32, torch.float16, False, False, "auto on CPU -> fp32")

def _print_precision_banner(pc: PrecisionCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cap = f"{_gpu_capability()[0]}.{_gpu_capability()[1]}" if torch.cuda.is_available() else "-"
    print("== Precision Config ==")
    print(f"Device: {dev} | {name} (SM {cap})")
    print(f"Requested: {pc.mode} | Selected model dtype: {pc.model_dtype} | 4-bit compute: {pc.bnb_compute_dtype}")
    print(f"Trainer flags -> bf16={pc.trainer_bf16} fp16={pc.trainer_fp16} | Note: {pc.note}")

# ======================= Args =======================

@dataclass
class Args:
    # Base + reward
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b-it")
    reward_model_id: str = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM-Base-3B")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Flow toggles
    run_sft: bool = True
    run_merge_after_ppo: bool = True
    rl_on_lora: bool = True  # keep for clarity; PPO trains LoRA + v-head only

    # Paths
    merged_path: str = "./gemma3_cot_sft_merged"
    lora_ckpt_dir: str = "./gemma3_cot_sft_lora"
    ppo_lora_dir: str = "./gemma3_cot_ppo_lora"
    output_dir: str = "./runs/gemma3_cot"

    # LoRA (SFT)
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # PRM loader options
    prm_load_in_4bit: bool = True
    prm_base_id: Optional[str] = os.environ.get("PRM_BASE_ID")  # e.g. "Qwen/Qwen2.5-3B"

    # SFT
    max_seq_len: int = 2048
    sft_epochs: int = 1
    sft_lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    load_in_4bit: bool = True
    seed: int = 42
    pack_sequences: bool = False

    # PPO core
    ppo_batch_size: int = 4
    ppo_mini_bs: int = 1
    ppo_epochs: int = 4
    ppo_lr: float = 5e-6
    ppo_target_kl: float = 6.0  # used only if kl_anneal != none
    ppo_max_new_tokens: int = 512

    # Sampling
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_top_k: int = 0
    repetition_penalty: float = 1.05
    min_new_tokens: int = 16
    length_penalty: float = 1.0

    # Reward sampling/averaging
    reward_samples: int = 1
    reward_temperature: float = 0.7

    # Data sampling caps (0 -> all)
    longtalk_n: int = 0
    gsm8k_n: int = 0
    mmlu_pro_n: int = 1000

    # Extra RL features
    ref_free: bool = False
    entropy_beta: float = 0.0
    kl_anneal: str = "none"             # "none" | "linear" | "cosine"
    kl_min: float = 1.0
    kl_max: float = 6.0
    eval_every: int = 100
    eval_gsm8k_n: int = 128
    reward_w_prm: float = 1.0
    reward_w_rule: float = 0.2
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    # Distributed / Hub
    fsdp: bool = False
    deepspeed_config: Optional[str] = None
    manifest_name: str = "manifest.json"
    jsonl_log: str = "metrics.jsonl"
    hf_repo: Optional[str] = os.environ.get("HF_REPO")
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")
    max_shard_size: str = "2GB"

    # Precision
    precision: str = os.environ.get("PRECISION", "auto")  # auto | bf16 | fp16 | fp32

def parse_args(argv: List[str]) -> Args:
    import argparse
    p = argparse.ArgumentParser(description="Gemma-3 CoT SFT + PPO on LoRA with KL anneal, entropy, eval, mixed rewards (precision-standardized)")
    p.add_argument("--prm-base-id", type=str, default=None)
    p.add_argument("--prm-load-in-4bit", action="store_true", default=None)
    p.add_argument("--no-prm-load-in-4bit", dest="prm_load_in_4bit", action="store_false")
    # Toggles
    p.add_argument("--run-sft", action="store_true", default=None)
    p.add_argument("--no-run-sft", dest="run_sft", action="store_false")
    p.add_argument("--run-merge-after-ppo", action="store_true", default=None)
    p.add_argument("--no-run-merge-after-ppo", dest="run_merge_after_ppo", action="store_false")
    p.add_argument("--ref-free", action="store_true", default=None)

    # Model IDs / paths
    p.add_argument("--base-model-id", type=str, default=None)
    p.add_argument("--reward-model-id", type=str, default=None)
    p.add_argument("--merged-path", type=str, default=None)
    p.add_argument("--lora-ckpt-dir", type=str, default=None)
    p.add_argument("--ppo-lora-dir", type=str, default=None)
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

    # Generation
    p.add_argument("--gen-temperature", type=float, default=None)
    p.add_argument("--gen-top-p", type=float, default=None)
    p.add_argument("--gen-top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--min-new-tokens", type=int, default=None)
    p.add_argument("--length-penalty", type=float, default=None)

    # Reward sampling
    p.add_argument("--reward-samples", type=int, default=None)
    p.add_argument("--reward-temperature", type=float, default=None)

    # Data caps
    p.add_argument("--longtalk-n", type=int, default=None)
    p.add_argument("--gsm8k-n", type=int, default=None)
    p.add_argument("--mmlu-pro-n", type=int, default=None)

    # Extras
    p.add_argument("--entropy-beta", type=float, default=None)
    p.add_argument("--kl-anneal", type=str, choices=["none","linear","cosine"], default=None)
    p.add_argument("--kl-min", type=float, default=None)
    p.add_argument("--kl-max", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-gsm8k-n", type=int, default=None)
    p.add_argument("--reward-w-prm", type=float, default=None)
    p.add_argument("--reward-w-rule", type=float, default=None)
    p.add_argument("--reward-clip-min", type=float, default=None)
    p.add_argument("--reward-clip-max", type=float, default=None)

    # Dist / Hub
    p.add_argument("--fsdp", action="store_true", default=None)
    p.add_argument("--deepspeed-config", type=str, default=None)
    p.add_argument("--hf-repo", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--max-shard-size", type=str, default=None)

    # Precision
    p.add_argument("--precision", type=str, choices=["auto","bf16","fp16","fp32"], default=None)

    ns = p.parse_args(argv)
    args = Args()
    for k, v in vars(ns).items():
        if v is not None:
            setattr(args, k, v)
    if args.kl_anneal != "none":
        args.ppo_target_kl = args.kl_max
    return args

# ======================= Data =======================

_SINGLE_ANGLE_OPEN = "‹"
_SINGLE_ANGLE_CLOSE = "›"

def _normalize_angle_tags(text: str) -> str:
    if not text:
        return text
    return text.replace(_SINGLE_ANGLE_OPEN, "<").replace(_SINGLE_ANGLE_CLOSE, ">")

def load_and_prepare_longtalk() -> Dataset:
    ds = load_dataset("kenhktsui/longtalk-cot-v0.1", split="train")
    def _convert(ex):
        messages = ex.get("chosen") or []
        user_msg      = next((m.get("content","") for m in messages if m.get("role")=="user"), "").strip()
        assistant_msg = next((m.get("content","") for m in messages if m.get("role")=="assistant"), "").strip()
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
        gen = _normalize_angle_tags(ex.get("generation","") or "")
        q = ex.get("question", "").strip()
        if not q:
            msgs = ex.get("messages") or []
            q = next((m.get("content","") for m in msgs if m.get("role")=="user"), "").strip()
        cot_parts = []
        for tag in ["thinking","reasoning","reflection","adjustment"]:
            m = re.search(fr"<{tag}>(.*?)</{tag}>", gen, re.DOTALL)
            if m:
                cot_parts.append(m.group(1).strip())
        chain = "\n\n".join([p for p in cot_parts if p])
        out_m = re.search(r"<output>(.*?)</output>", gen, re.DOTALL)
        final = (out_m.group(1).strip() if out_m else (ex.get("answer","") or "").strip())
        return {"prompt": q, "response": f"<think>{chain}</think>\n\n{final}"}
    return ds.map(_convert, remove_columns=ds.column_names)

def load_and_prepare_mmlu_pro() -> Dataset:
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
    def _convert(ex):
        q = ex.get("question","").strip()
        cot_list = ex.get("chain_of_thoughts") or []
        chain = "\n".join([str(s).strip() for s in cot_list if str(s).strip()])
        ans = ex.get("answer","").strip()
        return {"prompt": q, "response": f"<think>{chain}</think>\n\n{ans}"}
    return ds.map(_convert, remove_columns=ds.column_names)

def load_and_prepare_r1_distill(subset: str = "v0") -> Dataset:
    ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", subset, split="train")
    def _convert(ex):
        chain = re.sub(r"</?think>", "", ex.get("reannotated_assistant_content","")).strip()
        return {"prompt": (ex.get("problem","") or "").strip(),
                "response": f"<think>{chain}</think>\n\n{(ex.get('solution','') or '').strip()}"}
    return ds.map(_convert, remove_columns=ds.column_names)

def concatenate_and_cap(dsets: List[Dataset], cap: int, seed: int) -> Dataset:
    ds = concatenate_datasets(dsets).shuffle(seed=seed)
    if cap > 0 and len(ds) > cap:
        ds = ds.select(range(cap))
    return ds

def build_sft_dataset(args: Args) -> Dataset:
    sources: List[Dataset] = []
    for fn in (load_and_prepare_longtalk, load_and_prepare_gsm8k, load_and_prepare_mmlu_pro, load_and_prepare_r1_distill):
        try:
            d = fn()
            if len(d):
                sources.append(d)
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

# ======================= Tokenizer =======================

def prepare_tokenizer(path_or_id: str, left_pad: bool = False) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if tok.eos_token is None:
        if tok.sep_token is not None:
            tok.eos_token = tok.sep_token
        elif tok.pad_token is not None:
            tok.eos_token = tok.pad_token
        else:
            tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if left_pad:
        tok.padding_side = 'left'
        tok.truncation_side = 'left'
    return tok

# ======================= Collator & packing =======================

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
    packed, cur, cur_len = [], [], 0
    for t in texts:
        l = len(tokenizer.encode(t, add_special_tokens=False))
        if l > max_len:
            trunc = tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)[:max_len])
            if cur:
                packed.append("\n\n".join(cur)); cur, cur_len = [], 0
            packed.append(trunc)
            continue
        if cur_len + l <= max_len:
            cur.append(t); cur_len += l
        else:
            packed.append("\n\n".join(cur)); cur, cur_len = [t], l
    if cur:
        packed.append("\n\n".join(cur))
    return packed

# ======================= SFT =======================

def _dtype_for_model(model_id: str, pc: PrecisionCfg) -> torch.dtype:
    # Gemma-3 has issues with pure fp16 — prefer bf16 if available, else fp32.
    if _is_gemma3(model_id) and pc.model_dtype == torch.float16:
        return torch.float32
    return pc.model_dtype

def sft_train(args: Args, pc: PrecisionCfg) -> str:
    from unsloth import FastLanguageModel
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # IMPORTANT:
    # unsloth disallows dtype=torch.float32. On pre-Ampere with Gemma-3 our precision chooser
    # will select fp32; in that case we hand unsloth dtype=None so it can choose internally,
    # while we still quantize to 4-bit with fp16 compute.
    desired_dtype = _dtype_for_model(args.base_model_id, pc)
    dtype_for_unsloth: Optional[torch.dtype] = None if desired_dtype == torch.float32 else desired_dtype

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_len,
        dtype=dtype_for_unsloth,      # None on T4+Gemma3 to avoid AssertionError
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

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "sft"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.sft_epochs,
        learning_rate=args.sft_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=pc.trainer_bf16,
        fp16=pc.trainer_fp16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.lora_ckpt_dir)  # adapters only
    tokenizer.save_pretrained(args.lora_ckpt_dir)
    torch.cuda.empty_cache()
    return args.lora_ckpt_dir

# ======================= Reward (VersaPRM) =======================

class VersaPRM:
    """
    VersaPRM scoring helper.
    """
    def __init__(
        self,
        model_id: str,
        device: str,
        correct_label_candidates=(" CORRECT", " Correct", " correct", " YES", " Yes", " yes", " True", " true"),
        incorrect_label_candidates=(" INCORRECT", " Incorrect", " incorrect", " NO", " No", " no", " False", " false"),
        step_delim: str = " \n\n\n\n",
        load_in_4bit: Optional[bool] = None,
        base_model_id: Optional[str] = None,
        prm_compute_dtype: Optional[torch.dtype] = None,
    ):
        prm_dtype = prm_compute_dtype or (torch.bfloat16 if _is_bf16_supported() else torch.float16)

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.eos_token is None:
            if self.tok.sep_token is not None:
                self.tok.eos_token = self.tok.sep_token
            else:
                self.tok.add_special_tokens({"eos_token": "</s>"})
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"

        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=prm_dtype,
            )

        if base_model_id:
            base_kwargs: Dict[str, Any] = dict(
                torch_dtype=prm_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            if quant_cfg is not None:
                base_kwargs["quantization_config"] = quant_cfg
            base = AutoModelForCausalLM.from_pretrained(base_model_id, **base_kwargs)

            self.model = PeftModel.from_pretrained(
                base,
                model_id,
                is_trainable=False,
            )
        else:
            full_kwargs: Dict[str, Any] = dict(
                torch_dtype=prm_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            if quant_cfg is not None:
                full_kwargs["quantization_config"] = quant_cfg
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **full_kwargs)

        self.model.eval()

        self.step_delim = step_delim
        self.step_ids = self.tok.encode(self.step_delim, add_special_tokens=False)
        self.candidate_ids = self._resolve_candidate_ids(
            correct_label_candidates, incorrect_label_candidates
        )  # order: [INCORRECT, CORRECT]

    def _pick_single_token_id(self, candidates) -> Optional[int]:
        for s in candidates:
            ids = self.tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None

    def _fallback_last_id(self, s: str) -> int:
        ids = self.tok.encode(s, add_special_tokens=False)
        return ids[-1]

    def _resolve_candidate_ids(self, correct_cands, incorrect_cands):
        correct_id = self._pick_single_token_id(correct_cands)
        incorrect_id = self._pick_single_token_id(incorrect_cands)
        if correct_id is None:
            correct_id = self._fallback_last_id(correct_cands[0])
        if incorrect_id is None:
            incorrect_id = self._fallback_last_id(incorrect_cands[0])
        return [incorrect_id, correct_id]

    def _find_delim_tail_positions(self, input_ids_1d: torch.Tensor):
        ids = input_ids_1d.tolist()
        sub = self.step_ids
        L, M = len(ids), len(sub)
        positions = []
        if M == 0:
            return positions
        for i in range(L - M + 1):
            if ids[i : i + M] == sub:
                positions.append(i + M - 1)
        return positions

    @torch.no_grad()
    def score(self, question: str, steps_text: str) -> float:
        raw_lines = [ln.strip() for ln in (steps_text or "").split("\n") if ln.strip()]
        if not raw_lines:
            return 0.0
        joined = self.step_delim.join(raw_lines) + self.step_delim
        input_text = (question or "").strip() + " \n\n" + joined

        device = next(self.model.parameters()).device
        ids = self.tok.encode(input_text, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)

        out = self.model(input_ids)
        cand_logits = torch.stack(
            [out.logits[:, :, cid] for cid in self.candidate_ids], dim=-1
        )
        probs = torch.softmax(cand_logits, dim=-1)[0]

        pos_idx = self._find_delim_tail_positions(input_ids[0])
        if not pos_idx:
            tail_len = min(64, probs.shape[0])
            return float(probs[-tail_len:, 1].mean().item())
        return float(probs[pos_idx, 1].mean().item())

def extract_think(text: str) -> str:
    text = _normalize_angle_tags(text or "")
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text

# ======================= Rule/Safety Critic =======================

BAD_PHRASES = [
    "I can't help with that", "I'm unable to", "I cannot assist", "as an AI",
    "I'm just an AI", "sorry, I can't", "I'm sorry, I can't", "cannot comply"
]

def rule_safety_reward(prompt: str, output: str) -> float:
    if not output or not output.strip():
        return -0.5
    tail = output.split("</think>")[-1] if "</think>" in output else output
    tail = tail.strip()
    if any(p.lower() in output.lower() for p in BAD_PHRASES):
        return -0.5
    tok_est = len(re.findall(r"\S+", tail))
    len_score = 0.0
    if 1 <= tok_est <= 200:
        len_score = 0.2
    elif tok_est > 500:
        len_score = -0.2
    reps = sum(1 for m in re.finditer(r"(\b\w+\b)(?:\s+\1){3,}", tail, re.IGNORECASE))
    rep_score = -0.2 * min(reps, 3)
    has_number = bool(re.search(r"[-+]?\d[\d,\.]*", tail))
    fa_score = 0.1 if has_number else 0.0
    tool_like = bool(re.search(r"<\s*(tool|function|call)[^>]*>", tail, re.IGNORECASE))
    tool_pen = -0.2 if tool_like else 0.0
    score = len_score + rep_score + fa_score + tool_pen
    return float(max(-1.0, min(1.0, score)))

# ======================= PPO (LoRA-only training) =======================

def build_ppo_policy_with_lora(args: Args, pc: PrecisionCfg) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    compute_dtype = _dtype_for_model(args.base_model_id, pc)

    tok = AutoTokenizer.from_pretrained(args.lora_ckpt_dir, use_fast=True)
    if tok.eos_token is None:
        if tok.sep_token is not None:
            tok.eos_token = tok.sep_token
        else:
            tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=pc.bnb_compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model_id,
        **model_kwargs,
    )

    new_vocab = len(tok)
    base_vocab = policy.pretrained_model.get_input_embeddings().weight.shape[0]
    if new_vocab != base_vocab:
        policy.pretrained_model.resize_token_embeddings(new_vocab)

    policy.pretrained_model = PeftModel.from_pretrained(
        policy.pretrained_model,
        args.lora_ckpt_dir,
        is_trainable=True,
    )

    for n, p in policy.named_parameters():
        p.requires_grad = False
    for n, p in policy.named_parameters():
        if "lora_" in n or n.startswith("v_head"):
            p.requires_grad = True

    policy.config.pad_token_id = tok.pad_token_id
    policy.config.eos_token_id = tok.eos_token_id
    return policy, tok


def anneal_target_kl(kind: str, epoch_idx: int, total_epochs: int, kl_min: float, kl_max: float) -> float:
    if kind == "none" or total_epochs <= 1:
        return kl_max
    t = epoch_idx / (total_epochs - 1 + 1e-8)
    if kind == "linear":
        return kl_max + (kl_min - kl_max) * t
    if kind == "cosine":
        return kl_min + 0.5*(kl_max - kl_min)*(1 + np.cos(np.pi * (1 - t)))
    return kl_max

def parse_final_answer(text: str) -> Optional[str]:
    tail = text.split("</think>")[-1] if "</think>" in text else text
    tail = tail.strip()
    m = re.search(r"####\s*(.*)$", tail, re.MULTILINE)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d[\d,\.]*", tail)
    if nums:
        return nums[-1]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    return lines[-1] if lines else None

def evaluate_gsm8k_em(model: AutoModelForCausalLM, tok: AutoTokenizer, n: int = 128, device: str = "cuda") -> Dict[str, Any]:
    try:
        ref = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        return {"em": None, "n": 0}

    ref = ref.select(range(min(n, len(ref))))
    correct = 0
    total = len(ref)
    for ex in ref:
        q = ex["question"].strip()
        ans = ex["answer"].strip()
        with torch.no_grad():
            enc = tok([q], return_tensors="pt").to(device)
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = parse_final_answer(text) or ""
        gold_nums = re.findall(r"[-+]?\d[\d,\.]*", ans)
        gold = gold_nums[-1] if gold_nums else ans.strip()
        pred_nums = re.findall(r"[-+]?\d[\d,\.]*", pred)
        predn = pred_nums[-1] if pred_nums else pred.strip()
        correct += int(predn == gold)
    return {"em": correct / max(1, total), "n": total}

def ppo_train(args: Args, sft_dataset: Dataset, pc: PrecisionCfg):
    policy, tok = build_ppo_policy_with_lora(args, pc)
    ref_model = None if args.ref_free else create_reference_model(policy)

    prm = VersaPRM(
        args.reward_model_id,
        device=args.device,
        load_in_4bit=args.prm_load_in_4bit,
        base_model_id=args.prm_base_id,
        prm_compute_dtype=pc.bnb_compute_dtype if args.prm_load_in_4bit else (torch.bfloat16 if _is_bf16_supported() else torch.float16),
    )

    # PPOConfig -> pass precision flags ONLY when supported
    cfg = PPOConfig(
        learning_rate=args.ppo_lr,
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_bs,
        num_ppo_epochs=args.ppo_epochs,
        kl_coef=0.05,
        bf16=pc.trainer_bf16,
        fp16=pc.trainer_fp16,
    )

    
    try:
        trainer = PPOTrainer(
            ppo_config=cfg,           # new TRL versions expect 'ppo_config'
            model=policy,
            ref_model=ref_model,
            tokenizer=tok,
        )
    except TypeError:
        # fallback for older TRL versions that take positional 'config'
        trainer = PPOTrainer(
            cfg,
            model=policy,
            ref_model=ref_model,
            tokenizer=tok,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, args.jsonl_log)
    def log_jsonl(rec: Dict[str, Any]):
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    stop_flag = {"stop": False}
    def handle_sigint(sig, frame):
        print("\n[INFO] Caught signal, saving PPO LoRA and exiting...")
        if isinstance(policy.pretrained_model, PeftModel):
            policy.pretrained_model.save_pretrained(args.ppo_lora_dir)
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    prompts = sft_dataset["prompt"]
    bs = args.ppo_batch_size
    global_step = 0

    gen_kwargs = dict(
        max_new_tokens=args.ppo_max_new_tokens,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        top_k=args.gen_top_k,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        min_new_tokens=args.min_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    for epoch in range(args.ppo_epochs):
        if stop_flag["stop"]:
            break

        for i in range(0, len(prompts), bs):
            if stop_flag["stop"]:
                break

            batch_prompts = prompts[i : i + bs]
            enc = tok(batch_prompts, return_tensors="pt", padding=True).to(args.device)
            attn = enc["attention_mask"]

            num_samples = max(1, args.reward_samples)
            last_query_tensors, last_response_tensors = None, None
            rewards_accum: List[List[float]] = [[] for _ in range(len(batch_prompts))]
            dec_texts_cache: List[str] = [""] * len(batch_prompts)

            for s in range(num_samples):
                with torch.no_grad():
                    gen = policy.generate(**enc, **gen_kwargs)

                query_tensors = []
                response_tensors = []
                decoded_texts = []

                for j in range(len(batch_prompts)):
                    in_len = int(attn[j].sum().item())
                    full = gen[j]
                    query = enc["input_ids"][j, -in_len:].detach()
                    resp = full[in_len:].detach()
                    query_tensors.append(query)
                    response_tensors.append(resp)
                    dec_text = tok.decode(full, skip_special_tokens=False)
                    decoded_texts.append(dec_text)

                for j, text in enumerate(decoded_texts):
                    think = extract_think(text)
                    prm_score = prm.score(batch_prompts[j], think)
                    rule_score = rule_safety_reward(batch_prompts[j], text)
                    total = args.reward_w_prm * prm_score + args.reward_w_rule * rule_score
                    total = max(args.reward_clip_min, min(args.reward_clip_max, total))
                    rewards_accum[j].append(float(total))
                    dec_texts_cache[j] = text

                last_query_tensors = query_tensors
                last_response_tensors = response_tensors

            final_rewards = [float(np.mean(rs)) if len(rs) > 0 else 0.0 for rs in rewards_accum]

            stats = trainer.step(last_query_tensors, last_response_tensors, final_rewards)
            global_step += 1

            log_jsonl({
                "epoch": epoch,
                "global_step": global_step,
                "batch_index": i // bs,
                "reward_mean": float(np.mean(final_rewards)),
                "reward_min": float(np.min(final_rewards)),
                "reward_max": float(np.max(final_rewards)),
                "kl": float(stats.get("kl", 0.0)),
                "entropy_beta": args.entropy_beta,
                "loss/policy": float(stats.get("loss/policy", 0.0)) if isinstance(stats.get("loss/policy", 0.0), (int,float)) else 0.0,
                "loss/value": float(stats.get("loss/value", 0.0)) if isinstance(stats.get("loss/value", 0.0), (int,float)) else 0.0,
            })

            if args.eval_every and (global_step % args.eval_every == 0):
                try:
                    eval_res = evaluate_gsm8k_em(policy.pretrained_model, tok, n=args.eval_gsm8k_n, device=args.device)
                except Exception:
                    eval_res = evaluate_gsm8k_em(policy, tok, n=args.eval_gsm8k_n, device=args.device)
                log_jsonl({"eval_step": global_step, "gsm8k_em": eval_res.get("em"), "gsm8k_n": eval_res.get("n")})
                print(f"[EVAL] step {global_step}: GSM8K EM={eval_res.get('em')} on n={eval_res.get('n')}")

        print(f"[PPO] finished epoch {epoch+1}/{args.ppo_epochs}")

    os.makedirs(args.ppo_lora_dir, exist_ok=True)
    if isinstance(policy.pretrained_model, PeftModel):
        policy.pretrained_model.save_pretrained(args.ppo_lora_dir)
    tok.save_pretrained(args.ppo_lora_dir)

    final_dir = os.path.join(args.output_dir, "ppo_final_policy")
    trainer.save_pretrained(final_dir)

    if args.hf_repo:
        api = HfApi(token=args.hf_token)
        api.create_repo(args.hf_repo, exist_ok=True)
        api.upload_folder(folder_path=args.ppo_lora_dir, repo_id=args.hf_repo)

# ======================= Merge after PPO =======================

def merge_after_ppo(args: Args, pc: PrecisionCfg) -> str:
    print("[MERGE] Loading base and PPO LoRA for final merge...")
    dtype_for_merge = _dtype_for_model(args.base_model_id, pc)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype_for_merge,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, args.ppo_lora_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    os.makedirs(args.merged_path, exist_ok=True)
    merged.save_pretrained(args.merged_path, safe_serialization=True, max_shard_size=args.max_shard_size)
    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(args.merged_path)
    torch.cuda.empty_cache()
    print(f"[MERGE] Saved merged model to {args.merged_path}")
    return args.merged_path

# ======================= Manifest =======================

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

def write_manifest(args: Args, pc: PrecisionCfg, stage: str, extra: Dict[str, Any]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "git_commit": get_git_commit(),
        "stage": stage,
        "args": asdict(args),
        "precision": {
            "mode": pc.mode,
            "model_dtype": str(pc.model_dtype).split(".")[-1],
            "bnb_compute_dtype": str(pc.bnb_compute_dtype).split(".")[-1],
            "trainer_bf16": pc.trainer_bf16,
            "trainer_fp16": pc.trainer_fp16,
            "note": pc.note,
        },
        "versions": library_versions(),
        "gpu": gpu_info(),
    }
    manifest.update(extra)
    path = os.path.join(args.output_dir, args.manifest_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] wrote {path}")

# ======================= Main =======================

def main(argv: List[str]):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pc = choose_precision(args.precision, args.base_model_id)
    _print_precision_banner(pc)

    write_manifest(args, pc, stage="start", extra={})

    if args.run_sft:
        sft_dir = sft_train(args, pc)  # saves adapters to lora_ckpt_dir
        write_manifest(args, pc, stage="post_sft", extra={"sft_lora_dir": sft_dir})
    else:
        print("[INFO] Skipping SFT; expecting existing adapters in --lora-ckpt-dir")

    ds = build_sft_dataset(args)
    n = min(128000, len(ds))
    rl_ds = ds.shuffle(seed=args.seed).select(range(n))

    ppo_train(args, rl_ds, pc)
    write_manifest(args, pc, stage="post_ppo", extra={"ppo_lora_dir": args.ppo_lora_dir})

    if args.run_merge_after_ppo:
        merged = merge_after_ppo(args, pc)
        write_manifest(args, pc, stage="done", extra={"merged_path": merged})
    else:
        write_manifest(args, pc, stage="done", extra={"merged_path": None})

if __name__ == "__main__":
    main(sys.argv[1:])
