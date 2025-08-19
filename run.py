#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # we're not using JAX here
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import os, re, json, time, signal, warnings, random, subprocess, sys, inspect, math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

from packaging.version import parse as V
import trl as trl_lib
# 👇 place this right after "import trl as trl_lib"
import trl.trainer.utils as _trl_utils

if not getattr(_trl_utils, "_safe_generate_patched", False):
    _orig_generate = _trl_utils.generate

    def _safe_generate(*args, **kwargs):
        with no_dynamo():                 # hard-stop any Dynamo compile of the gen path
            return _orig_generate(*args, **kwargs)

    _trl_utils.generate = _safe_generate
    _trl_utils._safe_generate_patched = True
    print("[PATCH] TRL utils.generate wrapped with no_dynamo()")
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)
from huggingface_hub import HfApi
from peft import PeftModel

warnings.filterwarnings("ignore", category=UserWarning)
from contextlib import contextmanager
from contextlib import nullcontext

def no_dynamo():
    try:
        import torch._dynamo as dynamo
        return dynamo.disable()   # real context manager in PyTorch 2.x
    except Exception:
        return nullcontext()


def _force_return_dict_on_forward(m):
    if m is None:
        return
    base = (getattr(m, "pretrained_model", None)
            or getattr(m, "policy_model", None)
            or getattr(m, "actor_model", None)
            or getattr(m, "model", None)
            or m)
    try:
        base.config.return_dict = True
    except Exception:
        pass
    orig_forward = base.forward
    def _fwd(*args, **kwargs):
        kwargs.setdefault("return_dict", True)
        return orig_forward(*args, **kwargs)
    base.forward = _fwd

def _force_logit_object_forward(m):
    """
    Make sure forward(..., return_dict=True) and coerce tuple outputs to an
    object with .logits, so TRL's PPOTrainer doesn't crash.
    """
    if m is None or not hasattr(m, "forward"):
        return
    raw_fwd = m.forward

    def fwd(*a, **k):
        k.setdefault("return_dict", True)
        out = raw_fwd(*a, **k)
        if isinstance(out, tuple):
            # assume first element are the logits (HF default when return_dict=False)
            class _O: pass
            o = _O()
            o.logits = out[0]
            return o
        return out

    try:
        # also set the config flag if present
        if hasattr(m, "config"):
            m.config.return_dict = True
    except Exception:
        pass

    m.forward = fwd

# --- TRL 0.21+ wrappers: ensure .generate and writable configs on PolicyAndValueWrapper ---
try:
    try:
        from trl.trainer.ppo_trainer import PolicyAndValueWrapper as _PAVW
    except Exception:
        _PAVW = None
    if _PAVW is None:
        try:
            from trl.trainer.ppo_trainer import PolicyAndValueModel as _PAVW
        except Exception:
            _PAVW = None

    def _resolve_policy_like(self):
        return (getattr(self, "policy_model", None)
                or getattr(self, "actor_model", None)
                or getattr(self, "model", None))

    if _PAVW is not None:
        if not hasattr(_PAVW, "generate"):
            def _paw_generate(self, *args, **kwargs):
                pm = _resolve_policy_like(self)
                if pm is None or not hasattr(pm, "generate"):
                    raise AttributeError("Policy wrapper has no underlying policy_model with .generate")
                return pm.generate(*args, **kwargs)
            _PAVW.generate = _paw_generate

        def _get_generation_config(self):
            pm = _resolve_policy_like(self)
            return getattr(pm, "generation_config", getattr(self, "_shadow_generation_config", None))
        def _set_generation_config(self, value):
            pm = _resolve_policy_like(self)
            try:
                if pm is not None:
                    pm.generation_config = value
            finally:
                try:
                    self.__dict__["_shadow_generation_config"] = value
                except Exception:
                    pass
        _PAVW.generation_config = property(_get_generation_config, _set_generation_config)

        def _get_config(self):
            pm = _resolve_policy_like(self)
            return getattr(pm, "config", getattr(self, "_shadow_config", None))
        def _set_config(self, value):
            pm = _resolve_policy_like(self)
            try:
                if pm is not None:
                    pm.config = value
            finally:
                try:
                    self.__dict__["_shadow_config"] = value
                except Exception:
                    pass
        _PAVW.config = property(_get_config, _set_config)
except Exception as _e:
    print(f"[WARN] PolicyAndValueWrapper monkey-patch skipped: {_e}")

def _fix_trl_valuehead_base_prefix(model):
    try:
        getattr(model, "base_model_prefix")
        return model
    except AttributeError:
        pass
    if hasattr(model, "pretrained_model"):
        model.base_model_prefix = "pretrained_model"
    elif hasattr(model, "model"):
        model.base_model_prefix = "model"
    elif hasattr(model, "transformer"):
        model.base_model_prefix = "transformer"
    else:
        model.base_model_prefix = ""
    return model

# ----- Noop reward model stub (kept for compatibility) -----
class NoopRewardModel(nn.Module):
    def __init__(self, default_value: float = 0.0, device: str | None = None):
        super().__init__()
        self.default_value = float(default_value)
        self.register_buffer("_device_dummy", torch.tensor(0.0), persistent=False)
        self._device_hint = device

    def forward(self, *args, **kwargs) -> torch.Tensor:
        bs = 1
        for key in ("input_ids","completions","responses","samples","sequences","queries","prompts"):
            obj = kwargs.get(key, None)
            if obj is None:
                continue
            try:
                bs = len(obj)
                break
            except Exception:
                if hasattr(obj, "shape"):
                    bs = int(obj.shape[0])
                    break
        dev = self._device_hint or self._device_dummy.device
        return torch.full((bs,), self.default_value, dtype=torch.float32, device=dev)

def ensure_generation_config(model, tok):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        base = getattr(model, "pretrained_model", None)
        if base is not None and getattr(base, "generation_config", None) is not None:
            gen_cfg = base.generation_config
        else:
            gen_cfg = GenerationConfig.from_model_config(model.config)
        setattr(model, "generation_config", gen_cfg)

    if getattr(gen_cfg, "eos_token_id", None) is None:
        gen_cfg.eos_token_id = tok.eos_token_id
    if getattr(gen_cfg, "pad_token_id", None) is None:
        gen_cfg.pad_token_id = tok.pad_token_id

    base = getattr(model, "pretrained_model", None)
    if base is not None:
        base.generation_config = gen_cfg
        base.config.eos_token_id = tok.eos_token_id
        base.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id

# === version-flexible PPOTrainer constructor ===
def make_ppo_trainer(cfg, policy, ref_model, tok, train_dataset, data_collator=None, reward_model=None):
    actor = getattr(policy, "pretrained_model", None) or getattr(policy, "model", None) or policy
    critic = policy

    ensure_generation_config(critic, tok)
    if actor is not critic:
        ensure_generation_config(actor, tok)

    def value_for(name: str):
        critic = policy
        try:
            _dev = next(critic.parameters()).device
        except Exception:
            _dev = None
        if name in ("ppo_config", "config", "args"):
            return cfg
        if name in ("policy_model", "actor_model"):
            return actor
        if name in ("value_model", "critic_model"):
            return critic
        if name in ("model",):
            return critic
        if name in ("ref_model", "reference_model"):
            return ref_model
        if name in ("ref_policy_model",):
            return getattr(ref_model, "pretrained_model", ref_model) if ref_model is not None else None
        if name in ("processing_class", "tokenizer"):
            return tok
        if name in ("train_dataset", "dataset"):
            return train_dataset
        if name in ("data_collator",):
            return data_collator
        if name in ("reward_model",):
            return reward_model if reward_model is not None else NoopRewardModel(device=_dev)
        return None

    sig = inspect.signature(PPOTrainer.__init__)
    params = list(sig.parameters.values())[1:]

    pos_args = []
    kw_args: Dict[str, Any] = {}

    for p in params:
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            val = value_for(p.name)
            if val is None and p.default is inspect._empty:
                if p.name.lower().startswith("policy"):
                    val = actor
                elif p.name.lower().startswith("value"):
                    val = critic
                elif p.name.lower().startswith("ref"):
                    val = getattr(ref_model, "pretrained_model", ref_model) if ref_model is not None else None
                elif "config" in p.name:
                    val = cfg
            pos_args.append(val)
        else:
            val = value_for(p.name)
            if val is not None:
                kw_args[p.name] = val

    required_names = {p.name for p in params if (p.default is inspect._empty and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))}
    for rn in required_names:
        if rn not in kw_args:
            val = value_for(rn)
            if val is not None:
                kw_args[rn] = val

    if "data_collator" in kw_args and kw_args["data_collator"] is None:
        kw_args.pop("data_collator")

    try:
        return PPOTrainer(*pos_args, **kw_args)
    except TypeError as e1:
        try:
            return PPOTrainer(cfg, critic, ref_model, tokenizer=tok, dataset=train_dataset, data_collator=data_collator)
        except TypeError:
            kw2 = dict(kw_args)
            if "processing_class" in kw2:
                pc = kw2.pop("processing_class")
                kw2["tokenizer"] = pc
            try:
                return PPOTrainer(*pos_args, **kw2)
            except TypeError as e3:
                raise TypeError(f"PPOTrainer construction failed.\nFirst error: {e1}\nThird error: {e3}\nSignature seen: {sig}\npos_args={ [type(a).__name__ for a in pos_args] }\nkw={ list(kw_args.keys()) }")

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
    if not torch.cuda.is_available():
        return False
    major, _ = _gpu_capability()
    return major >= 8

def _is_bf16_supported() -> bool:
    return _bf16_hw_supported()

def _is_gemma3(model_id: str) -> bool:
    return "gemma-3" in (model_id or "").lower()

@dataclass
class PrecisionCfg:
    mode: str
    model_dtype: torch.dtype
    bnb_compute_dtype: torch.dtype
    trainer_bf16: bool
    trainer_fp16: bool
    note: str

def choose_precision(mode: Optional[str], base_model_id: str) -> PrecisionCfg:
    req = (mode or os.environ.get("PRECISION") or "auto").lower()
    cuda = torch.cuda.is_available()
    bf16_ok = _bf16_hw_supported()

    def cfg(model_dtype, bnb_dtype, bf16_flag, fp16_flag, note):
        return PrecisionCfg(req, model_dtype, bnb_dtype, bf16_flag, fp16_flag, note)

    if req not in {"auto","bf16","fp16","fp32"}:
        req = "auto"

    if req == "bf16":
        if not bf16_ok:
            if _is_gemma3(base_model_id):
                return cfg(torch.float32, torch.float16, False, False, "forced bf16 but unsupported; Gemma-3 -> fp32")
            return cfg(torch.float16, torch.float16, False, True, "forced bf16 but unsupported; fallback fp16")
        return cfg(torch.bfloat16, torch.bfloat16, True, False, "forced bf16")

    if req == "fp16":
        if _is_gemma3(base_model_id):
            return cfg(torch.float32, torch.float16, False, False, "Gemma-3 forbids fp16 -> using fp32")
        return cfg(torch.float16, torch.float16, False, True, "forced fp16")

    if req == "fp32":
        return cfg(torch.float32, torch.float16, False, False, "forced fp32")

    if cuda and bf16_ok:
        return cfg(torch.bfloat16, torch.bfloat16, True, False, "auto->bf16 (Ampere/Hopper)")
    if cuda:
        if _is_gemma3(base_model_id):
            return cfg(torch.float32, torch.float16, False, False, "auto on Turing + Gemma-3 -> fp32")
        return cfg(torch.float16, torch.float16, False, True, "auto on pre-Ampere -> fp16")
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
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b-it")
    reward_model_id: str = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM-Base-3B")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    run_sft: bool = True
    run_merge_after_ppo: bool = True
    rl_on_lora: bool = True

    merged_path: str = "./gemma3_cot_sft_merged"
    lora_ckpt_dir: str = "./gemma3_cot_sft_lora"
    ppo_lora_dir: str = "./gemma3_cot_ppo_lora"
    output_dir: str = "./runs/gemma3_cot"

    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    prm_load_in_4bit: bool = True
    prm_base_id: Optional[str] = os.environ.get("PRM_BASE_ID")

    max_seq_len: int = 2048
    sft_epochs: int = 1
    sft_lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    load_in_4bit: bool = True
    seed: int = 42
    pack_sequences: bool = False

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

    reward_samples: int = 1
    reward_temperature: float = 0.7

    longtalk_n: int = 0
    gsm8k_n: int = 0
    mmlu_pro_n: int = 1000

    ref_free: bool = False
    entropy_beta: float = 0.0
    kl_anneal: str = "none"
    kl_min: float = 1.0
    kl_max: float = 6.0
    eval_every: int = 100
    eval_gsm8k_n: int = 128
    reward_w_prm: float = 1.0
    reward_w_rule: float = 0.2  # repurposed below as w_format (kept for CLI compat)
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    fsdp: bool = False
    deepspeed_config: Optional[str] = None
    manifest_name: str = "manifest.json"
    jsonl_log: str = "metrics.jsonl"
    hf_repo: Optional[str] = os.environ.get("HF_REPO")
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")
    max_shard_size: str = "2GB"

    precision: str = os.environ.get("PRECISION", "auto")

    # ===== New: Faithfulness knobs =====
    reward_w_consistency: float = 0.6   # AFC (answer-from-chain) main driver
    reward_w_format: float = 0.2        # structure/anti-spam (reuses old --reward-w-rule if passed)
    reward_w_nli: float = 0.2           # entailment weight (auto-gated)
    use_nli: bool = True                # enable NLI component
    nli_model_id: str = "microsoft/deberta-base-mnli"
    consistency_num_tol: float = 1e-4   # numeric tolerance for AFC
    nli_dynamic_gate: bool = True       # auto-disable NLI for open-ended/code/writing

    # ===== New: length prior + compression knobs =====
    len_prior_mu_easy: int = 40         # target think length (tokens) for easy prompts
    len_prior_mu_hard: int = 220        # target think length for hard prompts
    len_prior_sigma: float = 0.8        # log-normal width
    len_prior_w: float = 0.25           # weight of length prior in format reward
    compression_w: float = 0.15         # weight for 1–2 sentence explanation tie-back
    compression_max_chars: int = 320    # only look at this many chars after </think> for tie-back

def parse_args(argv: List[str]) -> Args:
    import argparse
    p = argparse.ArgumentParser(description="Gemma-3 CoT SFT + PPO with faithfulness reward (AFC + gated NLI + format + optional PRM)")

    # PRM opts
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
    p.add_argument("--reward-w-rule", type=float, default=None)  # legacy; mapped to w_format
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

    # ===== New: Faithfulness knobs =====
    p.add_argument("--reward-w-consistency", type=float, default=None)
    p.add_argument("--reward-w-format", type=float, default=None)
    p.add_argument("--reward-w-nli", type=float, default=None)
    p.add_argument("--use-nli", action="store_true", default=None)
    p.add_argument("--no-use-nli", dest="use_nli", action="store_false")
    p.add_argument("--nli-model-id", type=str, default=None)
    p.add_argument("--consistency-num-tol", type=float, default=None)
    p.add_argument("--nli-dynamic-gate", action="store_true", default=None)
    p.add_argument("--no-nli-dynamic-gate", dest="nli_dynamic_gate", action="store_false")

    # ===== New: length prior + compression CLI =====
    p.add_argument("--len-prior-mu-easy", type=int, default=None)
    p.add_argument("--len-prior-mu-hard", type=int, default=None)
    p.add_argument("--len-prior-sigma", type=float, default=None)
    p.add_argument("--len-prior-w", type=float, default=None)
    p.add_argument("--compression-w", type=float, default=None)
    p.add_argument("--compression-max-chars", type=int, default=None)

    ns = p.parse_args(argv)
    args = Args()
    for k, v in vars(ns).items():
        if v is not None:
            setattr(args, k.replace("-", "_"), v)
    if args.kl_anneal != "none":
        args.ppo_target_kl = args.kl_max

    # keep legacy mapping: --reward-w-rule == format weight unless explicitly overridden
    if ns.reward_w_rule is not None and ns.reward_w_format is None:
        args.reward_w_format = ns.reward_w_rule

    return args

# ======================= Data =======================
_SINGLE_ANGLE_OPEN = "‹"
_SINGLE_ANGLE_CLOSE = "›"

def _normalize_angle_tags(text: str) -> str:
    if not text:
        return text
    return text.replace(_SINGLE_ANGLE_OPEN, "<").replace(_SINGLE_ANGLE_CLOSE, ">")

def upcast_linear_inputs_to_weight_dtype(module: nn.Module) -> None:
    if module is None:
        return
    for m in module.modules():
        if isinstance(m, nn.Linear):
            old_fwd = m.forward
            def new_fwd(x, *args, _old=old_fwd, _m=m, **kw):
                w = getattr(_m, "weight", None)
                if w is not None and x.dtype != w.dtype:
                    x = x.to(w.dtype)
                return _old(x, *args, **kw)
            m.forward = new_fwd

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
            m = re.search(r"<{tag}>(.*?)</{tag}>".format(tag=tag), gen, re.DOTALL)
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

# ======================= Tokenizer & collators =======================
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

def make_ppo_query_collator(tokenizer: AutoTokenizer, max_len: int):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    def collate(batch):
        prompts = []
        for b in batch:
            p = b.get("prompt") or b.get("query") or b.get("question") or ""
            prompts.append(str(p))
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    return collate

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
    if _is_gemma3(model_id) and pc.model_dtype == torch.float16:
        return torch.float32
    return pc.model_dtype

def sft_train(args: Args, pc: PrecisionCfg) -> str:
    from unsloth import FastLanguageModel
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    desired_dtype = _dtype_for_model(args.base_model_id, pc)
    dtype_for_unsloth: Optional[torch.dtype] = None if desired_dtype == torch.float32 else desired_dtype

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_len,
        dtype=dtype_for_unsloth,
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
    model.save_pretrained(args.lora_ckpt_dir)
    tokenizer.save_pretrained(args.lora_ckpt_dir)
    torch.cuda.empty_cache()
    return args.lora_ckpt_dir

# ======================= Reward (VersaPRM) =======================
class VersaPRM:
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
            self.model = PeftModel.from_pretrained(base, model_id, is_trainable=False)
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
        )

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

# ======================= Faithful Reasoning Reward =======================
# Local final-answer extractor (more robust for code)
def _extract_final_answer_from_tail(tail: str) -> str:
    s = tail.strip()
    if not s:
        return ""
    # prefer fenced code if present
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    # common labels
    for pat in [r"(?i)(?:final answer|answer|thus|therefore)\s*:\s*(.*)$"]:
        m = re.search(pat, s, re.DOTALL)
        if m:
            return m.group(1).strip()
    # ### pattern
    m = re.search(r"####\s*(.*)$", s, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # fallback: last non-empty paragraph
    parts = [p.strip() for p in s.split("\n\n") if p.strip()]
    if parts:
        return parts[-1]
    # absolute fallback: last non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def _extract_last_number(text: str) -> Optional[float]:
    nums = _NUM_RE.findall(text or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except Exception:
        return None

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _answer_type(final_answer: str, prompt: str) -> str:
    a = (final_answer or "").strip()
    if not a:
        return "empty"
    na = _normalize_text(a)
    if na in {"yes","no","true","false"}:
        return "bool"
    if re.search(r"\b[A-D]\)\s", prompt) or re.search(r"\b[A-D]\.\s", prompt):
        if re.fullmatch(r"[A-Da-d]", a.strip()):
            return "mc_letter"
    if _to_float(a) is not None:
        return "numeric"
    # If it's very long (> 300 chars), treat as freeform/code
    if len(a) >= 300 or "```" in a:
        return "freeform"
    return "string"

def _tokenize_text_simple(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (s or "").lower())

def _difficulty_score(prompt: str) -> float:
    """
    Crude 0..1 difficulty proxy to set the chain-length prior.
    Uses prompt length, presence of digits/operators, and a few math/physics keywords.
    """
    p = prompt or ""
    tokens = len(re.findall(r"\S+", p))
    digits = sum(ch.isdigit() for ch in p)
    eq = len(re.findall(r"[=+\-*/^]", p))
    flags = 0
    for k in ["prove", "derive", "integral", "limit", "matrix", "vector",
              "physics", "acceleration", "force", "charge", "momentum"]:
        flags += int(k in p.lower())
    s = 0.4 * min(1.0, tokens/120) + 0.3 * min(1.0, (digits+eq)/30) + 0.3 * min(1.0, flags/4)
    return max(0.0, min(1.0, s))

def _coverage_and_tail_overlap(chain: str, final_answer: str) -> float:
    """
    For free-text/code answers: reward when final answer is derivable from chain.
    - content-word coverage
    - tail overlap (last ~600 chars of chain)
    """
    ch = (chain or "")
    fa = (final_answer or "")
    if not ch or not fa:
        return 0.0
    ch_norm = _normalize_text(ch)
    tail = ch_norm[-800:]

    # content tokens from final (len >= 4)
    f_toks = [w for w in _tokenize_text_simple(fa) if len(w) >= 4]
    if not f_toks:
        return 1.0 if _normalize_text(fa) in tail else 0.0

    uniq = sorted(set(f_toks))
    hits = sum(1 for w in uniq if w in ch_norm)
    cov = hits / max(1, len(uniq))

    # encourage explicit presence of the answer (or big chunks) near tail
    near = 0.0
    if _normalize_text(fa) in tail:
        near = 1.0
    else:
        # n-gram-ish: check presence of long tokens in tail
        long_hits = sum(1 for w in uniq if (len(w) >= 6 and w in tail))
        near = min(1.0, long_hits / max(1, len([w for w in uniq if len(w) >= 6])))

    # combine coverage and tail presence
    return float(min(1.0, 0.6 * cov + 0.4 * near))

def _roleplay_or_creative(prompt: str) -> bool:
    """Detect RP/story/creative prompts where we want short planning and no entailment checks."""
    txt = (prompt or "").lower()
    return any(k in txt for k in [
        "roleplay", "in character", "write a story", "story about", "narrate",
        "poem", "song", "lyrics", "script", "as a ", "you are "
    ])

def _format_reward(text: str,
                   min_tokens: int = 8,
                   max_tokens: int = 4096,
                   max_repeat_ngram: int = 3,
                   prompt: str = "",
                   args: Optional[Args] = None) -> float:
    """
    Format/anti-spam + difficulty-aware length prior + explanation compression tie-back.
    Returns roughly in [-1, 1].
    """
    if not text or not text.strip():
        return -0.8
    lo = text.find("<think>")
    hi = text.find("</think>")
    if lo == -1 or hi == -1 or hi <= lo:
        return -0.6
    think_txt = text[lo + len("<think>"):hi]
    toks = re.findall(r"\S+", think_txt)
    T = len(toks)
    if T == 0:
        return -0.5

    # Base anti-spam (NRP + diversity)
    score = 0.2
    if T < min_tokens:
        score -= 0.2
    if T > max_tokens:
        score -= 0.2
    def rep_count(n):
        ngrams = [" ".join(toks[i:i+n]) for i in range(max(0, T-n+1))]
        if not ngrams:
            return 0
        from collections import Counter
        c = Counter(ngrams)
        return sum(v for v in c.values() if v >= max_repeat_ngram)
    score -= 0.12 * min(3, rep_count(2))
    score -= 0.08 * min(3, rep_count(3))
    uniq = len(set(toks)) / max(1, T)
    if uniq >= 0.35: score += 0.12
    elif uniq <= 0.15: score -= 0.12
    if re.search(r"<\s*(tool|function|call|api|request)\b", think_txt, re.I):
        score -= 0.2

    # Difficulty-aware length prior (log-normal over token count)
    if args is not None:
        if _roleplay_or_creative(prompt):
            mu = 24
        else:
            d = _difficulty_score(prompt)  # 0..1
            mu = getattr(args, "len_prior_mu_easy", 40) + d * (
                getattr(args, "len_prior_mu_hard", 220) - getattr(args, "len_prior_mu_easy", 40)
            )
        sigma = getattr(args, "len_prior_sigma", 0.8)
        w = getattr(args, "len_prior_w", 0.25)
        if T > 0:
            z = (math.log(max(1, T)) - math.log(max(1.0, mu))) / max(1e-6, sigma)
            prior = math.exp(-0.5 * z * z)
            prior = max(0.0, min(1.0, prior))
            score += w * (prior - 0.5)

    # Compression tie-back: short “why” near the end should reuse tail-of-chain tokens
    if args is not None:
        w_c = getattr(args, "compression_w", 0.15)
        tail = text.split("</think>")[-1] if "</think>" in text else text
        tail = (tail or "").strip()[: getattr(args, "compression_max_chars", 320)]
        ch_norm = _normalize_text(think_txt)[-800:]
        exp_norm = _normalize_text(tail)
        if exp_norm:
            toks_f = [w for w in _tokenize_text_simple(exp_norm) if len(w) >= 4]
            uniq_f = sorted(set(toks_f))
            hits = sum(1 for w in uniq_f if w in ch_norm)
            cov = hits / max(1, len(uniq_f))
            score += w_c * (cov - 0.5)

    return float(max(-1.0, min(1.0, score)))

# Optional NLI
class _Entailer:
    def __init__(self, model_id: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
    @torch.no_grad()
    def entail_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tok.encode_plus(premise, hypothesis, return_tensors="pt",
                                      truncation=True, max_length=2048).to(self.device)
        out = self.model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        return float(probs[-1].item())  # entailment

def _nli_should_apply(prompt: str, chain: str, final: str) -> bool:
    # Heuristics: turn off for code / very long open-ended answers
    txt = (prompt + " " + chain + " " + final).lower()
    if "```" in txt:
        return False
    if any(k in txt for k in ["def ", "class ", "import ", "#include", "public static", "fn ", "lambda ", "=>"]):
        return False
    if len(final) > 600:
        return False
    # If the answer is clearly numeric / MC / short QA, allow
    return True

class ReasoningFaithfulnessRewardModel(nn.Module):
    def __init__(self,
                 tok,
                 prm=None,
                 w_format: float = 0.2,
                 w_consistency: float = 0.6,
                 w_nli: float = 0.2,
                 use_nli: bool = True,
                 nli_model_id: str = "microsoft/deberta-base-mnli",
                 num_tol: float = 1e-4,
                 device: str = "cuda",
                 dynamic_gate: bool = True):
        super().__init__()
        self.tok = tok
        self.prm = prm
        self.w_format = float(w_format)
        self.w_consistency = float(w_consistency)
        self.w_nli = float(w_nli) if use_nli else 0.0
        self.num_tol = float(num_tol)
        self.dynamic_gate = bool(dynamic_gate)

        self.entailer = None
        if use_nli and self.w_nli > 0.0:
            try:
                self.entailer = _Entailer(nli_model_id, device=device)
            except Exception:
                self.entailer = None
                self.w_nli = 0.0

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        device = input_ids.device
        texts = self.tok.batch_decode(input_ids, skip_special_tokens=False)
        rewards: List[float] = []

        for text in texts:
            prompt = text.split("<think>")[0].strip()
            chain = extract_think(text)
            tail = text.split("</think>")[-1] if "</think>" in text else text
            final = _extract_final_answer_from_tail(tail)

            # 1) format (+ length prior + compression tie-back)
            r_fmt = _format_reward(text, prompt=prompt, args=getattr(self, "_args", None))  # [-1,1]

            # 2) AFC consistency
            t = _answer_type(final, prompt)
            if t == "numeric":
                fa = _to_float(final)
                lc = _extract_last_number(chain)
                if fa is None or lc is None:
                    r_afc = 0.0
                else:
                    denom = max(1.0, abs(fa))
                    ok = abs(fa - lc) <= max(self.num_tol, 1e-6 * denom)
                    r_afc = 1.0 if ok else 0.0
            elif t == "mc_letter":
                ch_norm = _normalize_text(chain)
                tail_norm = ch_norm[-600:]
                ans_norm = _normalize_text(final)
                if re.search(rf"(answer|thus|therefore|so)\s*(is|:)?\s*{ans_norm}\b", tail_norm):
                    r_afc = 1.0
                else:
                    r_afc = 0.6 if re.search(rf"\b{ans_norm}\b", ch_norm) else 0.0
            elif t == "bool":
                ch_norm = _normalize_text(chain)
                ans_norm = _normalize_text(final)
                tail_norm = ch_norm[-400:]
                if any(w in tail_norm for w in ["therefore", "so", "hence", "conclude"]) and re.search(rf"\b{ans_norm}\b", tail_norm):
                    r_afc = 1.0
                else:
                    r_afc = 0.6 if re.search(rf"\b{ans_norm}\b", ch_norm) else 0.0
            elif t in ("freeform", "string"):
                r_afc = _coverage_and_tail_overlap(chain, final)
            else:
                r_afc = 0.0

            # 3) NLI entailment (auto gated)
            r_nli = 0.0
            if self.entailer is not None and chain and final and self.w_nli > 0.0:
                if (not self.dynamic_gate) or _nli_should_apply(prompt, chain, final):
                    hyp = f"The answer is {final}."
                    try:
                        r_nli = self.entailer.entail_prob(chain, hyp)  # 0..1
                    except Exception:
                        r_nli = 0.0

            # 4) PRM step quality (small additive)
            r_prm = 0.0
            if self.prm is not None:
                try:
                    r_prm = float(self.prm.score(prompt, chain))  # 0..1
                except Exception:
                    r_prm = 0.0

            raw = (self.w_format * (0.5 + 0.5 * r_fmt) +
                   self.w_consistency * r_afc +
                   self.w_nli * r_nli +
                   max(0.0, 0.3 * r_prm))

            centered = (raw - 0.5) * 2.0
            centered = max(-1.0, min(1.0, centered))
            rewards.append(float(centered))

        logits = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        return SequenceClassifierOutput(logits=logits)

# ======================= PPO policy build =======================
from importlib import reload
import transformers.models.gemma3.modeling_gemma3 as _gemma3
reload(_gemma3)

def build_ppo_policy_with_lora(args: Args, pc: PrecisionCfg) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    is_gemma3 = _is_gemma3(args.base_model_id)
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

    compute_dtype = _dtype_for_model(args.base_model_id, pc)
    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.float32 if is_gemma3 else compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if (not is_gemma3) and args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pc.bnb_compute_dtype,
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

    upcast_linear_inputs_to_weight_dtype(policy.pretrained_model)

    if is_gemma3:
        policy.pretrained_model.to(dtype=torch.float32)
        policy.v_head.to(dtype=torch.float32)

    for n, p in policy.named_parameters():
        p.requires_grad = False
    for n, p in policy.named_parameters():
        if "lora_" in n or n.startswith("v_head"):
            p.requires_grad = True

    policy.config.pad_token_id = tok.pad_token_id
    policy.config.eos_token_id = tok.eos_token_id
    if hasattr(policy, "pretrained_model") and hasattr(policy.pretrained_model, "config"):
        policy.pretrained_model.config.pad_token_id = tok.pad_token_id
        policy.pretrained_model.config.eos_token_id = tok.eos_token_id
    policy.config.use_cache = False
    if hasattr(policy, "pretrained_model") and hasattr(policy.pretrained_model, "config"):
        policy.pretrained_model.config.use_cache = False

    _fix_trl_valuehead_base_prefix(policy)

    # Disable grad checkpointing everywhere
    _fully_disable_gc(policy)

    try:
        import torch._dynamo as dynamo
        if hasattr(policy, "forward"):
            policy.forward = dynamo.disable(policy.forward)
        if hasattr(policy, "generate"):
            policy.generate = dynamo.disable(policy.generate)
        if hasattr(policy, "pretrained_model") and hasattr(policy.pretrained_model, "forward"):
            policy.pretrained_model.forward = dynamo.disable(policy.pretrained_model.forward)
    except Exception:
        pass

    try:
        policy.config.return_dict = True
        if hasattr(policy, "pretrained_model"):
            policy.pretrained_model.config.return_dict = True
    except Exception:
        pass

    return policy, tok

def _fully_disable_gc(model) -> None:
    if model is None:
        return
    for m in (model, getattr(model, "pretrained_model", None)):
        if m is None:
            continue
        try:
            m.config.gradient_checkpointing = False
        except Exception:
            pass
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass
    backbone = getattr(model, "pretrained_model", model)
    try:
        for mod in backbone.modules():
            if hasattr(mod, "gradient_checkpointing"):
                try:
                    mod.gradient_checkpointing = False
                except Exception:
                    pass
            if not hasattr(mod, "_gradient_checkpointing_func"):
                setattr(mod, "_gradient_checkpointing_func",
                        lambda f, *a, **k: f(*a, **k))
    except Exception:
        pass

# ======================= PPO loop =======================
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
            with no_dynamo():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred = parse_final_answer(text) or ""
        gold_nums = re.findall(r"[-+]?\d[\d,\.]*", ans)
        gold = gold_nums[-1 if gold_nums else 0] if gold_nums else ans.strip()
        pred_nums = re.findall(r"[-+]?\d[\d,\.]*", pred)
        predn = pred_nums[-1 if pred_nums else 0] if pred_nums else pred.strip()
        correct += int(predn == gold)
    return {"em": correct / max(1, total), "n": total}

def ppo_train(args: Args, sft_dataset: Dataset, pc: PrecisionCfg):
    policy, tok = build_ppo_policy_with_lora(args, pc)
    ref_model = None if args.ref_free else create_reference_model(policy)
    if ref_model is not None:
        _fix_trl_valuehead_base_prefix(ref_model)
        _fully_disable_gc(ref_model)
        upcast_linear_inputs_to_weight_dtype(getattr(ref_model, "pretrained_model", ref_model))
        try:
            import torch._dynamo as dynamo
            if hasattr(ref_model, "forward"):
                ref_model.forward = dynamo.disable(ref_model.forward)
            if hasattr(ref_model, "generate"):
                ref_model.generate = dynamo.disable(ref_model.generate)
            if hasattr(ref_model, "pretrained_model") and hasattr(ref_model.pretrained_model, "forward"):
                ref_model.pretrained_model.forward = dynamo.disable(ref_model.pretrained_model.forward)
        except Exception:
            pass
    _force_return_dict_on_forward(policy)
    _force_return_dict_on_forward(ref_model)
    ensure_generation_config(policy, tok)

    # ---- Reward wiring ----
    prm = VersaPRM(
        args.reward_model_id,
        device=args.device,
        load_in_4bit=args.prm_load_in_4bit,
        base_model_id=args.prm_base_id,
        prm_compute_dtype=pc.bnb_compute_dtype if args.prm_load_in_4bit else (torch.bfloat16 if _is_bf16_supported() else torch.float16),
    ) if (args.reward_w_prm != 0.0) else None

    rm = ReasoningFaithfulnessRewardModel(
        tok,
        prm=prm,
        w_format=args.reward_w_format,
        w_consistency=args.reward_w_consistency,
        w_nli=args.reward_w_nli,
        use_nli=args.use_nli,
        nli_model_id=args.nli_model_id,
        num_tol=args.consistency_num_tol,
        device=args.device if torch.cuda.is_available() else "cpu",
        dynamic_gate=args.nli_dynamic_gate,
    )
    # pass Args to RM for length prior & compression
    rm._args = args

    ppo_collator = make_ppo_query_collator(tok, args.max_seq_len)

    cfg = PPOConfig(
        learning_rate=args.ppo_lr,
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_bs,
        num_ppo_epochs=args.ppo_epochs,
        kl_coef=0.05,
        bf16=pc.trainer_bf16,
        fp16=pc.trainer_fp16,
    )

    trainer = make_ppo_trainer(
        cfg, policy, ref_model, tok,
        train_dataset=sft_dataset,
        data_collator=ppo_collator,
        reward_model=rm
    )

    # --- make sure the *actual* reference model used by PPOTrainer returns dicts ---
    _force_return_dict_on_forward(getattr(trainer, "model", None))  # you already had this

    # Patch all plausible ref-model anchors the trainer might use
    for cand in (
        getattr(trainer, "ref_model", None),
        getattr(trainer, "reference_model", None),
        getattr(getattr(trainer, "model", None), "ref_model", None),
        getattr(getattr(trainer, "model", None), "reference_model", None),
    ):
        _force_return_dict_on_forward(cand)   # your existing helper
        _force_logit_object_forward(cand)     # NEW: tuple -> object with .logits

    m = getattr(trainer, "model", None)
    if m is not None and not hasattr(m, "generate"):
        policy_like = getattr(m, "policy_model", None) or getattr(m, "actor_model", None)
        if policy_like is not None and hasattr(policy_like, "generate"):
            def _delegate_generate(*args, _m=m, **kwargs):
                tgt = getattr(_m, "policy_model", None) or getattr(_m, "actor_model", None)
                return tgt.generate(*args, **kwargs)
            setattr(m, "generate", _delegate_generate)
            if not hasattr(m, "generation_config") and hasattr(policy_like, "generation_config"):
                m.generation_config = policy_like.generation_config
            if not hasattr(m, "config") and hasattr(policy_like, "config"):
                m.config = policy_like.config

    trainer.train()

    os.makedirs(args.ppo_lora_dir, exist_ok=True)
    if isinstance(policy.pretrained_model, PeftModel):
        policy.pretrained_model.save_pretrained(args.ppo_lora_dir)
    tok.save_pretrained(args.ppo_lora_dir)

    final_dir = os.path.join(args.output_dir, "ppo_final_policy")
    trainer.save_pretrained(final_dir)

    try:
        eval_res = evaluate_gsm8k_em(policy.pretrained_model, tok, n=args.eval_gsm8k_n, device=args.device)
        print(f"[EVAL] GSM8K EM={eval_res.get('em')} on n={eval_res.get('n')}")
    except Exception:
        pass

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

# ======================= Inference: Budgeted Reasoning Controller =======================
def budgeted_generate(model, tok, prompt: str,
                      budgets=(64, 160, 320, 640),
                      temperature: float = 0.7,
                      top_p: float = 0.9):
    """
    Budgeted Reasoning Controller: escalate think budget only if needed.
    For creative prompts we return on first pass (short plan).
    """
    device = next(model.parameters()).device
    for b in budgets:
        enc = tok([prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc,
                                 max_new_tokens=b,
                                 do_sample=True,
                                 temperature=temperature,
                                 top_p=top_p,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        # Minimal AFC-based confidence check
        chain = extract_think(text)
        final = _extract_final_answer_from_tail(text.split("</think>")[-1] if "</think>" in text else text)
        ok = False
        if final:
            t = _answer_type(final, prompt)
            if t == "numeric":
                fa = _to_float(final); lc = _extract_last_number(chain)
                ok = (fa is not None and lc is not None and abs(fa - lc) <= max(1e-4, 1e-6*max(1.0, abs(fa))))
            else:
                ok = _coverage_and_tail_overlap(chain, final) >= 0.5
        if ok or _roleplay_or_creative(prompt):
            return text
    return text

# ======================= Main =======================
def main(argv: List[str]):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pc = choose_precision(args.precision, args.base_model_id)
    _print_precision_banner(pc)
    write_manifest(args, pc, stage="start", extra={})

    if args.run_sft:
        sft_dir = sft_train(args, pc)
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
