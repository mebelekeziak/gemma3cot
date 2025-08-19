#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference for the merged final model.
Defaults to ./merged/g3-1b-faithful produced by your run.py.

Requirements (same env you trained in is fine):
  pip install -U transformers accelerate bitsandbytes

Examples:
  python infer.py --model-path ./merged/g3-1b-faithful --prompt "2+2?"
  python infer.py --model-path ./merged/g3-1b-faithful --chat --prompt "Explain transformers in 3 bullets."
  python infer.py --model-path ./merged/g3-1b-faithful --budgeted --budgets 64,160,320 --prompt "Solve: 37*58"
  python infer.py --model-path ./merged/g3-1b-faithful --repl
"""
import os, re, argparse, math
import torch
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.generation.streamers import TextStreamer

# --- small helpers (Gemma-3 prefers bf16/fp32 over fp16) ---
def _gpu_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)

def _bf16_supported():
    major, _ = _gpu_capability()
    return torch.cuda.is_available() and major >= 8  # Ampere+

def _is_gemma3(model_id: str) -> bool:
    return "gemma-3" in (model_id or "").lower()

def choose_dtype(base_model_id: str, precision: str):
    req = (precision or "auto").lower()
    if req == "bf16":
        return torch.bfloat16 if _bf16_supported() else (torch.float32 if _is_gemma3(base_model_id) else torch.float16)
    if req == "fp16":
        return torch.float32 if _is_gemma3(base_model_id) else torch.float16
    if req == "fp32":
        return torch.float32
    # auto
    if _bf16_supported():
        return torch.bfloat16
    return torch.float32 if _is_gemma3(base_model_id) else torch.float16

# --- minimal CoT utilities (for budgeted controller / optional answer extraction) ---
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def _extract_final_answer_from_tail(tail: str) -> str:
    s = tail.strip()
    if not s:
        return ""
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", s, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"(?i)(?:final answer|answer|thus|therefore)\s*:\s*(.*)$", s, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"####\s*(.*)$", s, re.MULTILINE)
    if m: return m.group(1).strip()
    parts = [p.strip() for p in s.split("\n\n") if p.strip()]
    if parts: return parts[-1]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _to_float(s: str) -> Optional[float]:
    try: return float(s.replace(",", ""))
    except Exception: return None

def _extract_last_number(text: str) -> Optional[float]:
    nums = _NUM_RE.findall(text or "")
    if not nums: return None
    try: return float(nums[-1].replace(",", ""))
    except Exception: return None

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _coverage_and_tail_overlap(chain: str, final_answer: str) -> float:
    ch = (chain or "")
    fa = (final_answer or "")
    if not ch or not fa: return 0.0
    ch_norm = _normalize_text(ch)
    tail = ch_norm[-800:]
    toks = [w for w in re.findall(r"[a-zA-Z0-9_]+", fa.lower()) if len(w) >= 4]
    if not toks: return 1.0 if _normalize_text(fa) in tail else 0.0
    uniq = sorted(set(toks))
    hits = sum(1 for w in uniq if w in ch_norm)
    cov = hits / max(1, len(uniq))
    near = 1.0 if _normalize_text(fa) in tail else min(1.0, sum(1 for w in uniq if len(w) >= 6 and w in tail) / max(1, len([w for w in uniq if len(w) >= 6])))
    return float(min(1.0, 0.6 * cov + 0.4 * near))

def budgeted_generate(model, tok, prompt: str, budgets=(64,160,320,640), temperature=0.7, top_p=0.9):
    device = next(model.parameters()).device
    last_text = ""
    for b in budgets:
        enc = tok([prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=b,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        last_text = text
        chain = extract_think(text)
        tail = text.split("</think>")[-1] if "</think>" in text else text
        final = _extract_final_answer_from_tail(tail)
        ok = False
        if final:
            fa = _to_float(final)
            lc = _extract_last_number(chain)
            if fa is not None and lc is not None:
                ok = abs(fa - lc) <= max(1e-4, 1e-6 * max(1.0, abs(fa)))
            else:
                ok = _coverage_and_tail_overlap(chain, final) >= 0.5
        if ok:
            return text
    return last_text

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

# --- main load + run ---
def load_model(path: str, precision: str, load_in_4bit: bool):
    dtype = choose_dtype(path, precision)
    quant = None
    if load_in_4bit:
        # compute dtype for 4-bit matmuls: prefer bf16 if available, else fp16
        compute = torch.bfloat16 if _bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute,
        )
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if tok.eos_token is None:
        if tok.sep_token is not None: tok.eos_token = tok.sep_token
        else: tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"

    kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if quant is not None:
        kwargs["quantization_config"] = quant
        # If quantized, prefer not to force fp32
        if dtype == torch.float32:
            kwargs["torch_dtype"] = torch.bfloat16 if _bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    # Make sure config has eos/pad
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    return model, tok

def format_prompt(messages: List[dict], tok: AutoTokenizer) -> str:
    """
    messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
    """
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_once(model, tok, prompt_text: str, args):
    if args.budgeted:
        raw = budgeted_generate(
            model, tok, prompt_text,
            budgets=tuple(int(x) for x in args.budgets.split(",")),
            temperature=args.temperature, top_p=args.top_p,
        )
    else:
        enc = tok([prompt_text], return_tensors="pt").to(next(model.parameters()).device)
        streamer = None
        if args.stream:
            streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=(args.top_k if args.top_k > 0 else None),
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                streamer=streamer,
            )
        raw = tok.decode(out[0], skip_special_tokens=True)

    return strip_think(raw) if args.hide_think else raw

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="./merged/g3-1b-faithful")
    p.add_argument("--precision", type=str, choices=["auto","bf16","fp16","fp32"], default="auto")
    p.add_argument("--load-in-4bit", action="store_true", help="Quantized 4-bit inference (NF4)")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--chat", action="store_true", help="Use chat template with roles")
    p.add_argument("--system", type=str, default="You are a helpful, concise assistant.")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--hide-think", action="store_true", help="Strip <think>…</think> from final print")
    p.add_argument("--budgeted", action="store_true", help="Enable budgeted reasoning controller")
    p.add_argument("--budgets", type=str, default="64,160,320,640")
    p.add_argument("--repl", action="store_true", help="Interactive loop")
    args = p.parse_args()

    model, tok = load_model(args.model_path, args.precision, args.load_in_4bit)

    def build_prompt(user_text: str) -> str:
        if args.chat:
            msgs = []
            if args.system: msgs.append({"role":"system","content":args.system})
            msgs.append({"role":"user","content":user_text})
            return format_prompt(msgs, tok)
        return user_text

    if args.repl:
        print(">> REPL mode (Ctrl+C to exit).")
        while True:
            try:
                user = input("\nUser > ").strip()
                if not user: continue
                prompt_text = build_prompt(user)
                out = run_once(model, tok, prompt_text, args)
                print("\nAssistant >", out)
            except KeyboardInterrupt:
                print("\nbye.")
                break
    else:
        if args.prompt is None:
            raise SystemExit("Provide --prompt or use --repl")
        prompt_text = build_prompt(args.prompt)
        out = run_once(model, tok, prompt_text, args)
        print(out)

if __name__ == "__main__":
    # keep kernels happy
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    main()
