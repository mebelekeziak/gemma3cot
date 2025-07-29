#!/usr/bin/env python
"""
Fine‑tune Gemma‑3 (or any HF causal‑LM) on chain‑of‑thought (CoT) data with
LoRA‑based supervised fine‑tuning (SFT) and PPO reinforcement learning (RL).

Changes vs. the original script
-------------------------------
1.  Uses **Unsloth** (`FastLanguageModel`) for SFT:
      * flash‑attn 2 kernels, paged attention, 4‑bit loading, RoPE scaling
      * 1‑line LoRA creation with gradient‑checkpointing support
2.  Automatically merges LoRA into a standard fp16/bf16 checkpoint so the
    PPO section (unchanged) can wrap it with a value‑head.
3.  Everything else—dataset prep, PPO loop, reward model—remains identical.

Hardware target: NVIDIA H100/H200 class GPUs (24 GB +).  VRAM usage with
`load_in_4bit=True` is ≈22 GB for Gemma‑3‑27B.

Dependencies:
    pip install "transformers>=4.50.0" peft datasets "trl>=0.19" \
                accelerate unsloth unsloth_zoo bitsandbytes -U
"""

import os, re, torch, warnings
from huggingface_hub import HfApi
from datasets import load_dataset, concatenate_datasets, Dataset

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BASE_MODEL_ID   = os.environ.get("BASE_MODEL_ID", "google/gemma-3-27b")
REWARD_MODEL_ID = os.environ.get("REWARD_MODEL_ID", "UW-Madison-Lee-Lab/VersaPRM")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Flip to True if you want to run SFT; False skips straight to PPO
RUN_SFT = True

# Where to save artefacts
MERGED_PATH         = "./gemma3_cot_sft_merged"
LORA_CHECKPOINT_DIR = "./gemma3_cot_sft_lora"

HF_REPO        = os.environ.get("HF_REPO")
HF_TOKEN       = os.environ.get("HF_TOKEN")
MAX_SHARD_SIZE = "10GB"

# LoRA hyper‑params
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_SEQ_LEN      = 2048        # Unsloth will extend RoPE if needed
SFT_EPOCHS       = 1
SFT_LR           = 1e-4

PPO_BATCH_SIZE   = 4
PPO_MINI_BS      = 1
PPO_EPOCHS       = 4
PPO_LR           = 5e-6
PPO_TARGET_KL    = 6.0

# ----------------------------------------------------------------------
# OPTIONAL: silence flash‑attn compile warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# TOKENIZER & BASE MODEL (Only loaded if RUN_SFT=True)
# ----------------------------------------------------------------------
if RUN_SFT:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL_ID,
        max_seq_length  = MAX_SEQ_LEN,
        dtype           = torch.bfloat16,   # auto‑upcasts critical layers
        load_in_4bit    = True,             # 4‑bit quant via bitsandbytes
    )

    # Add <think></think> delimiters used in CoT formatting
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_R,
        lora_alpha               = LORA_ALPHA,
        target_modules           = TARGET_MODULES,
        lora_dropout             = LORA_DROPOUT,
        use_gradient_checkpointing = True,
    )
else:
    tokenizer = None   # will load later for PPO

# ----------------------------------------------------------------------
# DATASET PREPARATION (unchanged)
# ----------------------------------------------------------------------
def load_and_prepare_longtalk() -> Dataset:
    ds = load_dataset("kenhktsui/longtalk-cot-v0.1", split="train")
    def _convert(ex):
        messages = ex["messages"]
        user_msg      = next((m["content"] for m in messages if m["role"]=="user"), "").strip()
        assistant_msg = next((m["content"] for m in messages if m["role"]=="assistant"), "").strip()

        answer_match = re.search(r"(?:Answer:|####)(.*)", assistant_msg)
        if answer_match:
            chain = assistant_msg[:answer_match.start()].strip()
            final = answer_match.group(1).strip()
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
        gen = ex.get("generation","") or ""
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

def build_sft_dataset() -> Dataset:
    sources = []
    for fn in (load_and_prepare_longtalk,
               load_and_prepare_gsm8k,
               load_and_prepare_mmlu_pro):
        try:
            d = fn()
            if len(d): sources.append(d)
        except Exception as e:
            print(f"[WARN] loading dataset failed: {e}")
    if not sources:
        raise RuntimeError("No data available for SFT.")
    return concatenate_datasets(sources).shuffle(seed=42)

# ----------------------------------------------------------------------
# SUPERVISED FINE‑TUNING (SFT) with UNSLOTH
# ----------------------------------------------------------------------
if RUN_SFT:
    from transformers import TrainingArguments, Trainer

    sft_dataset = build_sft_dataset()

    def data_collator(batch):
        texts  = [f["prompt"] + "\n\n" + f["response"] for f in batch]
        enc    = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SEQ_LEN)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"]==0] = -100
        return {"input_ids": enc["input_ids"],
                "labels": labels,
                "attention_mask": enc["attention_mask"]}

    train_args = TrainingArguments(
        output_dir               = "./gemma3_cot_sft",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_train_epochs         = SFT_EPOCHS,
        learning_rate            = SFT_LR,
        bf16                     = True,
        logging_steps            = 10,
        save_steps               = 500,
        save_total_limit         = 2,
        report_to                = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = train_args,
        train_dataset   = sft_dataset,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
    )

    trainer.train()
    trainer.save_model(LORA_CHECKPOINT_DIR)   # adapter for debugging / resume

    # Merge LoRA into base weights & save fp16/bf16 checkpoint
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        MERGED_PATH,
        max_shard_size=MAX_SHARD_SIZE,
    )
    tokenizer.save_pretrained(MERGED_PATH)
    if HF_REPO:
        merged_model.push_to_hub(
            HF_REPO,
            token=HF_TOKEN,
            max_shard_size=MAX_SHARD_SIZE,
        )
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)

    del model, merged_model
    torch.cuda.empty_cache()

else:
    print("RUN_SFT=False – skipping SFT and loading existing merged model.")

# ----------------------------------------------------------------------
# REINFORCEMENT LEARNING WITH PPO (unchanged)
# ----------------------------------------------------------------------
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

# Reload tokenizer if we skipped SFT
if tokenizer is None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MERGED_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
ppo_model.resize_token_embeddings(len(tokenizer))
ref_model = create_reference_model(ppo_model)

# Reward model
from transformers import AutoModelForCausalLM
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
reward_model     = AutoModelForCausalLM.from_pretrained(REWARD_MODEL_ID).to(DEVICE).eval()

incorrect_id = reward_tokenizer.convert_tokens_to_ids("<INCORRECT>")
correct_id   = reward_tokenizer.convert_tokens_to_ids("<CORRECT>")
step_id      = reward_tokenizer.convert_tokens_to_ids("<STEP>")
if step_id == reward_tokenizer.unk_token_id:
    step_id = reward_tokenizer.eos_token_id

def compute_step_scores(text: str) -> float:
    """
    Scores a generated CoT answer by averaging the reward model's
    <CORRECT>/<INCORRECT> logits at each <STEP> token.
    """
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    formatted = (" \n\n\n\n".join(ln.strip() for ln in m.group(1).split("\n")) +
                 " \n\n\n\n") if m else text
    ids = reward_tokenizer.encode(formatted, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = reward_model(ids).logits[..., [incorrect_id, correct_id]]
        probs  = logits.softmax(-1)[..., 1]          # P(correct)
    mask = (ids == step_id).float()
    return (probs * mask).sum().item() / max(mask.sum().item(), 1)

def build_rl_queries(ds: Dataset, num_samples: int = 100) -> list[str]:
    n = min(len(ds), num_samples)
    return ds.shuffle(seed=0).select(range(n))["prompt"]

if RUN_SFT:
    rl_prompts = build_rl_queries(sft_dataset, num_samples=128)
else:
    rl_prompts = build_rl_queries(build_sft_dataset(), num_samples=128)

ppo_cfg = PPOConfig(
    model_name        = "gemma3_cot_sft",
    learning_rate     = PPO_LR,
    batch_size        = PPO_BATCH_SIZE,
    mini_batch_size   = PPO_MINI_BS,
    adaptive_kl_ctrl  = True,
    target_kl         = PPO_TARGET_KL,
    ppo_epochs        = PPO_EPOCHS,
)

ppo_trainer = PPOTrainer(
    config     = ppo_cfg,
    model      = ppo_model,
    ref_model  = ref_model,
    tokenizer  = tokenizer,
)

def rl_finetune(num_epochs: int = 1):
    ppo_trainer.model.train()
    for ep in range(num_epochs):
        for prompt in rl_prompts:
            query_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            gen = ppo_model.generate(
                **query_ids,
                max_new_tokens = 512,
                do_sample      = True,
                temperature    = 0.7,
                top_p          = 0.9,
                eos_token_id   = tokenizer.eos_token_id,
            )
            reward = compute_step_scores(tokenizer.decode(gen[0], skip_special_tokens=True))
            ppo_trainer.step([query_ids["input_ids"][0]], [gen[0]], [reward])
        print(f"[PPO] finished epoch {ep+1}/{num_epochs}")

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

# To launch PPO training:
# rl_finetune(num_epochs=1)
