# ✅ Self-contained Colab cell: Hermes-3 (Llama 3.1-8B) 4-bit API + Public Tunnel (ngrok or Cloudflare binary)
# - If env NGROK_AUTHTOKEN is set -> ngrok
# - Else -> downloads cloudflared binary (no account) and exposes a Quick Tunnel
# - Prints public URL you can hit from your laptop UI

import os, sys, json, time, threading, signal, re, subprocess, shutil, urllib.request, stat
from packaging.version import parse as V

# --------------------------
# 1) Install Python deps (skip torch to avoid Colab conflicts)
# --------------------------
def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-input"] + list(args))

_pip(
    "fastapi==0.111.0", "uvicorn==0.30.1", "pydantic==2.8.2", "starlette==0.37.2",
    "transformers>=4.44.0", "accelerate>=0.33.0", "bitsandbytes>=0.43.0",
    "python-multipart", "orjson", "pyngrok==7.1.6"
)

# --------------------------
# 2) Imports
# --------------------------
import torch, bitsandbytes
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, __version__ as HF_VER
)

# --------------------------
# 3) Environment & config
# --------------------------
# If you want to silence bnb CUDA mismatch warnings, uncomment next line:
# os.environ["BNB_CUDA_VERSION"] = ""
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH",""))

API_KEY         = os.environ.get("API_KEY", "")  # optional bearer token
MODEL_ID        = os.environ.get("MODEL_ID", "NousResearch/Hermes-3-Llama-3.1-3B")
SYSTEM_PROMPT   = os.environ.get("SYSTEM_PROMPT", "You are a helpful, concise assistant.")
MAX_HISTORY     = int(os.environ.get("MAX_HISTORY_TURNS", "16"))
PORT            = int(os.environ.get("PORT", "8000"))
BNB_4BIT_TYPE   = os.environ.get("BNB_4BIT_TYPE", "fp4")   # 'fp4' or 'nf4'
TORCH_DTYPE_STR = os.environ.get("TORCH_DTYPE", "bfloat16") # 'bfloat16' or 'float16'
NGROK_TOKEN     = os.environ.get("NGROK_AUTHTOKEN", "").strip()

if not torch.cuda.is_available():
    raise SystemExit("❌ No CUDA GPU detected. Switch Colab runtime to GPU.")

if V(HF_VER) < V("4.44.0"):
    raise SystemExit("❌ Transformers too old. Please upgrade to >= 4.44.0.")

print(f"✅ Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPUs: {torch.cuda.device_count()}")
print(f"✅ Transformers: {HF_VER} | bitsandbytes: {bitsandbytes.__version__}")

# --------------------------
# 4) Load model (4-bit)
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=BNB_4BIT_TYPE,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if TORCH_DTYPE_STR=="bfloat16" else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if TORCH_DTYPE_STR=="bfloat16" else torch.float16,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
    )
except Exception as e:
    print("⚠️ SDPA path failed, falling back to eager attention:", e)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if TORCH_DTYPE_STR=="bfloat16" else torch.float16,
        quantization_config=bnb_config,
        attn_implementation="eager",
    )

# --------------------------
# 5) Helpers
# --------------------------
def truncate_history(msgs, max_pairs=16):
    sys_msgs = [m for m in msgs if m.get("role")=="system"]
    ua = [m for m in msgs if m.get("role")!="system"]
    return sys_msgs + ua[-2*max_pairs:]

def apply_template_and_tokenize(msgs):
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(txt, return_tensors="pt").to(model.device)

# --------------------------
# 6) FastAPI app
# --------------------------
app = FastAPI(title="Hermes-3 4-bit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def check_auth(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_ID}

@app.get("/v1/models")
def models():
    return {"data":[{"id": MODEL_ID, "object":"model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: dict = Body(...)):
    check_auth(request)
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Missing 'messages' (list)")

    if messages[0].get("role") != "system":
        messages = [{"role":"system","content":SYSTEM_PROMPT}] + messages
    messages = truncate_history(messages, MAX_HISTORY)

    inputs = apply_template_and_tokenize(messages)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    gen_args = dict(
        **inputs,
        max_new_tokens = int(body.get("max_tokens", 512)),
        temperature     = float(body.get("temperature", 0.7)),
        top_p           = float(body.get("top_p", 0.9)),
        repetition_penalty = float(body.get("repetition_penalty", 1.05)),
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    threading.Thread(target=model.generate, kwargs=gen_args, daemon=True).start()

    if body.get("stream", False):
        def event_stream():
            for tok in streamer:
                yield f"data: {json.dumps({'choices':[{'delta':{'content':tok}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        out = "".join([t for t in streamer]).strip()
        return JSONResponse({"choices":[{"message":{"role":"assistant","content":out}}]})

# --------------------------
# 7) Start Uvicorn in background
# --------------------------
import uvicorn

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()
time.sleep(2.0)  # give server a moment

# --------------------------
# 8) Public tunnel (prefer ngrok if token set, else Cloudflare binary)
# --------------------------
public_url = None
tunnel_proc = None

def start_ngrok(port: int, token: str):
    from pyngrok import ngrok
    if token:
        ngrok.set_auth_token(token)
    return str(ngrok.connect(addr=port))

def ensure_cloudflared_binary():
    # Try existing binary
    for cand in ("cloudflared", "/usr/local/bin/cloudflared", "/usr/bin/cloudflared", "/content/cloudflared"):
        if shutil.which(cand):
            return shutil.which(cand)

    # Download latest linux-amd64 binary from GitHub releases
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    dst = "/content/cloudflared"
    print("⬇️  Downloading cloudflared binary...")
    urllib.request.urlretrieve(url, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
    return dst

def start_cloudflared(port: int):
    bin_path = ensure_cloudflared_binary()
    cmd = [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate", "--protocol", "http2"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    url = None
    pattern = re.compile(r"(https://[a-z0-9.-]+trycloudflare\.com)")
    start_time = time.time()
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            raise RuntimeError("cloudflared exited before providing a URL.")
        if line:
            m = pattern.search(line)
            if m:
                url = m.group(1)
                break
        if time.time() - start_time > 90:
            raise TimeoutError("Timed out waiting for cloudflared URL.")
    return url, proc

try:
    if NGROK_TOKEN:
        public_url = start_ngrok(PORT, NGROK_TOKEN)
        print(f"🌍 Public URL (ngrok): {public_url}")
    else:
        url, proc = start_cloudflared(PORT)
        tunnel_proc = proc
        public_url = url
        print(f"🌍 Public URL (Cloudflare): {public_url}")
except Exception as e:
    print("❌ Tunnel start failed:", repr(e))
    print("Tip: set NGROK_AUTHTOKEN for ngrok, or rerun to retry Cloudflare.")
    raise

print("   Use this base URL in your client.")
print("   Auth:", "enabled (Bearer token required)" if API_KEY else "disabled")
print("   Health check:", f"{public_url}/health")

# Keep alive
def _keep_alive():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()

_keep_alive()
