#!/usr/bin/env python3
"""Download the cross-encoder reranker model and tokenizer into .cortyx/."""
import os, sys, shutil, pathlib, platform

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("Install huggingface_hub: pip install huggingface_hub")

REPO = "cross-encoder/ms-marco-MiniLM-L-2-v2"
OUT  = pathlib.Path(".cortyx")
OUT.mkdir(exist_ok=True)

# Pick the best quantized model for the current platform
arch = platform.machine().lower()
if arch == "arm64":
    model_file = "onnx/model_qint8_arm64.onnx"
elif "avx512" in (os.popen("sysctl -a 2>/dev/null || cat /proc/cpuinfo 2>/dev/null").read()).lower():
    model_file = "onnx/model_qint8_avx512.onnx"
else:
    model_file = "onnx/model.onnx"

print(f"Arch: {arch} → downloading {model_file} from {REPO} …")
model_src = hf_hub_download(REPO, filename=model_file)
shutil.copy(model_src, OUT / "reranker.onnx")
print(f"  → {OUT / 'reranker.onnx'}")

print(f"Downloading tokenizer from {REPO} …")
tok_src = hf_hub_download(REPO, filename="tokenizer.json")
shutil.copy(tok_src, OUT / "tokenizer.json")
print(f"  → {OUT / 'tokenizer.json'}")

print("Done. Build with: cargo build --release --features rerank")
