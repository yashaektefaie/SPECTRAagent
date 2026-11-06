import os
os.environ["HF_HOME"] = os.path.abspath("audit/weights/hf")
from huggingface_hub import snapshot_download
p = snapshot_download("biohub/ESMFold2",
    allow_patterns=["model.safetensors", "config.json", "ccd.pkl", "README.md"])
print("done", p, flush=True)
