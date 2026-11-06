"""Pilot probe (cheap-first): resolution axis on CRC-VAL-HE-7K balanced sample.
CONCH + OpenCLIP reference, levels s in {1,2,4,8}. Exploratory evidence only."""
import os, sys, json, hashlib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_common import (CLASSES, build_conch_text_feats, zeroshot_scores,
                             balanced_accuracy, load_prompt_ensemble)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "CRC-VAL-HE-7K")
OUT = os.path.join(ROOT, "artifacts", "pilot")
os.makedirs(OUT, exist_ok=True)

SEED = 20260722
PER_CLASS = 100
LEVELS = [1, 2, 4, 8]

# ---- balanced pilot sample ----
rng = np.random.RandomState(SEED)
rows = []
for ci, c in enumerate(CLASSES):
    files = sorted(os.listdir(os.path.join(DATA, c)))
    idx = rng.choice(len(files), PER_CLASS, replace=False)
    for j in idx:
        rows.append({"path": os.path.join(DATA, c, files[j]), "label": ci, "class": c,
                     "file": files[j]})
df = pd.DataFrame(rows)

# dedup diagnostic: exact md5
def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
df["md5"] = df["path"].map(md5)
assert df["md5"].is_unique, "exact duplicates in pilot sample!"
df.to_csv(os.path.join(OUT, "pilot_sample.csv"), index=False)
print("pilot sample:", df.shape)

device = "cuda"

# ---- CONCH ----
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
conch, pre_conch = create_model_from_pretrained(
    "conch_ViT-B-16", checkpoint_path=os.path.join(ROOT, "weights", "conch_pytorch_model.bin"))
conch = conch.to(device).eval()
tok = get_tokenizer()
W_conch = build_conch_text_feats(conch, lambda t: tokenize(tok, t), device)
print("CONCH text feats:", W_conch.shape)

# ---- OpenCLIP reference ----
import open_clip
oclip, _, pre_oclip = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
oclip = oclip.to(device).eval()
otok = open_clip.get_tokenizer("ViT-B-32")
W_oclip = build_conch_text_feats(oclip, lambda t: otok(t), device)

# ---- score ----
paths = df["path"].tolist(); y = df["label"].values
records = []
for s in LEVELS:
    for name, mdl, pre, W in [("CONCH", conch, pre_conch, W_conch),
                              ("OpenCLIP-B32-LAION", oclip, pre_oclip, W_oclip)]:
        probs, preds = zeroshot_scores(mdl.encode_image, pre, W, paths, s, device)
        ba, recs = balanced_accuracy(y, preds)
        acc = float((preds == y).mean())
        records.append({"model": name, "level_s": s, "eff_um_px": 0.5 * s,
                        "balanced_acc": ba, "top1_acc": acc,
                        **{f"recall_{c}": r for c, r in zip(CLASSES, recs)}})
        np.save(os.path.join(OUT, f"probs_{name.replace('-','_')}_s{s}.npy"), probs)
        df[f"pred_{name}_s{s}"] = preds
        print(f"{name} s={s}: bal-acc={ba:.4f} top1={acc:.4f}")

res = pd.DataFrame(records)
res.to_csv(os.path.join(OUT, "pilot_results.csv"), index=False)
df.to_csv(os.path.join(OUT, "pilot_per_image.csv"), index=False)
print(res[["model", "level_s", "balanced_acc", "top1_acc"]])
