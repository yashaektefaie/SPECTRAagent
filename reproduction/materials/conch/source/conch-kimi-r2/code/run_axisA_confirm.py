"""Axis A confirmation run on NCT-CRC-HE-100K sample (frozen spec v1).
Steps: (1) axis-order verification (SSIM + DINOv2 cosine vs original),
(2) DINOv2 features at every level for train_probe + eval,
(3) logistic regression trained on native train split, frozen, evaluated at all levels,
(4) CONCH + OpenCLIP zero-shot at all levels on eval,
(5) per-image CSVs + summary."""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_common import (CLASSES, build_conch_text_feats, zeroshot_scores,
                             balanced_accuracy, degrade_resolution)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "axisA_confirm")
os.makedirs(OUT, exist_ok=True)
LEVELS = [1, 2, 3, 4, 6, 8]
device = "cuda"

df = pd.read_csv(os.path.join(ROOT, "data", "NCT-sample", "nct_sample_manifest.csv"))
eval_df = df[df.split == "eval"].reset_index(drop=True)
train_df = df[df.split == "train_probe"].reset_index(drop=True)
print("eval:", eval_df.shape, "train:", train_df.shape)

# ---------------- DINOv2 ----------------
torch.hub.set_dir(os.path.join(ROOT, "hub"))
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device).eval()
DINO_SIZE = 224
from torchvision import transforms
pre_dino = transforms.Compose([
    transforms.Resize((DINO_SIZE, DINO_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

@torch.no_grad()
def dino_feats(paths, s, batch_size=128):
    feats = []
    for i in range(0, len(paths), batch_size):
        xs = []
        for p in paths[i:i + batch_size]:
            img = Image.open(p).convert("RGB")
            img = degrade_resolution(img, s)
            xs.append(pre_dino(img))
        x = torch.stack(xs).to(device)
        f = dino(x)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)

# ---------------- (1) axis-order verification on eval set ----------------
from skimage.metrics import structural_similarity as ssim_fn  # may need scikit-image
ver_rows = []
paths_eval = eval_df["path"].tolist()
orig_feats = dino_feats(paths_eval, 1)
orig_feats_n = orig_feats / np.linalg.norm(orig_feats, axis=1, keepdims=True)
for s in LEVELS:
    fs = dino_feats(paths_eval, s)
    fs_n = fs / np.linalg.norm(fs, axis=1, keepdims=True)
    cos = float(np.mean((fs_n * orig_feats_n).sum(1)))
    # SSIM on a fixed 300-image subsample (first 300, deterministic)
    ssims = []
    for p in paths_eval[:300]:
        a = np.asarray(Image.open(p).convert("RGB"))
        b = np.asarray(degrade_resolution(Image.open(p).convert("RGB"), s))
        ssims.append(ssim_fn(a, b, channel_axis=2, data_range=255))
    ver_rows.append({"level_s": s, "eff_um_px": 0.5 * s,
                     "dino_cosine_to_native": cos, "ssim_to_native": float(np.mean(ssims))})
    np.save(os.path.join(OUT, f"dino_eval_s{s}.npy"), fs)
    print("verify", ver_rows[-1])
ver = pd.DataFrame(ver_rows)
ver.to_csv(os.path.join(OUT, "axis_order_verification.csv"), index=False)
mono_cos = all(ver.dino_cosine_to_native[i] > ver.dino_cosine_to_native[i + 1] for i in range(len(ver) - 1))
mono_ssim = all(ver.ssim_to_native[i] > ver.ssim_to_native[i + 1] for i in range(len(ver) - 1))
print("monotone decreasing similarity: dino_cos:", mono_cos, "ssim:", mono_ssim)
assert mono_cos and mono_ssim, "axis order verification FAILED"

# ---------------- (2,3) R2: DINOv2 + frozen logistic regression ----------------
train_feats_native = dino_feats(train_df["path"].tolist(), 1)
np.save(os.path.join(OUT, "dino_train_native.npy"), train_feats_native)
clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=8).fit(train_feats_native, train_df["label"].values)
print("R2 probe trained on native-res train split. train acc:",
      float(clf.score(train_feats_native, train_df["label"].values)))

y_eval = eval_df["label"].values
r2_rows, r2_preds = [], {}
for s in LEVELS:
    fs = np.load(os.path.join(OUT, f"dino_eval_s{s}.npy"))
    preds = clf.predict(fs)
    ba, recs = balanced_accuracy(y_eval, preds)
    r2_rows.append({"model": "R2-DINOv2-logreg", "level_s": s, "eff_um_px": 0.5 * s,
                    "balanced_acc": ba, "top1_acc": float((preds == y_eval).mean()),
                    **{f"recall_{c}": r for c, r in zip(CLASSES, recs)}})
    r2_preds[s] = preds
    print("R2 s=", s, "bal-acc", round(ba, 4))

# ---------------- (4) CONCH + OpenCLIP zero-shot ----------------
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
conch, pre_conch = create_model_from_pretrained(
    "conch_ViT-B-16", checkpoint_path=os.path.join(ROOT, "weights", "conch_pytorch_model.bin"))
conch = conch.to(device).eval()
W_conch = build_conch_text_feats(conch, lambda t: tokenize(get_tokenizer(), t), device)

import open_clip
oclip, _, pre_oclip = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
oclip = oclip.to(device).eval()
otok = open_clip.get_tokenizer("ViT-B-32")
W_oclip = build_conch_text_feats(oclip, lambda t: otok(t), device)

zs_rows = []
per_image = eval_df[["path", "class", "label"]].copy()
for s in LEVELS:
    for name, mdl, pre, W in [("CONCH", conch, pre_conch, W_conch),
                              ("R1-OpenCLIP-B32-LAION", oclip, pre_oclip, W_oclip)]:
        probs, preds = zeroshot_scores(mdl.encode_image, pre, W, paths_eval, s, device)
        ba, recs = balanced_accuracy(y_eval, preds)
        zs_rows.append({"model": name, "level_s": s, "eff_um_px": 0.5 * s,
                        "balanced_acc": ba, "top1_acc": float((preds == y_eval).mean()),
                        **{f"recall_{c}": r for c, r in zip(CLASSES, recs)}})
        np.save(os.path.join(OUT, f"probs_{name.replace('-', '_')}_s{s}.npy"), probs)
        per_image[f"pred_{name}_s{s}"] = preds
        print(name, "s=", s, "bal-acc", round(ba, 4))
for s in LEVELS:
    per_image[f"pred_R2-DINOv2-logreg_s{s}"] = r2_preds[s]
per_image.to_csv(os.path.join(OUT, "per_image_eval.csv"), index=False)

summary = pd.concat([pd.DataFrame(zs_rows), pd.DataFrame(r2_rows)])
summary.to_csv(os.path.join(OUT, "axisA_summary.csv"), index=False)
print(summary[["model", "level_s", "balanced_acc", "top1_acc"]].to_string())
