"""Robustness: full CRC-VAL-HE-7K (7180 images, all 9 classes) across frozen Axis-A levels.
CONCH + R1 zero-shot; R2 frozen DINOv2 probe (trained on NCT native train split)."""
import os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_common import (CLASSES, build_conch_text_feats, zeroshot_scores,
                             balanced_accuracy, degrade_resolution)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "CRC-VAL-HE-7K")
OUT = os.path.join(ROOT, "artifacts", "axisA_crcval")
os.makedirs(OUT, exist_ok=True)
LEVELS = [1, 2, 3, 4, 6, 8]
device = "cuda"

rows = []
for ci, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(DATA, c))):
        rows.append({"path": os.path.join(DATA, c, f), "class": c, "label": ci})
df = pd.DataFrame(rows)
paths = df["path"].tolist(); y = df["label"].values
print("CRC-VAL full:", df.shape)

# models
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

# R2: retrain frozen probe identically to the confirmation run (NCT native train split)
torch.hub.set_dir(os.path.join(ROOT, "hub"))
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device).eval()
from torchvision import transforms
pre_dino = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

@torch.no_grad()
def dino_feats(ps, s, bs=128):
    out = []
    for i in range(0, len(ps), bs):
        xs = [pre_dino(degrade_resolution(Image.open(p).convert("RGB"), s)) for p in ps[i:i+bs]]
        out.append(dino(torch.stack(xs).to(device)).cpu().numpy())
    return np.concatenate(out)

train_feats = np.load(os.path.join(ROOT, "artifacts", "axisA_confirm", "dino_train_native.npy"))
mdf = pd.read_csv(os.path.join(ROOT, "data", "NCT-sample", "nct_sample_manifest.csv"))
y_train = mdf[mdf.split == "train_probe"]["label"].values
clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=8).fit(train_feats, y_train)

summary, per_image = [], df.copy()
for s in LEVELS:
    for name, mdl, pre, W in [("CONCH", conch, pre_conch, W_conch),
                              ("R1-OpenCLIP-B32-LAION", oclip, pre_oclip, W_oclip)]:
        probs, preds = zeroshot_scores(mdl.encode_image, pre, W, paths, s, device)
        ba, recs = balanced_accuracy(y, preds)
        summary.append({"model": name, "level_s": s, "eff_um_px": 0.5*s,
                        "balanced_acc": ba, "top1_acc": float((preds == y).mean()),
                        **{f"recall_{c}": r for c, r in zip(CLASSES, recs)}})
        per_image[f"pred_{name}_s{s}"] = preds
        print(name, "s=", s, "bal-acc", round(ba, 4))
    preds = clf.predict(dino_feats(paths, s))
    ba, recs = balanced_accuracy(y, preds)
    summary.append({"model": "R2-DINOv2-logreg", "level_s": s, "eff_um_px": 0.5*s,
                    "balanced_acc": ba, "top1_acc": float((preds == y).mean()),
                    **{f"recall_{c}": r for c, r in zip(CLASSES, recs)}})
    per_image[f"pred_R2-DINOv2-logreg_s{s}"] = preds
    print("R2-DINOv2-logreg s=", s, "bal-acc", round(ba, 4))

pd.DataFrame(summary).to_csv(os.path.join(OUT, "crcval_summary.csv"), index=False)
per_image.to_csv(os.path.join(OUT, "crcval_per_image.csv"), index=False)
