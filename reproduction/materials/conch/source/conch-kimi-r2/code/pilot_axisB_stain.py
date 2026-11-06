"""Pilot Axis B: HED stain perturbation on the CRC-VAL pilot sample (exploratory).
Magnitude alpha in {0, 0.1, 0.2, 0.4}: HED channels scaled by exp(N(0, alpha)), per-image seed.
Quantified by mean LAB delta-E vs original. CONCH + R1 OpenCLIP."""
import os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import rgb2hed, hed2rgb, rgb2lab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_common import CLASSES, build_conch_text_feats, balanced_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "pilot_axisB")
os.makedirs(OUT, exist_ok=True)
LEVELS = [0.0, 0.1, 0.2, 0.4]
device = "cuda"

sdf = pd.read_csv(os.path.join(ROOT, "artifacts", "pilot", "pilot_sample.csv"))
paths = sdf["path"].tolist(); y = sdf["label"].values

def stain_perturb(img, alpha, seed):
    if alpha == 0:
        return img, 0.0
    rng = np.random.RandomState(seed)
    a = np.asarray(img).astype(np.float32) / 255.0
    hed = rgb2hed(a)
    scale = np.exp(rng.normal(0, alpha, size=(1, 1, 3))).astype(np.float32)
    out = hed2rgb(np.clip(hed * scale, 0, None))
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    de = float(np.abs(rgb2lab(a) - rgb2lab(out.astype(np.float32)/255.0)).mean())
    return Image.fromarray(out), de

# cache transformed images to /var/tmp/opencode (scratch) as npy-free tensors per level
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

@torch.no_grad()
def score(model, pre, W, imgs, bs=256):
    probs = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([pre(im) for im in imgs[i:i+bs]]).to(device)
        f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        probs.append((100.0 * f @ W.t()).softmax(-1).cpu().numpy())
    return np.concatenate(probs)

rows = []
for li, alpha in enumerate(LEVELS):
    imgs, des = [], []
    for i, p in enumerate(paths):
        im, de = stain_perturb(Image.open(p).convert("RGB"), alpha, seed=hash((i, li)) % (2**31))
        imgs.append(im); des.append(de)
    for name, mdl, pre, W in [("CONCH", conch, pre_conch, W_conch),
                              ("R1-OpenCLIP-B32-LAION", oclip, pre_oclip, W_oclip)]:
        pr = score(mdl, pre, W, imgs)
        preds = pr.argmax(1)
        ba, _ = balanced_accuracy(y, preds)
        rows.append({"model": name, "alpha": alpha, "mean_lab_delta": float(np.mean(des)),
                     "balanced_acc": ba, "top1_acc": float((preds == y).mean())})
        print(name, "alpha=", alpha, "dE=", round(np.mean(des), 2), "bal-acc", round(ba, 4))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "axisB_pilot.csv"), index=False)
