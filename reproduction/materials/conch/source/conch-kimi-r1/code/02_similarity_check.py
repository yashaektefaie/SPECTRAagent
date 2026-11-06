"""Axis validity: measured native-vs-degraded feature similarity must decrease with s.

Uses ImageNet ResNet-50 features (independent of CONCH and of any labels/outcomes).
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms
from common import ART, LEVELS, SEED, degrade, load_index

DEVICE = "cuda"


def rn50():
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(DEVICE)


PRE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def feats(model, imgs):
    x = torch.stack([PRE(im) for im in imgs]).to(DEVICE)
    return torch.nn.functional.normalize(model(x), dim=1)


def main():
    df = load_index("test_index.csv")
    samp = (df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=100, random_state=SEED))
            .reset_index(drop=True))
    model = rn50()
    rows = []
    per_level_sims = {s: [] for s in LEVELS}
    B = 64
    paths = samp["path"].tolist()
    for i in range(0, len(paths), B):
        chunk = paths[i:i + B]
        native = [Image.open(p).convert("RGB") for p in chunk]
        f_nat = feats(model, native)
        for s in LEVELS:
            f_deg = feats(model, [degrade(im, s) for im in native])
            sim = (f_nat * f_deg).sum(1).cpu().numpy()
            per_level_sims[s].extend(sim.tolist())
        if i % (B * 4) == 0:
            print(f"{i}/{len(paths)}")
    out = {}
    prev = 2.0
    mono = True
    for s in LEVELS:
        arr = np.array(per_level_sims[s])
        out[str(s)] = {"mean_cosine_sim": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}
        if arr.mean() > prev:
            mono = False
        prev = arr.mean()
        print(s, out[str(s)]["mean_cosine_sim"])
    out["monotonic_decrease"] = bool(mono)
    json.dump(out, open(os.path.join(ART, "similarity_check.json"), "w"), indent=2)
    print("monotonic:", mono)


if __name__ == "__main__":
    main()
