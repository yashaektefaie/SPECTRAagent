"""Evaluate the ResNet-18 reference on the exact same images/levels as CONCH.

Usage:
  python scripts/eval_resnet.py --data-root data/CRC-VAL-HE-7K --tag val7k \
      [--file-list results/nct_discovery_files.csv]
If --file-list is given, evaluation is restricted to those files (per class dirs).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transforms_spec import TRANSFORMS

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
HUE_LEVELS = [0, 10, 20, 30, 45, 60, 90, 120, 180]
SAT_LEVELS = [1.0, 0.5, 0.25, 0.0]
VAL_LEVELS = [1.0, 0.75, 0.5, 0.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file-list", default=None)
    ap.add_argument("--weights", default="models/resnet18_nct.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda"
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(args.weights))
    model.to(device).eval()

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_t = transforms.ToTensor()

    if args.file_list:
        keep = set(pd.read_csv(args.file_list)["filename"])
    rows = []
    for cls in CLASSES:
        for f in sorted(os.listdir(os.path.join(args.data_root, cls))):
            if args.file_list and f not in keep:
                continue
            rows.append((cls, f))
    print(len(rows), "images")

    grid = [("hue", t) for t in HUE_LEVELS] + \
           [("sat", a) for a in SAT_LEVELS if a != 1.0] + \
           [("val", b) for b in VAL_LEVELS if b != 1.0]

    records = []
    for name, lvl in grid:
        fn = TRANSFORMS[name]
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            imgs = []
            for cls, f in batch:
                img = np.array(Image.open(os.path.join(args.data_root, cls, f)).convert("RGB"))
                imgs.append(norm(to_t(Image.fromarray(fn(img, lvl)))))
            x = torch.stack(imgs).to(device)
            with torch.inference_mode():
                preds = model(x).argmax(1).cpu().numpy()
            for (cls, f), p in zip(batch, preds):
                records.append({"axis": name, "level": lvl, "filename": f, "true": cls,
                                "pred": CLASSES[int(p)],
                                "correct": int(CLASSES[int(p)] == cls)})
        sub = pd.DataFrame([r for r in records if r["axis"] == name and r["level"] == lvl])
        print(f"{name} {lvl}: acc={sub['correct'].mean():.4f} "
              f"bacc={sub.groupby('true')['correct'].mean().mean():.4f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(f"results/{args.tag}_resnet_perimage.csv", index=False)
    summary = {}
    for (name, lvl), sub in df.groupby(["axis", "level"]):
        pc = sub.groupby("true")["correct"].mean()
        summary[f"{name}_{lvl}"] = {"axis": name, "level": float(lvl), "n": int(len(sub)),
                                    "acc": float(sub["correct"].mean()),
                                    "bacc": float(pc.mean()),
                                    "per_class_recall": pc.to_dict()}
    import json
    with open(f"results/{args.tag}_resnet_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
