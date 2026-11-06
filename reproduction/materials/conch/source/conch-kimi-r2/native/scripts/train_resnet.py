"""Train the fixed, CONCH-independent reference: ResNet-18 on clean NCT-CRC-HE-100K.

Training images: --train-per-class per class, drawn AFTER excluding the discovery
subset (results/nct_discovery_files.csv). Clean images only; geometric augs only
(no color/hue augmentation), per AXIS_SPEC.md section 6.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


class PatchDS(Dataset):
    def __init__(self, rows, root, tf):
        self.rows, self.root, self.tf = rows, root, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cls, f = self.rows[i]
        img = Image.open(os.path.join(self.root, cls, f)).convert("RGB")
        return self.tf(img), CLASSES.index(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/NCT-CRC-HE-100K")
    ap.add_argument("--train-per-class", type=int, default=4000)
    ap.add_argument("--exclude", default="results/nct_discovery_files.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="models/resnet18_nct.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    excl = set(pd.read_csv(args.exclude)["filename"]) if os.path.exists(args.exclude) else set()
    rows = []
    rng = np.random.RandomState(args.seed)
    for cls in CLASSES:
        files = sorted(f for f in os.listdir(os.path.join(args.data_root, cls)) if f not in excl)
        rows += [(cls, f) for f in rng.choice(files, size=args.train_per_class, replace=False)]
    print(f"{len(rows)} training images (excluded {len(excl)} discovery files)")

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda im: im.rotate(float(90 * np.random.randint(4)))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dl = DataLoader(PatchDS(rows, args.data_root, tf), batch_size=args.batch_size,
                    shuffle=True, num_workers=8, drop_last=True)

    device = "cuda"
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    lossf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        tot, cor, ls = 0, 0, 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step(); sched.step()
            tot += len(y); cor += (out.argmax(1) == y).sum().item(); ls += loss.item() * len(y)
        print(f"epoch {ep}: train acc={cor/tot:.4f} loss={ls/tot:.4f}", flush=True)

    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
