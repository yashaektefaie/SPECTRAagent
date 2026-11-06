"""Fixed model-independent reference: ImageNet ResNet-50 + kNN(k=20).

Primary condition: fit on native (s=1) train subset only, deploy frozen at all levels
(matches CONCH zero-shot frozen deployment). Sensitivity: refit per level.
Saves per-image predictions for every level.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms
from common import ART, CLASSES, LEVELS, RES, degrade, load_index

DEVICE = "cuda"
K = 20

PRE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def rn50():
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(DEVICE)


@torch.no_grad()
def extract(model, df, s, tag):
    cache = os.path.join(ART, f"rn50feat_{tag}_s{s}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    feats = []
    paths = df["path"].tolist()
    B = 128
    for i in range(0, len(paths), B):
        imgs = [degrade(Image.open(p).convert("RGB"), s) for p in paths[i:i + B]]
        x = torch.stack([PRE(im) for im in imgs]).to(DEVICE)
        f = torch.nn.functional.normalize(model(x), dim=1)
        feats.append(f.cpu().numpy())
        if i % (B * 20) == 0:
            print(tag, "s", s, f"{i}/{len(paths)}", flush=True)
    out = np.concatenate(feats)
    np.save(cache, out)
    return out


def knn_predict(train_f, train_y, test_f, k=K):
    # cosine similarity via normalized features; majority vote among top-k
    sim = test_f @ train_f.T  # (n_test, n_train)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    preds = []
    for row in idx:
        votes = train_y[row]
        vals, counts = np.unique(votes, return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def main():
    test = load_index("test_index.csv")
    train = load_index("train_subset_index.csv")
    cls2i = {c: i for i, c in enumerate(CLASSES)}
    y_test = test["label"].map(cls2i).values
    y_train = train["label"].map(cls2i).values

    model = rn50()
    train_f_by_s = {s: extract(model, train, s, "train") for s in LEVELS}

    results = {}
    pred_rows = []
    for s in LEVELS:
        test_f = extract(model, test, s, "test")
        for cond, tr_s in [("fit_native", 1), ("refit_per_level", s)]:
            pred = knn_predict(train_f_by_s[tr_s], y_train, test_f)
            acc = float((pred == y_test).mean())
            # balanced acc
            recs = [(pred[y_test == c] == c).mean() for c in range(len(CLASSES))]
            bacc = float(np.mean(recs))
            results[f"s{s}_{cond}"] = {"acc": acc, "balanced_acc": bacc}
            print(f"s={s} {cond}: acc={acc:.4f} bacc={bacc:.4f}", flush=True)
            if cond == "fit_native":
                pred_rows.append(pd.DataFrame({
                    "path": test["path"], "label": test["label"],
                    "level": s, "pred": [CLASSES[p] for p in pred],
                    "correct": pred == y_test}))
    pd.concat(pred_rows).to_csv(os.path.join(RES, "reference_per_image.csv"), index=False)
    json.dump(results, open(os.path.join(RES, "reference_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
