"""CONCH zero-shot evaluation across frozen magnification levels.

Official protocol from the CONCH repo notebook: conch_ViT-B-16, force_image_size=224,
prompt ensemble from prompts/crc100k_prompts_all_per_class.json.
Saves per-image logits and predictions for every level.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from conch.open_clip_custom import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier
from common import ART, CLASSES, LEVELS, RES, ROOT, degrade, load_index

DEVICE = "cuda"
CKPT = os.path.join(ROOT, "checkpoints", "pytorch_model.bin")
PROMPTS = os.path.join(ROOT, "code", "CONCH", "prompts", "crc100k_prompts_all_per_class.json")


class LevelDataset(Dataset):
    def __init__(self, df, s, preprocess):
        self.df, self.s, self.preprocess = df, s, preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df["path"].iloc[i]).convert("RGB")
        return self.preprocess(degrade(img, self.s))


def main():
    test = load_index("test_index.csv")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", CKPT, device=DEVICE, force_image_size=224)
    model.eval()

    with open(PROMPTS) as f:
        prompts = json.load(f)["0"]
    classnames = [prompts["classnames"][c] for c in CLASSES]
    templates = prompts["templates"]
    zw = zero_shot_classifier(model, classnames, templates, device=DEVICE)
    torch.save(zw.cpu(), os.path.join(ART, "zeroshot_weights.pt"))

    results = {}
    rows = []
    for s in LEVELS:
        ds = LevelDataset(test, s, preprocess)
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
        logits_all = []
        with torch.no_grad():
            for x in dl:
                x = x.to(DEVICE)
                f = model.encode_image(x, proj_contrast=True, normalize=True)
                logits_all.append((model.logit_scale.exp() * f @ zw).cpu().numpy())
        logits = np.concatenate(logits_all)
        np.save(os.path.join(ART, f"conch_logits_s{s}.npy"), logits)
        pred = logits.argmax(1)
        y = test["label"].map({c: i for i, c in enumerate(CLASSES)}).values
        acc = float((pred == y).mean())
        bacc = float(np.mean([(pred[y == c] == c).mean() for c in range(len(CLASSES))]))
        results[f"s{s}"] = {"acc": acc, "balanced_acc": bacc}
        print(f"s={s}: acc={acc:.4f} bacc={bacc:.4f}", flush=True)
        rows.append(pd.DataFrame({
            "path": test["path"], "label": test["label"], "level": s,
            "pred": [CLASSES[p] for p in pred], "correct": pred == y}))
    pd.concat(rows).to_csv(os.path.join(RES, "conch_per_image.csv"), index=False)
    json.dump(results, open(os.path.join(RES, "conch_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
