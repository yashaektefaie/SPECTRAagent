"""Evaluate CONCH zero-shot classification across the frozen axis grid.

Usage:
  python scripts/eval_conch.py --data-root data/CRC-VAL-HE-7K --tag val7k \
      [--subset-per-class N --seed 0] [--exclude-list path] [--features]

Saves:
  results/<tag>_conch_perimage.csv   per-image, per-level predictions
  results/<tag>_conch_summary.json   per-level balanced accuracy etc.
  results/<tag>_conch_features.npz   (optional) 768-d image features per level
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transforms_spec import TRANSFORMS, mean_delta_e, hue_hist_distance

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from conch.downstream.zeroshot_path import zero_shot_classifier

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

HUE_LEVELS = [0, 10, 20, 30, 45, 60, 90, 120, 180]
SAT_LEVELS = [1.0, 0.5, 0.25, 0.0]
VAL_LEVELS = [1.0, 0.75, 0.5, 0.25]


def build_file_list(root, subset_per_class=None, seed=0, exclude=None):
    rows = []
    excl = set()
    if exclude:
        excl = set(pd.read_csv(exclude)["filename"])
    for cls in CLASSES:
        files = sorted(os.listdir(os.path.join(root, cls)))
        files = [f for f in files if f not in excl]
        if subset_per_class:
            rng = np.random.RandomState(seed)
            files = list(rng.choice(files, size=subset_per_class, replace=False))
        rows += [(cls, f) for f in files]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-list", default=None)
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda"
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "models/conch/pytorch_model.bin", device=device)
    model.eval()
    tokenizer = get_tokenizer()

    with open("code/conch_repo/prompts/crc100k_prompts_all_per_class.json") as f:
        prompts = json.load(f)["0"]
    classnames = [prompts["classnames"][c] for c in CLASSES]
    zeroshot = zero_shot_classifier(model, classnames, prompts["templates"],
                                    tokenizer=tokenizer, device=device)

    rows = build_file_list(args.data_root, args.subset_per_class, args.seed,
                           args.exclude_list)
    print(f"{len(rows)} images")

    # measure prospective dissimilarity proxies on up to 200 images per hue level
    grid = [("hue", t) for t in HUE_LEVELS] + \
           [("sat", a) for a in SAT_LEVELS if a != 1.0] + \
           [("val", b) for b in VAL_LEVELS if b != 1.0]

    proxy_rows = []
    proxy_sample = rows[:200]
    for name, lvl in grid:
        des, hds = [], []
        fn = TRANSFORMS[name]
        for cls, f in proxy_sample:
            img = np.array(Image.open(os.path.join(args.data_root, cls, f)).convert("RGB"))
            t = fn(img, lvl)
            des.append(mean_delta_e(img, t))
            hds.append(hue_hist_distance(img, t))
        proxy_rows.append({"axis": name, "level": lvl,
                           "delta_e": float(np.mean(des)),
                           "hue_hist_dist": float(np.mean(hds))})
        print("proxy", name, lvl, proxy_rows[-1]["delta_e"], proxy_rows[-1]["hue_hist_dist"])
    pd.DataFrame(proxy_rows).to_csv(f"results/{args.tag}_dissimilarity.csv", index=False)

    records, feat_store = [], {}
    for name, lvl in grid:
        fn = TRANSFORMS[name]
        feats_lvl = []
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            imgs = []
            for cls, f in batch:
                img = np.array(Image.open(os.path.join(args.data_root, cls, f)).convert("RGB"))
                imgs.append(preprocess(Image.fromarray(fn(img, lvl))))
            x = torch.stack(imgs).to(device)
            with torch.inference_mode():
                im_emb = model.encode_image(x, proj_contrast=True, normalize=True)
                feat = model.encode_image(x, proj_contrast=False, normalize=False)
                logits = 100.0 * im_emb @ zeroshot
            probs = logits.softmax(-1).cpu().numpy()
            preds = probs.argmax(1)
            for (cls, f), p, pr in zip(batch, preds, probs):
                records.append({"axis": name, "level": lvl, "filename": f,
                                "true": cls, "pred": CLASSES[int(p)],
                                "correct": int(CLASSES[int(p)] == cls),
                                "conf": float(pr.max())})
            if args.features:
                feats_lvl.append(feat.cpu().numpy().astype(np.float16))
        if args.features:
            feat_store[f"{name}_{lvl}"] = np.concatenate(feats_lvl)
        sub = pd.DataFrame([r for r in records if r["axis"] == name and r["level"] == lvl])
        acc = sub["correct"].mean()
        bacc = sub.groupby("true")["correct"].mean().mean()
        print(f"{name} {lvl}: acc={acc:.4f} bacc={bacc:.4f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(f"results/{args.tag}_conch_perimage.csv", index=False)
    if args.features:
        np.savez_compressed(f"results/{args.tag}_conch_features.npz",
                            files=np.array([f for _, f in rows]),
                            trues=np.array([c for c, _ in rows]), **feat_store)

    summary = {}
    for (name, lvl), sub in df.groupby(["axis", "level"]):
        per_class = sub.groupby("true")["correct"].mean()
        summary[f"{name}_{lvl}"] = {
            "axis": name, "level": float(lvl), "n": int(len(sub)),
            "acc": float(sub["correct"].mean()),
            "bacc": float(per_class.mean()),
            "per_class_recall": per_class.to_dict(),
        }
    with open(f"results/{args.tag}_conch_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
