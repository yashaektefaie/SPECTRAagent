"""Reusable CONCH zero-shot inference for 9-class CRC tissue classification.

Loads the locally licensed CONCH checkpoint and the frozen upstream crc100k
prompt ensemble, builds the zero-shot text classifier, and scores a list of
image paths. Returns per-tile cosine-similarity logits, predicted class, and
probabilities. Used for both the development reproduction diagnostic and the
frozen confirmation run. No image is modified on disk.
"""
import sys, os, json, argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "CONCH"))
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
WEIGHTS = str(ROOT / "weights" / "pytorch_model.bin")
PROMPTS = ROOT / "code" / "CONCH" / "prompts" / "crc100k_prompts_all_per_class.json"


def load_model(device="cpu", threads=4):
    torch.set_num_threads(threads)
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", WEIGHTS, device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def build_classifier(model, device="cpu"):
    """Frozen crc100k ensemble: mean over classnames x templates, per class."""
    spec = json.loads(PROMPTS.read_text())["0"]
    classnames = spec["classnames"]  # dict class-> list of names
    templates = spec["templates"]
    tokenizer = get_tokenizer()
    weights = []
    for c in CLASSES:
        embs = []
        for name in classnames[c]:
            texts = [t.replace("CLASSNAME", name) for t in templates]
            tok = tokenize(tokenizer, texts).to(device)
            emb = model.encode_text(tok)
            embs.append(F.normalize(emb, dim=-1))
        ce = torch.stack(embs, dim=0).mean(dim=(0, 1))
        ce = ce / ce.norm()
        weights.append(ce)
    return torch.stack(weights, dim=1)  # [512, 9]


@torch.no_grad()
def score_paths(model, preprocess, classifier, paths, device="cpu", batch=32, log_every=20):
    """Returns dict: logits [N,9] (cosine sim), probs [N,9] (temp softmax), preds [N]."""
    all_logits = []
    temp = model.logit_scale.exp().item()
    buf = []
    def flush(buf):
        x = torch.stack(buf, dim=0).to(device)
        feats = model.encode_image(x, proj_contrast=True, normalize=True)
        lg = feats @ classifier
        return lg.cpu().numpy()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        buf.append(preprocess(img))
        if len(buf) == batch:
            all_logits.append(flush(buf)); buf = []
            if log_every and (i // batch) % log_every == 0:
                print(f"  scored {i+1}/{len(paths)}", flush=True)
    if buf:
        all_logits.append(flush(buf))
    logits = np.concatenate(all_logits, axis=0)
    probs = F.softmax(torch.from_numpy(logits) * temp, dim=1).numpy()
    preds = logits.argmax(axis=1)
    return {"logits": logits, "probs": probs, "preds": preds, "temp": temp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns: image_abspath,label_name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    paths = [r["image_abspath"] for r in rows]
    labels = [CLASS_TO_IDX[r["label_name"]] for r in rows]

    model, preprocess = load_model(device=args.device)
    classifier = build_classifier(model, device=args.device)
    res = score_paths(model, preprocess, classifier, paths, device=args.device, batch=args.batch)

    out = []
    for i, r in enumerate(rows):
        pred = int(res["preds"][i])
        row = {"image_abspath": r["image_abspath"], "label_name": r["label_name"],
               "label": labels[i], "pred": pred, "pred_label_name": CLASSES[pred],
               "correct": int(pred == labels[i])}
        for j, c in enumerate(CLASSES):
            row[f"logit_{c}"] = float(res["logits"][i, j])
            row[f"prob_{c}"] = float(res["probs"][i, j])
        out.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    acc = np.mean([o["correct"] for o in out])
    print(f"WROTE {args.out}  N={len(out)}  acc={acc:.4f}  temp={res['temp']:.3f}")


if __name__ == "__main__":
    main()
