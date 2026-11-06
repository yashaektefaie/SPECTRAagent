"""Shared utilities for the CONCH SPECTRA audit: prompts, transforms, zero-shot scoring."""
import json, os
import numpy as np
import torch
from PIL import Image

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

PROMPTS_JSON = os.path.join(os.path.dirname(__file__), "..", "code", "CONCH", "prompts",
                            "crc100k_prompts_all_per_class.json")


def load_prompt_ensemble():
    """Official CONCH CRC-100K prompt ensemble: all classnames x templates per class."""
    with open(PROMPTS_JSON) as f:
        d = json.load(f)["0"]
    classnames, templates = d["classnames"], d["templates"]
    texts = {c: [t.replace("CLASSNAME", n) for n in classnames[c] for t in templates]
             for c in CLASSES}
    return texts  # ~110 prompts per class


def degrade_resolution(img: Image.Image, s: int) -> Image.Image:
    """Downsample by factor s (area resample) then upsample back to 224 (bicubic).
    Deterministic, label-free, prospective input transform."""
    if s == 1:
        return img
    w, h = img.size
    small = img.resize((w // s, h // s), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


@torch.no_grad()
def build_conch_text_feats(model, tokenizer_fn, device):
    texts = load_prompt_ensemble()
    feats = {}
    for c in CLASSES:
        tok = tokenizer_fn(texts[c]).to(device)
        f = model.encode_text(tok)
        f = f / f.norm(dim=-1, keepdim=True)
        feats[c] = f.mean(dim=0, keepdim=True)
    W = torch.cat([feats[c] for c in CLASSES], dim=0)
    return W / W.norm(dim=-1, keepdim=True)  # [9, d]


@torch.no_grad()
def zeroshot_scores(model_encode_image, preprocess, W, image_paths, level_s, device,
                    batch_size=256, logit_scale=None):
    """Return softmax probs [N, 9] and preds for images degraded by factor level_s."""
    probs_all, preds_all = [], []
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for p in image_paths[i:i + batch_size]:
            img = Image.open(p).convert("RGB")
            img = degrade_resolution(img, level_s)
            batch.append(preprocess(img))
        x = torch.stack(batch).to(device)
        f = model_encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        scale = logit_scale if logit_scale is not None else 100.0
        logits = scale * f @ W.t()
        pr = logits.softmax(dim=-1)
        probs_all.append(pr.cpu().numpy())
        preds_all.append(pr.argmax(dim=-1).cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(preds_all)


def balanced_accuracy(y_true, y_pred, n_classes=9):
    recs = []
    for c in range(n_classes):
        m = np.array(y_true) == c
        recs.append((np.array(y_pred)[m] == c).mean())
    return float(np.mean(recs)), recs
