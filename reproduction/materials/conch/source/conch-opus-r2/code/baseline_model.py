"""Model-independent reference classifier for CRC 9-class tissue classification.

Backbone: ImageNet-pretrained ResNet50 (torchvision IMAGENET1K_V2), weights loaded
from a locally pinned .pth so the hash can be frozen. The penultimate 2048-d
feature feeds a multinomial logistic-regression head trained on NCT-CRC-HE-100K
reference tiles (sharp). This is a standard transfer-learning pathology baseline
that is architecturally and training-wise independent of CONCH. It is trained on
sharp reference tiles only (matching CONCH's implicitly sharp pretraining) and is
evaluated on the SAME blurred confirmation tiles as CONCH, so the two models'
degradation slopes under the frozen blur axis are directly comparable.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2

ROOT = Path(__file__).resolve().parents[1]
BACKBONE_PATH = ROOT / "weights" / "baseline_backbone" / "resnet50_IMAGENET1K_V2.pth"
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_backbone(device="cpu", threads=4):
    torch.set_num_threads(threads)
    import torchvision
    net = torchvision.models.resnet50(weights=None)
    sd = torch.load(str(BACKBONE_PATH), map_location="cpu")
    net.load_state_dict(sd)
    net.fc = nn.Identity()
    net.eval().to(device)
    return net


def _prep(img_rgb):
    # tiles are 224x224; resize defensively to 224 then ImageNet-normalize
    if img_rgb.shape[:2] != (224, 224):
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def extract_features(net, imgs_rgb, device="cpu", batch=32):
    feats = []
    buf = []
    def flush(buf):
        x = torch.stack(buf, 0).to(device)
        return net(x).cpu().numpy()
    for im in imgs_rgb:
        buf.append(_prep(im))
        if len(buf) == batch:
            feats.append(flush(buf)); buf = []
    if buf:
        feats.append(flush(buf))
    return np.concatenate(feats, 0)


def train_head(feats, labels, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial", random_state=seed),
    )
    clf.fit(feats, labels)
    return clf
