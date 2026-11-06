"""Shared utilities for the SPECTRA CONCH magnification audit."""
import os
from PIL import Image
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
RES = os.path.join(ROOT, "results")
TEST_DIR = os.path.join(DATA, "CRC-VAL-HE-7K")
TRAIN_DIR = os.path.join(DATA, "NCT-CRC-HE-100K-NONORM")
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
LEVELS = [1, 2, 4, 8, 16]
SEED = 0


def degrade(img: Image.Image, s: int) -> Image.Image:
    """Frozen axis transform: bicubic downsample by s, bicubic upsample back to 224."""
    if s == 1:
        return img
    w, h = img.size
    small = img.resize((w // s, h // s), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def load_index(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(ART, name))
