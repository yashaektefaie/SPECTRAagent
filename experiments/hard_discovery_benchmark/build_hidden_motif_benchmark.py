"""Build a hidden-axis regulatory DNA agent-discovery benchmark.

Agent-visible files contain only DNA sequences, labels, and predictions.
Hidden files retain HOCOMOCO motif/family provenance for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


DNA = "ACGT"


def parse_meme(path: Path) -> list[dict]:
    motifs = []
    current = None
    rows = []
    width = None
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("MOTIF "):
                if current and rows:
                    motifs.append(_finish_motif(current, rows))
                current = line.split()[1]
                rows = []
                width = None
            elif line.startswith("letter-probability matrix:"):
                parts = line.replace("=", " = ").split()
                if "w" in parts:
                    width = int(parts[parts.index("w") + 2])
            elif current and line[0].isdigit():
                vals = [float(x) for x in line.split()]
                if len(vals) == 4:
                    rows.append(vals)
                    if width is not None and len(rows) == width:
                        motifs.append(_finish_motif(current, rows))
                        current = None
                        rows = []
                        width = None
        if current and rows:
            motifs.append(_finish_motif(current, rows))
    return motifs


def _finish_motif(motif_id: str, rows: list[list[float]]) -> dict:
    pwm = np.asarray(rows, dtype=float)
    consensus = "".join(DNA[int(np.argmax(row))] for row in pwm)
    tf = motif_id.split(".")[0]
    family = "".join(ch for ch in tf if not ch.isdigit())
    if len(family) < 2:
        family = tf
    return {
        "motif_id": motif_id,
        "tf": tf,
        "family": family,
        "pwm": pwm,
        "consensus": consensus,
        "width": int(pwm.shape[0]),
    }


def weighted_consensus(pwm: np.ndarray, rng: random.Random) -> str:
    chars = []
    for row in pwm:
        chars.append(rng.choices(DNA, weights=row, k=1)[0])
    return "".join(chars)


def dinuc_background(length: int, rng: random.Random) -> str:
    # Keep backgrounds deliberately uninformative; the hidden axis should be
    # motif syntax, not composition.
    return "".join(rng.choice(DNA) for _ in range(length))


def shuffle_seq(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def insert_middle(background: str, insert: str) -> str:
    start = (len(background) - len(insert)) // 2
    return background[:start] + insert + background[start + len(insert):]


def kmers(seq: str, k: int) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build_dataset(motifs: list[dict], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for motif in motifs:
        if 6 <= motif["width"] <= 22:
            by_family[motif["family"]].append(motif)

    preferred_eval = ["FOX", "GATA", "KLF", "STAT", "SOX", "NFKB", "IRF", "CEBP"]
    eval_families = [fam for fam in preferred_eval if len(by_family.get(fam, [])) >= 3]
    if len(eval_families) < 5:
        ranked = sorted(by_family, key=lambda fam: len(by_family[fam]), reverse=True)
        for fam in ranked:
            if fam not in eval_families and len(by_family[fam]) >= 3:
                eval_families.append(fam)
            if len(eval_families) >= 6:
                break

    train_families = [fam for fam in by_family if fam not in set(eval_families) and len(by_family[fam]) >= 2]
    train_motifs_all = [m for fam in train_families for m in by_family[fam]]
    eval_hidden_motifs_all = [m for fam in eval_families for m in by_family[fam]]
    train_motifs = rng.sample(train_motifs_all, min(320, len(train_motifs_all)))
    eval_hidden_motifs = rng.sample(eval_hidden_motifs_all, min(180, len(eval_hidden_motifs_all)))
    eval_seen_like_motifs = rng.sample(train_motifs, min(120, len(train_motifs)))

    rows = []
    seq_len = 120
    samples_per_train_motif = 4
    samples_per_eval_motif = 5

    def add_rows(split: str, motif_list: list[dict], n_per: int, eval_group: str | None = None) -> None:
        for motif in motif_list:
            for rep in range(n_per):
                bg = dinuc_background(seq_len, rng)
                motif_seq = weighted_consensus(motif["pwm"], rng)
                neg_seq = shuffle_seq(motif_seq, rng)
                for label, inserted in [(1, motif_seq), (0, neg_seq)]:
                    seq = insert_middle(bg, inserted)
                    rows.append({
                        "sample_id": f"{split}_{len(rows):06d}",
                        "split": split,
                        "sequence": seq,
                        "y_true": label,
                        "motif_id": motif["motif_id"],
                        "tf": motif["tf"],
                        "family": motif["family"],
                        "motif_consensus": motif["consensus"],
                        "inserted_sequence": inserted,
                        "eval_group": eval_group or split,
                    })

    add_rows("train", train_motifs, samples_per_train_motif)
    add_rows("eval", eval_seen_like_motifs, samples_per_eval_motif, "seen_family")
    add_rows("eval", eval_hidden_motifs, samples_per_eval_motif, "heldout_family")

    df = pd.DataFrame(rows)

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(5, 6), lowercase=False, binary=True)
    x_train = vectorizer.fit_transform(df.loc[df.split == "train", "sequence"])
    y_train = df.loc[df.split == "train", "y_true"].to_numpy()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0015,
        max_iter=80,
        random_state=seed,
        class_weight="balanced",
        tol=1e-4,
    )
    model.fit(x_train, y_train)
    x_eval = vectorizer.transform(df.loc[df.split == "eval", "sequence"])
    df.loc[df.split == "eval", "y_pred"] = model.predict_proba(x_eval)[:, 1]
    df.loc[df.split == "train", "y_pred"] = model.predict_proba(x_train)[:, 1]

    train = df[df.split == "train"].copy()
    eval_df = df[df.split == "eval"].copy()

    train_pos_kmers = [kmers(seq, 6) for seq in train.loc[train.y_true == 1, "sequence"]]
    train_family_kmers = defaultdict(list)
    for _, row in train[train.y_true == 1].iterrows():
        train_family_kmers[row["family"]].append(kmers(row["sequence"], 6))

    hidden_rows = []
    for _, row in eval_df.iterrows():
        seq_k = kmers(row["sequence"], 6)
        max_train_sim = max(jaccard(seq_k, k) for k in train_pos_kmers)
        family_sims = []
        for fam, family_sets in train_family_kmers.items():
            family_sims.append((fam, max(jaccard(seq_k, k) for k in family_sets)))
        nearest_family, nearest_family_sim = max(family_sims, key=lambda item: item[1])
        hidden_rows.append({
            "sample_id": row["sample_id"],
            "motif_id": row["motif_id"],
            "tf": row["tf"],
            "family": row["family"],
            "motif_consensus": row["motif_consensus"],
            "inserted_sequence": row["inserted_sequence"],
            "eval_group": row["eval_group"],
            "max_train_positive_6mer_jaccard": max_train_sim,
            "nearest_train_family_by_6mer": nearest_family,
            "nearest_train_family_similarity": nearest_family_sim,
        })
    hidden_eval = pd.DataFrame(hidden_rows)

    return train, eval_df, hidden_eval


def metric_summary(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    if df["y_true"].nunique() == 2:
        out["auroc"] = float(roc_auc_score(df["y_true"], df["y_pred"]))
        out["auprc"] = float(average_precision_score(df["y_true"], df["y_pred"]))
    out["mean_positive_score"] = float(df.loc[df.y_true == 1, "y_pred"].mean())
    out["mean_negative_score"] = float(df.loc[df.y_true == 0, "y_pred"].mean())
    out["brier"] = float(np.mean((df["y_pred"] - df["y_true"]) ** 2))
    return out


def write_outputs(train: pd.DataFrame, eval_df: pd.DataFrame, hidden_eval: pd.DataFrame, output_dir: Path) -> None:
    visible = output_dir / "agent_visible"
    hidden = output_dir / "hidden_for_scoring"
    visible.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)

    train[["sample_id", "sequence", "y_true"]].to_csv(visible / "train.csv", index=False)
    eval_df[["sample_id", "sequence", "y_true", "y_pred"]].to_csv(visible / "eval_predictions.csv", index=False)
    (visible / "model_description.json").write_text(json.dumps({
        "model_name": "trained_dna_sequence_classifier",
        "task": "predict binary regulatory activity from DNA sequence",
        "visible_training_rows": int(len(train)),
        "visible_eval_rows": int(len(eval_df)),
        "sequence_length": int(eval_df["sequence"].str.len().iloc[0]),
        "target": "binary activity label",
        "prediction": "probability of active regulatory sequence"
    }, indent=2) + "\n")
    (visible / "README.md").write_text(
        "# Hidden-Axis Regulatory DNA Model Evaluation Bundle\n\n"
        "You are given artifacts for a trained DNA-sequence prediction model.\n\n"
        "Files:\n\n"
        "- `train.csv`: training sequences and binary labels.\n"
        "- `eval_predictions.csv`: held-out sequences, binary labels, and model predictions.\n"
        "- `model_description.json`: model/task metadata.\n\n"
        "The task is binary regulatory activity prediction from raw DNA sequence.\n",
        encoding="utf-8",
    )

    train[["sample_id", "motif_id", "tf", "family", "motif_consensus", "inserted_sequence"]].to_csv(
        hidden / "train_hidden_motif_labels.csv", index=False
    )
    hidden_eval.to_csv(hidden / "eval_hidden_motif_labels.csv", index=False)

    eval_joined = eval_df.merge(hidden_eval, on="sample_id", suffixes=("", "_hidden"))
    if "eval_group_hidden" in eval_joined.columns:
        eval_joined["eval_group"] = eval_joined["eval_group_hidden"]
    if "family_hidden" in eval_joined.columns:
        eval_joined["family"] = eval_joined["family_hidden"]
    metrics = {
        "overall": metric_summary(eval_joined),
        "by_hidden_eval_group": {
            key: metric_summary(group)
            for key, group in eval_joined.groupby("eval_group", observed=True)
        },
        "by_hidden_family": {
            key: metric_summary(group)
            for key, group in eval_joined.groupby("family", observed=True)
        },
        "family_counts": Counter(eval_joined["family"]),
    }
    # Performance as a function of hidden train-positive sequence similarity.
    curve = []
    for threshold in [1.0, 0.18, 0.16, 0.14, 0.12, 0.10]:
        subset = eval_joined[eval_joined["max_train_positive_6mer_jaccard"] <= threshold]
        if len(subset) >= 20:
            row = metric_summary(subset)
            row["threshold"] = threshold
            row["mean_max_train_positive_6mer_jaccard"] = float(subset["max_train_positive_6mer_jaccard"].mean())
            curve.append(row)
    metrics["hidden_similarity_curve"] = curve
    (hidden / "hidden_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    motifs = parse_meme(args.meme)
    train, eval_df, hidden_eval = build_dataset(motifs, args.seed)
    write_outputs(train, eval_df, hidden_eval, args.output_dir)


if __name__ == "__main__":
    main()
