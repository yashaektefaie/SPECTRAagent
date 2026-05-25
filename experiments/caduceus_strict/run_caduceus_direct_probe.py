#!/usr/bin/env python3
"""Run a small direct Caduceus frozen-embedding SPECTRA probe.

This is the recovery path the exploratory agent should have attempted after
finding that the default SPECTRA env lacked torch/transformers.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


SEED = 20260513
PRIME = np.uint64(4294967311)
HASH_A = np.array([2654435761 + 2246822519 * i for i in range(8)], dtype=np.uint64)
HASH_B = np.array([3266489917 + 668265263 * i for i in range(8)], dtype=np.uint64)


def read_manifest_row(manifest_path: Path, suite: str, task: str) -> dict[str, str]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_suite"] == suite and row["task_name"] == task:
                return row
    raise ValueError(f"Task not found in manifest: {suite}/{task}")


def load_task(row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["sample_id", "sequence", "label"]
    train = pd.read_csv(row["train_path"], usecols=cols)
    test = pd.read_csv(row["test_path"], usecols=cols)
    train["label"] = train["label"].astype(str)
    test["label"] = test["label"].astype(str)
    return train, test


def stratified_cap(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    parts = []
    per_class = max(1, max_rows // df["label"].nunique())
    for _, group in df.groupby("label", sort=True):
        parts.append(group.sample(n=min(len(group), per_class), random_state=SEED))
    capped = pd.concat(parts, axis=0)
    if len(capped) < max_rows:
        remaining = df.drop(capped.index)
        capped = pd.concat(
            [capped, remaining.sample(n=min(len(remaining), max_rows - len(capped)), random_state=SEED)],
            axis=0,
        )
    return capped.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def minhash_signature(seq: str, k: int = 6, n_hashes: int = 8) -> np.ndarray:
    seq = str(seq).upper()
    if len(seq) < k:
        kmers = [seq]
    else:
        kmers = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    base = np.array([zlib.crc32(kmer.encode("ascii", "ignore")) for kmer in kmers], dtype=np.uint64)
    vals = (base[:, None] * HASH_A[:n_hashes][None, :] + HASH_B[:n_hashes][None, :]) % PRIME
    return vals.min(axis=0).astype(np.uint64)


def signatures(seqs: list[str]) -> np.ndarray:
    arr = np.empty((len(seqs), 8), dtype=np.uint64)
    for idx, seq in enumerate(seqs):
        arr[idx] = minhash_signature(seq)
    return arr


def train_to_test_minhash_similarity(train_seqs: list[str], test_seqs: list[str]) -> np.ndarray:
    """Return each train sequence's nearest estimated 6-mer Jaccard to test."""
    train_sig = signatures(train_seqs)
    test_sig = signatures(test_seqs)
    sims = np.zeros(len(train_seqs), dtype=np.float32)
    chunk = 1024
    for start in range(0, len(train_sig), chunk):
        stop = min(start + chunk, len(train_sig))
        # Hamming agreement between 8-hash signatures estimates Jaccard.
        agree = train_sig[start:stop, None, :] == test_sig[None, :, :]
        sims[start:stop] = agree.mean(axis=2).max(axis=1)
    return sims


def metric_value(metric_name: str, y_true: list[str], y_pred: list[str]) -> float:
    if metric_name == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric_name == "f1_binary":
        labels = sorted(set(y_true) | set(y_pred))
        return float(f1_score(y_true, y_pred, pos_label=labels[-1], zero_division=0))
    return float(accuracy_score(y_true, y_pred))


def embed_sequences(
    model_dir: Path,
    seqs: list[str],
    output_path: Path,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    if output_path.exists():
        print(f"Loading cached embeddings: {output_path}", flush=True)
        return np.load(output_path)

    print(f"Loading tokenizer/model from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Embedding {len(seqs)} sequences on {device} with batch_size={batch_size}", flush=True)

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            if start % max(batch_size * 10, 1) == 0:
                print(f"  embedded {start}/{len(seqs)}", flush=True)
            batch = seqs[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = encoded.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.vstack(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb)
    print(f"Saved embeddings: {output_path}", flush=True)
    return emb


def run_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    metric_name: str,
    thresholds: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_sim = train_to_test_minhash_similarity(
        train["sequence"].astype(str).tolist(),
        test["sequence"].astype(str).tolist(),
    )
    perf_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    y_test = test["label"].astype(str).tolist()

    for threshold in thresholds:
        if threshold > 1:
            keep = np.ones(len(train), dtype=bool)
            split_name = "original"
        else:
            keep = train_sim < threshold
            split_name = f"remove_train_sim_ge_{threshold:.2f}"

        x_train = train_embeddings[keep]
        y_train = train.loc[keep, "label"].astype(str).tolist()
        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                random_state=SEED,
                class_weight="balanced",
            ),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(test_embeddings).tolist()
        score = metric_value(metric_name, y_test, pred)

        retained_train_sim = train_sim[keep]
        perf_rows.append(
            {
                "split_name": split_name,
                "threshold": "" if threshold > 1 else threshold,
                "metric": metric_name,
                "score": score,
                "fresh_model": "Frozen Caduceus mean-pooled embeddings + SGD logistic probe",
            }
        )
        split_rows.append(
            {
                "split_name": split_name,
                "n_train_retained": int(keep.sum()),
                "n_train_removed": int((~keep).sum()),
                "n_test": int(len(test)),
                "train_to_test_kmer6_nn_mean": float(retained_train_sim.mean()),
                "train_to_test_frac_ge_0_80": float((retained_train_sim >= 0.80).mean()),
                "train_to_test_frac_ge_0_90": float((retained_train_sim >= 0.90).mean()),
            }
        )
    return perf_rows, split_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("/ewsc/yektefai/spectra_caduceus_strict_20260513"))
    parser.add_argument("--suite", default="nucleotide_transformer")
    parser.add_argument("--task", default="enhancers")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.bundle / "agents" / "caduceus_direct_probe_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.bundle / "agent_inputs" / "caduceus_strict_visible" / "tasks_manifest.csv"
    model_dir = args.bundle / "model" / "caduceus_ps"
    row = read_manifest_row(manifest_path, args.suite, args.task)
    train, test = load_task(row)
    train = stratified_cap(train, args.max_train)
    test = stratified_cap(test, args.max_test)
    print(
        f"Loaded {args.suite}/{args.task}: train={len(train)} test={len(test)} "
        f"metric={row['original_metric']}",
        flush=True,
    )

    cap_suffix = ""
    if args.max_train is not None or args.max_test is not None:
        cap_suffix = f"_captrain{args.max_train or 'all'}_captest{args.max_test or 'all'}"
    prefix = f"{args.suite}_{args.task}{cap_suffix}"
    train_embeddings = embed_sequences(
        model_dir,
        train["sequence"].astype(str).tolist(),
        out_dir / f"{prefix}_train_caduceus_mean.npy",
        args.batch_size,
        args.max_length,
    )
    test_embeddings = embed_sequences(
        model_dir,
        test["sequence"].astype(str).tolist(),
        out_dir / f"{prefix}_test_caduceus_mean.npy",
        args.batch_size,
        args.max_length,
    )
    perf_rows, split_rows = run_probe(
        train,
        test,
        train_embeddings,
        test_embeddings,
        row["original_metric"],
        thresholds=[1.01, 0.95, 0.85, 0.75],
    )
    print("Finished fresh Caduceus probe fits", flush=True)

    perf_path = out_dir / f"{prefix}_performance_by_overlap.csv"
    split_path = out_dir / f"{prefix}_split_stats.csv"
    pd.DataFrame(perf_rows).to_csv(perf_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    summary = {
        "suite": args.suite,
        "task": args.task,
        "metric": row["original_metric"],
        "model_dir": str(model_dir),
        "environment_python": str(Path(torch.__file__).parents[1]),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "performance_by_overlap": str(perf_path),
        "split_stats": str(split_path),
        "rows": perf_rows,
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
