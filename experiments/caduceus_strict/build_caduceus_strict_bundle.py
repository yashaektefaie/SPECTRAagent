"""Build a strict Caduceus/SPECTRA agent bundle.

The strict bundle contains only Caduceus paper/repo/model context and datasets
that the Caduceus repository names as original evaluation data. It intentionally
does not include DART-Eval papers, results, or conclusions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


CADUCEUS_PS_MODEL = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
CADUCEUS_REPO_URL = "https://github.com/kuleshov-group/caduceus.git"
CADUCEUS_PAPER_URL = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/schiff24a/schiff24a.pdf"

GENOMIC_BENCHMARK_TASKS = [
    "dummy_mouse_enhancers_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]

NUCLEOTIDE_TRANSFORMER_TASKS = [
    "enhancers",
    "enhancers_types",
    "H3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K14ac",
    "H3K36me3",
    "H3K79me3",
    "H4",
    "H4ac",
    "promoter_all",
    "promoter_no_tata",
    "promoter_tata",
    "splice_sites_acceptors",
    "splice_sites_all",
    "splice_sites_donors",
]

NT_METRICS = {
    "enhancers": "mcc",
    "enhancers_types": "mcc",
    "H3": "mcc",
    "H3K4me1": "mcc",
    "H3K4me2": "mcc",
    "H3K4me3": "mcc",
    "H3K9ac": "mcc",
    "H3K14ac": "mcc",
    "H3K36me3": "mcc",
    "H3K79me3": "mcc",
    "H4": "mcc",
    "H4ac": "mcc",
    "promoter_all": "f1_binary",
    "promoter_no_tata": "f1_binary",
    "promoter_tata": "f1_binary",
    "splice_sites_acceptors": "f1_binary",
    "splice_sites_all": "accuracy",
    "splice_sites_donors": "f1_binary",
}

MODEL_FILES = [
    "README.md",
    "config.json",
    "configuration_caduceus.py",
    "model.safetensors",
    "modeling_caduceus.py",
    "modeling_rcps.py",
    "special_tokens_map.json",
    "tokenization_caduceus.py",
    "tokenizer_config.json",
]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def json_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def clone_or_update(repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", CADUCEUS_REPO_URL, str(repo_dir)], check=True)


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def download_genomic_benchmark(task: str, data_dir: Path) -> dict[str, Path]:
    repo = f"katarinagresova/Genomic_Benchmarks_{task}"
    meta = json_url(f"https://huggingface.co/api/datasets/{repo}")
    siblings = [item["rfilename"] for item in meta.get("siblings", [])]
    split_paths: dict[str, Path] = {}
    for split in ["train", "test"]:
        matches = [name for name in siblings if name.startswith(f"data/{split}-") and name.endswith(".parquet")]
        if not matches:
            raise RuntimeError(f"Could not find {split} parquet for {repo}")
        rel = matches[0]
        out = data_dir / "genomic_benchmarks" / task / f"{split}.parquet"
        download(f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}", out)
        split_paths[split] = out
    return split_paths


def download_nt_task(task: str, data_dir: Path) -> dict[str, Path]:
    split_paths = {}
    for split in ["train", "test"]:
        out = data_dir / "nucleotide_transformer" / task / f"{split}.parquet"
        download(
            f"https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/resolve/main/{task}/{split}.parquet",
            out,
        )
        split_paths[split] = out
    return split_paths


def normalize_parquet(path: Path, suite: str, task: str, split: str, out_path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    sequence_col = "sequence" if "sequence" in df.columns else "seq"
    name_col = "name" if "name" in df.columns else None
    label_col = "label"
    out = pd.DataFrame(
        {
            "sample_id": [f"{suite}:{task}:{split}:{idx:07d}" for idx in range(len(df))],
            "source_suite": suite,
            "task_name": task,
            "original_split": split,
            "sequence": df[sequence_col].astype(str).str.upper(),
            "label": df[label_col].astype(int),
        }
    )
    if name_col:
        out["source_name"] = df[name_col].astype(str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "rows": int(len(out)),
        "classes": int(out["label"].nunique()),
        "min_len": int(out["sequence"].str.len().min()),
        "median_len": float(out["sequence"].str.len().median()),
        "max_len": int(out["sequence"].str.len().max()),
    }


def build_bundle(root: Path, limit_tasks: int | None = None) -> None:
    repo_dir = root / "repos" / "caduceus"
    assets_dir = root / "assets"
    data_dir = root / "data"
    model_dir = root / "model" / "caduceus_ps"
    visible_dir = root / "agent_inputs" / "caduceus_strict_visible"
    task_dir = visible_dir / "tasks"
    clone_or_update(repo_dir)
    download(CADUCEUS_PAPER_URL, assets_dir / "caduceus_schiff24a.pdf")
    for filename in MODEL_FILES:
        download(
            f"https://huggingface.co/{CADUCEUS_PS_MODEL}/resolve/main/{filename}",
            model_dir / filename,
        )
    download(
        "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/resolve/main/variant_effect_causal_eqtl/All_Tissues.csv",
        data_dir / "lrb" / "variant_effect_causal_eqtl" / "All_Tissues.csv",
    )

    tasks = []
    gb_tasks = GENOMIC_BENCHMARK_TASKS if limit_tasks is None else GENOMIC_BENCHMARK_TASKS[:limit_tasks]
    nt_tasks = NUCLEOTIDE_TRANSFORMER_TASKS if limit_tasks is None else NUCLEOTIDE_TRANSFORMER_TASKS[:limit_tasks]

    for task in gb_tasks:
        splits = download_genomic_benchmark(task, data_dir)
        rows = {"source_suite": "genomic_benchmarks", "task_name": task, "original_metric": "accuracy"}
        for split, parquet_path in splits.items():
            out_csv = task_dir / "genomic_benchmarks" / task / f"{split}.csv"
            stats = normalize_parquet(parquet_path, "genomic_benchmarks", task, split, out_csv)
            rows[f"{split}_path"] = str(out_csv)
            rows[f"n_{split}"] = stats["rows"]
            rows[f"{split}_classes"] = stats["classes"]
            rows[f"{split}_median_len"] = stats["median_len"]
            rows[f"{split}_max_len"] = stats["max_len"]
        tasks.append(rows)

    for task in nt_tasks:
        splits = download_nt_task(task, data_dir)
        rows = {"source_suite": "nucleotide_transformer", "task_name": task, "original_metric": NT_METRICS[task]}
        for split, parquet_path in splits.items():
            out_csv = task_dir / "nucleotide_transformer" / task / f"{split}.csv"
            stats = normalize_parquet(parquet_path, "nucleotide_transformer", task, split, out_csv)
            rows[f"{split}_path"] = str(out_csv)
            rows[f"n_{split}"] = stats["rows"]
            rows[f"{split}_classes"] = stats["classes"]
            rows[f"{split}_median_len"] = stats["median_len"]
            rows[f"{split}_max_len"] = stats["max_len"]
        tasks.append(rows)

    lrb_in = data_dir / "lrb" / "variant_effect_causal_eqtl" / "All_Tissues.csv"
    lrb_out = visible_dir / "variant_effect_causal_eqtl_metadata.csv"
    lrb = pd.read_csv(lrb_in)
    lrb.to_csv(lrb_out, index=False)

    pd.DataFrame(tasks).to_csv(visible_dir / "tasks_manifest.csv", index=False)
    model_context = {
        "model_name": "Caduceus-PS",
        "huggingface_model": CADUCEUS_PS_MODEL,
        "local_model_dir": str(model_dir),
        "checkpoint_path": str(model_dir / "model.safetensors"),
        "paper_pdf": str(assets_dir / "caduceus_schiff24a.pdf"),
        "repo_path": str(repo_dir),
        "repo_commit": git_commit(repo_dir),
        "strict_condition": "No DART-Eval paper, result tables, or conclusions are included.",
        "foundation_model_note": (
            "For a pretrained foundation model, SPECTRA benchmark mode means "
            "fresh downstream probes/fine-tuning heads per split, not pretraining "
            "Caduceus from scratch."
        ),
    }
    visible_dir.mkdir(parents=True, exist_ok=True)
    (visible_dir / "model_context.json").write_text(json.dumps(model_context, indent=2) + "\n", encoding="utf-8")
    (visible_dir / "README.md").write_text(
        "# Strict Caduceus SPECTRA Bundle\n\n"
        "This bundle contains Caduceus paper/repo/checkpoint context and original "
        "evaluation data sources named by the Caduceus repository. It intentionally "
        "excludes DART-Eval papers, conclusions, and result tables.\n\n"
        "Primary task: evaluate whether Caduceus generalizes on its original "
        "evaluation data using the SPECTRA protocol.\n\n"
        "Important benchmark-mode rule: because labeled train/test sequences are "
        "available, primary SPECTRA evidence should use controlled splits with a "
        "fresh downstream probe/head/baseline per split. Fixed-score binning is "
        "diagnostic only.\n\n"
        "Files:\n\n"
        "- `model_context.json`: Caduceus paper, repo, and checkpoint paths.\n"
        "- `tasks_manifest.csv`: normalized original evaluation tasks.\n"
        "- `tasks/`: train/test CSVs with `sample_id`, `sequence`, `label`, and original split.\n"
        "- `variant_effect_causal_eqtl_metadata.csv`: original long-range benchmark eQTL metadata; sequences require genome extraction.\n",
        encoding="utf-8",
    )
    summary = {
        "visible_dir": str(visible_dir),
        "task_count": len(tasks),
        "genomic_benchmark_task_count": len(gb_tasks),
        "nucleotide_transformer_task_count": len(nt_tasks),
        "lrb_eqtl_rows": int(len(lrb)),
        "strict_condition": True,
    }
    (visible_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()
    build_bundle(args.root, limit_tasks=args.limit_tasks)


if __name__ == "__main__":
    main()
