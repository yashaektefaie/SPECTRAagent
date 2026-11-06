#!/usr/bin/env python3
"""Rerun a few ESMFold2 examples and save best-by-pTM CA overlays."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.SVDSuperimposer import SVDSuperimposer
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_reviewer_proofing_20260602")
EX = ROOT / "example_structures"
BASE_SCRIPT_DIR = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602/scripts")
sys.path.insert(0, str(BASE_SCRIPT_DIR))
import evaluate_esmfold2_scaling as base  # noqa: E402
import evaluate_esmfold2_scaling_multisample as multi  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_scalar(output: dict, key: str, sample_idx: int) -> float:
    value = output[key].detach().float().cpu()
    if value.ndim == 0:
        return float(value)
    if value.shape[0] > sample_idx:
        return float(value[sample_idx].mean())
    return float(value.mean())


def aligned_ca(query: str, pred_ca: np.ndarray, ref_seq: str, ref_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs, _ident, _cov = base.alignment_pairs(query, ref_seq)
    valid = [(q, r) for q, r in pairs if q < len(pred_ca) and r < len(ref_coords) and np.isfinite(pred_ca[q]).all()]
    qidx = np.asarray([q for q, _r in valid], dtype=int)
    ridx = np.asarray([r for _q, r in valid], dtype=int)
    return pred_ca[qidx].astype(np.float64), ref_coords[ridx].astype(np.float64)


def superpose(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    sup = SVDSuperimposer()
    sup.set(ref, pred)
    sup.run()
    return sup.get_transformed()


def write_ca_pdb(path: Path, ref: np.ndarray, pred_fit: np.ndarray, target: str) -> None:
    lines = [f"REMARK CA-only reference/prediction overlay for {target}"]
    atom_id = 1
    for i, xyz in enumerate(ref, start=1):
        x, y, z = xyz
        lines.append(f"ATOM  {atom_id:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
        atom_id += 1
    lines.append("TER")
    for i, xyz in enumerate(pred_fit, start=1):
        x, y, z = xyz
        lines.append(f"ATOM  {atom_id:5d}  CA  ALA B{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
        atom_id += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overlay(path: Path, ref: np.ndarray, pred_fit: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="#1f77b4", lw=1.5, label="experimental")
    ax.plot(pred_fit[:, 0], pred_fit[:, 1], pred_fit[:, 2], color="#d62728", lw=1.5, label="ESMFold2")
    both = np.vstack([ref, pred_fit])
    center = both.mean(axis=0)
    span = float(np.max(np.ptp(both, axis=0)))
    if span <= 0 or not np.isfinite(span):
        span = 1.0
    for setter, c in [(ax.set_xlim, center[0]), (ax.set_ylim, center[1]), (ax.set_zlim, center[2])]:
        setter(c - span / 2, c + span / 2)
    ax.set_axis_off()
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    base.configure_torch()
    seed = 0
    seed_everything(seed)
    examples = pd.read_csv(EX / "example_structure_targets.csv").head(3)
    out_dir = EX / "predicted_best_by_ptm"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2", local_files_only=True).cuda().eval()
    for _, rec in examples.iterrows():
        target = str(rec["target_id"])
        pdb_out = out_dir / f"{target}.ca_overlay.pdb"
        png_out = out_dir / f"{target}.ca_overlay.png"
        t0 = time.time()
        seed_everything(seed)
        with torch.inference_mode(), torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False
        ):
            output = model.infer_protein(
                str(rec["sequence"]),
                num_loops=10,
                num_sampling_steps=68,
                num_diffusion_samples=5,
            )
        pred_samples = multi.pred_ca_samples_from_output(output)
        ref_chain = str(rec.get("actual_ref_chain", rec.get("ref_chain", "")))
        if not ref_chain or ref_chain == "nan":
            ref_chain = str(rec.get("ref_chain", ""))
        ref_seq, ref_coords = base.ref_ca(Path(str(rec["pdb_path"])), ref_chain)
        scored = []
        for sample_idx, pred_ca in enumerate(pred_samples):
            metrics = base.score_prediction(str(rec["sequence"]), pred_ca, ref_seq, ref_coords)
            metrics.update(
                {
                    "sample_index": sample_idx,
                    "ptm": sample_scalar(output, "ptm", sample_idx),
                    "plddt_mean": sample_scalar(output, "plddt", sample_idx),
                    "plddt_ca_mean": sample_scalar(output, "plddt_ca", sample_idx),
                }
            )
            scored.append(metrics)
        best = max(scored, key=lambda r: r["ptm"])
        best_idx = int(best["sample_index"])
        pred, ref = aligned_ca(str(rec["sequence"]), pred_samples[best_idx], ref_seq, ref_coords)
        pred_fit = superpose(pred, ref)
        write_ca_pdb(pdb_out, ref, pred_fit, target)
        title = f"{target}: CA-lDDT {best['ca_lddt']:.3f}, pTM {best['ptm']:.3f}"
        plot_overlay(png_out, ref, pred_fit, title)
        rows.append(
            {
                "target_id": target,
                "best_by_ptm_sample_index": best_idx,
                "ptm": best["ptm"],
                "plddt_mean": best["plddt_mean"],
                "ca_lddt": best["ca_lddt"],
                "gdt_ts": best["gdt_ts"],
                "ca_rmsd": best["ca_rmsd"],
                "aligned_ca": best["aligned_ca"],
                "wall_sec": round(time.time() - t0, 3),
                "ca_overlay_pdb": str(pdb_out),
                "ca_overlay_png": str(png_out),
            }
        )
        pd.DataFrame(rows).to_csv(EX / "example_prediction_overlay_summary.csv", index=False)
        print(json.dumps(rows[-1]), flush=True)
        del output
        torch.cuda.empty_cache()
    (EX / "example_prediction_config.json").write_text(
        json.dumps(
            {
                "model": "biohub/ESMFold2",
                "num_loops": 10,
                "num_sampling_steps": 68,
                "num_diffusion_samples": 5,
                "seed": seed,
                "selection": "best_by_ptm",
                "output": "CA-only superposed reference/prediction overlays",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
