#!/usr/bin/env python3
"""Construct ProteinGym v1.3 human DMS panels joined to official AM scores."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from build_panels import load_transcripts, splice_geometry, axis_level

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
AA_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def protein_split(entry: str) -> str:
    value = int.from_bytes(hashlib.sha256(
        ("spectra-alphamissense-dms-v1|" + entry).encode()).digest()[:4], "big") % 10
    return "discovery" if value < 4 else "confirmation"


def load_dms_rows() -> tuple[pd.DataFrame, dict, Counter]:
    ref = pd.read_csv(ROOT / "model/ProteinGym/reference_files/DMS_substitutions.csv")
    ref = ref[ref.taxon.eq("Human")].copy()
    mapping = pd.read_csv(ROOT / "metadata/proteingym_human_uniprot_mapping.tsv", sep="\t")
    candidates = mapping.groupby("Entry Name")["Entry"].apply(list).to_dict()
    qc = Counter()
    rows = []
    archive = RAW / "DMS_ProteinGym_v1.3_substitutions.zip"
    with zipfile.ZipFile(archive) as zf:
        for _, meta in ref.iterrows():
            name = f"DMS_ProteinGym_substitutions/{meta.DMS_filename}"
            with zf.open(name) as handle:
                assay = pd.read_csv(handle)
            for record in assay.itertuples(index=False):
                qc["dms_rows_scanned"] += 1
                mutant = str(record.mutant)
                match = AA_RE.match(mutant)
                if not match:
                    qc["excluded_non_single_or_noncanonical_mutant"] += 1
                    continue
                ref_aa, pos_text, alt_aa = match.groups()
                pos = int(pos_text)
                if pos == 1:
                    qc["excluded_start_loss_position1"] += 1
                    continue
                target = str(meta.target_seq)
                if pos > len(target) or target[pos - 1] != ref_aa:
                    qc["excluded_proteingym_reference_mismatch"] += 1
                    continue
                rows.append({
                    "dms_id": meta.DMS_id, "entry_name": meta.UniProt_ID,
                    "protein_partition": protein_split(meta.UniProt_ID),
                    "mutant": mutant, "ref_aa": ref_aa, "aa_pos": pos, "alt_aa": alt_aa,
                    "dms_score": float(record.DMS_score),
                    "dms_score_bin": int(record.DMS_score_bin),
                    "dms_deleterious": 1 - int(record.DMS_score_bin),
                    "selection_type": meta.selection_type,
                    "coarse_selection_type": meta.coarse_selection_type,
                    "msa_neff_l": meta.MSA_Neff_L, "msa_depth_category": meta.MSA_Neff_L_category,
                    "protein_length": len(target),
                })
    df = pd.DataFrame(rows)
    qc["eligible_human_single_mutant_rows"] = len(df)
    qc["human_assays"] = df.dms_id.nunique()
    qc["human_proteins"] = df.entry_name.nunique()
    return df, candidates, qc


def score_keys(df: pd.DataFrame, candidates: dict) -> tuple[dict, Counter]:
    wanted = set()
    for entry, mutant in df[["entry_name", "mutant"]].drop_duplicates().itertuples(index=False):
        for accession in candidates[entry]:
            wanted.add((accession, mutant))
    found = defaultdict(list)
    qc = Counter()
    with gzip.open(RAW / "AlphaMissense_aa_substitutions.tsv.gz", "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for f in reader:
            if not f or f[0].startswith("#") or f[0] == "uniprot_id":
                continue
            qc["aa_score_rows_scanned"] += 1
            key = (f[0], f[1])
            if key in wanted:
                found[key].append((float(f[2]), f[3]))
    qc["wanted_accession_mutation_keys"] = len(wanted)
    qc["matched_accession_mutation_keys"] = len(found)
    return found, qc


def resolve_scores(df: pd.DataFrame, candidates: dict, found: dict, qc: Counter) -> pd.DataFrame:
    accessions, scores, classes = [], [], []
    for entry, mutant in df[["entry_name", "mutant"]].itertuples(index=False):
        options = []
        for accession in candidates[entry]:
            for score, cls in found.get((accession, mutant), []):
                options.append((accession, score, cls))
        unique = {(round(x[1], 8), x[2]) for x in options}
        if not options:
            accessions.append(None); scores.append(np.nan); classes.append(None)
            continue
        if len(unique) > 1:
            qc["ambiguous_accession_score_rows"] += 1
            accessions.append(None); scores.append(np.nan); classes.append(None)
            continue
        accessions.append("|".join(sorted({x[0] for x in options})))
        scores.append(options[0][1]); classes.append(options[0][2])
    out = df.copy()
    out["uniprot_accession"] = accessions
    out["am_pathogenicity"] = scores
    out["am_class"] = classes
    qc["dms_rows_with_am_score"] = int(out.am_pathogenicity.notna().sum())
    return out[out.am_pathogenicity.notna()].copy()


def genomic_mapping(df: pd.DataFrame, qc: Counter) -> pd.DataFrame:
    wanted = set()
    for acc_text, mutant in df[["uniprot_accession", "mutant"]].drop_duplicates().itertuples(index=False):
        for acc in acc_text.split("|"):
            wanted.add((acc, mutant))
    hits = defaultdict(list)
    transcripts = set()
    with gzip.open(RAW / "AlphaMissense_hg38.tsv.gz", "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for f in reader:
            if not f or f[0].startswith("#"):
                continue
            qc["hg38_rows_scanned_for_dms"] += 1
            key = (f[5], f[7])
            if key in wanted:
                rec = {"chrom": f[0], "pos": int(f[1]), "ref": f[2], "alt": f[3],
                       "transcript_id": f[6], "score": float(f[8])}
                hits[key].append(rec); transcripts.add(f[6])
    tx = load_transcripts(transcripts)
    geometry = defaultdict(list)
    for key, records in hits.items():
        for rec in records:
            info = tx.get(rec["transcript_id"])
            geom = splice_geometry(rec["pos"], info["exons"]) if info else None
            if geom:
                d, exon, terminal = geom
                lv, name = axis_level(d)
                geometry[key].append((lv, name, d, rec["transcript_id"], rec["chrom"], rec["pos"]))
    map_rows = []
    for (acc, mutant), values in geometry.items():
        levels = {x[0] for x in values}
        if len(levels) != 1:
            qc["excluded_dms_mutation_crossing_axis_bins"] += 1
            continue
        map_rows.append({
            "accession": acc, "mutant": mutant, "axis_level": values[0][0],
            "axis_name": values[0][1], "d_splice_bp_min": min(x[2] for x in values),
            "d_splice_bp_median": float(np.median([x[2] for x in values])),
            "genomic_encodings": len(values),
            "transcript_ids": "|".join(sorted({x[3] for x in values})),
        })
    mapping = pd.DataFrame(map_rows)
    lookup = {(r.accession, r.mutant): r for r in mapping.itertuples(index=False)}
    out_rows = []
    for r in df.itertuples(index=False):
        options = [lookup[(a, r.mutant)] for a in r.uniprot_accession.split("|")
                   if (a, r.mutant) in lookup]
        signatures = {(x.axis_level, x.d_splice_bp_min, x.d_splice_bp_median) for x in options}
        if len(signatures) != 1:
            continue
        x = options[0]
        row = r._asdict()
        row.update({"axis_level": x.axis_level, "axis_name": x.axis_name,
                    "d_splice_bp_min": x.d_splice_bp_min,
                    "d_splice_bp_median": x.d_splice_bp_median,
                    "genomic_encodings": x.genomic_encodings,
                    "transcript_ids": x.transcript_ids})
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    qc["dms_rows_with_unambiguous_genomic_axis"] = len(out)
    return out


def main() -> None:
    state = json.loads((ROOT / "controller/state.json").read_text())
    if not state.get("confirmation_accessible"):
        raise SystemExit("DMS confirmation outcomes remain sealed")
    df, candidates, qc = load_dms_rows()
    found, scan_qc = score_keys(df, candidates); qc.update(scan_qc)
    df = resolve_scores(df, candidates, found, qc)
    df = genomic_mapping(df, qc)
    blosum = substitution_matrices.load("BLOSUM62")
    df["blosum62"] = [float(blosum[a, b]) for a, b in zip(df.ref_aa, df.alt_aa)]
    df["blosum_pathogenicity"] = -df.blosum62
    df.sort_values(["protein_partition", "entry_name", "dms_id", "mutant"], inplace=True)
    df.to_parquet(OUT / "proteingym_dms_human_scored.parquet", index=False)
    df[df.protein_partition.eq("discovery")].to_parquet(
        OUT / "proteingym_dms_discovery_scored.parquet", index=False)
    df[df.protein_partition.eq("confirmation")].to_parquet(
        OUT / "proteingym_dms_confirmation_scored.parquet", index=False)
    qc["discovery_rows"] = int(df.protein_partition.eq("discovery").sum())
    qc["confirmation_rows"] = int(df.protein_partition.eq("confirmation").sum())
    qc["protein_overlap"] = len(set(df.loc[df.protein_partition.eq("discovery"), "entry_name"]) &
                                set(df.loc[df.protein_partition.eq("confirmation"), "entry_name"]))
    (ROOT / "artifacts/tables/dms_panel_build_qc.json").write_text(
        json.dumps(dict(qc), indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(qc), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
