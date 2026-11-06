#!/usr/bin/env python3
"""Build leakage-controlled ClinVar panels under the sealed splice-axis contract.

The script may read the full raw sources, but writes confirmation labels and target
scores only to a sealed file. Human-facing diagnostics contain no confirmation
outcome summaries until the controller explicitly unlocks confirmation.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
STATE = ROOT / "controller/state.json"
CONTRACT = ROOT / "config/frozen_axis.json"
AA_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
ALLOWED_PATH = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
ALLOWED_BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard() -> dict:
    state = json.loads(STATE.read_text())
    if not state.get("axis_contract_sha256"):
        raise SystemExit("Axis contract is not sealed")
    if digest(CONTRACT) != state["axis_contract_sha256"]:
        raise SystemExit("Frozen axis contract hash mismatch")
    if not state.get("labels_accessible") or not state.get("target_outcomes_accessible"):
        raise SystemExit("Discovery outcomes are not unlocked")
    return state


def label_value(value: str) -> int | None:
    if value in ALLOWED_PATH:
        return 1
    if value in ALLOWED_BENIGN:
        return 0
    return None


def review_stars(status: str) -> int:
    s = status.lower()
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "multiple submitters" in s and "no conflicts" in s:
        return 2
    if "criteria provided" in s and "conflict" not in s:
        return 1
    return 0


def load_clinvar() -> tuple[dict, Counter]:
    path = RAW / "clinvar_variant_summary_2026-07-07.txt.gz"
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    qc = Counter()
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qc["clinvar_rows"] += 1
            if row["Assembly"] != "GRCh38":
                continue
            qc["grch38_rows"] += 1
            if row["Type"].lower() != "single nucleotide variant":
                continue
            label = label_value(row["ClinicalSignificance"])
            if label is None:
                continue
            stars = review_stars(row["ReviewStatus"])
            if stars < 1:
                qc["excluded_review_lt_1"] += 1
                continue
            chrom = row["Chromosome"]
            pos = row["PositionVCF"]
            ref = row["ReferenceAlleleVCF"].upper()
            alt = row["AlternateAlleleVCF"].upper()
            if chrom not in {str(i) for i in range(1, 23)} | {"X", "Y"}:
                qc["excluded_contig"] += 1
                continue
            if not pos.isdigit() or len(ref) != 1 or len(alt) != 1 or ref == alt:
                qc["excluded_bad_vcf_allele"] += 1
                continue
            key = ("chr" + chrom, int(pos), ref, alt)
            grouped[key].append({
                "label": label,
                "clinical_significance": row["ClinicalSignificance"],
                "review_status": row["ReviewStatus"],
                "review_stars": stars,
                "last_evaluated": row["LastEvaluated"],
                "clinvar_gene": row["GeneSymbol"],
                "variation_id": row["VariationID"],
                "allele_id": row["#AlleleID"],
                "number_submitters": row["NumberSubmitters"],
                "origin": row["OriginSimple"],
            })
    eligible = {}
    for key, rows in grouped.items():
        labels = {r["label"] for r in rows}
        genes = {r["clinvar_gene"] for r in rows}
        if len(labels) != 1 or len(genes) != 1:
            qc["excluded_clinvar_duplicate_disagreement"] += 1
            continue
        rows.sort(key=lambda r: (-r["review_stars"], r["variation_id"], r["allele_id"]))
        chosen = rows[0].copy()
        chosen["clinvar_duplicate_rows"] = len(rows)
        chosen["variation_ids"] = "|".join(sorted({r["variation_id"] for r in rows}))
        eligible[key] = chosen
    qc["eligible_unique_clinvar_alleles"] = len(eligible)
    return eligible, qc


def match_alphamissense(keys: set[tuple], qc: Counter) -> dict:
    path = RAW / "AlphaMissense_hg38.tsv.gz"
    matched: dict[tuple, list[dict]] = defaultdict(list)
    names = ["chrom", "pos", "ref", "alt", "genome", "uniprot_id",
             "transcript_id", "protein_variant", "am_pathogenicity", "am_class"]
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields or fields[0].startswith("#"):
                continue
            qc["alphamissense_rows_scanned"] += 1
            key = (fields[0], int(fields[1]), fields[2], fields[3])
            if key in keys:
                matched[key].append(dict(zip(names, fields)))
    qc["clinvar_alleles_with_am_match"] = len(matched)
    unique = {}
    for key, rows in matched.items():
        signatures = {(r["uniprot_id"], r["transcript_id"], r["protein_variant"],
                       r["am_pathogenicity"]) for r in rows}
        if len(signatures) != 1:
            qc["excluded_ambiguous_am_mapping"] += 1
            continue
        unique[key] = rows[0]
    return unique


def parse_attrs(text: str) -> dict:
    out = {}
    for item in text.rstrip(";").split("; "):
        if " " in item:
            key, value = item.split(" ", 1)
            out[key] = value.strip('"')
    return out


def load_transcripts(wanted: set[str]) -> dict:
    tx = defaultdict(lambda: {"exons": [], "gene": None, "strand": None})
    with gzip.open(RAW / "gencode.v32.annotation.gtf.gz", "rt") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "exon":
                continue
            attrs = parse_attrs(f[8])
            tid = attrs.get("transcript_id")
            if tid not in wanted:
                continue
            tx[tid]["exons"].append((int(f[3]), int(f[4])))
            tx[tid]["gene"] = attrs.get("gene_name")
            tx[tid]["strand"] = f[6]
    for value in tx.values():
        value["exons"].sort()
    return dict(tx)


def load_sequences(wanted: set[str]) -> dict[str, str]:
    seqs = {}
    current = None
    chunks = []
    with gzip.open(RAW / "gencode.v32.pc_translations.fa.gz", "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if current in wanted:
                    seqs[current] = "".join(chunks)
                parts = line[1:].rstrip().split("|")
                current = parts[1] if len(parts) > 1 else None
                chunks = []
            else:
                chunks.append(line.strip())
        if current in wanted:
            seqs[current] = "".join(chunks)
    return seqs


def splice_geometry(pos: int, exons: list[tuple[int, int]]) -> tuple[int, int, bool] | None:
    if len(exons) < 2:
        return None
    hits = []
    for i, (start, end) in enumerate(exons):
        if start <= pos <= end:
            candidates = []
            if i > 0:
                candidates.append(pos - start)
            if i < len(exons) - 1:
                candidates.append(end - pos)
            if candidates:
                hits.append((min(candidates), i + 1, i in {0, len(exons) - 1}))
    if len(hits) != 1:
        return None
    return hits[0]


def axis_level(distance: int) -> tuple[int, str]:
    if distance >= 50:
        return 0, "far"
    if distance >= 20:
        return 1, "mid"
    if distance >= 3:
        return 2, "proximal"
    return 3, "splice_region"


def split_cluster(gene: str) -> str:
    value = int.from_bytes(hashlib.sha256(
        ("spectra-alphamissense-splice-v1|" + gene).encode()).digest()[:4], "big") % 10
    return "discovery" if value < 4 else "confirmation"


def main() -> None:
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    clinvar, qc = load_clinvar()
    am = match_alphamissense(set(clinvar), qc)
    wanted = {r["transcript_id"] for r in am.values()}
    tx = load_transcripts(wanted)
    seqs = load_sequences(wanted)
    blosum = substitution_matrices.load("BLOSUM62")
    rows = []
    for key, model in am.items():
        meta = clinvar[key]
        match = AA_RE.match(model["protein_variant"])
        if not match:
            qc["excluded_bad_protein_variant"] += 1
            continue
        ref_aa, aa_pos_text, alt_aa = match.groups()
        aa_pos = int(aa_pos_text)
        if aa_pos == 1:
            qc["excluded_start_loss_position1"] += 1
            continue
        transcript = model["transcript_id"]
        if transcript not in tx:
            qc["excluded_transcript_missing_gtf"] += 1
            continue
        sequence = seqs.get(transcript)
        if not sequence or aa_pos > len(sequence):
            qc["excluded_translation_missing_or_short"] += 1
            continue
        if sequence[aa_pos - 1] != ref_aa:
            qc["excluded_reference_aa_mismatch"] += 1
            continue
        geom = splice_geometry(key[1], tx[transcript]["exons"])
        if geom is None:
            qc["excluded_no_unique_internal_splice_distance"] += 1
            continue
        distance, exon_ordinal, terminal_exon = geom
        level, level_name = axis_level(distance)
        gene = tx[transcript]["gene"] or meta["clinvar_gene"] or model["uniprot_id"]
        partition = split_cluster(gene)
        rows.append({
            "variant_key": f"{key[0]}:{key[1]}:{key[2]}:{key[3]}",
            "chrom": key[0], "pos": key[1], "ref": key[2], "alt": key[3],
            "gene": gene, "clinvar_gene": meta["clinvar_gene"],
            "uniprot_id": model["uniprot_id"], "transcript_id": transcript,
            "protein_variant": model["protein_variant"], "ref_aa": ref_aa,
            "aa_pos": aa_pos, "alt_aa": alt_aa, "protein_length": len(sequence),
            "position_fraction": aa_pos / len(sequence), "strand": tx[transcript]["strand"],
            "exon_ordinal_genomic": exon_ordinal, "terminal_exon": terminal_exon,
            "d_splice_bp": distance, "axis_level": level, "axis_name": level_name,
            "partition": partition, "review_status": meta["review_status"],
            "review_stars": meta["review_stars"], "last_evaluated": meta["last_evaluated"],
            "number_submitters": meta["number_submitters"], "origin": meta["origin"],
            "variation_ids": meta["variation_ids"],
            "blosum62": float(blosum[ref_aa, alt_aa]),
            "blosum_pathogenicity": -float(blosum[ref_aa, alt_aa]),
            "label": meta["label"], "clinical_significance": meta["clinical_significance"],
            "am_pathogenicity": float(model["am_pathogenicity"]), "am_class": model["am_class"],
        })
    df = pd.DataFrame(rows).sort_values(["partition", "gene", "variant_key"])
    qc["final_panel_rows"] = len(df)
    qc["final_panel_genes"] = df["gene"].nunique()
    qc["discovery_rows"] = int((df.partition == "discovery").sum())
    qc["discovery_genes"] = int(df.loc[df.partition == "discovery", "gene"].nunique())
    qc["confirmation_rows_sealed"] = int((df.partition == "confirmation").sum())
    qc["confirmation_genes_sealed"] = int(df.loc[df.partition == "confirmation", "gene"].nunique())
    outcome = ["label", "clinical_significance", "am_pathogenicity", "am_class"]
    prospective = df.drop(columns=outcome)
    discovery = df[df.partition == "discovery"].copy()
    confirmation = df[df.partition == "confirmation"].copy()
    prospective.to_parquet(OUT / "clinvar_all_prospective.parquet", index=False)
    discovery.to_parquet(OUT / "clinvar_discovery_scored.parquet", index=False)
    confirmation.to_parquet(OUT / "clinvar_confirmation_outcomes_sealed.parquet", index=False)
    public_qc = dict(qc)
    # Do not reveal confirmation label/score distributions before unlock.
    (ROOT / "artifacts/tables/panel_build_qc.json").write_text(
        json.dumps(public_qc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(public_qc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
