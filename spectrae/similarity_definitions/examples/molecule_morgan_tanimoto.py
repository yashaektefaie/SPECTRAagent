#!/usr/bin/env python3
"""Create pairwise_similarity.csv from Morgan Tanimoto similarity.

If RDKit is unavailable, this script falls back to transparent SMILES character
n-gram Jaccard similarity. The fallback is useful for plumbing tests but should
not be reported as chemical Morgan/Tanimoto similarity.
"""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def char_ngrams(value, lengths=(2, 3, 4)):
    text = str(value)
    features = set()
    for length in lengths:
        if len(text) < length:
            features.add(text)
        else:
            for index in range(len(text) - length + 1):
                features.add(text[index : index + length])
    return features


def jaccard(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def fallback_pairwise(train_rows, eval_rows, args):
    train_features = [
        (row[args.train_id_col], char_ngrams(row[args.smiles_col]))
        for row in train_rows
    ]
    for eval_row in eval_rows:
        eval_features = char_ngrams(eval_row[args.smiles_col])
        for train_id, features in train_features:
            yield {
                "sample_id": eval_row[args.eval_id_col],
                "train_id": train_id,
                "similarity": jaccard(eval_features, features),
            }


def rdkit_pairwise(train_rows, eval_rows, args):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    train_fps = []
    for row in train_rows:
        mol = Chem.MolFromSmiles(row[args.smiles_col])
        if mol is None:
            raise ValueError("Could not parse train SMILES for %s" % row[args.train_id_col])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.n_bits)
        train_fps.append((row[args.train_id_col], fp))

    for eval_row in eval_rows:
        mol = Chem.MolFromSmiles(eval_row[args.smiles_col])
        if mol is None:
            raise ValueError("Could not parse eval SMILES for %s" % eval_row[args.eval_id_col])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.n_bits)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, [item[1] for item in train_fps])
        for (train_id, _), similarity in zip(train_fps, similarities):
            yield {
                "sample_id": eval_row[args.eval_id_col],
                "train_id": train_id,
                "similarity": float(similarity),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    if args.fallback:
        rows = fallback_pairwise(train_rows, eval_rows, args)
    else:
        try:
            rows = rdkit_pairwise(train_rows, eval_rows, args)
        except ImportError:
            rows = fallback_pairwise(train_rows, eval_rows, args)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
