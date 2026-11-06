"""Independence / duplicate check between external NCT-CRC-HE-100K pools and the
exposed development set (CRC-VAL-HE-7K).

For every tile we compute:
  - pixel SHA256 (exact-duplicate detection, format-independent: hashes decoded
    RGB bytes so TIF-vs-PNG encoding does not matter)
  - 64-bit dHash perceptual hash (near-duplicate detection)

Reports: exact overlaps (identical pixels) and near-duplicate pairs (Hamming
distance <= threshold) between {reference, confirmation} and development. Also
reports intra-pool exact duplicates. Independence requires zero exact overlaps
with development and no near-duplicate leakage below threshold.
"""
import sys, csv, hashlib, argparse
from pathlib import Path
import numpy as np
import cv2


def pixel_sha(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest(), im


def dhash(im_rgb, hs=8):
    g = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    r = cv2.resize(g, (hs + 1, hs), interpolation=cv2.INTER_AREA)
    diff = r[:, 1:] > r[:, :-1]
    bits = np.packbits(diff.flatten())
    return int.from_bytes(bits.tobytes(), "big")


def load_manifest(p):
    return list(csv.DictReader(open(p)))


def hashes_for(rows, label):
    px = {}; dh = []
    for i, r in enumerate(rows):
        s, im = pixel_sha(r["image_abspath"])
        px.setdefault(s, []).append(r["sample_id"])
        dh.append((r["sample_id"], dhash(im)))
        if i % 1000 == 0:
            print(f"  {label}: {i}/{len(rows)}", flush=True)
    return px, dh


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--pools", nargs="+", required=True, help="name=manifest.csv ...")
    ap.add_argument("--near_thresh", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev_rows = load_manifest(args.dev)
    print(f"hashing development ({len(dev_rows)})")
    dev_px, dev_dh = hashes_for(dev_rows, "dev")
    dev_dh_arr = np.array([h for _, h in dev_dh], dtype=np.uint64)

    report = {"dev_n": len(dev_rows), "near_thresh": args.near_thresh, "pools": {}}
    for spec in args.pools:
        name, mp = spec.split("=", 1)
        rows = load_manifest(mp)
        print(f"hashing pool {name} ({len(rows)})")
        px, dh = hashes_for(rows, name)
        # exact overlaps with dev
        exact = [s for s in px if s in dev_px]
        # intra-pool exact dups
        intra = {s: ids for s, ids in px.items() if len(ids) > 1}
        # near-dup vs dev
        near_pairs = []
        min_hd_list = []
        for sid, h in dh:
            hds = np.array([bin(int(h) ^ int(d)).count("1") for d in dev_dh_arr])
            mn = int(hds.min())
            min_hd_list.append(mn)
            if mn <= args.near_thresh:
                j = int(hds.argmin())
                near_pairs.append({"pool_id": sid, "dev_id": dev_dh[j][0], "hamming": mn})
        report["pools"][name] = {
            "n": len(rows),
            "exact_overlap_with_dev": len(exact),
            "exact_overlap_examples": exact[:10],
            "intra_pool_exact_dup_groups": len(intra),
            "near_dup_pairs_below_thresh": len(near_pairs),
            "near_dup_examples": near_pairs[:10],
            "min_hamming_to_dev_median": float(np.median(min_hd_list)),
            "min_hamming_to_dev_p01": float(np.percentile(min_hd_list, 1)),
            "min_hamming_to_dev_min": int(np.min(min_hd_list)),
        }
        print(f"  {name}: exact_overlap={len(exact)} near_dup(<= {args.near_thresh})={len(near_pairs)} "
              f"min_hd_median={np.median(min_hd_list):.0f} min_hd_min={np.min(min_hd_list)}")

    import json
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
