#!/usr/bin/env python3
"""03_features.py — compute prospective features + fixed reference scores.

Per AXIS_SPEC: windowed Shannon entropy (21-aa window, clipped at termini,
min 11), protein length, distance to nearest terminus; references BLOSUM62
(score, higher=more benign) and -Grantham (higher=more benign).
Wild-type residue from AM protein_variant must match UniProt canonical seq.
Output: data/processed/panel_frozen.tsv
"""
import gzip, csv, math, re

FASTA = "data/raw/uniprot_human_proteome.fasta.gz"
IN = "data/processed/am_matched.tsv"
OUT = "data/processed/panel_frozen.tsv"

BLOSUM62_STR = """   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4"""

_aa = BLOSUM62_STR.split("\n")[0].split()
B62 = {}
for line in BLOSUM62_STR.split("\n")[1:]:
    t = line.split()
    for j, a2 in enumerate(_aa):
        B62[(t[0], a2)] = int(t[1 + j])

# Grantham (1974) distance matrix, transcribed from the published triangle
# (columns: R L P T A V G I F Y C H Q N K D E M W; rows S,R,L,P,T,A,V,G,I,F,Y,C,H,Q,N,K,D,E,M).
_GCOLS = "RLPTAVGIF YCHQNKDEMW".replace(" ", "")
_GTRI = [
    ("S", [110, 145, 74, 58, 99, 124, 56, 142, 155, 144, 112, 89, 68, 46, 121, 65, 80, 135, 177]),
    ("R", [102, 103, 71, 112, 96, 125, 97, 97, 77, 180, 29, 43, 86, 26, 96, 54, 91, 101]),
    ("L", [98, 92, 96, 32, 138, 5, 22, 36, 198, 99, 113, 153, 107, 172, 138, 15, 61]),
    ("P", [38, 27, 68, 42, 95, 114, 110, 169, 77, 76, 91, 103, 108, 93, 87, 147]),
    ("T", [58, 69, 59, 89, 103, 92, 149, 47, 42, 65, 78, 85, 65, 81, 128]),
    ("A", [64, 60, 94, 113, 112, 195, 86, 91, 111, 106, 126, 107, 84, 148]),
    ("V", [109, 29, 50, 55, 192, 84, 96, 133, 97, 152, 121, 21, 88]),
    ("G", [135, 153, 147, 159, 98, 87, 80, 127, 94, 98, 127, 184]),
    ("I", [21, 33, 198, 94, 109, 149, 102, 168, 134, 10, 61]),
    ("F", [22, 205, 100, 116, 158, 102, 177, 140, 28, 40]),
    ("Y", [194, 83, 99, 143, 85, 160, 122, 36, 37]),
    ("C", [174, 154, 139, 202, 154, 170, 196, 215]),
    ("H", [24, 68, 32, 81, 40, 87, 115]),
    ("Q", [46, 53, 61, 29, 101, 130]),
    ("N", [94, 23, 42, 142, 174]),
    ("K", [101, 56, 95, 110]),
    ("D", [45, 160, 181]),
    ("E", [126, 152]),
    ("M", [67]),
]
GRANTHAM_PAIRS = {}
for _ri, (_raa, _vals) in enumerate(_GTRI):
    for _ci in range(_ri, len(_GCOLS)):
        GRANTHAM_PAIRS[(_raa, _GCOLS[_ci])] = _vals[_ci - _ri]
assert len(GRANTHAM_PAIRS) == 190, len(GRANTHAM_PAIRS)
def grantham(a, b):
    if a == b:
        return 0
    return GRANTHAM_PAIRS.get((a, b), GRANTHAM_PAIRS.get((b, a)))

# load proteome
seqs = {}
name = None
buf = []
with gzip.open(FASTA, "rt") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name = line[1:].split("|")[1]
            buf = []
        else:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
print(f"proteome_seqs={len(seqs)}")

AA = "ACDEFGHIKLMNPQRSTVWY"
LOG2 = math.log(2)
def entropy(win):
    n = len(win)
    h = 0.0
    for a in set(win):
        p = win.count(a) / n
        h -= p * math.log(p) / LOG2
    return h

PV = re.compile(r"^([A-Z])(\d+)([A-Z])$")
WIN = 21
HALF = WIN // 2

n_in = n_noseq = n_nomatch = n_shortwin = n_badpv = n_ok = 0
with open(IN) as f, open(OUT, "w", newline="") as o:
    rd = csv.DictReader(f, delimiter="\t")
    w = csv.writer(o, delimiter="\t")
    w.writerow(rd.fieldnames + ["wt_aa", "alt_aa", "aa_pos", "protein_length",
                                "term_dist", "window_size", "entropy",
                                "blosum62", "neg_grantham"])
    for row in rd:
        n_in += 1
        seq = seqs.get(row["uniprot_id"])
        if seq is None:
            n_noseq += 1
            continue
        m = PV.match(row["protein_variant"])
        if not m:
            n_badpv += 1
            continue
        wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
            n_nomatch += 1
            continue
        lo, hi = max(0, pos - 1 - HALF), min(len(seq), pos + HALF)
        win = seq[lo:hi]
        if len(win) < 11 or any(a not in AA for a in win):
            n_shortwin += 1
            continue
        g = grantham(wt, alt)
        if g is None or (wt, alt) not in B62:
            n_badpv += 1
            continue
        n_ok += 1
        w.writerow(list(row.values()) + [wt, alt, pos, len(seq),
                                         min(pos - 1, len(seq) - pos), len(win),
                                         f"{entropy(win):.6f}", B62[(wt, alt)], -g])

print(f"in={n_in} ok={n_ok} no_seq={n_noseq} wt_mismatch={n_nomatch} "
      f"short_window_or_nonstandard_aa={n_shortwin} bad_pv_or_pair={n_badpv}")
