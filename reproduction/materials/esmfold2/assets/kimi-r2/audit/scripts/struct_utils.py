#!/usr/bin/env python
"""Shared structure utilities: CA extraction from mmCIF and CA-only PDB writing."""
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M',
    'SEC':'U','PYL':'O',
}

def extract_ca(cif_path, auth_chain=None, entity_id=None):
    """Return (seq, coords, resids) for model 1 CA atoms.
    Select by auth_chain (auth_asym_id) or entity_id (label_entity_id)."""
    d = MMCIF2Dict(cif_path)
    def get(key):
        for k in (f"_atom_site.{key}", key):
            if k in d:
                return d[k]
        raise KeyError(key)
    group = get("group_PDB")
    atom = get("label_atom_id")
    comp = get("label_comp_id")
    auth_asym = get("auth_asym_id")
    label_ent = get("label_entity_id")
    auth_seq = get("auth_seq_id")
    alt = get("label_alt_id")
    model = get("pdbx_PDB_model_num")
    xs = get("Cartn_x"); ys = get("Cartn_y"); zs = get("Cartn_z")
    seq, coords, resids = [], [], []
    seen = set()
    n = len(group)
    for i in range(n):
        if model[i] != "1" or group[i] != "ATOM" or atom[i] != "CA":
            continue
        if auth_chain is not None and auth_asym[i] != auth_chain:
            continue
        if entity_id is not None and label_ent[i] != str(entity_id):
            continue
        if alt[i] not in (".", "?", "A"):
            continue
        r = (auth_asym[i], auth_seq[i])
        if r in seen:
            continue
        seen.add(r)
        aa = THREE_TO_ONE.get(comp[i].upper(), "X")
        seq.append(aa)
        coords.append((float(xs[i]), float(ys[i]), float(zs[i])))
        resids.append(auth_seq[i])
    return "".join(seq), coords, resids

def ca_to_pdb(seq, coords, chain="A"):
    lines = []
    for i, (aa, (x, y, z)) in enumerate(zip(seq, coords), start=1):
        rname = {v: k for k, v in THREE_TO_ONE.items()}.get(aa, "UNK")
        if rname == "UNK" and aa in THREE_TO_ONE.values():
            rname = "UNK"
        lines.append(
            f"ATOM  {i:5d}  CA  {rname:>3s} {chain}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C")
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"

def selector_from_seqres_id(seqres_id):
    """Map a pdb_seqres-style id like 1abc_1 or 10af_A to (auth_chain, entity_id)."""
    suffix = seqres_id.split("_")[1]
    if suffix.isdigit():
        return None, suffix
    return suffix, None
