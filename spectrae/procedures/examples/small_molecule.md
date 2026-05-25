# Example: Small-Molecule Property Prediction

Domain: small_molecule

Dataset samples are molecules represented by SMILES strings. The prediction target is a molecular property such as toxicity, solubility, or activity.

Recommended spectral property:

- Morgan fingerprint Tanimoto similarity.

Recommended graph:

- Weighted graph for richer analysis.
- Binary graph if the scientific question needs a threshold, such as Tanimoto similarity greater than or equal to 0.7.

Implementation options:

- Exact all-pairs Tanimoto for small datasets.
- Chunked fingerprint comparison for medium datasets.
- Approximate nearest neighbors over fingerprints for large datasets.

Quality checks:

- Cross-split Tanimoto similarity should decrease as spectral parameter increases.
- Report whether scaffold or chemotype leakage remains after splitting.
- Compare random split performance against low-overlap SPECTRA splits.
