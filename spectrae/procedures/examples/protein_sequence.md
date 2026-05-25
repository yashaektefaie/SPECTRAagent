# Example: Protein Sequence Prediction

Domain: protein_sequence

Dataset samples are protein sequences. The prediction target may be function, stability, expression, or mutational fitness.

Recommended spectral property:

- Sequence identity, edit distance, alignment score, or embedding similarity, depending on the scientific claim.

Recommended graph:

- Binary graph when there is a meaningful identity threshold.
- Weighted graph when gradual similarity decay matters.

Implementation options:

- Exact pairwise alignment for small datasets.
- k-mer indexes or approximate methods for larger datasets.
- Embedding nearest neighbors when alignment is too expensive and embeddings are validated for the task.

Quality checks:

- Cross-split sequence similarity should decrease across spectral parameters.
- Report homolog leakage risk.
- Explain whether the spectral property captures the expected failure mode.
