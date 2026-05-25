# Example: Single-Cell Perturbation Response

Domain: single_cell_perturbation

Dataset samples are perturbation conditions, genes, cells, or condition-cell pairs. The unit of analysis must be declared before splitting.

Recommended spectral property:

- Gene pathway membership, perturbation target similarity, expression response similarity, or learned perturbation embedding similarity.

Recommended graph:

- Binary graph for shared pathways or shared target families.
- Weighted graph for expression or embedding similarity.

Implementation options:

- Exact similarity for curated perturbation sets.
- Sparse pathway membership joins for binary properties.
- Approximate nearest neighbors for embedding-based similarity.

Quality checks:

- Verify that unseen perturbations in the test set are less similar to training perturbations at higher spectral parameters.
- Report whether cell-type leakage is controlled separately from perturbation similarity.
- Distinguish biological novelty from batch or assay artifacts.
