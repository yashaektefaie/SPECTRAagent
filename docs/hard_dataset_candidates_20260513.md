# Hard Dataset Candidates for SPECTRA Agent Ablations

Date: 2026-05-13

The BOOM, NABench, and local PerturBench capsules were too easy as agent
discovery tests. In each case, the natural generalization axis was almost
spelled out by the visible schema: SMILES for molecules, mutation positions for
sequence fitness, and perturbation components for combination prediction.
Those tasks are useful SPECTRA sanity checks, but they are weak tests of
whether `/spectra` helps an agent discover scientifically meaningful axes.

This document defines the replacement search target: datasets where the useful
generalization axis is latent, domain-specific, or requires external scientific
structure.

## Admission Criteria

A hard SPECTRA agent-ablation dataset should satisfy most of the following:

1. The agent-visible bundle contains only the trained model artifacts, training
   data, held-out predictions, and ordinary sample metadata.
2. The correct or most useful audit axis is not directly named by a simple
   column such as `smiles`, `mutation_position`, or `geneA_geneB_support`.
3. A strong audit requires at least one external structure source, such as a
   motif database, gene ontology, pathway graph, cell ontology, chemical target
   annotation, scanner/acquisition metadata, morphology embedding, material
   composition graph, or simulation parameter.
4. The benchmark is executable without impossible access barriers. Credentialed
   access is acceptable only for a later-stage result, not for the next fast
   ablation.
5. There is a clear blinded-agent task: "Evaluate whether this model
   generalizes" with no prompt mention of novelty, similarity, distance,
   SPECTRA, or testing.

## Priority Shortlist

| Priority | Candidate | Domain | Why it is a hard SPECTRA test | Feasibility | Recommended SPECTRA axes |
| --- | --- | --- | --- | --- | --- |
| 1 | Open Problems single-cell perturbation prediction / NeurIPS 2023 PBMC DGE | Single-cell drug perturbation | The obvious columns are cell type and compound, but the useful axes require chemical structure, drug target/pathway, donor/cell-state context, and baseline multiome priors. Vanilla agents may report aggregate error by cell type/drug; `/spectra` should propose chemical, pathway, and cell-state similarity. | Medium. Public task page reports a 174.68 MiB DGE dataset and public method metadata. The raw benchmark includes SMILES, so the hard blinded capsule should expose only compound names/LINCS IDs and require agents to decide whether external chemical structure should be retrieved. Need fetch/verify prediction availability. | Morgan or ChemBERTa drug similarity, LINCS signature similarity, target/pathway overlap, cell-type ontology distance, donor baseline expression/accessibility similarity. |
| 2 | DART-Eval regulatory DNA tasks | Regulatory genomics | Sequence strings alone do not tell an agent which TF motifs, motif families, cell-type programs, or variant-effect contexts matter. `/spectra` should use motif libraries and regulatory annotations; vanilla agents may stop at GC/k-mer/chromosome checks. | Medium to high friction. Code is public; processed data and genome references are on Synapse. A synthetic HOCOMOCO-derived footprinting capsule is feasible without full Synapse data. | HOCOMOCO/JASPAR motif-family similarity, motif-disruption score, cell-type TF program similarity, chromosome/genomic distance, GC/k-mer controls, conservation. |
| 3 | PerturBench covariate-transfer datasets, especially Srivatsan20, Jiang24, McFalineFigueroa23 | Single-cell perturbation biology | The prior local PerturBench demo used component support, which is too obvious. The hard version should use unseen perturbations or biological states where the useful axis is gene/pathway/drug/cytokine/cell-line relatedness, not exact component overlap. | Medium. Processed PerturBench h5ad files are on Hugging Face; Norman19 is about 395 MB compressed. Larger datasets vary. Code and split logic are public. | GO/Reactome/KEGG overlap, STRING/PPI distance, perturbation-response signature similarity, chemical target/pathway similarity, cell-line/treatment/cytokine state similarity. |
| 4 | GeSS, especially QMOF fidelity shift and Track signal/pileup shift | Materials / particle physics / geometric ML | The useful axis is a physical or data-generating process parameter: DFT fidelity, composition/structure similarity, signal curvature, pileup/noise level. These are not generic tabular columns and require scientific interpretation. | Medium. GeSS code is public; QMOF processed data is about 138 MB on Zenodo. Track data is much larger, but Par-Label signal-shift archives are around 330-390 MB. | Material composition/structure similarity, DFT-fidelity gap, local atomic environment similarity, event point-count/pileup similarity, signal-track curvature/radius similarity. |
| 5 | COOS-7 microscopy out-of-sample cells | Bioimage generalization | The shift is experimental replication over time and instruments. If test-set names are hidden, a vanilla agent may only report aggregate image accuracy; `/spectra` can use acquisition/style/morphology embeddings and instrument/time metadata. | Medium to high. Public Zenodo data totals roughly 4.3 GB across train and four test hdf5 files. Small sampled capsules are feasible. | Image embedding similarity, illumination/intensity histogram similarity, morphology profile similarity, acquisition-time/instrument domain, batch/style distance. |
| 6 | WILDS RxRx1 or Camelyon17 | Cellular imaging / clinical pathology | These are established distribution-shift benchmarks. They are useful if metadata labels are hidden and agents must infer batch/site/style axes from images or embeddings. | Medium to high. WILDS gives loaders and prediction evaluation; downloads are 7 GB for RxRx1 and 10 GB for Camelyon17. | Batch/plate/site similarity, image embedding/style distance, scanner/staining statistics, hospital/site domain, label-prevalence controls. |
| 7 | BBBC021 / Cell Painting MoA prediction | Cell painting / chemical biology | The real axis is not just compound identity; it is chemical structure, concentration, mechanism-of-action, and phenotypic profile similarity. This is a strong non-obvious axis test if image features or profiles are used. | High if using raw images. The Hugging Face mirror reports 45.7 GB. Search for smaller precomputed morphology profiles before using it. | Compound fingerprint similarity, MoA hierarchy similarity, concentration distance, morphology profile similarity, plate/batch similarity. |
| 8 | TableShift public healthcare datasets, e.g. diabetes readmission or NHANES lead | Clinical/tabular distribution shift | Useful as a low-friction tabular control, but weaker scientifically than the above unless we construct site/time/population axes that are not visible as named split labels. | Low to medium. TableShift supports public datasets and standard ID/OOD splits. | Patient trajectory similarity, EHR concept similarity, demographic/population shift, site/time proxies, phenotype-definition shift. |

## Execution Order

The next ablation should use three hard capsules:

1. **Open Problems PBMC drug perturbation**: best balance of scientific depth,
   moderate data size, and non-obvious axes. This is the strongest replacement
   for the too-obvious PerturBench component-support capsule.
2. **DART-Eval motif-footprinting derivative**: best test of registry value,
   because motif-family similarity is exactly the kind of external scientific
   structure a vanilla agent may not operationalize from raw sequences.
3. **GeSS QMOF fidelity or Track signal shift**: expands SPECTRA beyond
   biology and tests physical/geometric similarity axes.

COOS-7 is the best fourth capsule if we want an imaging result. It is less
urgent than the first three because the data footprint is larger.

## Local Asset Status

Repos cloned for hard-capsule feasibility checks:

- `/ewsc/yektefai/spectra_hard_datasets_20260513/repos/task_perturbation_prediction`
- `/ewsc/yektefai/spectra_hard_datasets_20260513/repos/DART-Eval`
- `/ewsc/yektefai/spectra_hard_datasets_20260513/repos/GeSS`
- `/ewsc/yektefai/spectra_hard_datasets_20260513/repos/wilds`

Immediate feasibility notes:

- Open Problems PBMC: repo documents `de_train.h5ad`, `de_test.h5ad`,
  `id_map.csv`, `prediction.h5ad`, and model-output files. The source AnnData
  includes `cell_type`, `sm_name`, `sm_lincs_id`, and `SMILES`; the hard
  capsule should withhold `SMILES` from the vanilla visible bundle.
- DART-Eval: repo is local, but real processed task `data.h5` files and genome
  references are still Synapse-hosted. The low-friction path is a
  HOCOMOCO-derived footprinting capsule that uses the DART-Eval task structure
  without requiring full Synapse data.
- GeSS: repo is local. Zenodo lists `QMOF_processed.zip` at about 138 MB,
  `Bio_processed.zip` at about 334 MB, and Track archives from hundreds of MB
  to multi-GB. QMOF is the first target.
- WILDS: repo is local. `rxrx1` is about 7 GB and `camelyon17` about 10 GB, so
  these are viable later but not first.

## Blinded Prompt Constraint

For all hard capsules, the vanilla condition must receive only:

> You are given files for a trained scientific prediction model. Evaluate
> whether the model generalizes. Use only the listed files. Write a JSON report.

The vanilla prompt must not mention:

- SPECTRA
- similarity
- novelty
- distance from training data
- overlap
- spectral curves
- paper reproduction
- benchmarking or testing the agent

The `/spectra` condition receives the same visible files plus the SPECTRA MCP
skill and registries.

## Scoring Changes

The scoring rubric should add a "latent-axis discovery" category:

- Did the agent identify that a surface schema split was insufficient?
- Did it retrieve or construct an external scientific similarity source?
- Did it distinguish schema-obvious axes from biologically/physically meaningful axes?
- Did it run at least one failed axis and use that failure to choose a better axis?
- Did it produce a spectral curve using the externally grounded axis?

This is the test that the previous BOOM/NABench/PerturBench capsules did not
actually perform.

## Paperclip Search Records

Relevant Paperclip result sets from this pass:

- `s_7c23c5f6`: regulatory DNA, DART-Eval, motif and cell-type regulatory tasks.
- `s_687593b2`: perturbation-response benchmarks, pathway/ontology/gene-network generalization.
- `s_5d9ed79e`: medical imaging scanner/site/domain shift.
- `s_d8647d8a`: WILDS, GeSS, COOS-7, scientific distribution shift.
- `s_3ef237b6`: GeSS, QMOF/materials, DrugOOD, force-field OOD.
- `s_5d1c0d9d`: eICU/MIMIC multicenter clinical generalization.
- `s_37b38831`: COOS-7 and microscopy out-of-sample generalization.
- `s_80b3294e`: BBBC021, Cell Painting, morphology profiling, leave-one-compound/MoA evaluation.
