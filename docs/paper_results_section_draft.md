# Results Section Draft

## 10 Results

We evaluated SPECTRA in two complementary ways. First, we ran deterministic
audits on benchmark capsules where model predictions and train-evaluation
relationships could be computed directly. Second, we compared agent behavior
with and without `/spectra` on anonymous model-evaluation tasks where agents
were given training data and held-out predictions but not paper context, split
labels, or precomputed train-test similarity.

### 10.1 Spectral Curves Reveal Novelty-Dependent Failure Across Domains

The deterministic benchmark suite now contains executable capsules in three
scientific settings: molecular property prediction, nucleotide fitness
prediction, and perturbation-response modeling. Each capsule starts from a fixed
trained or fitted model and evaluates prediction error as a function of a
measured train-set similarity axis. These experiments test the central SPECTRA
claim that named splits can be converted into reviewable performance-over-
novelty curves once the scientific unit and similarity definition are explicit.

The clearest molecular result is the BOOM density audit. We trained a
lightweight random-forest regressor on Morgan fingerprints for the BOOM 10k
density task and evaluated performance on the BOOM density OOD split. The model
achieved ID RMSE 0.0443, while full OOD RMSE was 0.2296. Within the OOD set,
RMSE increased as maximum train-set Morgan Tanimoto similarity decreased: the
lowest-overlap OOD subset had mean max Tanimoto 0.390 and RMSE 0.2534. The
negative-RMSE AUSPC was -0.2410. This converts the split-level BOOM OOD finding
into a spectral chemical-novelty curve.

For sequence fitness, we reran three NABench assays using a position-aware ridge
baseline and contiguous mutational-region holdouts. The revised runner now
prefers leakage-free supported axes over post-hoc label-based axes when curve
support is comparable. Two assays produced prospective SPECTRA curves. In
Gregory 2018 mRNA, mutation-position support was selected; contiguous RMSE was
1.1582 and the lowest-overlap subset rose to RMSE 1.5723. In Martin 2018 MYC
enhancer, mutation-centered local window identity was selected; contiguous RMSE
was 0.0763 and the lowest-overlap subset rose to RMSE 0.0899. The Pitt 2010
ribozyme assay did not produce a supported prospective axis in the tested
candidate set; its strongest curve used a post-hoc position-fitness composite
axis and should be treated as diagnostic rather than prospective.

For perturbation biology, we ran a local PerturBench development capsule using
the bundled `devel.h5ad`. Train units were K562 single-perturbation expression
response profiles, evaluation units were two-gene combination profiles, and the
baseline predicted a combination response by summing available single-
perturbation responses. The prospective similarity axis was component-support
similarity, the fraction of combination components observed as single
perturbations in training. Mean profile RMSE decreased monotonically with
component support: 0.0978 when no components were supported, 0.0740 when one of
two components was supported, and 0.0199 when both components were supported.
The profile-level negative-RMSE AUSPC was -0.0919.

| Domain | Task | Model | Prospective axis | Reference performance | High-novelty performance | AUSPC | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Molecules | BOOM 10k density | RF Morgan fingerprints | max Morgan Tanimoto | 0.0443 ID RMSE; 0.2296 OOD RMSE | 0.2534 RMSE at mean max Tanimoto 0.390 | -0.2410 | prospective |
| Sequence fitness | NABench Gregory 2018 mRNA | position-aware ridge | mutation-position support | 1.1582 contiguous RMSE | 1.5723 lowest-overlap RMSE | -1.2536 | prospective |
| Sequence fitness | NABench Martin 2018 MYC enhancer | position-aware ridge | mutation-centered window identity | 0.0763 contiguous RMSE | 0.0899 lowest-overlap RMSE | -0.0936 | prospective |
| Sequence fitness | NABench Pitt 2010 ribozyme | position-aware ridge | none selected prospectively | 0.0137 contiguous RMSE | 0.0165 post-hoc lowest-overlap RMSE | -0.0214 | diagnostic only |
| Perturbation biology | PerturBench devel combination prediction | additive single-perturbation baseline | component support | 0.0199 RMSE with both components supported | 0.0978 RMSE with no components supported | -0.0919 | prospective |

These results are sufficient for a preliminary cross-domain empirical claim:
SPECTRA can turn fixed model predictions into leakage-aware spectral audits in
molecules, sequence fitness, and perturbation biology. The result set does not
yet support a claim over all registries or all AI-for-science domains.

### 10.2 Failed Axes Are Audit Findings

The NABench runs show why SPECTRA should be described as an iterative audit loop
rather than a single metric. In the earlier NABench pass, the runner used a
narrower candidate list and selected axes primarily by observed effect size. That
caused post-hoc fitness-support axes to dominate even when leakage-free axes
were available. The revised runner adds a prospective local-window sequence
axis, marks label-using axes as post-hoc, and ranks leakage-free supported axes
ahead of post-hoc axes.

This changed the interpretation. Gregory is now explained by a leakage-free
mutation-position support axis. Martin is now explained by a leakage-free local
sequence-window axis. Pitt remains unresolved prospectively under the tested
axes; the diagnostic post-hoc curve suggests the model error is related to
fitness support, but that axis cannot be used for prospective split design. This
is a valid SPECTRA outcome: when a biologically plausible axis fails or is
non-evaluable, the audit records that failure and motivates the next similarity
hypothesis.

### 10.3 `/spectra` Improves Agent-Executed Audits

We then ran a fresh blinded agent-ablation experiment across the three
executable domains. For each domain, one vanilla agent received only visible
train data, held-out predictions, and a broad instruction to evaluate
generalization. A separate `/spectra` agent received the same visible files plus
the SPECTRA audit loop and permission to use the local SPECTRA package, docs,
and registries. Agents were not given paper context, repository context,
grading-only files, split labels, precomputed train-test similarities, or
deterministic SPECTRA summaries.

Reports were scored on a fixed 12-dimension rubric covering scientific unit,
aggregate error, axis discovery, pairwise similarity, overlap validation,
performance curves, AUSPC or equivalent summaries, failed-axis reporting,
adaptive iteration, leakage classification, domain interpretation, and artifact
quality. Each dimension was scored from 0 to 2.

| Domain | Vanilla | `/spectra` | Delta | Main behavioral difference |
| --- | ---: | ---: | ---: | --- |
| Molecules | 17 / 24 | 24 / 24 | +7 | Both found novelty-dependent error; `/spectra` added pairwise artifacts, overlap validation, leaky-axis classification, and audit cards. |
| Sequence fitness | 17 / 24 | 23 / 24 | +6 | Both found constant-prediction failure; `/spectra` tested multiple axes and reported non-explanatory/saturated axes with AUSPC artifacts. |
| Perturbation biology | 18 / 24 | 24 / 24 | +6 | Both found component-support failure; `/spectra` formalized it as a leakage-aware spectral audit with AUSPC and failed-axis reporting. |

Because the first vanilla prompt still requested fields such as
similarity/novelty analyses and performance by novelty, we also reran the
vanilla side with a strict-naive prompt that only asked agents to evaluate
whether the model generalizes. Under this stricter condition, vanilla agents
still found the main failure in all three domains, and the `/spectra` advantage
shrank:

| Domain | Strict-naive vanilla | `/spectra` | Delta |
| --- | ---: | ---: | ---: |
| Molecules | 21 / 24 | 24 / 24 | +3 |
| Sequence fitness | 22 / 24 | 23 / 24 | +1 |
| Perturbation biology | 22 / 24 | 24 / 24 | +2 |

This result is not that vanilla agents failed. They discovered the dominant
generalization failure in all three executable domains, even without being asked
to search for similarity or novelty. The supported claim is that `/spectra` made
the audits more standardized and reproducible. In particular, `/spectra`
consistently produced pairwise similarity artifacts, validated overlap, computed
spectral summaries, classified leakage risk, reported failed or non-evaluable
axes, and wrote reusable audit cards.

The sequence-fitness result is especially important for framing. The `/spectra`
agent did not force a monotonic curve. It selected a scientifically direct
mutation-position axis, validated that train-test overlap decreased, and then
reported that error did not monotonically worsen with measured novelty. It
therefore concluded that the dominant failure was global constant prediction
rather than a clean novelty-specific degradation under the tested prospective
axes.

### 10.4 Data-Access Outcomes for Planned Capsules

DART-Eval was inspected but not executed. The local repository contains code,
while the processed task files and genome references are stored on Synapse. A
runnable regulatory-DNA capsule therefore requires authenticated Synapse data
access and local genome-reference setup. We do not count DART-Eval as an
executed result in this paper draft.

PerturBench was initially treated as blocked because the full benchmark uses
external processed h5ad assets. A second pass found that the repository bundles a
small `devel.h5ad`, which was sufficient for a local perturbation-combination
capsule. That result is useful cross-domain evidence, but it should be described
as a development capsule, not as a reproduction of the full PerturBench model
suite.

### 10.5 Limitations of Current Experiments

The current result set supports a broad but bounded empirical claim. SPECTRA was
executed in three domains, but each capsule uses a lightweight baseline and a
small number of tasks. The strict-naive ablation further shows that these
capsules are too schema-obvious to support a strong agent-discovery claim:
SMILES strings, mutation labels, and perturbation-combination names made the
natural axes easy for capable vanilla agents to infer. BOOM is a strong
prospective molecular result. NABench provides two prospective sequence-fitness
curves and one diagnostic-only failure case. PerturBench provides a local
perturbation-biology capsule, but not the full benchmark suite. DART-Eval
remains data-blocked.

The empirical claim should therefore be stated conservatively: SPECTRA converts
fixed model predictions and agent-defined train-test relationships into
standardized spectral generalization audits across multiple scientific domains,
while `/spectra` improves the completeness, leakage disclosure, and
reproducibility of agent-executed audits.
