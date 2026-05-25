# Caduceus SPECTRA Generalizability Finding

Date: 2026-05-17

## Revised Distiller Routing

This document should now be read as a checkpoint synthesis for the local cCRE
support hypothesis, not as the terminal Caduceus SPECTRA finding. The matched
follow-up showed that local cCRE support attenuated to hash-control scale, but
the same evidence contains a much larger unresolved composition-regime signal.
Under the updated Distiller policy, that larger signal should route back to the
Investigator as the new primary axis.

New route: `return_to_investigator_new_primary_axis`

New primary hypothesis: frozen Caduceus enhancer-like probes generalize poorly
across GC/CpG/length sequence-composition regimes, and this effect should be
tested after matching cCRE class, label balance, local cCRE support,
promoter-distance, chromosome/provenance, and overlap.

## Model-Paper Context

The Caduceus paper establishes a bidirectional/equivariant long-range DNA
sequence model and evaluates it on downstream genomic benchmarks. The SPECTRA
audit asks a narrower and different question: whether frozen Caduceus
representations support enhancer-like classification under controlled
regulatory-context novelty, and whether an apparent degradation signal survives
composition, provenance, and overlap controls.

This should be framed as an extension of the model paper's evaluation, not as a
replacement for it. The model paper shows that Caduceus can be useful under its
reported benchmark settings. The SPECTRA result tests whether one plausible
generalization axis is independently responsible for performance degradation in
a controlled ENCODE cCRE setting.

## Paper-Ready Finding

For ENCODE cCRE V4 1024 bp windows, the apparent degradation of frozen Caduceus
enhancer-like probes on low local cCRE support is not supported as a strong
independent regulatory-context generalization axis once cCRE class, sequence
composition, interval length, promoter-distance proxies, label balance, and
cross-split overlap are controlled. The matched middle-to-low ROC-AUC delta
(`0.016665`) is essentially the same size as the same-pool hash-control delta
(`0.017131`), and the matched AP delta is small (`0.006481`).

## Results Paragraph

In frozen Caduceus representation probes on ENCODE cCRE V4 1024 bp windows, an
initial split by within-class local cCRE support showed a monotonic decrease in
enhancer-like classification performance from high-support training to
low-support testing. The middle-to-low ROC-AUC delta was `0.0386`, while the
original hash control did not show the same pattern. A subsequent GC/CpG/length
control produced a larger degradation, with an absolute low/high ROC-AUC gap of
`0.1621`, motivating matched split construction. After matching cCRE class,
label balance, GC/CpG composition, interval length, promoter-distance proxy, and
eliminating cross-split cCRE/window overlap, the local-support middle-to-low
ROC-AUC delta fell to `0.0167`, comparable to a same-pool hash-control delta of
`0.0171`. Thus, under frozen Caduceus probe evidence, local cCRE support is not
validated as a strong independent novelty axis for this task; the earlier signal
is better interpreted as composition/proxy-sensitive and weakened after
matching.

## Interpretation

The interesting scientific result is not simply that the first curve weakened.
The audit shows how an apparently biological generalization story can collapse
into a more careful claim after controls. Low local cCRE support initially
looked like a regulatory-context novelty axis. But composition and matched
controls showed that the signal was entangled with sequence composition,
promoter-distance structure, cCRE class mixture, local density, and overlap
properties.

This does not mean sequence composition is "just an artifact." GC/CpG structure
and related composition features are themselves part of regulatory sequence
biology. The controlled claim is more precise: in this experiment, SPECTRA did
not isolate local cCRE support as an independent regulatory-context axis beyond
those sequence-composition and dataset-construction variables. Therefore the
result weakens a clean regulatory-context generalization claim for frozen
Caduceus probes, while leaving open the possibility that composition-linked
regulatory regimes are biologically meaningful deployment axes.

## Comparison To Vanilla Agent

The vanilla agent found useful but generic split-quality concerns, including
exact/reverse-complement overlap and source-family leakage in the provided
benchmark bundle. It did not recover the working Caduceus environment, run the
target model through controlled cCRE probes, construct ENCODE cCRE matched
splits, or convert the proxy signal into a controlled attenuation result.

The SPECTRA loop therefore adds more than formatting. It turns an open
"does this generalize?" question into target-model evidence, negative findings,
controls, dataset construction, and a scoped interpretation of what can and
cannot be concluded.

## Evidence To Claim

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Initial local cCRE-support split | ROC-AUC fell from high support to low support; middle-to-low delta `0.0386` | Plausible regulatory-context novelty signal |
| Original hash control | Validation-to-test ROC-AUC delta `-0.0015` | Initial signal looked non-random before stronger controls |
| GC/CpG/length control | Absolute low/high ROC-AUC gap `0.1621` | Composition/proxy structure could explain more than local cCRE support |
| Matched regulatory split | Middle-to-low ROC-AUC delta `0.0167`; AP delta `0.0065` | Local-support effect becomes weak after matching |
| Matched same-pool hash control | Validation-to-test ROC-AUC delta `0.0171` | Residual regulatory delta is hash-control scale |
| Leakage checks | Zero cross-split 1024 bp window overlap, interval overlap, duplicate cCRE IDs, and duplicate sequence MD5s | Matched attenuation is not explained by obvious cross-split overlap |

## Claim Boundary

- Target model evidence: frozen Caduceus mean-pooled embeddings with logistic
  probes.
- Not claimed: full Caduceus fine-tuning or adapter-tuning behavior.
- Task: ENCODE SCREEN cCRE V4 GRCh38 enhancer-like versus other classification.
- Scientific unit: 1024 bp sequence windows centered on cCRE intervals.
- Matched evaluation size: deterministic bounded sampling with `n=1600` per
  evaluated matched split.
- Scope: one constructed ENCODE cCRE hypothesis-test setting, not a complete
  characterization of all Caduceus regulatory generalization.

## What Not To Claim

- Do not claim that Caduceus globally fails to learn regulatory biology.
- Do not claim that Caduceus globally generalizes biologically.
- Do not claim that local cCRE support is never a meaningful axis.
- Do not present frozen-probe results as full fine-tuning evidence.
- Do not present composition controls as biologically irrelevant; composition
  can be a real regulatory covariate.
- Do not claim the original Caduceus paper is invalid. The SPECTRA audit asks a
  different, more controlled generalization question.

## Manuscript-Level Claim

In a Caduceus case study, SPECTRA converted an initially plausible regulatory
novelty curve into a controlled attenuation finding, then surfaced a larger
candidate axis that should become the next primary hypothesis. A vanilla broad
audit identified split leakage and surrogate-model concerns, whereas SPECTRA
recovered target-model execution, built a coordinate-backed ENCODE cCRE
hypothesis-test dataset, tested composition and matched controls, and revised
the active scientific question. The local cCRE-support degradation is not
independent of composition/proxy structure under the current controls; the next
headline candidate is whether frozen Caduceus probes degrade across GC/CpG/length
composition regimes after stronger matching. This supports the process claim
that SPECTRA helps agents turn broad generalizability questions into
interpretable, evidence-bounded scientific audits and then keep pushing toward
the strongest unresolved axis.
