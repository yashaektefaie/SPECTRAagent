# SPECTRA Value Ablation

This experiment tests whether `/spectra` adds value beyond simply prompting an
agent to evaluate performance as a function of distance from the training set.

## Conditions

1. **Broad generalizability**
   - Prompt: `prompt_broad_generalizability.md`
   - Core question: "Does the model generalize?"

2. **Explicit distance-from-train**
   - Prompt: `prompt_distance_from_train.md`
   - Core question: "How does performance vary with distance from train?"
   - This condition controls for the central SPECTRA idea being stated directly.

3. **Full `/spectra`**
   - Prompt: `prompt_spectra_protocol.md`
   - Core question: "Execute the SPECTRA audit protocol."
   - This condition tests whether the protocol, validation requirements, domain
     defaults, and report schema add value beyond a natural-language hint.

## Interpretation

If condition 3 only matches condition 2, then `/spectra` is mainly a prompt
packaging layer for this task.

If condition 3 produces more valid axes, better overlap validation, richer
performance-vs-novelty curves, and reusable audit artifacts across tasks, then
`/spectra` is functioning as infrastructure rather than just wording.
