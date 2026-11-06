# Analysis records for agents

Start with `index.json`. It lists 28 investigations and links to 143 analysis records. Paths resolve relative to this directory. There is no website renderer or hosted service in this change.

Every analysis has five sections:

1. `dataset_split`: source identities, retained sample/split tables, input inventory and public data sources.
2. `axis_levels`: measured property and the code assigning levels.
3. `model_evaluation`: individual measurements and what is known about training or fine-tuning.
4. `curve`: aggregation and uncertainty calculations.
5. `insight`: comparisons, controls and confirmation procedures.

`reconstruction_notes` identifies missing inputs, original design details that cannot be recovered, and specific numerical discrepancies. These records do not assign scientific verdicts or grades. All 143 entries remain represented, including candidates with no retained measurements.

## Inputs and code

- `sources.json` lists public upstream downloads and repositories checked without credentials. Released STATE score tables use pinned repository revisions. A current annotation release is not necessarily the original snapshot.
- `samples/<model>/index.json` describes compressed JSON tables of retained sample identities and recorded split/level assignments. Each table identifies its original source file and hash. These are identity/assignment columns, not replacement predictions. Run the linked analysis code to apply its exact inclusion rules; a source table can contain a larger pool than a particular axis uses.
- `inputs/<model>.json` records original input filenames, sizes and hashes. `download_source: null` means no exact anonymous download was established. Do not substitute an upstream dataset for a missing original sample selection, intermediate feature table or prediction file. No private bucket links, account-specific paths or login instructions are supplied.
- `materials/` contains the retained supporting scripts, requirements and code-only notebook exports. Python exports preserve notebook cell order and identify original cell numbers. Use the notebook environment for cells requiring IPython features. The notebooks have no saved outputs or execution claims; the earlier executed records remain separate historical artifacts. Published notebooks are identified by SHA-256 in their analysis record.
- `code/<model>.json` identifies original supporting scripts, hashes and published hashes. Where necessary, account-specific home-directory prefixes were replaced by `/path/to/audit-home`. Trailing whitespace was normalized only where the parsed Python syntax tree remained identical. These scripts require path configuration and their specified inputs. Their presence does not establish that every historical command was executed.

To inspect the records:

```sh
python reproduction/validate.py
python reproduction/fetch_source.py --list
python reproduction/fetch_source.py --source state-score-001 --output /tmp/state-score-001.csv
```

Fetching a public upstream file does not rebuild an audit automatically. Use its analysis record, sample tables and input hashes to determine what still needs reconstruction. Pass `--sha256 <original-hash>` when the required file hash is known; the command refuses to save mismatched bytes. Exact original arrays, predictions or annotation versions may remain unavailable even when a related public dataset exists. Never silently generate a different split and call it the original experiment.

## Training and interpretation

`target_weight_updates_in_replay` refers only to the retained notebook calculation. `target_fine_tuning_during_original_audit: null` means the original event was not established by the retained evidence. It is not a statement that fine-tuning did not occur. Upstream checkpoint training is separate from audit-specific training.

Comparator fitting is recorded separately. For example, the CONCH Opus blur comparison used frozen ImageNet ResNet50 features with a trained logistic-regression head. Another retained CONCH script fine-tunes ResNet18 parameters; execution and use of that checkpoint in the reported comparison were not established. Neither operation should be described as fine-tuning CONCH.

`reconstruction.json` collects the per-analysis reconstruction notes. The AlphaMissense Opus coordinate-join error and the ESMFold v1 target mismatch remain explicit. A historical replay, a corrected same-panel sensitivity and an independent confirmation are different analyses.

`validate.py` checks the record structure, identifiers, published file hashes, code syntax, sample-table dimensions and links within this directory. It does not execute the scientific analyses, reproduce the 40 analyses discussed in the reviewer inventory, or certify a scientific conclusion.

The existing website and scientific protocol are unchanged. No data have been imported into the hosted API. A maintainer can consume these JSON files without a frontend or authentication to this repository.
