# Paperclip Dataset Scan

Date: 2026-05-21

## Status

Paperclip CLI is working in this environment:

```sh
/ewsc/yektefai/envs/envs/boltz/bin/paperclip config --show
```

Authenticated user reported by Paperclip: `yektefai@broadinstitute.org`.

The official Paperclip Codex skill installer is not working right now:

```sh
paperclip install --dir /home/unix/yektefai/SPECTRA/SPECTRA
```

It attempts to fetch `https://paperclip.gxl.ai/skills/codex.md`, which returns
404. The repository-local Codex skill was therefore repaired manually at
`.agents/skills/paperclip/SKILL.md` to document the working CLI path and usage.

## Codex MCP Configuration

The public Paperclip endpoint advertises MCP metadata at:

```sh
curl https://paperclip.gxl.ai/mcp
```

and returns server info for `paperclip` version `1.0.0`. Direct Codex OAuth
registration requires a browser flow, so this repository now includes a small
stdio MCP shim that forwards one MCP tool to the locally authenticated CLI:

```sh
scripts/paperclip_mcp_stdio.py
```

Codex was configured globally with:

```sh
codex mcp add paperclip -- python3 /home/unix/yektefai/SPECTRA/SPECTRA/scripts/paperclip_mcp_stdio.py
```

The shim exposes a single MCP tool:

```json
{"name": "paperclip", "arguments": {"command": "search \"CRISPR\""}}
```

Manual MCP smoke test:

```sh
python3 scripts/paperclip_mcp_stdio.py <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"manual","version":"0"}}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"paperclip","arguments":{"command":"config --show"}}}
EOF
```

The tool call returned authenticated Paperclip config for
`yektefai@broadinstitute.org`.

## Smoke Test

Command:

```sh
/ewsc/yektefai/envs/envs/boltz/bin/paperclip search \
  "SPECTRA generalizability dataset benchmark similarity axis"
```

Result ID: `s_2ae62971`

The search returned 17 papers, including:

- `bio_6bc9ce1576eb`: Evaluating generalizability of artificial intelligence
  models for molecular datasets.
- `arx_2312.04078`: Methods for Quantifying Dataset Similarity.
- `arx_2002.12105`: Data Representativeness Criterion.
- `arx_2410.08643`: SOAK cross-validation.

Paperclip filesystem read also worked:

```sh
paperclip cat /papers/bio_6bc9ce1576eb/meta.json
paperclip grep -i "dataset" /papers/arx_2505.01912/content.lines
```

## Dataset-Mining Rounds

### Round 1: Bioimage and Microscopy

Command:

```sh
paperclip search "bioimage microscopy out-of-sample generalization dataset benchmark WILDS Camelyon17 RxRx1 COOS BBBC021 cell painting"
paperclip map --from s_b3dc0bbb "Extract reusable datasets or benchmarks for SPECTRA..."
```

Search ID: `s_b3dc0bbb`
Map ID: `m_dd4da469`

Useful hits:

- COOS-7 out-of-sample microscopy benchmark.
- RxRx1/RxRx1-WILDS batch-effect microscopy benchmark.
- BBBC021 Cell Painting benchmark.
- RxRx2 Cell Painting benchmark from Recursion.
- CytoImageNet and BSCCM as larger bioimage resources.
- Several segmentation resources such as LiveCell, CellPose, Cell Tracking
  Challenge, and Aitslab Bioimaging.

Catalog implication:

- Existing catalog entries cover COOS-7, RxRx1, and BBBC021.
- `rxrx2_cell_painting` is a reasonable future catalog entry, but should be
  marked large and subset/embedding-first.
- Segmentation datasets are useful for image-model audits, but less central to
  the first SPECTRA generalization-audit dataset catalog unless the target model
  is a segmentation model.

### Round 2: Proteins, Materials, and Graphs

Command:

```sh
paperclip search "protein fitness mutation generalization benchmark ProteinGym materials distribution shift Matbench graph OOD GOOD dataset"
paperclip map --from s_b1997b23 "Extract reusable datasets or benchmarks for SPECTRA..."
```

Search ID: `s_b1997b23`
Map ID: `m_901f3fda`

Useful hits:

- ProteinGym: DMS assays and clinical variant benchmarks.
- FLIP: GB1, AAV, and thermostability protein fitness landscapes.
- FLIP2: newer protein fitness benchmark with mutation-number, mutation-position,
  mutation-identity, fitness, and wild-type generalization axes.
- GeSS for geometric deep learning under scientific distribution shifts.
- Materials OOD benchmark papers and Matbench/MatFold context.
- Graph benchmark/generator papers such as SkyMap and TU/DrugOOD graph datasets.

Catalog implication:

- Existing catalog covers ProteinGym, Matbench, GOOD, GeSS, and DrugOOD.
- `flip_protein_fitness` and `flip2_protein_fitness` are strong future additions
  because they are explicit split/generalization benchmarks and appear agent
  accessible.

### Round 3: Clinical, Tabular, Time-Series, and Geospatial

Command:

```sh
paperclip search "clinical tabular geospatial time series distribution shift benchmark dataset generalization external validation WILDS TableShift MIMIC eICU"
paperclip map --from s_bcaa8d4e "Extract reusable datasets or benchmarks for SPECTRA..."
```

Search ID: `s_bcaa8d4e`
Map ID: `m_6070b7e2`

Useful hits:

- TableShift tabular distribution-shift benchmark.
- WILDS benchmark.
- MIMIC-IV and eICU external-validation clinical resources.
- Wild-Time temporal distribution-shift benchmark.
- WOODS time-series OOD benchmark.
- EuroCropsML geospatial crop-type benchmark.
- BEDS-Bench and LCD Benchmark for EHR/text distribution-shift evaluation.

Catalog implication:

- Existing catalog covers TableShift, WILDS, MIMIC/eICU, UCR/UEA time-series,
  and WILDS PovertyMap.
- `eurocropsml_zenodo`, `wild_time_benchmark`, and `woods_time_series` are
  reasonable future entries.
- Clinical resources should remain credentialed/authorization-gated. Some
  Paperclip map output described MIMIC/eICU as agent-pullable, but the stricter
  catalog policy should stand: PhysioNet data requires credentialing, training,
  and a data use agreement.

### Round 4: Molecular, Regulatory, and Perturbation

Command:

```sh
paperclip search "molecular drug discovery OOD benchmark dataset DrugOOD BOOM perturbation response generalization PerturBench Systema DART-Eval"
paperclip map --from s_e0dc18d1 "Extract reusable datasets or benchmarks for SPECTRA..."
```

Search ID: `s_e0dc18d1`
Map ID: `m_5d9b27a0`

Useful hits:

- BOOM molecular OOD benchmark.
- DrugOOD molecular/domain OOD benchmark.
- DART-Eval regulatory DNA benchmark.
- PerturBench cellular perturbation benchmark.
- WelQrate small-molecule drug-discovery benchmark.
- ADMEOOD drug-property OOD benchmark.
- CandidateDrug4Cancer molecular graph benchmark.
- Community drug response prediction cross-dataset benchmark.
- PRnet/transcriptional response datasets.

Catalog implication:

- Existing catalog covers BOOM, DrugOOD, DART-Eval, PerturBench, Systema, and
  regulatory perturbation resources.
- `welqrate_drug_discovery`, `admeood_drug_property`, and a standardized
  cross-dataset drug-response resource are possible future additions after
  official-source verification.

## Outcome

Paperclip is usable for SPECTRA mining through the CLI. The actual scans support
the original 23-entry dataset catalog and identified a second wave of useful
entries that have now been promoted into the portable dataset cache where source
and access constraints are acceptable:

- `rxrx2_cell_painting`
- `flip_protein_fitness`
- `flip2_protein_fitness`
- `eurocropsml_zenodo`
- `wild_time_benchmark`
- `woods_time_series`
- `welqrate_drug_discovery`
- `admeood_drug_property`

The dataset cache has now been populated with the strongest second-wave entries:

- `rxrx2_cell_painting`
- `flip_protein_fitness`
- `flip2_protein_fitness`
- `eurocropsml_zenodo`
- `wild_time_benchmark`
- `woods_time_series`
- `welqrate_drug_discovery`
- `admeood_drug_property`

The catalog keeps strict scale and credentialing rules rather than trusting
Paperclip map output when it calls controlled-access clinical data
"autonomous-agent pullable." `admeood_drug_property` is intentionally marked
`needs_source_verification` because Paperclip found the benchmark paper, but the
official packaged data source needs confirmation before automatic use.

## Follow-Up Cache Population Run

Additional Paperclip searches/maps used to populate the cache:

| Area | Search ID | Map ID | Entries populated |
| --- | --- | --- | --- |
| Protein fitness | `s_ac058eed` | `m_8847498b` | `flip_protein_fitness`, `flip2_protein_fitness` |
| Protein fitness targeted | `s_bdecd278` | included in protein evidence | `flip2_protein_fitness` |
| Crop time-series | `s_5c956f9a` | `m_a029787f` | `eurocropsml_zenodo` |
| Time-series OOD | `s_20d1c9ef` | `m_06adbca5` | `woods_time_series` |
| Temporal shift | `s_ebdb52dc` | included in temporal evidence | `wild_time_benchmark` |
| Drug discovery | `s_731e9b16` | search only; map failed due transient Paperclip result lookup issue | `welqrate_drug_discovery`, `admeood_drug_property` with official-source verification |

Official-source checks were used for entries where Paperclip had only title or
partial indexed text. In those cases, the catalog keeps the Paperclip search ID
as discovery provenance and the official URL as access provenance.

## Additional Mining Passes

After the first cache population pass, additional Paperclip searches/maps were
run across medical imaging, single-cell perturbation, regulatory genomics,
earth observation, SciML/PDE simulation, and materials/catalysis. These passes
expanded the cache from 31 to 39 entries.

| Area | Search IDs | Map IDs | Entries populated |
| --- | --- | --- | --- |
| Medical imaging/CXR robustness | `s_6c9b4571`, `s_42b782d8` | `m_7ebd871f`, `m_47958686` | `chexphoto_cxr_robustness` |
| Single-cell perturbation/GRN | `s_8c2cafec`, `s_c8099614` | `m_ee210067`, `m_50d01e90` | `causalbench_single_cell_grn`, `scperturb_harmonized` |
| Regulatory genomics/cross-species | `s_68eb3407` | `m_1af5f230` | `cross_species_tf_binding_morale` |
| Earth observation/foundation models | `s_4d7eeb00`, `s_2974afed` | `m_9d30698a`, `m_c52d8c04` | `geobench_earth_monitoring` |
| Land cover/geospatial | `s_5110991d` | `m_f8743c48` | `landcovernet_radiant` |
| SciML/PDE simulation | `s_ab5e3dfc`, `s_54517283` | `m_4d843701`; targeted map had a transient lookup issue | `pdebench_sciml` |
| Materials/catalysis | `s_6b5119bf` | `m_f8b6693f` | `open_catalyst_oc20` |

The added entries are intentionally conservative about scale. PDEBench and OC20
must not be full-downloaded by default; their SPECTRA defaults point to selected
equations, parameters, metadata, official split manifests, or validation subsets.
