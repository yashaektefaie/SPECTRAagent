# Runtime Artifacts

This repository tracks the read-only SPECTRA Knowledge MCP server, protocol
resource, deployment config, and structured finding summaries.

`data/provenance.json` is also tracked. It contains compact source, model,
checkpoint, dataset, metadata, retrieval, environment, and known-gap records for
the stored findings. This is the first place an external agent should look when
it needs to know where a model, model outputs, data, or metadata came from.

The curated run artifacts under `data/artifacts/` are intentionally not tracked
in Git. They are deployed on the hosted MCP instance and referenced by stable
artifact ids in `data/store.json`.

Large or reproducibility-critical outputs should be exposed through the public
download manifest, not through MCP text responses. Run:

```sh
python build_download_manifest.py --materialize --clean
```

This writes `data/downloads.json` and a deployment-only
`data/public_downloads/` tree. Caddy serves that tree at
`https://spectra.yashaektefaie.com/downloads/...`.

Current hosted endpoint:

```text
https://spectra.yashaektefaie.com/mcp
```

Current deployed artifact families:

- `conch`: CONCH CRC ROI TUM tissue/stain-fraction tail finding.
- `esmfold2`: ESMFold2 high disulfide/cysteine capacity and class-II E/E2 finding.
- `state`: STATE pathway/module train-support finding.

To rebuild the hosted artifact set, copy curated artifacts from the source run
roots recorded in `data/store.json` into `data/artifacts/<model>/`, then deploy
the `data/` directory to the MCP host.

When adding a new artifact family, update `data/provenance.json` and run the
MCP tool/function `validate_spectra_provenance`. A finding without provenance is
not ready for the hosted knowledge server.

Also update `data/downloads.json` and materialize public downloads for any raw
tables required to reproduce the reported analysis from per-target or
per-example rows.
