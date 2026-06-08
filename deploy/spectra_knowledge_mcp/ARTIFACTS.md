# Runtime Artifacts

This repository tracks the read-only SPECTRA Knowledge MCP server, protocol
resource, deployment config, and structured finding summaries.

The curated run artifacts under `data/artifacts/` are intentionally not tracked
in Git. They are deployed on the hosted MCP instance and referenced by stable
artifact ids in `data/store.json`.

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
