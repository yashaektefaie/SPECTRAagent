"""Build the static SPECTRA Knowledge MCP website from data/store.json."""

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
STORE_PATH = ROOT / "data" / "store.json"
PROVENANCE_PATH = ROOT / "data" / "provenance.json"
DOWNLOADS_PATH = ROOT / "data" / "downloads.json"
SITE_ROOT = ROOT / "site"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def slug(value: str) -> str:
    out = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            out.append(char)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "item"


def human_decision(decision: str) -> str:
    return decision.replace("__", " / ").replace("_", " ")


def metric_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:.2e}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, list):
        return "[" + ", ".join(metric_value(item) for item in value) + "]"
    return str(value)


def nested_get(data: Dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


KEY_METRICS = {
    "CONCH": [
        ("Discovery tiles", ("evidence", "discovery_pool_crc_val", "n")),
        ("CRC-VAL balanced accuracy", ("evidence", "discovery_pool_crc_val", "balanced_accuracy")),
        ("Confirmation TUM tiles", ("evidence", "fresh_nonorm_tum_confirmation", "n")),
        ("Expanded TUM tiles", ("evidence", "expanded_nonorm_tum_panel", "n")),
        ("Low-tail recall gap", ("evidence", "expanded_nonorm_tum_panel", "levels_1_to_7_minus_8_to_10_gap")),
        ("Expanded Spearman", ("evidence", "expanded_nonorm_tum_panel", "spearman_level_vs_tum_recall")),
    ],
    "ESMFold2": [
        ("Primary rows", ("evidence", "primary_exact_sequence_rows")),
        ("High-disulfide n", ("evidence", "global_high_disulfide_effect", "high_disulfide_n")),
        ("Lower-disulfide n", ("evidence", "global_high_disulfide_effect", "lower_disulfide_n")),
        ("Raw CA-lDDT gap", ("evidence", "global_high_disulfide_effect", "raw_ca_lddt_gap")),
        ("Matched-control gap", ("evidence", "matched_controls", "mean_high_minus_control_ca_lddt")),
        ("Class-II CA-lDDT gap", ("evidence", "localized_class_ii_effect", "ca_lddt_gap")),
    ],
    "STATE": [
        ("Scored rows", ("evidence", "official_scored_rows_used")),
        ("Few-shot HVG gap", ("evidence", "primary_replogle_fewshot_confirmation", "hvg_high_low_gap")),
        ("Few-shot SE gap", ("evidence", "primary_replogle_fewshot_confirmation", "se_high_low_gap")),
        ("Zero-shot HVG gap", ("evidence", "replogle_zeroshot_confirmation", "HVG", "high_low_gap")),
        ("Zero-shot SE gap", ("evidence", "replogle_zeroshot_confirmation", "SE", "high_low_gap")),
        ("OLS support coefficient", ("evidence", "confound_checks", "ols_support_coefficient_overlap_at_n")),
    ],
}


def render_spc_svg() -> str:
    return """
<svg class="spc-graphic" viewBox="0 0 720 360" role="img" aria-label="Spectral performance curve diagram">
  <defs>
    <linearGradient id="curveFill" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#1f9d8a" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#d04735" stop-opacity="0.14"/>
    </linearGradient>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0 0 L8 4 L0 8 Z" fill="#6c7783"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="720" height="360" rx="12" fill="#fbfcfd"/>
  <line x1="84" y1="286" x2="654" y2="286" stroke="#2d3640" stroke-width="2"/>
  <line x1="84" y1="286" x2="84" y2="54" stroke="#2d3640" stroke-width="2"/>
  <g stroke="#d8dee4" stroke-width="1">
    <line x1="84" y1="232" x2="654" y2="232"/>
    <line x1="84" y1="178" x2="654" y2="178"/>
    <line x1="84" y1="124" x2="654" y2="124"/>
    <line x1="198" y1="54" x2="198" y2="286"/>
    <line x1="312" y1="54" x2="312" y2="286"/>
    <line x1="426" y1="54" x2="426" y2="286"/>
    <line x1="540" y1="54" x2="540" y2="286"/>
  </g>
  <path d="M84 118 C170 116 210 126 270 145 S390 184 454 218 S570 259 654 264 L654 286 L84 286 Z" fill="url(#curveFill)"/>
  <path d="M84 118 C170 116 210 126 270 145 S390 184 454 218 S570 259 654 264" fill="none" stroke="#1f6f68" stroke-width="5" stroke-linecap="round"/>
  <g fill="#ffffff" stroke="#1f6f68" stroke-width="4">
    <circle cx="84" cy="118" r="9"/>
    <circle cx="198" cy="122" r="9"/>
    <circle cx="312" cy="159" r="9"/>
    <circle cx="426" cy="210" r="9"/>
    <circle cx="540" cy="250" r="9"/>
    <circle cx="654" cy="264" r="9"/>
  </g>
  <g fill="#4c5966" font-family="Inter, Arial, sans-serif" font-size="16">
    <text x="80" y="326">high similarity</text>
    <text x="514" y="326">low similarity</text>
    <text x="18" y="72" transform="rotate(-90 18 72)">performance</text>
  </g>
  <g fill="#202831" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="700">
    <text x="112" y="82">freeze prospective axis</text>
    <text x="404" y="126">audit degradation</text>
    <text x="484" y="240">confirm boundary</text>
  </g>
  <path d="M180 90 L126 112" stroke="#6c7783" stroke-width="2" marker-end="url(#arrow)"/>
</svg>
"""


def render_metrics(finding: Dict[str, Any]) -> str:
    rows: List[str] = []
    for label, path in KEY_METRICS.get(finding.get("model", ""), []):
        value = nested_get(finding, path)
        if value is None:
            continue
        rows.append(
            '<div class="metric"><span class="metric-label">%s</span><strong>%s</strong></div>'
            % (esc(label), esc(metric_value(value)))
        )
    return "\n".join(rows)


def render_artifacts(ids: List[str]) -> str:
    items = "\n".join(
        '<li><code>%s</code></li>' % esc(artifact_id) for artifact_id in ids[:8]
    )
    if len(ids) > 8:
        items += '<li class="muted">and %d more key artifacts in the MCP store</li>' % (len(ids) - 8)
    return "<ul class=\"artifact-list\">%s</ul>" % items


def render_downloads(downloads: List[Dict[str, Any]]) -> str:
    if not downloads:
        return ""
    ordered = sorted(downloads, key=lambda item: int(item.get("bytes") or 0), reverse=True)
    items = []
    for item in ordered[:8]:
        label = item.get("relative_path", item.get("artifact_id", "download"))
        details = []
        if item.get("rows") is not None:
            details.append("%s rows" % metric_value(item.get("rows")))
        if item.get("bytes") is not None:
            details.append("%s bytes" % metric_value(item.get("bytes")))
        detail = " · ".join(details)
        items.append(
            '<li><a href="%s">%s</a><span>%s</span></li>'
            % (esc(item.get("download_url", "")), esc(label), esc(detail))
        )
    if len(ordered) > 8:
        items.append('<li class="muted">and %d more downloadable artifacts</li>' % (len(ordered) - 8))
    return """
  <details class="download-box" open>
    <summary>Public downloads</summary>
    <ul class="download-list">%s</ul>
  </details>
""" % "\n".join(items)


def render_provenance(provenance: Dict[str, Any]) -> str:
    if not provenance:
        return """
  <div class="provenance-box missing">
    <h4>Source Provenance</h4>
    <p>No normalized provenance record is currently linked to this finding.</p>
  </div>
"""
    model_source = provenance.get("model_source", {})
    weights = model_source.get("weights_or_checkpoint", {})
    dataset_names = [item.get("name", "") for item in provenance.get("dataset_sources", [])[:4]]
    metadata_names = [item.get("name", "") for item in provenance.get("metadata_sources", [])[:4]]
    gaps = provenance.get("known_gaps", [])
    gap_items = "\n".join("<li>%s</li>" % esc(item) for item in gaps[:3])
    dataset_items = "\n".join("<li>%s</li>" % esc(item) for item in dataset_names if item)
    metadata_items = "\n".join("<li>%s</li>" % esc(item) for item in metadata_names if item)
    return """
  <div class="provenance-box">
    <div class="provenance-head">
      <h4>Source Provenance</h4>
      <span>%s</span>
    </div>
    <dl class="provenance-grid">
      <div><dt>Model</dt><dd>%s</dd></div>
      <div><dt>Weights / Scores</dt><dd>%s</dd></div>
      <div><dt>Execution</dt><dd>%s</dd></div>
    </dl>
    <div class="source-lists">
      <div><strong>Data</strong><ul>%s</ul></div>
      <div><strong>Metadata</strong><ul>%s</ul></div>
    </div>
    %s
  </div>
""" % (
        esc(provenance.get("status", "unknown")),
        esc(model_source.get("source_url_or_repo", "")),
        esc(weights.get("source_url_or_repo", weights.get("filename", ""))),
        esc(model_source.get("execution_mode", "")),
        dataset_items or "<li>No dataset sources recorded.</li>",
        metadata_items or "<li>No metadata sources recorded.</li>",
        (
            '<details class="provenance-gaps"><summary>Known provenance gaps</summary><ul>%s</ul></details>'
            % gap_items
            if gap_items
            else ""
        ),
    )


def render_finding_card(
    finding: Dict[str, Any],
    provenance: Dict[str, Any],
    downloads: List[Dict[str, Any]],
) -> str:
    model = finding.get("model", "")
    decision = finding.get("validity", {}).get("decision", "")
    limitations = finding.get("validity", {}).get("limitations", [])
    limit_items = "\n".join("<li>%s</li>" % esc(item) for item in limitations[:3])
    nonprimary = finding.get("negative_or_downgraded_axes", [])
    nonprimary_items = "\n".join(
        "<li><strong>%s:</strong> %s</li>" % (esc(item.get("axis", "")), esc(item.get("status", "")))
        for item in nonprimary[:3]
    )
    return """
<article class="finding-card" id="%s">
  <div class="finding-head">
    <div>
      <p class="eyebrow">%s</p>
      <h3>%s</h3>
    </div>
    <span class="decision-pill">%s</span>
  </div>
  <p class="summary">%s</p>
  <div class="axis-box">
    <span>Primary Axis</span>
    <strong>%s</strong>
    <p>%s</p>
  </div>
  <div class="metric-grid">%s</div>
  <div class="claim">
    <h4>Claim Boundary</h4>
    <p>%s</p>
  </div>
  %s
  %s
  <div class="two-col">
    <div>
      <h4>Limitations</h4>
      <ul>%s</ul>
    </div>
    <div>
      <h4>Downgraded Axes</h4>
      <ul>%s</ul>
    </div>
  </div>
  <details>
    <summary>Key MCP artifact IDs</summary>
    %s
  </details>
</article>
""" % (
        esc(slug(finding.get("finding_id", ""))),
        esc(model),
        esc(finding.get("finding_id", "")),
        esc(human_decision(decision)),
        esc(finding.get("summary", "")),
        esc(finding.get("axis", {}).get("name", "")),
        esc(finding.get("axis", {}).get("definition", finding.get("axis", {}).get("interpretation", ""))),
        render_metrics(finding),
        esc(finding.get("claim_boundary", "")),
        render_provenance(provenance),
        render_downloads(downloads),
        limit_items or "<li>No explicit limitations recorded.</li>",
        nonprimary_items or "<li>No downgraded axes recorded.</li>",
        render_artifacts(finding.get("key_artifact_ids", [])),
    )


def render_model_links(findings: List[Dict[str, Any]]) -> str:
    links = []
    for finding in findings:
        links.append(
            '<a href="#%s"><span>%s</span><strong>%s</strong></a>'
            % (
                esc(slug(finding.get("finding_id", ""))),
                esc(finding.get("model", "")),
                esc(finding.get("axis", {}).get("name", "")),
            )
        )
    return "\n".join(links)


def build() -> None:
    store = json.loads(STORE_PATH.read_text())
    provenance_records = []
    if PROVENANCE_PATH.exists():
        provenance_records = json.loads(PROVENANCE_PATH.read_text()).get("records", [])
    download_records = []
    if DOWNLOADS_PATH.exists():
        download_records = json.loads(DOWNLOADS_PATH.read_text()).get("records", [])
    provenance_by_finding = {
        item.get("finding_id"): item for item in provenance_records if item.get("finding_id")
    }
    downloads_by_finding: Dict[str, List[Dict[str, Any]]] = {}
    for item in download_records:
        for finding_id in item.get("finding_ids", []):
            downloads_by_finding.setdefault(finding_id, []).append(item)
    findings = store.get("findings", [])
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SITE_ROOT / "assets").mkdir(parents=True, exist_ok=True)

    finding_cards = "\n".join(
        render_finding_card(
            finding,
            provenance_by_finding.get(finding.get("finding_id"), {}),
            downloads_by_finding.get(finding.get("finding_id"), []),
        )
        for finding in findings
    )
    model_links = render_model_links(findings)
    model_count = len(store.get("models", []))
    finding_count = len(findings)
    artifact_count = sum(len(run.get("artifact_ids", [])) for run in store.get("runs", []))

    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPECTRA Knowledge MCP</title>
  <meta name="description" content="SPECTRA generalizability audit protocol, current findings, and public MCP endpoint.">
  <link rel="stylesheet" href="/assets/site.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/">
      <span class="brand-mark" aria-hidden="true"></span>
      <span>SPECTRA</span>
    </a>
    <nav aria-label="Primary">
      <a href="#protocol">Protocol</a>
      <a href="#provenance">Provenance</a>
      <a href="#downloads">Downloads</a>
      <a href="#findings">Findings</a>
      <a href="#connect">Connect</a>
      <a href="/mcp">MCP</a>
    </nav>
  </header>

  <main>
    <section class="intro-band">
      <div class="intro-copy">
        <p class="eyebrow">Generalizability audits for foundation models</p>
        <h1>SPECTRA turns model generalization claims into validated spectral performance curves.</h1>
        <p class="lede">This hosted knowledge server exposes the SPECTRA protocol, prior audit findings, and artifact references for agents and humans. Agents connect at <code>https://spectra.yashaektefaie.com/mcp</code>.</p>
        <div class="quick-stats" aria-label="Current SPECTRA knowledge store counts">
          <div><strong>%d</strong><span>models</span></div>
          <div><strong>%d</strong><span>findings</span></div>
          <div><strong>%d</strong><span>artifact refs</span></div>
        </div>
      </div>
      <div class="visual-panel">
        %s
      </div>
    </section>

    <section id="protocol" class="section-grid">
      <div>
        <p class="eyebrow">Protocol</p>
        <h2>What SPECTRA Requires</h2>
      </div>
      <div class="protocol-steps">
        <div><span>01</span><strong>Define the scientific unit</strong><p>Choose the model, data, task, metric, and prospective novelty axis before target scoring.</p></div>
        <div><span>02</span><strong>Freeze and validate the split</strong><p>Measure that train-test or pretraining-test similarity decreases across levels.</p></div>
        <div><span>03</span><strong>Score baselines and model</strong><p>Run fixed baselines where labels exist, then evaluate the target model on frozen levels.</p></div>
        <div><span>04</span><strong>Confirm live hypotheses</strong><p>Outcome-mined axes are exploratory until frozen and confirmed on fresh or adequate evidence.</p></div>
        <div><span>05</span><strong>Ledger weak and negative axes</strong><p>Non-explanatory curves are findings that route back into prospective-axis discovery.</p></div>
        <div><span>06</span><strong>State the claim boundary</strong><p>A valid SPC closes only the deployment or mechanism boundary it actually tested.</p></div>
        <div><span>07</span><strong>Record provenance</strong><p>Publish model, weight, dataset, metadata, download, cache, and known-gap records with every stored finding.</p></div>
      </div>
    </section>

    <section id="provenance" class="section-grid">
      <div>
        <p class="eyebrow">Provenance</p>
        <h2>Where the evidence came from</h2>
      </div>
      <div class="provenance-intro">
        <p>Agents should call <code>get_spectra_provenance</code> or <code>list_spectra_sources</code> before using a stored finding. New findings are incomplete unless they include source repositories, checkpoint or score provenance, dataset access routes, metadata resources, cache roots, and explicit known gaps.</p>
        <pre><code>{
  "tool": "get_spectra_provenance",
  "arguments": {
    "model": "ESMFold2"
  }
}</code></pre>
      </div>
    </section>

    <section id="downloads" class="section-grid">
      <div>
        <p class="eyebrow">Downloads</p>
        <h2>Raw tables leave MCP as files</h2>
      </div>
      <div class="provenance-intro">
        <p>Large CSVs and reproducibility-critical artifacts are published as normal HTTPS downloads with row counts and SHA-256 checksums. Agents should call <code>list_spectra_downloads</code>, then use <code>curl</code>, <code>wget</code>, or a dataframe loader on the returned URL.</p>
        <pre><code>{
  "tool": "list_spectra_downloads",
  "arguments": {
    "model": "STATE",
    "query": "target_model_results"
  }
}</code></pre>
      </div>
    </section>

    <section class="model-strip" aria-label="Current finding navigation">
      %s
    </section>

    <section id="findings" class="findings">
      <div class="section-heading">
        <p class="eyebrow">Current Evidence</p>
        <h2>Stored SPECTRA Findings</h2>
        <p>These summaries are human-readable views of the same structured records exposed through the MCP tools.</p>
      </div>
      %s
    </section>

    <section id="connect" class="connect-section">
      <div>
        <p class="eyebrow">Agent Access</p>
        <h2>Connect Claude Code or another MCP client</h2>
        <p>The server uses streamable HTTP and exposes read-only tools for listing models, retrieving findings, reading protocol sections, fetching text previews, retrieving model/data provenance, and returning public download URLs.</p>
      </div>
      <pre><code>{
  "mcpServers": {
    "spectra": {
      "type": "http",
      "url": "https://spectra.yashaektefaie.com/mcp"
    }
  }
}</code></pre>
    </section>
  </main>

  <footer>
    <span>SPECTRA Knowledge MCP</span>
    <span>Read-only protocol and findings server</span>
  </footer>
</body>
</html>
""" % (
        model_count,
        finding_count,
        artifact_count,
        render_spc_svg().strip(),
        model_links.strip(),
        finding_cards.strip(),
    )
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    (SITE_ROOT / "index.html").write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    build()
