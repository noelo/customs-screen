"""Web UI for the Border Classifier.

One page: a search box for the goods description, a Classify button, and a
results panel that lists every candidate's probability sorted highest first.

Serves two things from a single process:
  GET  /            -> the HTML page (inline, no build step)
  POST /classify    -> {"query": "..."} -> full classification JSON
  GET  /healthz     -> liveness (no model load)

Run:
  ./.venv/bin/python src/server.py           # port 8000
  TMM_MAAS_API_KEY must be in the env or ~/.hermes/.env (see jev.py).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from classifier import CustomsClassifier, ScoringError

app = FastAPI(title="Border Classifier")

# The catalog (~30k codes + BM25) loads once at startup; first request after
# that only pays for the LLM calls.
_classifier: CustomsClassifier | None = None
_load_error: str | None = None


@app.on_event("startup")
def _load() -> None:
    global _classifier, _load_error
    t = time.perf_counter()
    try:
        _classifier = CustomsClassifier()
        print(f"catalog + scorer ready in {time.perf_counter() - t:.1f}s")
    except Exception as e:  # keep the server up; classify() reports the error
        _load_error = f"{type(e).__name__}: {e}"
        print(f"startup load failed: {_load_error}")


class Query(BaseModel):
    query: str


@app.get("/healthz")
def healthz() -> dict:
    return {"ready": _classifier is not None, "load_error": _load_error}


@app.post("/classify")
def classify(q: Query) -> JSONResponse:
    query = (q.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")
    if _classifier is None:
        raise HTTPException(status_code=503,
                            detail=f"classifier not ready: {_load_error}")
    try:
        result = _classifier.classify(query)
    except ScoringError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse(content=_result_payload(result))


def _result_payload(result) -> dict:
    """Flat JSON the page renders: candidates with probabilities, best first."""
    # The distribution's label alphabet indexes the FINAL scoring round's set
    # (`scored`): the full pool when it fit one logprob window, the bracket
    # survivors otherwise. Resolve labels against `scored`, never the pool.
    ranked = sorted(result.distribution.items(), key=lambda kv: -kv[1])
    by_label = {c["label"]: c for c in result.scored}
    rows = []
    for lab, prob in ranked:
        match = by_label.get(lab)
        rows.append({
            "label": lab,
            "code": match["code"] if match else None,
            "description": match["description"] if match else None,
            "probability": prob,
        })
    return {
        "query": result.query,
        "top_code": result.code,
        "top_description": result.description,
        "confidence": result.confidence,
        "margin": result.margin,
        "escalated": result.escalated,
        "tier_used": result.tier_used,
        "rationale": result.rationale,
        "filter_headings": result.filter_headings,
        "timings_ms": result.timings_ms,
        "results": rows,
        "notes": result.notes,
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Border Classifier</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #e6edf3;
    max-width: 860px; margin: 0 auto; padding: 2rem 1rem;
  }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  .sub { color: #8b949e; font-size: .9rem; margin-bottom: 1.5rem; }
  .sub code { color: #79c0ff; }
  form { display: flex; gap: .5rem; }
  input[type=text] {
    flex: 1; padding: .7rem 1rem; font-size: 1rem;
    background: #161b22; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 8px;
  }
  input[type=text]:focus { outline: none; border-color: #1f6feb; }
  button {
    padding: .7rem 1.4rem; font-size: 1rem; font-weight: 600;
    background: #238636; color: #fff; border: 0; border-radius: 8px;
    cursor: pointer;
  }
  button:disabled { background: #1b2e22; color: #4a5a50; cursor: wait; }
  #status { margin: 1rem 0; color: #8b949e; font-size: .9rem; min-height: 1.2em; }
  .err {
    background: #3d1418; border: 1px solid #f85149; color: #ffa198;
    padding: .8rem 1rem; border-radius: 8px; white-space: pre-wrap;
  }
  .top {
    background: #161b22; border: 1px solid #30363d; border-left: 4px solid #238636;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;
  }
  .top .code { font-size: 1.3rem; font-weight: 700; font-family: ui-monospace, monospace; }
  .top .meta { color: #8b949e; font-size: .85rem; margin-top: .4rem; }
  .esc {
    background: #3d2e00; border: 1px solid #d29922; color: #e3b341;
    padding: .6rem 1rem; border-radius: 8px; font-size: .9rem; margin: .8rem 0;
  }
  .rat {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: .9rem 1.2rem; margin: .8rem 0; font-size: .92rem; white-space: pre-wrap;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: .9rem; }
  th {
    text-align: left; color: #8b949e; font-weight: 500;
    padding: .45rem .6rem; border-bottom: 1px solid #30363d;
  }
  td { padding: .45rem .6rem; border-bottom: 1px solid #21262d; vertical-align: top; }
  td.p { font-family: ui-monospace, monospace; text-align: right; white-space: nowrap; }
  tr.winner td { background: #12261e; }
  .bar {
    position: relative; background: #21262d; border-radius: 4px;
    height: 18px; min-width: 120px;
  }
  .bar > div {
    background: #1f6feb; border-radius: 4px; height: 100%;
    min-width: 1px;
  }
  .bar > span {
    position: absolute; left: 6px; top: 1px; font-size: .78rem;
    font-family: ui-monospace, monospace; color: #e6edf3;
  }
  .lab {
    display: inline-block; font-family: ui-monospace, monospace; font-weight: 700;
    color: #79c0ff; width: 1.6em;
  }
  .note { color: #8b949e; font-size: .8rem; margin-top: 1rem; }
  details { margin-top: .8rem; color: #8b949e; font-size: .85rem; }
  details pre { white-space: pre-wrap; }
</style>
</head>
<body>
<h1>Border Classifier</h1>
<div class="sub">Three-tier HTS classification &mdash; BM25 recall, logprob scoring over
<code>glm-53-flash</code>, escalation on flat spreads. Probabilities are
<em>spread</em>, not calibrated accuracy.</div>

<form id="f">
  <input type="text" id="q" placeholder="e.g. USB-C charging cables, retail packaging"
         autocomplete="off" autofocus>
  <button id="go" type="submit">Classify</button>
</form>
<div id="status"></div>
<div id="out"></div>

<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const go = document.getElementById('go');
const status = document.getElementById('status');
const out = document.getElementById('out');

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = q.value.trim();
  if (!query) return;
  go.disabled = true;
  status.textContent = 'Classifying \u2014 semantic filter \u2192 recall \u2192 scoring\u2026';
  out.innerHTML = '';
  try {
    const r = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || ('HTTP ' + r.status));
    render(d);
    status.textContent = '';
  } catch (err) {
    out.innerHTML = '<div class="err">' + esc(String(err.message || err)) + '</div>';
    status.textContent = '';
  } finally {
    go.disabled = false;
    go.focus();
  }
});

function esc(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function render(d) {
  const top = d.results[0] || {};
  let html = '';
  html += '<div class="top">';
  html += '<span class="code">' + esc(d.top_code || 'no code') + '</span>';
  html += '<div>' + esc(d.top_description || '') + '</div>';
  html += '<div class="meta">confidence ' + (d.confidence*100).toFixed(1) + '%'
       +  ' &middot; margin ' + (d.margin*100).toFixed(1) + ' pts'
       +  ' &middot; tier ' + d.tier_used
       +  (d.filter_headings && d.filter_headings.length
            ? ' &middot; headings ' + d.filter_headings.join(', ') : '')
       +  (d.timings_ms ? ' &middot; ' + Object.entries(d.timings_ms)
            .map(([k,v]) => k.replace('_ms','') + ' ' + Math.round(v) + 'ms').join(', ') : '')
       +  '</div></div>';
  if (d.escalated) html += '<div class="esc">&#9888; Escalated: the distribution is too flat to trust &mdash; verify before use.</div>';
  if (d.rationale) html += '<div class="rat">' + esc(d.rationale) + '</div>';

  html += '<table><thead><tr><th></th><th>HTS code</th><th>Description</th>'
       +  '<th style="text-align:right">Probability</th></tr></thead><tbody>';
  for (const r of d.results) {
    const pct = (r.probability*100).toFixed(2);
    const w = r.label === top.label ? ' class="winner"' : '';
    html += '<tr' + w + '><td><span class="lab">' + esc(r.label) + '</span></td>'
         +  '<td>' + esc(r.code || '\u2014') + '</td>'
         +  '<td>' + esc(r.description || 'scored subset only (bracket mode)') + '</td>'
         +  '<td class="p"><div class="bar"><div style="width:' + pct + '%"></div>'
         +  '<span>' + pct + '%</span></div></td></tr>';
  }
  html += '</tbody></table>';
  if (d.notes && d.notes.length)
    html += '<details><summary>pipeline notes</summary><pre>' + esc(d.notes.join('\\n')) + '</pre></details>';
  out.innerHTML = html;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="info")
