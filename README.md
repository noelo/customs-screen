# Customs Classifier ("The Border Classifier" demo)

Three-tier HTS classification demo of the Jev pattern: cheap typed scoring
proposes the code, an LLM verifies only when the spread says so, near-ties
escalate.

## Tiers

1. RECALL — BM25 over the 29,860-code USITC HTS index (rank_bm25, local, no GPU).
   `src/recall.py`, index built by `src/build_index.py` from `data/hts_raw.json`.
1b. SEMANTIC FILTER — one cheap chat call maps the description to plausible
   4-digit headings; recall is constrained to them (rescues lexical blind spots:
   "laptop" never appears in HTS text, "automatic data processing machines" does).
2. JEV SCORE — the decision. Raw `/v1/completions` first-token logprob read over
   single-token labels (A, B, C...), restricted softmax over declared candidates.
   `src/jev.py`. Pools are capped at 12 labels per call (endpoint caps logprobs
   at 20 and label tokens crowd out beyond ~12); pools >12 run a bracket
   tournament (groups of 12, winners advance to a final round).
3. VERIFY/ESCALATE — escalate when `top1 < 0.70` or `margin < 0.20`; the chat
   model then verifies the winner and writes the rationale.

Endpoint: MAAS vLLM serving `glm-53-flash`
(`https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/glm-53-flash/v1`).
Key: `TMM_MAAS_API_KEY` in the environment or `~/.hermes/.env`.

## Usage

```bash
cd ~/customs-classifier
./.venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
from classifier import CustomsClassifier; \
r = CustomsClassifier().classify('USB-C charging cables, retail packaging'); \
print(r.to_json())"
```

## Data (not in git)

The USITC export and built index are regenerable and not committed.
Download the raw JSON, then build:

```bash
curl -o data/hts_raw.json "https://hts.usitc.gov/reststop/exportList?from=0100&to=9999&format=JSON&styles=false"
./.venv/bin/python src/build_index.py
```

## Verified 2026-09-23 (end to end, real endpoint)

- "Assorted metal fasteners, zinc-plated, mixed sizes, for construction"
  → 7318.16.00.15, conf 0.98, margin 0.97, no escalation (tier 2).
  Bracket: 3 groups of 12/12/6 → 3 survivors. ~6.4 s total (filter 5.2 s, Jev 1.1 s).
  Correct heading (7318, iron/steel fasteners); the earlier "buttons" recall bug is gone.
- "USB-C charging cables, retail packaging, 2 metre"
  → 8544.42.90.90, conf 0.63, margin 0.29 → escalated (tier 3), verifier ran
  (~13 s) and produced a rationale. Correct heading (8544, insulated conductors).

## Known limitations

- Confidence is spread, not calibrated accuracy — treat flat distributions as
  "ask someone else", not "91% likely correct".
- The verifier's rationale can read as raw GLM reasoning ("We need answer...")
  when the endpoint returns `reasoning` instead of `content`; cosmetic.
- The semantic filter sometimes emits duplicate headings (dedup happens at recall).
- Statistical-suffix picks within a correct heading are a coin toss for genuinely
  ambiguous goods — that is what tier 3 exists for.
