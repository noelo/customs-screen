"""Jev-style typed decision layer.

The mechanism (per the decision-router research): instead of *generating* an
answer, read the model's probability distribution over a small set of single-token
candidate labels and return that distribution. No autoregressive loop, no
generated tokens -> near-zero cost and sub-second latency.

Two hard-won implementation facts about this endpoint (MAAS vLLM serving
glm-53-flash), both verified empirically:

1. `/v1/chat/completions` ALWAYS emits reasoning first, even with
   `enable_thinking=False` / `thinking=False` / `reasoning_effort=none` /
   a forced system prompt. The first generated token is reasoning text
   ("The..."), never the answer label, so a first-token logprob read is useless
   there.

2. `/v1/completions` (raw text completion, no chat template) does NOT reason.
   The first generated token is the label itself (" A", " B", ...). This is the
   path that makes logit-scoring work.

So: build a plain-text prompt ending in "Answer:", POST to /v1/completions,
read `choices[0].logprobs.top_logprobs[0]`, keep only the tokens matching our
declared labels, and apply a RESTRICTED softmax (renormalise over just those
labels -- NOT the whole vocabulary). That renormalised mass is the decision
distribution.

Confidence here is *spread*, not calibrated accuracy. Callers must treat a flat
distribution as "ask someone else", not as "the top pick is 91% likely correct".
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/glm-53-flash/v1"
DEFAULT_MODEL = "glm-53-flash"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# The endpoint caps `logprobs` at 20, and label tokens crowd each other out as
# the candidate set grows (verified: 12/12 labels surface at 12 candidates,
# only ~9/20 at 20 candidates). 12 is the largest pool that reliably scores.
MAX_LABELS_PER_CALL = 12


class ScoringError(RuntimeError):
    """Raised when the scoring layer cannot produce a trustworthy distribution."""


def _load_key() -> str:
    key = os.environ.get("TMM_MAAS_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("TMM_MAAS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ScoringError("TMM_MAAS_API_KEY not found in environment or ~/.hermes/.env")


@dataclass
class Decision:
    """The result of a scored decision."""

    choice: str                      # winning label, e.g. "C"
    index: int                       # position of the winner in the candidate list
    probabilities: dict[str, float]  # restructured label -> probability
    confidence: float                # probability of the top label (spread, not accuracy)
    margin: float                    # top1 - top2
    escalate: bool                   # True when the distribution is too flat to trust
    latency_ms: float
    raw_top: list[tuple[str, float]] = field(default_factory=list)


class JevScorer:
    """Reads label distributions from a vLLM /v1/completions logprobs response."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: int = 60, confidence_threshold: float = 0.70,
                 margin_threshold: float = 0.20):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self._key = _load_key()

    # ---- prompt construction -------------------------------------------------

    @staticmethod
    def build_prompt(question: str, candidates: list[str],
                     labels: list[str] | None = None,
                     instructions: str | None = None) -> str:
        """Render a plain-text MCQ prompt ending so the next token IS the label.

        Note the trailing "Answer:" with no newline: empirically the model
        continues with a leading-space token (" A"), which we normalise.
        """
        if labels is None:
            labels = list(LETTERS[: len(candidates)])
        if len(labels) != len(candidates):
            raise ScoringError("labels and candidates must be the same length")
        lines = [instructions or
                 "You are a customs classification engine. Choose the single best "
                 "Harmonized Tariff Schedule category for the goods described below.",
                 "", f"Goods: {question}", "", "Candidates:"]
        for lab, cand in zip(labels, candidates):
            lines.append(f"{lab}. {cand}")
        lines += ["", "Answer with the letter only.", "Answer:"]
        return "\n".join(lines)

    @staticmethod
    def _label_token(labels: list[str]) -> dict[str, str]:
        """Map each label to the token form the model actually emits.

        Verified: the model emits a LEADING SPACE before the letter (" A"), not
        the bare letter. We accept both and prefer the spaced form.
        """
        token_for: dict[str, str] = {}
        for lab in labels:
            token_for[lab] = " " + lab
        return token_for

    # ---- scoring -------------------------------------------------------------

    def _post(self, prompt: str, top_logprobs: int = 12) -> dict:
        # This endpoint hard-caps `logprobs` at 20 (verified: HTTP 400
        # "Requested sample logprobs of 52, which is greater than max allowed: 20").
        # Empirically, label tokens crowd each other out as the candidate set
        # grows -- at 12 candidates all 12 labels surface; at 20 only ~9 do. So
        # callers must keep candidate groups at or below MAX_LABELS_PER_CALL.
        capped = min(top_logprobs, 20)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": capped,
        }
        req = urllib.request.Request(
            self.base_url + "/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self._key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            raise ScoringError(f"scoring request failed: HTTP {e.code} {body}") from e
        except Exception as e:  # network/timeout
            raise ScoringError(f"scoring request failed: {type(e).__name__}: {e}") from e

    def score(self, question: str, candidates: list[str],
              labels: list[str] | None = None,
              instructions: str | None = None) -> Decision:
        """Return a Decision with a restricted-softmax distribution over labels.

        Keeps the candidate set at or below 12 so every label token surfaces in
        the endpoint's 20-wide logprob window.
        """
        if not candidates:
            raise ScoringError("no candidates supplied")
        if labels is None:
            labels = list(LETTERS[: len(candidates)])
        if len(labels) != len(candidates):
            raise ScoringError("labels and candidates must be the same length")
        if len(labels) > MAX_LABELS_PER_CALL:
            raise ScoringError(
                f"at most {MAX_LABELS_PER_CALL} candidates per scoring call "
                f"(got {len(labels)}); use score_bracketed for larger sets"
            )

        prompt = self.build_prompt(question, candidates, labels, instructions)
        t0 = time.perf_counter()
        data = self._post(prompt, top_logprobs=20)
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            first = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise ScoringError(f"malformed scoring response: {json.dumps(data)[:300]}") from e

        raw_top = sorted(first.items(), key=lambda kv: -kv[1])
        seen: dict[str, float] = {}
        wanted = {lab.upper(): lab for lab in labels}
        for tok, lp in raw_top:
            key = tok.strip().upper()
            lab = wanted.get(key)
            if lab is not None and lab not in seen:
                seen[lab] = lp

        if not seen:
            raise ScoringError(
                "model did not emit any declared label token; "
                "the prompt/endpoint is not in single-label mode"
            )

        # Restricted softmax over ONLY the declared labels. Labels the model
        # never surfaced get a floor derived from the weakest observed logprob,
        # so "never considered" reads as very unlikely but not literally
        # impossible (assigning them zero would overstate certainty).
        if len(seen) < len(labels):
            floor = min(seen.values()) - 2.0
            for lab in labels:
                seen.setdefault(lab, floor)

        exp = {lab: math.exp(lp) for lab, lp in seen.items()}
        total = sum(exp.values())
        probs = {lab: v / total for lab, v in exp.items()}

        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top1_lab, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1 - top2

        return Decision(
            choice=top1_lab,
            index=labels.index(top1_lab),
            probabilities={lab: round(p, 6) for lab, p in probs.items()},
            confidence=round(top1, 6),
            margin=round(margin, 6),
            escalate=(top1 < self.confidence_threshold or margin < self.margin_threshold),
            latency_ms=round(latency_ms, 1),
            raw_top=[(t, round(l, 3)) for t, l in raw_top[:8]],
        )

    def score_bracketed(self, question: str, candidates: list[str],
                        instructions: str | None = None,
                        relay: int = 12) -> Decision:
        """Score a candidate set larger than one logprob window can hold.

        Round 1 splits the candidates into groups of `relay` and scores each
        independently (each group gets its own label alphabet). The top pick from
        every group advances to round 2, which is scored as a single set.

        Honest caveat: this is a tournament, not a joint distribution. All
        candidates in a group are compared against each other, but a group's
        internal probabilities are not calibrated against another group's, so
        cross-group probability values should NOT be read as comparable. The
        final round's distribution IS a true restricted softmax over the
        survivors, and that is what the caller should trust.
        """
        if len(candidates) <= relay:
            return self.score(question, candidates, instructions=instructions)

        groups = [candidates[i:i + relay] for i in range(0, len(candidates), relay)]
        survivors: list[str] = []
        total_ms = 0.0
        rounds: list[dict] = []
        for gi, group in enumerate(groups):
            labels = list(LETTERS[: len(group)])
            d = self.score(question, group, labels=labels, instructions=instructions)
            total_ms += d.latency_ms
            rounds.append({"group": gi, "size": len(group), "winner": group[d.index],
                           "confidence": d.confidence})
            survivors.append(group[d.index])

        final = self.score(question, survivors, instructions=instructions)
        final.latency_ms = round(total_ms + final.latency_ms, 1)
        # Surface the bracket so a caller can see the tournament happened.
        final.raw_top = final.raw_top
        setattr(final, "bracket", rounds)
        return final
