"""Step 4 (forecast) and step 5 (guard): model call, code-enforced clamps, saved output."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from . import prompts
from .fetch import DATA_DIR, market_probs

FORECAST_DIR = DATA_DIR / "forecasts"
CACHE_DIR = DATA_DIR / "cache"

# USD per 1M tokens (input, output). Thinking tokens are billed as output.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

BAND = {"strong": 0.20}  # any other confidence level gets the default band
DEFAULT_BAND = 0.10
FLOOR, CEIL = 0.01, 0.99


def forecast_model() -> str:
    return os.environ.get("FORECAST_MODEL", "claude-opus-5-5")


def is_independent(question: dict) -> bool:
    """Ballot measures are independent Yes/No markets per outcome: never renormalize."""
    return question["office"] == "measure"


def cost_usd(model: str, usage: dict) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out) / 1e6


# ---------- prompt ----------

def _prices(question: dict, venue: str) -> str:
    vals = {o["name"]: o.get(venue) for o in question["outcomes"]}
    if all(v is None for v in vals.values()):
        return "not listed"
    return ", ".join(f"{k}: {'n/a' if v is None else v}" for k, v in vals.items())


def build_prompt(question: dict, brief: str | None) -> str:
    outcomes = ", ".join(
        f"{o['name']} ({o['party']})" if o.get("party") else o["name"] for o in question["outcomes"]
    )
    return prompts.FORECAST_PROMPT.format(
        title=question["title"],
        outcomes=outcomes,
        structure_note=prompts.INDEPENDENT_NOTE if is_independent(question) else prompts.EXCLUSIVE_NOTE,
        kalshi_prices=_prices(question, "kalshi"),
        polymarket_prices=_prices(question, "polymarket"),
        brief=brief or prompts.NO_BRIEF,
    )


def output_schema(question: dict) -> dict:
    """Per-event JSON schema so the model must use the exact outcome names."""
    names = [o["name"] for o in question["outcomes"]]
    return {
        "type": "object",
        "properties": {
            "probabilities": {
                "type": "object",
                "properties": {n: {"type": "number"} for n in names},
                "required": names,
                "additionalProperties": False,
            },
            "deviation_reason": {"type": "string"},
            "confidence_in_deviation": {"type": "string", "enum": ["none", "weak", "strong"]},
            "rationale": {"type": "string"},
        },
        "required": ["probabilities", "deviation_reason", "confidence_in_deviation", "rationale"],
        "additionalProperties": False,
    }


# ---------- model call (cached) ----------

async def call_model(client: anthropic.AsyncAnthropic, question: dict, prompt: str, force: bool = False) -> dict:
    """Return {"output": parsed-or-None, "usage": {...}, "model": ..., "error": ...}.

    Cached per event by prompt hash, so an unchanged prompt never re-bills.
    """
    model = forecast_model()
    key = hashlib.sha256((model + prompt).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / question["ticker"] / f"forecast-{key}.json"
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        cached["cached"] = True
        return cached

    result = {"output": None, "usage": {}, "model": model, "error": None, "cached": False}
    try:
        resp = await client.beta.messages.create(
            model=model,
            max_tokens=16000,
            output_config={
                "effort": os.environ.get("FORECAST_EFFORT", "high"),
                "format": {"type": "json_schema", "schema": output_schema(question)},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        result["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        result["served_by"] = resp.model
        if resp.stop_reason == "refusal":
            result["error"] = f"refusal: {getattr(resp.stop_details, 'category', None)}"
        elif resp.stop_reason == "max_tokens":
            result["error"] = "max_tokens"
        else:
            text = next((b.text for b in resp.content if b.type == "text"), "")
            result["output"] = json.loads(text)
    except (anthropic.APIError, json.JSONDecodeError) as e:
        result["error"] = f"{type(e).__name__}: {e}"

    if result["error"] is None:  # only cache successes; failures should be retried
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2))
    return result


# ---------- guard ----------

def prior(question: dict) -> dict[str, float]:
    """Market prior, clipped to [0.01, 0.99]; renormalized unless outcomes are independent."""
    probs = {k: _clip(v if v is not None else FLOOR) for k, v in market_probs(question).items()}
    if is_independent(question):
        return {k: round(v, 4) for k, v in probs.items()}
    return _normalize({k: _clip(v) for k, v in _normalize(probs).items()})


def _clip(v: float) -> float:
    return min(CEIL, max(FLOOR, v))


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Scale to sum 1, round to 4 dp, and put the rounding residual on the largest outcome."""
    total = sum(probs.values())
    out = {k: round(v / total, 4) for k, v in probs.items()}
    top = max(out, key=out.get)
    out[top] = round(out[top] + 1 - sum(out.values()), 4)
    return out


def guard(question: dict, output: dict | None) -> tuple[dict[str, float], list[str]]:
    """Enforce CLAUDE.md rules in code. Returns (final probabilities, notes on what was changed)."""
    names = [o["name"] for o in question["outcomes"]]
    market = prior(question)

    probs = (output or {}).get("probabilities")
    if (
        not isinstance(probs, dict)
        or sorted(probs) != sorted(names)
        or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in probs.values())
    ):
        return market, ["invalid or missing model output: used market prior"]

    if output.get("confidence_in_deviation") == "none":
        drift = max(abs(float(probs[n]) - market[n]) for n in names)
        notes = [f"confidence 'none' but max drift {drift:.3f}: snapped to market"] if drift >= 5e-4 else []
        return market, notes

    band = BAND.get(output.get("confidence_in_deviation"), DEFAULT_BAND)
    notes = []
    final = {}
    for n in names:
        v = float(probs[n])
        lo, hi = market[n] - band, market[n] + band
        clamped = _clip(min(hi, max(lo, v)))
        if abs(clamped - v) >= 5e-4:
            notes.append(f"{n}: {v:.3f} -> {clamped:.3f} (band ±{band:.2f}, bounds [{FLOOR}, {CEIL}])")
        final[n] = clamped

    if is_independent(question):
        return {k: round(v, 4) for k, v in final.items()}, notes
    total = sum(final.values())
    if abs(total - 1) >= 1e-3:
        notes.append(f"renormalized from sum {total:.4f}")
    return _normalize(final), notes


# ---------- per-event pipeline ----------

async def forecast_event(
    client: anthropic.AsyncAnthropic,
    question: dict,
    brief: str | None = None,
    trace: list | None = None,
    force: bool = False,
    research: dict | None = None,
) -> dict:
    """Forecast one event, apply the guard, save data/forecasts/<ticker>.json."""
    trace = list(trace or [])
    prompt = build_prompt(question, brief)
    res = await call_model(client, question, prompt, force=force)
    out = res["output"]

    final, notes = guard(question, out)
    if out is not None:
        trace.append({"type": "final", "probabilities": out["probabilities"], "rationale": out["rationale"]})
    if out is None or notes:
        before = out["probabilities"] if out else None
        trace.append({"type": "guard", "before": before, "after": final})

    rationale = out["rationale"] if out else "Model output unavailable; submitting the market average."
    if out and out.get("deviation_reason") not in (None, "", "none"):
        rationale += f" Deviation basis: {out['deviation_reason']}"

    record = {
        "ticker": question["ticker"],
        "title": question["title"],
        "office": question["office"],
        "independent": is_independent(question),
        "market": prior(question),
        "probabilities": final,
        "rationale": rationale,
        "confidence_in_deviation": (out or {}).get("confidence_in_deviation", "none"),
        "deviation_reason": (out or {}).get("deviation_reason", "none"),
        "guard_notes": notes,
        "model": res["model"],
        "served_by": res.get("served_by"),
        "error": res["error"],
        "usage": res["usage"],
        "cached": res.get("cached", False),
        "cost_usd": 0.0 if res.get("cached") else cost_usd(res["model"], res["usage"]),
        "research": research,
        "brief": brief,
        "prices_at": question.get("prices_at"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace": trace,
    }
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    (FORECAST_DIR / f"{question['ticker']}.json").write_text(json.dumps(record, indent=2))
    return record


def load_forecasts() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FORECAST_DIR.glob("*.json"))]
