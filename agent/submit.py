"""Build, sanitize, validate and batch saved forecasts for POST /election/forecasts.

Dry run only for now: nothing is sent.
"""

import base64
import gzip
import json
import math
import re
from datetime import datetime, timezone

from dotenv import dotenv_values

from .fetch import load_events
from .forecast import load_forecasts

MAX_RATIONALE = 10_000
MAX_TRACE_RAW = 10 * 1024 * 1024
MAX_ITEMS_PER_REQUEST = 100
MAX_TRACED_PER_REQUEST = 10
MAX_REQUEST_BYTES = 1_500_000  # stay well under the 2 MB server limit
SUM_TOL = 1e-6

TRACE_KEYS = {
    "tool_call": {"type", "tool", "args", "result"},
    "brief": {"type", "text"},
    "final": {"type", "probabilities", "rationale"},
    "guard": {"type", "before", "after"},
}

# Key-shaped tokens. Require a long tail so words like "risk-" never match.
KEY_PATTERNS = [
    re.compile(r"pa_weekly_[A-Za-z0-9_\-]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
]
REDACTED = "[REDACTED]"


# ---------- sanitize ----------

def _secret_values() -> list[str]:
    """Actual values from .env that must never leave the machine (keys, not model names)."""
    vals = dotenv_values(".env")
    return [v for k, v in vals.items() if v and k.endswith("_API_KEY")]


def scrub(text: str, secrets: list[str]) -> str:
    text = text.replace("\u0000", "")
    for s in secrets:
        text = text.replace(s, REDACTED)
    for pat in KEY_PATTERNS:
        text = pat.sub(REDACTED, text)
    return text


def scrub_obj(obj, secrets: list[str]):
    if isinstance(obj, str):
        return scrub(obj, secrets)
    if isinstance(obj, list):
        return [scrub_obj(x, secrets) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v, secrets) for k, v in obj.items()}
    return obj


# ---------- trace packing ----------

def pack_trace(trace: list) -> str:
    return base64.b64encode(gzip.compress(json.dumps(trace).encode())).decode()


def unpack_trace(packed: str) -> list:
    return json.loads(gzip.decompress(base64.b64decode(packed)))


def fit_trace(trace: list, budget_b64: int = 1_400_000) -> tuple[str, bool]:
    """Pack the trace; if too large, truncate the longest search results but keep every step."""
    packed = pack_trace(trace)
    truncated = False
    limit = 20_000
    while len(packed) > budget_b64 and limit > 200:
        truncated = True
        for e in trace:
            if e.get("type") == "tool_call" and isinstance(e.get("result"), str) and len(e["result"]) > limit:
                e["result"] = e["result"][:limit] + " …[truncated]"
        packed = pack_trace(trace)
        limit //= 2
    return packed, truncated


# ---------- payload ----------

def build_item(record: dict, secrets: list[str]) -> tuple[dict, dict]:
    """Return (payload item, info) for one saved forecast."""
    trace = scrub_obj(record.get("trace") or [], secrets)
    item = {
        "question": record["ticker"],
        "probabilities": record["probabilities"],
        "rationale": scrub(record["rationale"], secrets)[:MAX_RATIONALE],
    }
    info = {"trace_entries": len(trace), "trace_raw_bytes": 0, "trace_truncated": False}
    if trace:
        packed, truncated = fit_trace(trace)
        item["trace_gzip_base64"] = packed
        info.update(trace_raw_bytes=len(json.dumps(trace).encode()), trace_truncated=truncated)
    return item, info


def validate_item(item: dict, question: dict | None, secrets: list[str]) -> list[str]:
    """Check one item against the contract in CLAUDE.md. Returns a list of problems."""
    errs = []
    if question is None:
        return [f"unknown ticker {item['question']}"]
    if question.get("status") != "open":
        errs.append(f"question status is {question.get('status')}, not open")

    names = [o["name"] for o in question["outcomes"]]
    probs = item["probabilities"]
    if sorted(probs) != sorted(names):
        errs.append(f"outcome names {sorted(probs)} != event outcomes {sorted(names)}")
    for k, v in probs.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1:
            errs.append(f"{k}: probability {v!r} not in [0, 1]")
    independent = question["office"] == "measure"
    if not independent and abs(sum(probs.values()) - 1) > SUM_TOL:
        errs.append(f"mutually exclusive outcomes sum to {sum(probs.values()):.6f}, not 1")

    r = item["rationale"]
    if not r.strip():
        errs.append("empty rationale")
    if len(r) > MAX_RATIONALE:
        errs.append(f"rationale {len(r)} chars > {MAX_RATIONALE}")
    if "\u0000" in r:
        errs.append("rationale contains NUL")

    if "trace" in item and "trace_gzip_base64" in item:
        errs.append("both trace and trace_gzip_base64 set")
    if "trace_gzip_base64" in item:
        try:
            raw = gzip.decompress(base64.b64decode(item["trace_gzip_base64"]))
            trace = json.loads(raw)
            if len(raw) > MAX_TRACE_RAW:
                errs.append(f"trace {len(raw)} bytes uncompressed > 10 MB")
            for i, e in enumerate(trace):
                allowed = TRACE_KEYS.get(e.get("type"))
                if allowed is None:
                    errs.append(f"trace[{i}] has unknown type {e.get('type')!r}")
                elif set(e) - allowed:
                    errs.append(f"trace[{i}] ({e['type']}) has extra keys {sorted(set(e) - allowed)}")
        except Exception as ex:  # noqa: BLE001 - any decode failure is a validation failure
            errs.append(f"trace does not decode: {ex}")

    for forbidden in ("agent", "anonymous"):
        if forbidden in item:
            errs.append(f"item must not set {forbidden!r}")

    blob = json.dumps(item)
    if "trace_gzip_base64" in item:
        blob += json.dumps(unpack_trace(item["trace_gzip_base64"]))
    if any(s in blob for s in secrets) or any(p.search(blob) for p in KEY_PATTERNS):
        errs.append("secret-like token still present after sanitizing")
    return errs


def make_batches(items: list[dict]) -> list[list[dict]]:
    """Split into requests: ≤100 items, ≤10 with traces, ≤~1.5 MB each."""
    batches, cur, cur_bytes, cur_traced = [], [], 0, 0
    for it in items:
        size = len(json.dumps(it).encode())
        traced = "trace_gzip_base64" in it
        if cur and (
            len(cur) >= MAX_ITEMS_PER_REQUEST
            or cur_bytes + size > MAX_REQUEST_BYTES
            or (traced and cur_traced >= MAX_TRACED_PER_REQUEST)
        ):
            batches.append(cur)
            cur, cur_bytes, cur_traced = [], 0, 0
        cur.append(it)
        cur_bytes += size
        cur_traced += traced
    if cur:
        batches.append(cur)
    return batches


def prepare(tickers: list[str] | None = None) -> dict:
    """Build and validate every saved forecast. Returns everything a dry run needs to report."""
    events = load_events()
    by_ticker = {q["ticker"]: q for q in events["questions"]}
    secrets = _secret_values()

    records = load_forecasts()
    if tickers:
        records = [r for r in records if r["ticker"] in set(tickers)]

    items, infos, problems = [], {}, {}
    seen = set()
    for rec in records:
        item, info = build_item(rec, secrets)
        errs = validate_item(item, by_ticker.get(rec["ticker"]), secrets)
        if rec["ticker"] in seen:
            errs.append("duplicate ticker")
        seen.add(rec["ticker"])
        if rec.get("error"):
            info["note"] = f"model error, market prior used: {rec['error']}"
        infos[rec["ticker"]] = info
        if errs:
            problems[rec["ticker"]] = errs
        else:
            items.append(item)

    closes_at = next((r["closes_at"] for r in events.get("rounds", []) if r["key"] == "early"), None)
    return {
        "items": items,
        "infos": infos,
        "problems": problems,
        "batches": make_batches(items),
        "closes_at": closes_at,
        "now": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
