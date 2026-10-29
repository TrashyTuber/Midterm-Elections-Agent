"""Steps 1-3 per event: plan queries, run searches, digest results into an evidence brief."""

import asyncio
import hashlib
import json
import os
import re
from datetime import date

import anthropic
from tavily import AsyncTavilyClient

from . import prompts
from .fetch import DATA_DIR
from .forecast import cost_usd

CACHE_DIR = DATA_DIR / "cache"
MAX_QUERIES = 5
RESULTS_PER_QUERY = 5
SNIPPET_CHARS = 1500  # per result, in the digest prompt
TAVILY_CREDITS = {"basic": 1, "advanced": 2}


def cheap_model() -> str:
    return os.environ.get("CHEAP_MODEL", "claude-haiku-5-5")


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _outcomes(question: dict) -> str:
    return ", ".join(f"{o['name']} ({o['party']})" if o.get("party") else o["name"] for o in question["outcomes"])


# ---------- search interface (swappable) ----------

async def search(query: str) -> list[dict]:
    """search(query) -> list of {title, url, content, published_date, score}. Tavily backend."""
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = await client.search(
        query,
        search_depth=os.environ.get("SEARCH_DEPTH", "advanced"),
        time_range="month",
        max_results=RESULTS_PER_QUERY,
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "published_date": r.get("published_date"),
            "score": r.get("score"),
        }
        for r in resp.get("results", [])
    ]


# ---------- cheap model helper ----------

async def _cheap_call(client: anthropic.AsyncAnthropic, prompt: str, effort: str) -> tuple[str, dict, str | None]:
    model = cheap_model()
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        return "", {}, f"{type(e).__name__}: {e}"
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    if resp.stop_reason in ("refusal", "max_tokens"):
        return "", usage, resp.stop_reason
    return next((b.text for b in resp.content if b.type == "text"), ""), usage, None


def _parse_queries(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in data if isinstance(q, str) and q.strip()][:MAX_QUERIES]


# ---------- per-event pipeline ----------

async def research_event(client: anthropic.AsyncAnthropic, question: dict, force: bool = False) -> dict:
    """Return {"brief", "trace", "usage": {model: {...}}, "cost_usd", "tavily_credits", "errors"}."""
    ticker = question["ticker"]
    cdir = CACHE_DIR / ticker
    today = date.today().isoformat()
    model = cheap_model()
    usage = {"input_tokens": 0, "output_tokens": 0}
    cost, credits, errors, trace = 0.0, 0, [], []

    def add_usage(u):
        nonlocal cost
        usage["input_tokens"] += u.get("input_tokens", 0)
        usage["output_tokens"] += u.get("output_tokens", 0)
        cost += cost_usd(model, u)

    # 1. plan
    plan_prompt = prompts.PLANNER_PROMPT.format(title=question["title"]) + "\n" + prompts.PLANNER_CONTEXT.format(today=today)
    plan_path = cdir / f"plan-{_key(model, plan_prompt)}.json"
    plan = None if force else _read(plan_path)
    if plan is None:
        text, u, err = await _cheap_call(client, plan_prompt, effort="low")
        add_usage(u)
        queries = _parse_queries(text)
        if err or not queries:
            errors.append(f"plan: {err or 'unparseable output'}; using fallback queries")
            queries = [f"{question['title'].rstrip('?')} 2026 poll", f"{question['title'].rstrip('?')} 2026 latest news"]
        plan = {"queries": queries, "raw": text}
        _write(plan_path, plan)

    # 2. search (each query cached on its own)
    async def run_query(q: str):
        path = cdir / f"search-{_key(q)}.json"
        hit = None if force else _read(path)
        if hit is not None:
            return q, hit, 0
        try:
            results = await search(q)
        except Exception as e:  # noqa: BLE001 - search failures should not kill the event
            return q, {"error": f"{type(e).__name__}: {e}", "results": []}, 0
        hit = {"query": q, "date": today, "results": results}
        _write(path, hit)
        return q, hit, TAVILY_CREDITS.get(os.environ.get("SEARCH_DEPTH", "advanced"), 1)

    searched = await asyncio.gather(*(run_query(q) for q in plan["queries"]))
    seen, kept = set(), []
    for q, hit, c in searched:
        credits += c
        if hit.get("error"):
            errors.append(f"search {q!r}: {hit['error']}")
        trace.append({
            "type": "tool_call",
            "tool": "web_search",
            "args": {"query": q},
            "result": json.dumps(hit.get("results", []), ensure_ascii=False) if not hit.get("error") else hit["error"],
        })
        for r in hit.get("results", []):
            if r["url"] not in seen:
                seen.add(r["url"])
                kept.append(r)

    # 3. digest
    if not kept:
        errors.append("no search results; no brief")
        return {"brief": None, "trace": trace, "usage": usage, "model": model,
                "cost_usd": cost, "tavily_credits": credits, "errors": errors, "queries": plan["queries"]}

    results_text = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\nPublished: {r.get('published_date') or 'unknown'}\n{r['content'][:SNIPPET_CHARS]}"
        for i, r in enumerate(kept, 1)
    )
    digest_prompt = prompts.DIGEST_PROMPT.format(today=today, title=question["title"],
                                                 outcomes=_outcomes(question), results=results_text)
    digest_path = cdir / f"digest-{_key(model, digest_prompt)}.json"
    digest = None if force else _read(digest_path)
    if digest is None:
        text, u, err = await _cheap_call(client, digest_prompt, effort="medium")
        add_usage(u)
        if err or not text.strip():
            errors.append(f"digest: {err or 'empty output'}")
            digest = {"brief": None}
        else:
            digest = {"brief": text.strip()}
            _write(digest_path, digest)
    if digest["brief"]:
        trace.append({"type": "brief", "text": digest["brief"]})

    return {"brief": digest["brief"], "trace": trace, "usage": usage, "model": model,
            "cost_usd": cost, "tavily_credits": credits, "errors": errors, "queries": plan["queries"]}
