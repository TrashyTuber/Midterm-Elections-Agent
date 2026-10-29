"""Steps 1-3 per event: plan queries, run searches, digest results into an evidence brief.

Two search backends (SEARCH_BACKEND):
  tavily  plan -> Tavily search -> digest (default). Falls back to claude for an event
          if every Tavily search fails (e.g. out of credits), unless SEARCH_FALLBACK=0.
  claude  one cheap-model call with Anthropic's server-side web search tool that
          searches and writes the brief with citations.
"""

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
from .forecast import cost_usd, forecast_model

CACHE_DIR = DATA_DIR / "cache"
MAX_QUERIES = 5
RESULTS_PER_QUERY = 5
SNIPPET_CHARS = 1500  # per result, in the digest prompt
TAVILY_CREDITS = {"basic": 1, "advanced": 2}
WEB_SEARCH_USD = 0.01  # Anthropic web search: $10 per 1,000 searches
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_QUERIES}
MAX_CONTINUATIONS = 3


def cheap_model() -> str:
    return os.environ.get("CHEAP_MODEL", "claude-haiku-5-5")


def search_backend() -> str:
    return os.environ.get("SEARCH_BACKEND", "tavily")


def _key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _outcomes(question: dict) -> str:
    return ", ".join(f"{o['name']} ({o['party']})" if o.get("party") else o["name"] for o in question["outcomes"])


class Tally:
    """Token, dollar and search-credit accounting for one event's research."""

    def __init__(self):
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.cost = 0.0
        self.tavily_credits = 0
        self.web_searches = 0

    def add(self, model: str, usage: dict):
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)
        self.cost += cost_usd(model, usage)

    def add_web_searches(self, n: int):
        self.web_searches += n
        self.cost += n * WEB_SEARCH_USD


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


# ---------- model helper ----------

async def _call(client: anthropic.AsyncAnthropic, model: str, prompt: str, effort: str) -> tuple[str, dict, str | None]:
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


def _result(brief, trace, tally, errors, queries, backend):
    return {"brief": brief, "trace": trace, "usage": tally.usage, "model": cheap_model(),
            "cost_usd": tally.cost, "tavily_credits": tally.tavily_credits,
            "web_searches": tally.web_searches, "backend": backend,
            "errors": errors, "queries": queries}


# ---------- tavily backend: plan -> search -> digest ----------

async def _research_tavily(client, question: dict, force: bool) -> dict:
    cdir = CACHE_DIR / question["ticker"]
    today = date.today().isoformat()
    model = cheap_model()
    tally, errors, trace = Tally(), [], []

    # 1. plan
    plan_prompt = prompts.PLANNER_PROMPT.format(title=question["title"]) + "\n" + prompts.PLANNER_CONTEXT.format(today=today)
    plan_path = cdir / f"plan-{_key(model, plan_prompt)}.json"
    plan = None if force else _read(plan_path)
    if plan is None:
        text, u, err = await _call(client, model, plan_prompt, effort="low")
        tally.add(model, u)
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
        tally.tavily_credits += c
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
        res = _result(None, trace, tally, errors, plan["queries"], "tavily")
        res["all_searches_failed"] = all(hit.get("error") for _, hit, _ in searched)
        return res

    def digest_prompt(results: list[dict]) -> str:
        text = "\n\n".join(
            f"[{i}] {r['title']}\nURL: {r['url']}\nPublished: {r.get('published_date') or 'unknown'}\n{r['content'][:SNIPPET_CHARS]}"
            for i, r in enumerate(results, 1)
        )
        return prompts.DIGEST_PROMPT.format(today=today, title=question["title"], outcomes=_outcomes(question), results=text)

    full_prompt = digest_prompt(kept)
    digest_path = cdir / f"digest-{_key(model, full_prompt)}.json"
    digest = None if force else _read(digest_path)
    if digest is None:
        digest = {"brief": None}
        # Cheap model first; if it refuses or fails, retry once with the forecast model.
        for m in dict.fromkeys([model, forecast_model()]):
            text, u, err = await _call(client, m, full_prompt, effort="medium")
            tally.add(m, u)
            if not err and text.strip():
                digest = {"brief": text.strip(), "model": m}
                break
            errors.append(f"digest ({m}): {err or 'empty output'}")
        if digest["brief"] is None:
            # Something in the results trips a refusal. Digest each query's results on its
            # own and keep the parts that succeed, so one bad result only costs its group.
            groups = [(q, [r for r in hit.get("results", [])]) for q, hit, _ in searched if hit.get("results")]
            parts = await asyncio.gather(*(_call(client, model, digest_prompt(rs), effort="medium") for _, rs in groups))
            kept_parts = []
            for (q, _), (text, u, err) in zip(groups, parts):
                tally.add(model, u)
                if not err and text.strip():
                    kept_parts.append(f"### Results for: {q}\n\n{text.strip()}")
                else:
                    errors.append(f"digest part {q!r}: {err or 'empty output'}")
            if kept_parts:
                digest = {"brief": "\n\n".join(kept_parts), "model": model, "split": True}
        if digest["brief"]:
            _write(digest_path, digest)
    if digest["brief"]:
        trace.append({"type": "brief", "text": digest["brief"]})
    return _result(digest["brief"], trace, tally, errors, plan["queries"], "tavily")


# ---------- claude backend: server-side web search + brief in one call ----------

def _claude_trace(content: list) -> tuple[list, str]:
    """Turn response blocks into step-log entries and the brief text. Only what came back."""
    blocks = [b.model_dump() for b in content]
    results = {b["tool_use_id"]: b["content"] for b in blocks if b["type"] == "web_search_tool_result"}
    cited = {}
    for b in blocks:
        if b["type"] == "text":
            for c in b.get("citations") or []:
                if c.get("url"):
                    cited.setdefault(c["url"], []).append(c.get("cited_text", ""))

    trace = []
    for b in blocks:
        if b["type"] == "server_tool_use" and b.get("name") == "web_search":
            res = results.get(b["id"])
            if isinstance(res, list):
                payload = [{"title": r.get("title"), "url": r.get("url"), "page_age": r.get("page_age"),
                            "cited_text": cited.get(r.get("url"), [])} for r in res]
                result = json.dumps(payload, ensure_ascii=False)
            else:
                result = f"error: {(res or {}).get('error_code', 'no result')}"
            trace.append({"type": "tool_call", "tool": "web_search", "args": {"query": b["input"].get("query")}, "result": result})
    brief = "".join(b["text"] for b in blocks if b["type"] == "text").strip()
    return trace, brief


async def _research_claude(client, question: dict, force: bool) -> dict:
    cdir = CACHE_DIR / question["ticker"]
    today = date.today().isoformat()
    model = cheap_model()
    tally, errors = Tally(), []
    prompt = prompts.CLAUDE_RESEARCH_PROMPT.format(today=today, title=question["title"], outcomes=_outcomes(question))
    path = cdir / f"claude-research-{_key(model, prompt)}.json"
    cached = None if force else _read(path)
    if cached is not None:
        return _result(cached["brief"], cached["trace"], tally, errors, cached["queries"], "claude")

    messages = [{"role": "user", "content": prompt}]
    content = []
    try:
        for _ in range(MAX_CONTINUATIONS + 1):
            resp = await client.messages.create(
                model=model, max_tokens=16000, output_config={"effort": "medium"},
                tools=[WEB_SEARCH_TOOL], messages=messages,
            )
            tally.add(model, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})
            stu = getattr(resp.usage, "server_tool_use", None)
            tally.add_web_searches(getattr(stu, "web_search_requests", 0) or 0)
            content += resp.content
            if resp.stop_reason != "pause_turn":
                break
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
        if resp.stop_reason in ("refusal", "max_tokens"):
            errors.append(f"claude research: {resp.stop_reason}")
    except anthropic.APIError as e:
        errors.append(f"claude research: {type(e).__name__}: {e}")

    trace, brief = _claude_trace(content)
    queries = [e["args"]["query"] for e in trace]
    if brief:
        trace.append({"type": "brief", "text": brief})
        _write(path, {"brief": brief, "trace": trace, "queries": queries})
    else:
        errors.append("claude research: no brief")
    return _result(brief or None, trace, tally, errors, queries, "claude")


# ---------- entry point ----------

async def research_event(client: anthropic.AsyncAnthropic, question: dict, force: bool = False) -> dict:
    """Return {"brief", "trace", "usage", "cost_usd", "tavily_credits", "web_searches", "backend", "errors", "queries"}."""
    if search_backend() == "claude":
        return await _research_claude(client, question, force)
    res = await _research_tavily(client, question, force)
    if res.pop("all_searches_failed", False) and os.environ.get("SEARCH_FALLBACK", "1") != "0":
        fb = await _research_claude(client, question, force)
        fb["errors"] = res["errors"] + ["tavily failed; used claude web search"] + fb["errors"]
        fb["cost_usd"] += res["cost_usd"]
        return fb
    return res
