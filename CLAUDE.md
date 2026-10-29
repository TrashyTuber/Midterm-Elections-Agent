# Prophet Arena Election Agent

A forecasting agent for the Prophet Arena 2026 Midterms Election Challenge. It pulls 100 election events, researches each one, produces market-anchored probabilities with a rationale, and submits them through the Prophet Arena API.

## Context: the challenge

- Site: https://www.prophetarena.co/election
- Docs: https://www.prophetarena.co/docs/election
- Prophet Arena is a live AI forecasting benchmark on real prediction markets. The election challenge covers 100 events for November 3, 2026: chamber control, Senate, Governor, close House seats, and a few ballot measures.
- **Deadline: Monday, Nov 2, 2026, 5:00 PM ET (2026-11-02T22:00:00Z, 4:00 PM Central).** There is one round, keyed `early` in responses. Resubmitting replaces earlier forecasts for that event. Whatever is on file at the deadline gets scored against market prices frozen at that moment. The server clock decides, so do not cut it close.
- **Visibility:** the entry's display name and scores are public, and after the deadline its forecast numbers appear on event pages (by name, or anonymized if `anonymous` is true). Rationales, traces and email are never public. New entries are scored privately until a human at Prophet Arena approves them.

## One-time setup (done by the user in a browser, not by code)

1. Create a free account at https://prophetarena.co.
2. On https://prophetarena.co/election, open **Submit your forecasts** and register the entry (name, email, affiliation, agent display name).
3. Copy the `pa_weekly_...` key. It is shown once; if lost, regenerate at https://prophetarena.co/profile/api-keys.
4. Put it in `.env` as `ARENA_API_KEY`. Never commit it, print it, or log it.

If a request returns `election_entry_required` or `unauthorized`, stop and tell the user to redo this setup. Do not try to register or create accounts from code.
- **Scoring:** difficulty-adjusted Arena score, 0 to 100, based on Brier. Only answered events are scored, so skipping an event costs nothing. At least 10 settled answers are needed to be ranked. Kalshi and Polymarket appear on the leaderboard as reference rows, so the real goal is to beat the market average.

## API

Base URL: `https://api.prophetarena.co/api/v1`. This section is the complete contract. Do not fetch any prophetarena.co pages; everything needed is here.

### GET /election (no auth)

```json
{
  "rounds": [{"key": "early", "label": "Monday", "closes_at": "2026-11-02T22:00:00+00:00", "status": "published"}],
  "questions": [
    {
      "ticker": "SENATEGA-26",
      "title": "Georgia Senate winner?",
      "office": "senate",
      "state": "GA",
      "status": "open",
      "outcomes": [
        {"name": "Jon Ossoff", "party": "D", "kalshi": 0.955, "polymarket": 0.9615},
        {"name": "Mike Collins", "party": "R", "kalshi": 0.045, "polymarket": 0.037}
      ],
      "prices_at": "..."
    }
  ]
}
```

- `office` is one of `control`, `senate`, `governor`, `house`, `measure`.
- Outcomes are mutually exclusive (exactly one wins), **except** "Which ballot measures will pass in X?" questions (`office == "measure"` with several outcomes, e.g. `KXSTATEBALLOTMEASURE-NE26`, `KXSTATEBALLOTMEASURE-MT`). There each outcome is its own independent Yes/No market, so several can pass and prices do not sum to 1.
- A ballot measure may have a **single `Yes` outcome**. Its probability is forecast on its own and must NOT be renormalized to 1.
- Rule of thumb: any `office == "measure"` question is independent per outcome (never renormalized); every other office is mutually exclusive.
- Kalshi settles every question. `polymarket` is `null` where unmatched.
- Only forecast questions with `status == "open"`.

### POST /election/forecasts

Header: `Authorization: Bearer $ARENA_API_KEY` (format `pa_weekly_...`). Body:

```json
{"forecasts": [{
  "question": "SENATEGA-26",
  "probabilities": {"Jon Ossoff": 0.93, "Mike Collins": 0.07},
  "rationale": "...",
  "trace_gzip_base64": "..."
}]}
```

- `question`: the event `ticker`.
- `probabilities`: every outcome exactly once, keyed by outcome name, each in [0, 1]. The server never renormalizes, so mutually exclusive multi-outcome events must sum to 1 (ballot measures are independent per outcome and do not).
- `rationale`: up to 10,000 characters.
- `trace` or `trace_gzip_base64` (never both): the step log (see "Step log" below), up to 10 MB uncompressed. Always send gzipped: `base64(gzip(json.dumps(trace)))`. One request carries about 1.4 MB of gzip once base64-encoded, and JSON usually compresses 5 to 10x, so most traces fit easily. If a packed trace is still too big, truncate the longest search result texts in it but keep every step.
- **Sanitize before sending.** Remove any API keys or tokens from traces and rationales (scan for `pa_weekly_`, `sk-`, `tvly-` and the actual values from `.env`). Strip NUL characters (`\u0000`) from `rationale` text, since a request containing one is refused whole.
- `agent`: omit (implied by the weekly key).
- `anonymous`: default false; the latest request's value applies to everything on file. Ask the user before setting it.
- Limits: 100 forecasts per request, 2 MB per request, 120 submissions per minute. Batch under ~1.5 MB and at most 10 items with traces per request.
- On 429, sleep `Retry-After` seconds and resend the same batch. Submissions are upserts per round, so retrying on network errors and 5xx is safe.

Response:

```json
{
  "accepted": 1, "rejected": 0, "rounds": ["early"], "listed": true,
  "results": [
    {"question": "SENATEGA-26", "ok": true, "rounds": [
      {"round": "early", "ok": true, "replaces_previous": false,
       "receipt": {"fingerprint": "0x...", "code": "0x..."}}
    ]}
  ]
}
```

Each item is accepted or rejected on its own. Log every rejected item with its `error`. Save every `receipt` (from `results[].rounds[].receipt`) to `data/receipts/`; they cannot be recovered later.

Errors: `validation_failed` (422, fix and retry), `window_closed` (422, deadline passed), `not_found` (404), `duplicate` (same event twice in one request), `question_settled`, `election_entry_required` (403, user must register on the site), `forbidden` (403), `unauthorized` (401, bad key), `rate_limited` (429), HTTP 413 (request too large, split it).

### GET /election/me (auth)

The user's own record: what is on file with receipts, and after settlement, Brier per event vs the markets plus Arena score and rank. Use it to verify submissions landed.

### GET /election/questions/{ticker}

Public per-event page data: both venues' prices and price history.

## Architecture

```
fetch events -> for each event (5 to 10 concurrent):
    1. plan     cheap model writes 3 to 5 search queries
    2. search   run queries, keep top results
    3. digest   cheap model condenses results into a ~1k token evidence brief
    4. forecast strong model returns probabilities + rationale as JSON
    5. guard    code validates, clamps, renormalizes
-> save to data/forecasts/<ticker>.json
-> separate submit step reads saved files and POSTs them
```

Rules:

- **Market price is the prior.** `market_probs` = average of Kalshi and Polymarket for each outcome (fall back to whichever exists).
- **The guard is enforced in code, never trusted to the model.** If the model reports `confidence_in_deviation == "none"`, use the market prior exactly. Otherwise clamp each outcome to within ±0.03 of market for `"weak"` and ±0.10 for `"strong"`, clamp to [0.01, 0.99], then renormalize to sum to 1. **Exception: ballot measures (`office == "measure"`), whether a lone `Yes` or a multi-measure "which will pass" question, are clamped per outcome but never renormalized.** Outcome names in the output must exactly match the event's outcome names. If the model output is missing or invalid, use the market prior.
- **Polymarket can be null.** When it is, use Kalshi alone as the prior.
- **Research and submission are separate commands.** Never re-run searches just to resubmit.
- **Cache everything.** Search results and model outputs are saved per event so a single event can be rerun without touching the others.
- **Step log.** The `trace` field holds a simple activity log of what the pipeline did, as a list of entries: `{"type": "tool_call", "tool": "web_search", "args": {...}, "result": "..."}` for each search, `{"type": "brief", "text": "..."}` for the digest, `{"type": "final", "probabilities": {...}, "rationale": "..."}` for the answer, and `{"type": "guard", "before": {...}, "after": {...}}` when clamping changed anything. Nothing else goes in it. Never fabricate entries.

## Stack

- Python 3.11+, `httpx` (async), `anthropic` SDK, `python-dotenv`
- Search (`SEARCH_BACKEND`): `tavily` (default) runs plan, search, digest, behind a small `search(query) -> list[result]` interface. `claude` uses Anthropic's server-side web search tool (`web_search_20250305` on the cheap model, $10 per 1,000 searches) to search and write the brief in one call. If every Tavily search for an event fails (for example, out of credits), that event falls back to `claude` unless `SEARCH_FALLBACK=0`. The free Tavily plan is 1,000 credits a month; `SEARCH_DEPTH=basic` costs 1 per query, `advanced` 2.
- If the digest model refuses, the digest is retried with `FORECAST_MODEL`, then per query group, keeping the parts that succeed.
- Models (configurable via env): `CHEAP_MODEL` for plan and digest, `FORECAST_MODEL` for step 4, `FORECAST_EFFORT` (default `high`)
- Env vars in `.env` (gitignored): `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `ARENA_API_KEY`, `CHEAP_MODEL`, `FORECAST_MODEL`, optional `FORECAST_EFFORT`, `SEARCH_BACKEND`, `SEARCH_DEPTH`, `SEARCH_FALLBACK`

## Layout

```
agent/
  fetch.py       pull events, save data/events.json
  research.py    plan, search, digest per event
  forecast.py    forecasting prompt + guard
  submit.py      POST saved forecasts, save receipts
  prompts.py     all prompt templates
  cli.py         entry points
data/            events, cache, forecasts, receipts (gitignored)
```

CLI:

```
python -m agent.cli fetch
python -m agent.cli run --limit 5          # research + forecast first N events
python -m agent.cli run --tickers A,B      # specific events
python -m agent.cli run --office senate    # one office type
python -m agent.cli run --competitive      # only events with an outcome priced 0.15-0.85
python -m agent.cli run --no-search        # market-anchored forecasts, no research
python -m agent.cli report                 # table: ticker, market, ours, shift, confidence
python -m agent.cli rebase [--dry-run]     # refetch prices, re-anchor saved forecasts (no model calls)
python -m agent.cli submit --dry-run       # validate payloads without sending
python -m agent.cli submit                 # send; asks to type SUBMIT (--yes skips), then checks /me
python -m agent.cli me                     # what is on file vs local forecasts
```

`rebase` keeps each saved deviation from market, measured from the new price. If the market has moved toward our view, the deviation shrinks by that move (never flips sign), on the theory that the evidence is now priced in.

## Deadline runbook (Mon Nov 2, deadline 5:00 PM ET = 22:00 UTC)

Scored against market prices at the deadline, so stale anchors cost more than research earns.

1. Before Monday: run all events, review, `submit` once as insurance (resubmitting replaces).
2. Monday morning: `fetch`, then `run --competitive` (fresh research on races where it can matter; about 58 events). Review `report`, then `submit`.
3. Monday about 3:30 PM ET: `rebase`, `submit --dry-run`, `submit`, then `me`. Leave at least an hour of buffer; the server clock decides.
4. If anything breaks late: `fetch` then `run --no-search` puts market-anchored forecasts on file in minutes.

## Forecasting prompt (step 4)

```
You are forecasting the outcome of a US election event on November 3, 2026.

EVENT: {title}
OUTCOMES: {outcomes}
PREDICTION MARKET PRICES:
  Kalshi: {kalshi_prices}
  Polymarket: {polymarket_prices}

EVIDENCE BRIEF (from searches run today):
{brief}

The market prices are your starting point. They aggregate many informed traders
and are hard to beat. Only move away from them if the evidence brief contains
something specific the market may be underweighting, such as a very recent poll,
a late-breaking event, or a clear structural factor.

Rules:
- If the evidence gives no specific reason to disagree, return the market average.
- Every shift must be tied to a specific item in the brief. General partisan lean
  does not count; the market already knows that.
- Be skeptical of single polls, especially partisan or low-quality ones.
- If the evidence is something the market has likely already seen and priced in
  (for example, a widely reported poll or rating change), or you think it may be,
  return the market average.
- Never put any outcome above 0.99 or below 0.01.

Respond with JSON only:
{
  "probabilities": {"<outcome>": <float>, ...},
  "deviation_reason": "<the specific evidence that moved you, or 'none'>",
  "confidence_in_deviation": "none" | "weak" | "strong",
  "rationale": "<3 to 5 sentences explaining the forecast>"
}
```

## Planner prompt (step 1)

```
Write 3 to 5 web search queries to find the most recent information relevant to
forecasting: {title}. Prioritize polls from the last 2 weeks, early vote data,
and late-breaking news. Return a JSON list of strings only.
```

## Working rules

- Develop against 5 events. Do not run all 100 until outputs have been reviewed.
- Log token usage and estimated cost per run.
- Sanity check after each run: if more than about a third of events deviate from market, the prompt is too loose.
- Never commit `.env` or anything under `data/`.
- Do not submit to the live API unless explicitly asked. Default to `--dry-run`.
