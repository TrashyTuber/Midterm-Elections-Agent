"""Command-line entry points: python -m agent.cli <command>."""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from collections import Counter

import anthropic
from dotenv import load_dotenv

import json

from . import fetch, forecast, research, submit

CONCURRENCY = 8
SHIFT_EPS = 0.005  # shifts smaller than this count as "at market"


def cmd_fetch(args: argparse.Namespace) -> None:
    payload = fetch.fetch_events()
    path = fetch.save_events(payload)
    questions = payload.get("questions", [])
    statuses = Counter(q.get("status") for q in questions)
    offices = Counter(q.get("office") for q in questions)
    print(f"Saved {len(questions)} questions to {path}")
    print(f"  status: {dict(statuses)}")
    print(f"  office: {dict(offices)}")
    for r in payload.get("rounds", []):
        print(f"  round {r['key']}: closes {r['closes_at']} ({r['status']})")


def select_questions(args: argparse.Namespace) -> list[dict]:
    questions = fetch.open_questions(fetch.load_events())
    if args.tickers:
        wanted = [t.strip() for t in args.tickers.split(",") if t.strip()]
        by_ticker = {q["ticker"]: q for q in questions}
        missing = [t for t in wanted if t not in by_ticker]
        if missing:
            raise SystemExit(f"Unknown or non-open tickers: {', '.join(missing)}")
        return [by_ticker[t] for t in wanted]
    if args.office:
        questions = [q for q in questions if q["office"] == args.office]
    if getattr(args, "competitive", False):
        questions = [q for q in questions if is_competitive(q)]
    return questions[: args.limit] if args.limit else questions


def is_competitive(q: dict) -> bool:
    """Some outcome priced between 0.15 and 0.85: where research can plausibly matter."""
    return any(0.15 <= v <= 0.85 for v in forecast.prior(q).values())


async def _run(questions: list[dict], force: bool, no_search: bool) -> list[dict]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(q):
        async with sem:
            res = None if no_search else await research.research_event(client, q, force=force)
            rec = await forecast.forecast_event(
                client, q,
                brief=res["brief"] if res else None,
                trace=res["trace"] if res else None,
                force=force,
                research={k: v for k, v in res.items() if k not in ("brief", "trace")} if res else None,
            )
            status = "ERROR " + rec["error"] if rec["error"] else ("cached" if rec["cached"] else "ok")
            if res and res["errors"]:
                status += f" (research: {'; '.join(res['errors'])})"
            print(f"  {q['ticker']}: {status}")
            return rec

    return await asyncio.gather(*(one(q) for q in questions))


def cmd_run(args: argparse.Namespace) -> None:
    questions = select_questions(args)
    print(f"Forecasting {len(questions)} events with {forecast.forecast_model()}")
    records = asyncio.run(_run(questions, args.force, args.no_search))
    inp = sum(r["usage"].get("input_tokens", 0) for r in records if not r["cached"])
    out = sum(r["usage"].get("output_tokens", 0) for r in records if not r["cached"])
    cost = sum(r["cost_usd"] for r in records)
    print(f"Forecast ({forecast.forecast_model()}, new calls): {inp:,} in / {out:,} out, ${cost:.4f}")
    rs = [r["research"] for r in records if r.get("research")]
    if rs:
        rin = sum(x["usage"]["input_tokens"] for x in rs)
        rout = sum(x["usage"]["output_tokens"] for x in rs)
        rcost = sum(x["cost_usd"] for x in rs)
        credits = sum(x["tavily_credits"] for x in rs)
        web = sum(x.get("web_searches", 0) for x in rs)
        print(f"Research ({research.cheap_model()}, new calls): {rin:,} in / {rout:,} out, ${rcost:.4f}; "
              f"Tavily credits: {credits}; Claude web searches: {web}")
        cost += rcost
    print(f"Total est. model cost this run: ${cost:.4f}")
    print_report(records)


def cmd_report(args: argparse.Namespace) -> None:
    print_report(forecast.load_forecasts())


def print_report(records: list[dict]) -> None:
    rows = []
    deviating = 0
    for r in records:
        shifts = {k: r["probabilities"][k] - r["market"][k] for k in r["probabilities"]}
        if any(abs(s) >= SHIFT_EPS for s in shifts.values()):
            deviating += 1
        for i, (name, ours) in enumerate(r["probabilities"].items()):
            rows.append((
                r["ticker"] if i == 0 else "",
                name[:40],
                f"{r['market'][name]:.3f}",
                f"{ours:.3f}",
                f"{shifts[name]:+.3f}",
                r["confidence_in_deviation"] if i == 0 else "",
                ("guard" if r["guard_notes"] else "") if i == 0 else "",
            ))
    header = ("ticker", "outcome", "market", "ours", "shift", "confidence", "notes")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    n = len(records)
    if n:
        print(f"\n{deviating}/{n} events deviate from market (|shift| >= {SHIFT_EPS}).")
        if deviating > n / 3:
            print("WARNING: more than a third of events deviate; the prompt may be too loose.")


def cmd_submit(args: argparse.Namespace) -> None:
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    plan = submit.prepare(tickers)

    mode = "DRY RUN: nothing will be sent." if args.dry_run else "LIVE SUBMISSION."
    print(f"{mode} Now {plan['now']}, round 'early' closes {plan['closes_at']}.")
    for ticker, info in plan["infos"].items():
        status = "INVALID" if ticker in plan["problems"] else "valid"
        extra = f", trace {info['trace_entries']} entries / {info['trace_raw_bytes']:,} B raw"
        extra += " (truncated)" if info["trace_truncated"] else ""
        extra += f", {info['note']}" if info.get("note") else ""
        print(f"  {ticker}: {status}{extra}")
        for err in plan["problems"].get(ticker, []):
            print(f"      - {err}")
    print(f"{len(plan['items'])} valid, {len(plan['problems'])} invalid")
    for i, b in enumerate(plan["batches"], 1):
        size = len(json.dumps({"forecasts": b}).encode())
        traced = sum("trace_gzip_base64" in it for it in b)
        print(f"  request {i}: {len(b)} forecasts ({traced} with traces), {size:,} bytes")

    if args.show:
        item = next((it for it in plan["items"] if it["question"] == args.show), None)
        if item is None:
            raise SystemExit(f"{args.show} is not among the valid payloads")
        print(f"\n--- payload for {args.show} (as sent) ---")
        print(json.dumps({"forecasts": [item]}, indent=2))
        if "trace_gzip_base64" in item:
            print(f"\n--- trace_gzip_base64 decoded ---")
            print(json.dumps(submit.unpack_trace(item["trace_gzip_base64"]), indent=2))

    if args.dry_run:
        return
    if plan["closes_at"] and datetime.now(timezone.utc) >= datetime.fromisoformat(plan["closes_at"]):
        raise SystemExit("The deadline has passed; not sending.")
    if not plan["items"]:
        raise SystemExit("Nothing valid to send.")
    if not args.yes:
        answer = input(f"Send {len(plan['items'])} forecasts to Prophet Arena? Type SUBMIT to confirm: ")
        if answer.strip() != "SUBMIT":
            raise SystemExit("Not sent.")
    try:
        result = submit.send(plan)
    except submit.SetupError as e:
        raise SystemExit(f"STOP: {e}")
    print(f"Accepted {len(result['accepted'])}, rejected {len(result['rejected'])}, "
          f"receipts saved: {result['receipts']} (data/receipts/)")
    for t, err in result["rejected"]:
        print(f"  REJECTED {t}: {err}")
    verify(plan["items"])


def verify(sent: list[dict]) -> None:
    try:
        me = submit.fetch_me()
    except submit.SetupError as e:
        raise SystemExit(f"STOP: {e}")
    if not submit.find_on_file(me):
        # /election/me reports per-round counts, not per-forecast values.
        answered = {r["key"]: r.get("answered") for r in me.get("rounds", [])}
        print(f"/election/me answered per round: {answered} (sent {len(sent)} this time; "
              "the endpoint does not return values, so they cannot be compared one by one).")
        return
    problems = submit.compare_on_file(me, sent)
    if not problems:
        print(f"Verified: all {len(sent)} forecasts are on file with matching probabilities.")
    else:
        print(f"Verification found {len(problems)} issue(s) (raw response saved in data/receipts/):")
        for p in problems[:20]:
            print(f"  {p}")


def cmd_export(args: argparse.Namespace) -> None:
    """Write the upload-panel JSON file from the same validated payloads `submit` would send."""
    plan = submit.prepare([t.strip() for t in args.tickers.split(",")] if args.tickers else None)
    for t, errs in plan["problems"].items():
        print(f"  SKIPPED {t}: {'; '.join(errs)}")
    path = submit.export_upload(plan)
    print(f"Wrote {len(plan['items'])} forecasts to {path} ({path.stat().st_size / 1e6:.1f} MB). "
          f"Prices as of the last fetch/rebase; nothing was sent.")


def cmd_me(args: argparse.Namespace) -> None:
    try:
        me = submit.fetch_me()
    except submit.SetupError as e:
        raise SystemExit(f"STOP: {e}")
    on_file = submit.find_on_file(me)
    print(f"Entry: {me.get('entry', {}).get('name')} ({me.get('entry', {}).get('status')})")
    for r in me.get("rounds", []):
        print(f"  round {r.get('key')}: answered {r.get('answered')}, scored {r.get('scored')}, "
              f"brier {r.get('brier')}, closes {r.get('closes_at')}")
    print(f"Saved locally: {len(forecast.load_forecasts())} forecasts. Raw response saved in data/receipts/.")
    if not on_file:
        return  # per-forecast values are not part of this response
    saved = {r["ticker"]: r["probabilities"] for r in forecast.load_forecasts()}
    stale = [t for t, p in saved.items() if t in on_file and any(abs(on_file[t].get(k, -1) - v) > 1e-6 for k, v in p.items())]
    print(f"Saved locally but not on file: {len(set(saved) - set(on_file))}; on file but differs from local: {len(stale)}")


def cmd_rebase(args: argparse.Namespace) -> None:
    """Refetch prices and re-anchor every saved forecast to them. No model or search calls."""
    payload = fetch.fetch_events()
    by_ticker = {q["ticker"]: q for q in payload["questions"]}
    records = forecast.load_forecasts()
    if args.tickers:
        wanted = {t.strip() for t in args.tickers.split(",")}
        records = [r for r in records if r["ticker"] in wanted]
    updated, rows = [], []
    for r in records:
        q = by_ticker.get(r["ticker"])
        new, note = forecast.rebase(r, q) if q else (None, "skipped: not in /election")
        rows.append((r["ticker"], note))
        if new:
            updated.append(new)
    moved = [(t, n) for t, n in rows if not n.startswith("rebased, max change 0.000")]
    print(f"Prices as of {payload['questions'][0].get('prices_at') if payload['questions'] else '?'}: "
          f"{len(updated)} rebased, {len(rows) - len(updated)} skipped.")
    for t, n in moved:
        print(f"  {t}: {n}")
    if args.dry_run:
        print("DRY RUN: forecasts and data/events.json not changed.")
        return
    fetch.save_events(payload)
    for rec in updated:
        forecast.save_forecast(rec)
    print_report(updated)


def main() -> None:
    load_dotenv()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in log files
    parser = argparse.ArgumentParser(prog="agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="pull events to data/events.json").set_defaults(func=cmd_fetch)

    run = sub.add_parser("run", help="research + forecast events")
    run.add_argument("--limit", type=int, help="first N open events")
    run.add_argument("--tickers", help="comma-separated tickers")
    run.add_argument("--office", choices=["control", "senate", "governor", "house", "measure"], help="only this office type")
    run.add_argument("--competitive", action="store_true", help="only events with an outcome priced 0.15-0.85")
    run.add_argument("--force", action="store_true", help="ignore all caches (re-plan, re-search, re-forecast)")
    run.add_argument("--no-search", action="store_true", help="skip research; forecast from market prices only")
    run.set_defaults(func=cmd_run)

    sub.add_parser("report", help="table of saved forecasts").set_defaults(func=cmd_report)

    sm = sub.add_parser("submit", help="validate and send saved forecasts (asks to confirm)")
    sm.add_argument("--dry-run", action="store_true", help="build and validate payloads without sending")
    sm.add_argument("--tickers", help="comma-separated tickers (default: all saved forecasts)")
    sm.add_argument("--show", metavar="TICKER", help="print one full payload with its trace decoded")
    sm.add_argument("--yes", action="store_true", help="skip the SUBMIT confirmation prompt")
    sm.set_defaults(func=cmd_submit)

    sub.add_parser("me", help="what is on file at Prophet Arena vs local forecasts").set_defaults(func=cmd_me)

    ex = sub.add_parser("export", help="write a JSON file for the website's Upload JSON panel")
    ex.add_argument("--tickers", help="comma-separated tickers (default: all saved forecasts)")
    ex.set_defaults(func=cmd_export)

    rb = sub.add_parser("rebase", help="refetch prices and re-anchor saved forecasts (no model calls)")
    rb.add_argument("--tickers", help="comma-separated tickers (default: all saved forecasts)")
    rb.add_argument("--dry-run", action="store_true", help="show what would change without saving")
    rb.set_defaults(func=cmd_rebase)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
