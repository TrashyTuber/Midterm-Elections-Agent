"""Command-line entry points: python -m agent.cli <command>."""

import argparse
import asyncio
from collections import Counter

import anthropic
from dotenv import load_dotenv

import json

from . import fetch, forecast, submit

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
    return questions[: args.limit] if args.limit else questions


async def _run(questions: list[dict], force: bool) -> list[dict]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(q):
        async with sem:
            rec = await forecast.forecast_event(client, q, brief=None, force=force)
            status = "ERROR " + rec["error"] if rec["error"] else ("cached" if rec["cached"] else "ok")
            print(f"  {q['ticker']}: {status}")
            return rec

    return await asyncio.gather(*(one(q) for q in questions))


def cmd_run(args: argparse.Namespace) -> None:
    questions = select_questions(args)
    print(f"Forecasting {len(questions)} events with {forecast.forecast_model()}")
    records = asyncio.run(_run(questions, args.force))
    inp = sum(r["usage"].get("input_tokens", 0) for r in records if not r["cached"])
    out = sum(r["usage"].get("output_tokens", 0) for r in records if not r["cached"])
    cost = sum(r["cost_usd"] for r in records)
    print(f"Tokens (new calls): {inp:,} in / {out:,} out, est. cost ${cost:.4f}")
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
    if not args.dry_run:
        raise SystemExit("Live submission is not enabled yet. Use --dry-run.")
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    plan = submit.prepare(tickers)

    print(f"DRY RUN: nothing will be sent. Now {plan['now']}, round 'early' closes {plan['closes_at']}.")
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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="pull events to data/events.json").set_defaults(func=cmd_fetch)

    run = sub.add_parser("run", help="forecast events (research added in step 5)")
    run.add_argument("--limit", type=int, help="first N open events")
    run.add_argument("--tickers", help="comma-separated tickers")
    run.add_argument("--force", action="store_true", help="ignore cached model outputs")
    run.set_defaults(func=cmd_run)

    sub.add_parser("report", help="table of saved forecasts").set_defaults(func=cmd_report)

    sm = sub.add_parser("submit", help="validate (and later send) saved forecasts")
    sm.add_argument("--dry-run", action="store_true", help="build and validate payloads without sending")
    sm.add_argument("--tickers", help="comma-separated tickers (default: all saved forecasts)")
    sm.add_argument("--show", metavar="TICKER", help="print one full payload with its trace decoded")
    sm.set_defaults(func=cmd_submit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
