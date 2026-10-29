"""Command-line entry points: python -m agent.cli <command>."""

import argparse
from collections import Counter

from . import fetch


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="pull events to data/events.json").set_defaults(func=cmd_fetch)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
