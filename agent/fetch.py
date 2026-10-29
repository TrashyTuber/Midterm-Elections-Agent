"""Pull election events from the Prophet Arena API and save them to data/events.json."""

import json
from pathlib import Path

import httpx

BASE_URL = "https://api.prophetarena.co/api/v1"
DATA_DIR = Path("data")
EVENTS_PATH = DATA_DIR / "events.json"


def fetch_events() -> dict:
    """GET /election (no auth) and return the parsed response."""
    resp = httpx.get(f"{BASE_URL}/election", timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_events(payload: dict, path: Path = EVENTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_events(path: Path = EVENTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m agent.cli fetch` first")
    return json.loads(path.read_text())


def open_questions(payload: dict) -> list[dict]:
    """Only questions with status == "open" may be forecast."""
    return [q for q in payload.get("questions", []) if q.get("status") == "open"]


def market_probs(question: dict) -> dict[str, float]:
    """Prior: average of Kalshi and Polymarket per outcome, falling back to whichever exists."""
    probs = {}
    for o in question["outcomes"]:
        prices = [p for p in (o.get("kalshi"), o.get("polymarket")) if p is not None]
        probs[o["name"]] = sum(prices) / len(prices) if prices else None
    return probs
