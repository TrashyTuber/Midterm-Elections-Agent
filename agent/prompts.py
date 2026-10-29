"""All prompt templates. Text follows CLAUDE.md; edit there first, then here."""

FORECAST_PROMPT = """\
You are forecasting the outcome of a US election event on November 3, 2026.

EVENT: {title}
OUTCOMES: {outcomes}
{structure_note}
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
- Never put any outcome above 0.99 or below 0.01.

Respond with JSON only:
{{
  "probabilities": {{"<outcome>": <float>, ...}},
  "deviation_reason": "<the specific evidence that moved you, or 'none'>",
  "confidence_in_deviation": "none" | "weak" | "strong",
  "rationale": "<3 to 5 sentences explaining the forecast>"
}}"""

# Inserted as {structure_note}: tells the model whether outcomes are exclusive.
EXCLUSIVE_NOTE = "Exactly one outcome wins, so the probabilities must sum to 1.\n"
INDEPENDENT_NOTE = (
    "Each outcome is a separate ballot measure with its own independent Yes/No market.\n"
    "Give the probability that each one passes. Several can pass, so the probabilities\n"
    "do NOT need to sum to 1.\n"
)

NO_BRIEF = "(No evidence brief: search is not enabled for this run. Treat this as no new evidence.)"

PLANNER_PROMPT = """\
Write 3 to 5 web search queries to find the most recent information relevant to
forecasting: {title}. Prioritize polls from the last 2 weeks, early vote data,
and late-breaking news. Return a JSON list of strings only."""
