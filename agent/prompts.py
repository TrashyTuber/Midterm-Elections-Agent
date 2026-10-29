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
- If the evidence is something the market has likely already seen and priced in
  (for example, a widely reported poll or rating change), or you think it may be,
  return the market average.
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

# Appended to PLANNER_PROMPT so queries target this cycle, not 2024.
PLANNER_CONTEXT = "Today is {today}. The election is on November 3, 2026."

DIGEST_PROMPT = """\
Today is {today}. You are preparing an evidence brief for a forecaster predicting
this US election event on November 3, 2026.

EVENT: {title}
OUTCOMES: {outcomes}

Below are web search results gathered today. Condense them into a brief of at most
about 700 words that a forecaster can act on.

Include, when present:
- Polls: pollster, field dates, sample size/type (LV/RV), and the numbers. Note
  partisan sponsors and low-quality or internal polls.
- Early vote or turnout data, with dates.
- Late-breaking news: scandals, withdrawals, endorsements, court rulings, ad spending,
  debates, ballot or rule changes.
- Expert ratings (Cook, Sabato, Inside Elections) and any recent rating changes.

Rules:
- Only state what the results say. Do not add outside knowledge or a forecast of your own.
- Give each item its date and source URL. Put the most recent and most decision-relevant
  items first.
- Say clearly if the results are old, thin, off-topic, or about a different race or year.
- Ignore prediction market prices; the forecaster already has them.

SEARCH RESULTS:
{results}"""

# SEARCH_BACKEND=claude: one call that searches with the web search tool and writes the brief.
CLAUDE_RESEARCH_PROMPT = """\
Today is {today}. You are preparing an evidence brief for a forecaster predicting
this US election event on November 3, 2026.

EVENT: {title}
OUTCOMES: {outcomes}

Use the web search tool 3 to 5 times to find the most recent information. Prioritize
polls from the last 2 weeks, early vote data, and late-breaking news. Then write a
brief of at most about 700 words that a forecaster can act on.

Include, when present:
- Polls: pollster, field dates, sample size/type (LV/RV), and the numbers. Note
  partisan sponsors and low-quality or internal polls.
- Early vote or turnout data, with dates.
- Late-breaking news: scandals, withdrawals, endorsements, court rulings, ad spending,
  debates, ballot or rule changes.
- Expert ratings (Cook, Sabato, Inside Elections) and any recent rating changes.

Rules:
- Only state what the search results say. Do not add outside knowledge or a forecast of your own.
- Give each item its date and source. Put the most recent and most decision-relevant items first.
- Say clearly if the results are old, thin, off-topic, or about a different race or year.
- Ignore prediction market prices; the forecaster already has them."""
