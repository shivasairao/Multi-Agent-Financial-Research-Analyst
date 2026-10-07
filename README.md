# Multi-Agent Financial Research Analyst

A four-agent pipeline that researches stocks and writes a **fully cited** report, plus a Streamlit viewer to explore the results. Every claim in the report must point to a numbered source (`[S1]`, `[S2]`, ...), and a Critic agent audits the report against those sources before it is accepted.

Built on the Google Gemini API **free tier**, with market data from Yahoo Finance via `yfinance`.

> Educational / portfolio project. Not investment advice.

<!-- Add a screenshot or GIF of the Streamlit app here -->
<!-- ![demo](docs/demo.png) -->

---

## How it works

```
Question + tickers
        |
        v
   [ Planner ]  -> 3-5 focused research questions (JSON)
        |
        v
   [ Data agent ]  -> Gemini function calling over 4 tools:
        |             price stats, fundamentals, income statements, news
        |             Every tool result is registered as a numbered source [S1], [S2], ...
        v
   [ Analyst ]  -> writes a markdown report using ONLY the registered sources,
        |          citing every factual or numeric claim
        v
   [ Critic ]  -> 1) deterministic citation audit (no LLM)
        |          2) LLM verification of claims against the cited sources
        |
        +-- fail -> back to the Analyst with feedback (bounded revision loop)
        |
        v
   reports/<TICKERS>_<timestamp>.pkl  ->  Streamlit viewer
```

### Design choices

- **Grounding by construction:** the Analyst only sees a numbered source registry. Any citation that does not exist, and any numeric line with no citation, is flagged automatically by the deterministic audit before the LLM Critic even runs.
- **Prompt-injection hygiene:** news and other source text are fenced as `<untrusted_source>` data, and every agent is told never to follow instructions found inside it.
- **Free-tier resilience:**
  - calls are spaced out (`--min-interval`) to stay under per-minute limits
  - short 429s are waited out using the delay the server asks for
  - when a model's daily quota is used up, the pipeline switches to the next model in a fallback chain (free quotas are per model)
  - 5xx and network errors are retried
- **Full observability:** every LLM call and tool call is traced with tokens, latency and agent name, and shown in the app's "Trace & cost" tab.
- **Decoupled viewer:** the Streamlit app makes no LLM or market-data calls. It only reads a pickle, so it can be deployed without any API key.
- **Plain-types pickle contract:** `report_io.py` validates a schema-versioned payload made only of dicts, lists, strings and numbers, so old reports keep loading.

---

## Project structure

```
.
├── agentic_ai.py            # the pipeline (Planner -> Data -> Analyst -> Critic), CLI entry point
├── app.py                   # Streamlit viewer for saved reports
├── report_io.py             # pickle schema, save / load / validate
├── reports/                 # generated .pkl reports (read by app.py)
├── requirements.txt         # viewer dependencies (what Streamlit Cloud installs)
├── requirements-agent.txt   # pipeline dependencies (adds yfinance, google-genai, dotenv)
├── requirements-dev.txt     # pytest + ruff
└── .env                     # your GEMINI_API_KEY (never commit this)
```

---

## Quick start

### 1. Install

```bash
python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate        # macOS / Linux

pip install -r requirements-agent.txt
```

### 2. Add your free Gemini API key

Get a key at https://aistudio.google.com/apikey and save it in a file named `.env` next to `agentic_ai.py`:

```
GEMINI_API_KEY=your_key_here
```

### 3. Generate a report

```bash
python agentic_ai.py --tickers NVDA AMD --question "Compare growth, profitability and valuation. What are the key risks?"
```

The report is saved to `reports/<TICKERS>_<timestamp>.pkl`.

### 4. View it

```bash
streamlit run app.py
```

---

## CLI options

| Option | Default | Description |
|---|---|---|
| `--tickers` | `NVDA AMD` | Up to 4 tickers, space- or comma-separated |
| `--question` | growth / profitability / valuation / risks | The research question |
| `--period` | `1y` | Price window: `3mo`, `6mo`, `1y`, `2y`, `5y` |
| `--model` | `gemini-3.8-flash` | Primary model (also settable via `GEMINI_MODEL`) |
| `--fallback-models` | three other free models | Models to switch to when one runs out of free quota. Use `none` to disable |
| `--min-interval` | `4.0` | Minimum seconds between API calls (guards the per-minute limit) |
| `--thinking-level` | model default | `minimal`, `low`, `medium` or `high` |
| `--max-revisions` | `2` | Max Analyst rewrites after Critic feedback |
| `--out-dir` | `reports` | Where to write the pickle |

Model names on the free tier change often. If you see a "model not found" error, pass `--model` with a current free-tier model from Google AI Studio.

---

## The Streamlit viewer

| Tab | What it shows |
|---|---|
| **Report** | The cited markdown report, a download button, any open Critic issues and the research plan |
| **Chart** | Price performance for every ticker, rebased to 100 |
| **Evidence** | Every `[S#]` source with its kind, ticker, URL, retrieval time and full text |
| **Trace & cost** | Input/output tokens, wall time and a step-by-step trace of every agent action |

A banner at the top says whether the Critic passed the report or still has open issues.

### Deploying

Push the repo to GitHub, commit a sample report in `reports/`, and deploy `app.py` on [Streamlit Community Cloud](https://streamlit.io/cloud). It installs `requirements.txt` and needs no secrets, because the viewer never calls an API.

---

## Security notes

- **Never commit `.env`.** Add it to `.gitignore`. If a key was ever shared or committed, revoke it and create a new one.
- **Pickle files can execute code when loaded.** Only open reports you generated yourself or got from a source you trust, and never add an "upload a pickle" widget to a public app.
- **Free-tier data use:** Google may use free-tier prompts and outputs to improve its products. This project only sends public market data, but don't put confidential text into it.

---

## Limitations

- Data comes from Yahoo Finance through `yfinance`, an unofficial API that can rate-limit, change shape or return gaps. Missing data is reported as "data unavailable".
- The Critic is an LLM, so it can miss subtle errors. The deterministic audit only checks that citations exist and that numeric lines are cited, not that a number is correct. Reports that still have open issues are labelled in the app.
- Capped at 4 tickers per report to keep the report and the context window manageable.
- Free-tier quotas limit how many reports you can generate per day.

## Ideas for next steps

- Unit tests for the citation audit, ticker parsing and retry logic (`requirements-dev.txt` already includes pytest and ruff)
- Number-level verification: re-check each cited figure directly against the source text
- Add filings (SEC EDGAR) and earnings-call transcripts as sources
- Replace pickle with JSON for safer sharing

---

## Disclaimer

For educational purposes only. Nothing here is investment advice.
