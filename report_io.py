"""Pickle contract between the agent pipeline (writer) and the Streamlit app (reader).

The pickle contains ONLY plain Python types (dict/list/str/float/bool), never custom
classes or DataFrames. That keeps it independent of this codebase's class definitions
and of pandas versions, so an old pickle keeps loading.

SECURITY: pickle can execute arbitrary code when loaded. Only load files you
generated yourself or that come from a source you trust. Never expose an
"upload a pickle" widget in a public app.
"""

from __future__ import annotations

import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

REQUIRED_KEYS = {
    "schema_version",
    "created_at",
    "model",
    "question",
    "tickers",
    "period",
    "report",
    "passed",
    "issues",
    "revisions",
    "plan",
    "sources",
    "trace",
    "totals",
    "prices",
}


def validate_payload(payload: Any) -> dict[str, Any]:
    """Raise ValueError if the object is not a report payload this app understands."""
    if not isinstance(payload, dict):
        raise ValueError("Report file does not contain a dict.")
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Report is missing keys: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {payload['schema_version']} (expected {SCHEMA_VERSION}).")
    return payload


def save_report(payload: dict[str, Any], out_dir: str | Path) -> Path:
    """Validate and write a payload to <out_dir>/<TICKERS>_<UTC timestamp>.pkl."""
    validate_payload(payload)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tickers = "-".join(payload["tickers"])
    tickers = re.sub(r"[^A-Za-z0-9\-]", "", tickers)[:40] or "report"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"{tickers}_{stamp}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return path


def load_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a report pickle (trusted files only, see module docstring)."""
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as exc:
        raise ValueError(f"Could not read pickle {path}: {exc}") from exc
    return validate_payload(payload)


def list_reports(report_dir: str | Path) -> list[Path]:
    """All .pkl files in the directory, newest first."""
    d = Path(report_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
