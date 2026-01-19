"""
Issue tracking LLM pipeline for Ecoplus.

Input schema (WhatsApp-style JSONL; one JSON object per line):
- chat_id
- message_id
- timestamp
- sender_number
- sender_name
- message_text

Output:
- DataFrame returned
- JSONL + CSV saved to output_dir with original fields + issue fields + debug fields
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from llm_router import call_llm_json


# --------- Config ---------

ISSUE_CATEGORIES = [
    "Delivery Delay",
    "Product Quality",
    "Stock Issues",
    "Service Issues",
    "Fulfilment Error",
    "Payments",
    "Other",
]

RESOLUTION_SCORES = [
    "Not resolved yet",
    "Satisfied",
    "Neutral",
    "Dissatisfied",
]

OUTPUT_KEYS = [
    "issue",
    "customer",
    "created_by",
    "created_at",
    "to_inform",
    "assigned_to",
    "resolved_at",
    "status_to_close",
    "closed_at",
    "resolution_score",
    "comments",
]

REQUIRED_INPUT_COLS = [
    "chat_id",
    "message_id",
    "timestamp",
    "sender_number",
    "sender_name",
    "message_text",
]

SYSTEM_PROMPT = """
You are an assistant helping Ecoplus, a B2B pipes/foundry SME in Singapore,
convert raw WhatsApp messages into a structured issue-tracking row.

You will receive ONE WhatsApp message at a time, in JSON with these fields:
- chat_id: conversation ID
- message_id: message ID
- timestamp: when the message was sent (string)
- sender_number: customer's phone number
- sender_name: customer's company name or contact name
- message_text: full WhatsApp message text

Your job is to infer, as far as possible, the ISSUE and basic workflow fields
used by Ecoplus' internal issue tracker.

CRITICAL RULES:
- Output ONLY a single valid JSON object (no surrounding text).
- Do NOT invent specific dates or times that are not given.
- If a field cannot be inferred from the message, use an empty string "".
- Do NOT make up specific person names (Andy, Darren, etc.).
- Use generic roles (e.g. "AM", "Cust Svc", "Purchase", "Finance", "Site") where needed.
- Keep strings short and suitable for a table cell.

The JSON object MUST have exactly these keys:

- "issue": one of the following categories
  ["Delivery Delay", "Product Quality", "Stock Issues",
   "Service Issues", "Fulfilment Error", "Payments", "Other"].

- "customer": short name for the customer, usually from sender_name.

- "created_by": who raised the issue.
  For WhatsApp client messages, use "Customer" or "".

- "created_at": when the issue was raised.
  For WhatsApp messages, you may copy the timestamp string or leave "".

- "to_inform": which roles should be notified, as a short comma-separated string.
  Use roles such as "AM", "Cust Svc", "Purchase", "Finance", "Site".
  Example: "AM, Cust Svc".

- "assigned_to": the primary role that should handle this issue.
  Use roles like "AM", "Cust Svc", "Purchase", "Finance", "Site".

- "resolved_at": keep as "" (WhatsApp messages alone cannot tell this).

- "status_to_close": who should confirm closure, as a role or "".

- "closed_at": keep as "".

- "resolution_score": one of
  ["Not resolved yet", "Satisfied", "Neutral", "Dissatisfied"].
  For a single complaint message, usually "Not resolved yet".

- "comments": a short internal note (1–2 lines) summarising
  what Ecoplus needs to do or follow up.

Ensure the JSON is syntactically valid and can be parsed by json.loads().
""".strip()


# --------- Paths ---------

def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    return start


HERE = Path(__file__).resolve().parent
ROOT = find_project_root(HERE)

INPUT_PATH_DEFAULT = ROOT / "data" / "issue_tracking" / "val" / "claude_48.jsonl"
OUTPUT_DIR_DEFAULT = (ROOT / "models" / "issue_tracking" / "results")


# --------- Helpers ---------

def _ensure_output_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce EXACT output keys. Missing -> "", extra keys dropped.
    """
    out: Dict[str, Any] = {}
    for k in OUTPUT_KEYS:
        v = d.get(k, "")
        out[k] = "" if v is None else str(v)
    return out


def schema_ok(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Non-expert schema validation:
    - exact keys present (at least required ones)
    - enums valid where applicable
    """
    missing = [k for k in OUTPUT_KEYS if k not in d]
    if missing:
        return False, missing

    bad: List[str] = []
    issue = str(d.get("issue", "")).strip()
    if issue and issue not in ISSUE_CATEGORIES:
        bad.append("issue")

    rs = str(d.get("resolution_score", "")).strip()
    if rs and rs not in RESOLUTION_SCORES:
        bad.append("resolution_score")

    return (len(bad) == 0), bad


def load_input_df(input_path: Path) -> pd.DataFrame:
    """
    Loads WhatsApp JSONL and checks required columns exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_json(input_path, lines=True)

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    return df


def call_issue_model(payload: Dict[str, Any], *, model: str, max_retries: int = 3, retry_sleep: float = 2.0) -> Dict[str, Any]:
    """
    Provider-agnostic via llm_router. Returns dict or {"_error": "..."}.
    """
    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            data = call_llm_json(
                text=json.dumps(payload, ensure_ascii=False),
                system_prompt=SYSTEM_PROMPT,
                model=model,
            )
            if not isinstance(data, dict):
                return {"_error": "NON_DICT_OUTPUT", "_raw": str(data)}
            return data
        except Exception as e:
            last_error = f"Attempt {attempt} failed: {e}"
            time.sleep(retry_sleep)

    return {"_error": last_error or "Unknown error"}


# --------- Runner ---------

def run_issue_tracking_pipeline(
    input_path: Path = INPUT_PATH_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    model: str = "gpt-5",
    test_mode: bool = True,
    max_rows: int = 30,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """
    Run issue tracking pipeline on WhatsApp JSONL.

    Saves:
    - issue_tracking_results_<model>.jsonl
    - issue_tracking_results_<model>.csv
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input from: {input_path}")
    df = load_input_df(input_path)

    if test_mode:
        df = df.head(max_rows)
        print(f"TEST_MODE=True → processing first {len(df)} rows only")

    records: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        payload = {
            "chat_id": str(row.get("chat_id", "")),
            "message_id": str(row.get("message_id", "")),
            "timestamp": str(row.get("timestamp", "")),
            "sender_number": str(row.get("sender_number", "")),
            "sender_name": str(row.get("sender_name", "")),
            "message_text": str(row.get("message_text", "")),
        }

        print(f"[{i+1}/{len(df)}] Processing message_id={payload['message_id']} ...")

        out = call_issue_model(payload, model=model)

        # Normalize / validate
        if "_error" not in out:
            out = _ensure_output_keys(out)
            ok, bad_fields = schema_ok(out)
        else:
            ok, bad_fields = (False, ["_error"])

        rec = payload.copy()
        rec.update({
            "model": model,
            "raw_model_output": json.dumps(out, ensure_ascii=False),
            "schema_ok": bool(ok),
            "schema_bad_fields": bad_fields,
            "error": out.get("_error", "") if isinstance(out, dict) else "UNKNOWN_ERROR",
        })

        # explode issue fields for easy CSV filtering
        if isinstance(out, dict) and "_error" not in out:
            for k in OUTPUT_KEYS:
                rec[k] = out.get(k, "")
        else:
            for k in OUTPUT_KEYS:
                rec[k] = ""

        records.append(rec)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    out_df = pd.DataFrame(records)

    model_safe = model.replace("/", "_").replace(".", "_").replace("-", "_")
    jsonl_out = output_dir / f"issue_tracking_results_{model_safe}.jsonl"
    csv_out = output_dir / f"issue_tracking_results_{model_safe}.csv"

    print(f"Writing JSONL to: {jsonl_out}")
    out_df.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)

    print(f"Writing CSV to: {csv_out}")
    out_df.to_csv(csv_out, index=False)

    print("Done.")
    return out_df


if __name__ == "__main__":
    run_issue_tracking_pipeline(test_mode=True)