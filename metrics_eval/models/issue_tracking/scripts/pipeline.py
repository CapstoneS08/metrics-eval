"""
Issue tracking LLM pipeline for Ecoplus.

Takes WhatsApp-style client messages (JSONL) and turns each into a
single issue row in Ecoplus' template.

Input file schema (one JSON object per line):
- chat_id
- message_id
- timestamp
- sender_number
- sender_name
- message_text

Output: DataFrame + JSONL + CSV with original fields + issue fields.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI


# --------- Config ---------

ISSUE_CATEGORIES = [
    "Delivery Delay",
    "Product Quality",
    "Stock Issues",
    "Service Issues",
    "Fulfilment Error",
    "Payments",
]

RESOLUTION_SCORES = [
    "Not resolved yet",
    "Satisfied",
    "Neutral",
    "Dissatisfied",
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
- Use generic roles (e.g. "AM", "Cust Svc", "Purchase", "Site") where needed.
- Keep strings short and suitable for a table cell.

The JSON object MUST have exactly these keys:

- "issue": one of the following categories
  ["Delivery Delay", "Product Quality", "Stock Issues",
   "Service Issues", "Fulfilment Error", "Payments", "Other"].
  Pick the best fit, or "Other" if none apply.

- "customer": short name for the customer, usually from sender_name.
  Example: "ABC Engineering", "Hock Seng Trading".

- "created_by": who raised the issue.
  For WhatsApp client messages, use "Customer" or empty string "".

- "created_at": when the issue was raised.
  For WhatsApp messages, you may copy the timestamp string or leave "".

- "to_inform": which roles should be notified, as a short comma-separated string.
  Use roles such as "AM", "Cust Svc", "Purchase", "Finance", "Site".
  Example: "AM, Cust Svc".

- "assigned_to": the primary role that should handle this issue.
  Use roles like "AM", "Cust Svc", "Purchase", "Finance", "Site".
  Example: "Cust Svc" for delivery/complaint issues.

- "resolved_at": keep as "" (WhatsApp messages alone cannot tell this).

- "status_to_close": who should confirm closure, as a role or "".

- "closed_at": keep as "".

- "resolution_score": one of
  ["Not resolved yet", "Satisfied", "Neutral", "Dissatisfied"].
  For a single complaint message, usually "Not resolved yet".

- "comments": a short internal note (1–2 lines) summarising
  what Ecoplus needs to do or follow up.
  Example: "Customer chasing for delivery ETA; inform driver and update customer."

Ensure the JSON is syntactically valid and can be parsed by json.loads().
"""


@dataclass
class IssueTrackingResult:
    # original fields
    chat_id: str
    message_id: str
    timestamp: str
    sender_number: str
    sender_name: str
    message_text: str

    # model outputs
    issue: Optional[str] = None
    customer: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    to_inform: Optional[str] = None
    assigned_to: Optional[str] = None
    resolved_at: Optional[str] = None
    status_to_close: Optional[str] = None
    closed_at: Optional[str] = None
    resolution_score: Optional[str] = None
    comments: Optional[str] = None

    # plumbing / debugging
    raw_model_output: Optional[str] = None
    error: Optional[str] = None


# --------- OpenAI client helper ---------

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment. "
            "Set it or load it from a .env file before running the pipeline."
        )
    return OpenAI(api_key=api_key)


# --------- Core LLM call ---------

def call_issue_model(
    client: OpenAI,
    row: pd.Series,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> IssueTrackingResult:
    """
    Call the LLM for a single message row, and parse into IssueTrackingResult.
    """

    base = IssueTrackingResult(
        chat_id=str(row.get("chat_id", "")),
        message_id=str(row.get("message_id", "")),
        timestamp=str(row.get("timestamp", "")),
        sender_number=str(row.get("sender_number", "")),
        sender_name=str(row.get("sender_name", "")),
        message_text=str(row.get("message_text", "")),
    )

    payload = {
        "chat_id": base.chat_id,
        "message_id": base.message_id,
        "timestamp": base.timestamp,
        "sender_number": base.sender_number,
        "sender_name": base.sender_name,
        "message_text": base.message_text,
    }

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Convert this WhatsApp message into the required JSON:\n\n"
                            + json.dumps(payload, ensure_ascii=False)
                        ),
                    },
                ],
            )
            content = resp.choices[0].message.content or ""
            base.raw_model_output = content

            # Try to parse JSON
            parsed = json.loads(content)

            # Map fields safely
            base.issue = parsed.get("issue")
            base.customer = parsed.get("customer")
            base.created_by = parsed.get("created_by")
            base.created_at = parsed.get("created_at")
            base.to_inform = parsed.get("to_inform")
            base.assigned_to = parsed.get("assigned_to")
            base.resolved_at = parsed.get("resolved_at")
            base.status_to_close = parsed.get("status_to_close")
            base.closed_at = parsed.get("closed_at")
            base.resolution_score = parsed.get("resolution_score")
            base.comments = parsed.get("comments")

            base.error = None
            return base

        except Exception as e:  # noqa: BLE001
            last_error = f"Attempt {attempt} failed: {e}"
            time.sleep(retry_sleep)

    base.error = last_error or "Unknown error"
    return base


# --------- Pipeline runner ---------

def run_issue_tracking_pipeline(
    input_path: Path,
    output_dir: Path,
    model: str = "gpt-4.1-mini",
    test_mode: bool = True,
    max_rows: int = 30,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """
    Run the issue-tracking pipeline on a JSONL file of WhatsApp messages.

    Parameters
    ----------
    input_path : Path
        Path to JSONL file with WhatsApp-style client messages.
    output_dir : Path
        Directory where results (JSONL + CSV) will be saved.
    model : str
        OpenAI model name.
    test_mode : bool
        If True, only process the first `max_rows` rows.
    max_rows : int
        Number of rows to process in test_mode.
    sleep_sec : float
        Optional sleep between API calls to be gentle on rate limits.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading input from: {input_path}")
    df = pd.read_json(input_path, lines=True)

    required_cols = [
        "chat_id",
        "message_id",
        "timestamp",
        "sender_number",
        "sender_name",
        "message_text",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if test_mode:
        df = df.head(max_rows)
        print(f"TEST_MODE=True → processing first {len(df)} rows only")

    client = get_openai_client()

    results: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing message_id={row['message_id']} ...")
        result_obj = call_issue_model(client, row, model=model)
        results.append(asdict(result_obj))

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    result_df = pd.DataFrame(results)

    # Save outputs
    model_safe = model.replace(".", "_").replace("-", "_")
    jsonl_out = output_dir / f"issue_tracking_results_{model_safe}.jsonl"
    csv_out = output_dir / f"issue_tracking_results_{model_safe}.csv"

    print(f"Writing JSONL to: {jsonl_out}")
    result_df.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)

    print(f"Writing CSV to: {csv_out}")
    result_df.to_csv(csv_out, index=False)

    print("Done.")
    return result_df