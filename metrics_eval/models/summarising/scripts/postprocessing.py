"""
Post-processing / aggregation for CX improvement comments.

Takes a DataFrame with an 'improvement_comment' column (plus anything else),
sends them to an LLM, and returns + optionally saves a JSON summary:

{
  "summary_overall": [...],
  "categories": {
    "Product": [
      {"theme": "...", "issue_summary": "...", "action": "..."},
      ...
    ],
    "Service": [...],
    "Delivery": [...],
    "Payment": [...]
  }
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # so it picks up OPENAI_API_KEY from your .env


HERE = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = HERE / "results" / "cx_improvement_aggregate.json"


AGGREGATION_SYSTEM_PROMPT = """
You are a customer experience (CX) analyst.

You will receive a numbered list of short improvement suggestions from customers.
Each line is one suggestion, written as an action Ecoplus could take.

Your tasks:

1. Assign each suggestion to one or more of these CX aspects:
   - Product
   - Service
   - Delivery
   - Payment
   (Infer the aspect from the content. If unclear, choose the best guess.)

2. Cluster the suggestions into 2–5 themes per aspect.
   Focus on recurring patterns, not one-off comments.

3. For each theme, provide:
   - "theme": a very short label (max 5 words)
   - "issue_summary": one clear sentence describing what customers are unhappy about
   - "action": one concrete, actionable step Ecoplus can take to improve this

4. Also provide 2–3 bullet points summarising the most important recurring issues overall.
   These should be cross-cutting takeaways a manager should see first.

IMPORTANT:
- Be concise and businesslike.
- Do not quote customers verbatim unless necessary.
- If an aspect has no relevant suggestions, return an empty list for that aspect.

Return STRICTLY valid JSON, with this exact structure:

{
  "summary_overall": ["...", "..."],
  "categories": {
    "Product": [
      {"theme": "...", "issue_summary": "...", "action": "..."}
    ],
    "Service": [
      {"theme": "...", "issue_summary": "...", "action": "..."}
    ],
    "Delivery": [
      {"theme": "...", "issue_summary": "...", "action": "..."}
    ],
    "Payment": [
      {"theme": "...", "issue_summary": "...", "action": "..."}
    ]
  }
}

Do not include any text before or after the JSON.
""".strip()


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def _build_input_text(improvements: Iterable[str]) -> str:
    """
    Turn a list of improvement sentences into a numbered block of text
    to send to the LLM.
    """
    lines: List[str] = []
    for i, s in enumerate(improvements, start=1):
        s = s.strip()
        if not s:
            continue
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def aggregate_from_df(
    df: pd.DataFrame,
    *,
    improvement_col: str = "improvement_comment",
    model: str = "gpt-4.1-mini",
    max_items: Optional[int] = None,
    save_path: Optional[Path] = DEFAULT_SAVE_PATH,
) -> Dict[str, Any]:
    """
    Run aggregation on a DataFrame that has an `improvement_comment` column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an `improvement_comment` column.
    improvement_col : str
        Name of the column with improvement sentences.
    model : str
        OpenAI model name.
    max_items : Optional[int]
        If set, only the first `max_items` non-empty, non-NONE/ERROR comments are used.
    save_path : Optional[Path]
        If provided, aggregated JSON is saved there.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON structure from the LLM.
    """
    if improvement_col not in df.columns:
        raise ValueError(f"Column '{improvement_col}' not found in DataFrame.")

    improvements: List[str] = []
    for s in df[improvement_col].astype(str).tolist():
        s = s.strip()
        if not s or s in ("NONE", "ERROR"):
            continue
        improvements.append(s)

    if max_items is not None:
        improvements = improvements[:max_items]

    if not improvements:
        raise ValueError("No valid improvement comments found to aggregate.")

    input_text = _build_input_text(improvements)

    client = get_openai_client()
    resp = client.responses.create(
        model=model,
        instructions=AGGREGATION_SYSTEM_PROMPT,
        input=input_text,
    )

    raw = resp.output_text.strip()

    try:
        agg = json.loads(raw)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, optionally save raw text for debugging
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                f.write(raw)
        raise RuntimeError(f"Failed to parse aggregation JSON: {e}\nRaw:\n{raw}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)

    return agg
