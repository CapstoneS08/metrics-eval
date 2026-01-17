"""
Summarising pipeline: extract CX improvement comments from processed feedback.

INPUT (default):
    <root>/data/processed/summarising/summarising_input.jsonl

Each JSONL line:
    {
      "Comment_ID": ...,
      "Comment": "..."
    }

OUTPUT:
    JSONL files saved to:
        <this folder>/results/
    - cx_improvement_full.jsonl
    - cx_improvement_only.jsonl
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI


# ---------- project / path helpers ----------

def find_project_root(start: Path) -> Path:
    """
    Walk upwards from `start` until we find a folder containing 'data'.
    Falls back to `start` if not found (up to 6 levels).
    """
    cur = start
    for _ in range(6):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    return start


# This file lives in: notebooks/summarising/cel/pipeline.py
HERE = Path(__file__).resolve().parent
ROOT_DIR = find_project_root(HERE)

INPUT_PATH_DEFAULT = ROOT_DIR / "data" / "processed" / "summarising" / "summarising_input.jsonl"
OUTPUT_DIR_DEFAULT = HERE / "results"

SYSTEM_PROMPT = """
You are a CX analyst.

TASK:
Given one raw customer message, extract ONLY the actionable improvement suggestion.

RULES:
- If there is an improvement → rewrite clearly in 1 short sentence.
- If NO actionable improvement (no complaint, no suggestion) → return: NONE
- No greetings, no extra text.
- NO explanations.
- Output MUST be valid JSON of the form:
{"improvement_comment": "<text>"}
""".strip()


# ---------- OpenAI helper ----------

def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client. Requires OPENAI_API_KEY in env.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def get_json_improvement(
    text: str,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Call the LLM and return {"improvement_comment": "..."}.

    If input is empty or an error occurs, returns:
        {"improvement_comment": "NONE"} or {"improvement_comment": "ERROR"}.
    """
    if not isinstance(text, str) or not text.strip():
        return {"improvement_comment": "NONE"}

    try:
        resp = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=text.strip(),
        )
        raw = resp.output_text.strip()

        # Try parse as JSON; otherwise wrap in JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"improvement_comment": raw}

        if "improvement_comment" not in data:
            data = {"improvement_comment": "ERROR"}

        return data

    except Exception as e:
        print("ERROR in get_json_improvement:", e)
        return {"improvement_comment": "ERROR"}


# ---------- core pipeline ----------

def load_input_df(path: Path) -> pd.DataFrame:
    """
    Load processed input and normalise to columns:
        - Comment_ID
        - Comment

    Format is expected to be JSONL produced by analysis.ipynb.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".jsonl", ".json"]:
        df = pd.read_json(path, lines=True)
    elif suffix in [".xlsx", ".xls"]:
        # fallback if someone passes Excel directly
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    lower_map = {c.lower(): c for c in df.columns}
    id_col = lower_map.get("comment_id") or lower_map.get("id")
    text_col = lower_map.get("comment") or lower_map.get("text") or lower_map.get("message")

    if id_col is None or text_col is None:
        raise ValueError(
            f"Could not find Comment_ID/Comment columns in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(columns={id_col: "Comment_ID", text_col: "Comment"})
    return df[["Comment_ID", "Comment"]]


def run_summarising_pipeline(
    input_path: Path = INPUT_PATH_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    test_mode: bool = True,
    max_rows: int = 30,
    model: str = "gpt-4.1-mini",
    sleep_sec: float = 0.0,
) -> Dict[str, Path]:
    """
    Run the summarising model over the dataset.

    Returns dict of output file paths:
      {
        "full": <path to cx_improvement_full.jsonl>,
        "only": <path to cx_improvement_only.jsonl>,
      }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading processed data from: {input_path}")
    df = load_input_df(input_path)
    print(f"Rows in dataset: {len(df)}")

    if test_mode:
        df = df.head(max_rows)
        print(f"TEST_MODE=True → processing first {len(df)} rows only")

    client = get_openai_client()
    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        comment_id = row["Comment_ID"]
        text = row["Comment"]

        print(f"\n--- Row {idx} (Comment_ID={comment_id}) ---")
        print("Input:", text)

        output_json = get_json_improvement(text, client=client, model=model)
        print("JSON output:", output_json)

        results.append(
            {
                "Comment_ID": int(comment_id) if pd.notna(comment_id) else None,
                "Comment": str(text) if pd.notna(text) else "",
                "model_output": {
                    "improvement_comment": str(
                        output_json.get("improvement_comment", "")
                    )
                },
            }
        )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # Filter to only rows with non-trivial improvements
    improvement_only: List[Dict[str, Any]] = [
        r
        for r in results
        if r["model_output"]["improvement_comment"]
        not in ["NONE", "ERROR", ""]
    ]

    # ---- Save JSONL ----
    full_file = output_dir / "cx_improvement_full.jsonl"
    only_file = output_dir / "cx_improvement_only.jsonl"

    with full_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    with only_file.open("w", encoding="utf-8") as f:
        for r in improvement_only:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"\n✅ Saved JSONL:")
    print(f"  - All rows:      {full_file}")
    print(f"  - Improvements:  {only_file}")

    return {
        "full": full_file,
        "only": only_file,
    }


if __name__ == "__main__":
    # quick manual test: python pipeline.py
    run_summarising_pipeline(test_mode=True)