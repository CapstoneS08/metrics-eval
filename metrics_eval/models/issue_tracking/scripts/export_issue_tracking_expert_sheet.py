import random
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# What experts fill (your “metrics” columns)
EXPERT_COLS = [
    "Schema_Validity (Y/N)",
    "Issue_Validity (Y/N)",
    "Overall_Issue_Accuracy (Y/N)",
    "Notes",
]

# Field-level checks (optional but useful)
FIELD_COLS = [
    "issue_field_ok (Y/N/NA)",
    "customer_field_ok (Y/N/NA)",
    "to_inform_field_ok (Y/N/NA)",
    "assigned_to_field_ok (Y/N/NA)",
    "status_to_close_field_ok (Y/N/NA)",
    "resolution_score_field_ok (Y/N/NA)",
    "comments_field_ok (Y/N/NA)",
]


def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=RANDOM_SEED).copy()


def make_sheet(df: pd.DataFrame, output_col: str) -> pd.DataFrame:
    out = df.copy()

    # compact JSON-like output column (so experts see structured fields)
    # we keep only the important output fields in one string
    def _pack(row):
        keys = ["issue","customer","to_inform","assigned_to","status_to_close","resolution_score","comments"]
        packed = {k: row.get(k, "") for k in keys}
        return str(packed)

    out[output_col] = out.apply(_pack, axis=1)

    keep = ["message_id", "sender_name", "message_text", output_col]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].rename(columns={"message_text": "input"})

    for c in EXPERT_COLS[:-1]:  # all except Notes (add later)
        out[c] = ""

    for c in FIELD_COLS:
        out[c] = ""

    out["Notes"] = ""
    return out


def export_workbook(
    *,
    gpt_df: pd.DataFrame,
    claude_df: pd.DataFrame,
    out_path: Path,
    sample_n: int = 40,
):
    wb = Workbook()
    ws = wb.active
    ws.title = "ReadMe"

    lines = [
        "Issue Tracking Expert Validation",
        "",
        "Each sheet contains outputs from ONE model on the same task.",
        "Fill Schema_Validity / Issue_Validity / Overall_Issue_Accuracy with Y or N.",
        "",
        "Field-level columns: mark Y if correct, N if incorrect, NA if not inferable from input.",
        "Notes: optional comments.",
    ]
    for i, line in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=line)

    gpt_sheet = make_sheet(sample_df(gpt_df, sample_n), output_col="output_gpt")
    claude_sheet = make_sheet(sample_df(claude_df, sample_n), output_col="output_claude")

    ws_gpt = wb.create_sheet("GPT")
    for r in dataframe_to_rows(gpt_sheet, index=False, header=True):
        ws_gpt.append(r)

    ws_claude = wb.create_sheet("Claude")
    for r in dataframe_to_rows(claude_sheet, index=False, header=True):
        ws_claude.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print("Saved:", out_path)
