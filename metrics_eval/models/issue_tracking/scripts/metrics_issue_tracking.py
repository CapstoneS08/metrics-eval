import json
from pathlib import Path
from collections import Counter
import pandas as pd


def read_results(jsonl_path: Path) -> pd.DataFrame:
    return pd.read_json(jsonl_path, lines=True)


def compute_nonexpert_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n_rows": 0}

    schema_ok_pct = round(df["schema_ok"].mean() * 100, 2) if "schema_ok" in df.columns else 0.0
    error_rate_pct = round((df["error"].astype(str).str.strip() != "").mean() * 100, 2) if "error" in df.columns else 0.0

    # output completeness: how many empty strings in key fields (rough sanity)
    issue_empty_pct = round((df["issue"].astype(str).str.strip() == "").mean() * 100, 2) if "issue" in df.columns else 0.0
    assigned_empty_pct = round((df["assigned_to"].astype(str).str.strip() == "").mean() * 100, 2) if "assigned_to" in df.columns else 0.0

    # comments length sanity
    comments_len = df["comments"].fillna("").astype(str).str.len() if "comments" in df.columns else pd.Series([], dtype=int)
    avg_comments_len = round(float(comments_len.mean()), 2) if len(comments_len) else 0.0
    median_comments_len = float(comments_len.median()) if len(comments_len) else 0.0

    # bad field counts
    bad_counter = Counter()
    if "schema_bad_fields" in df.columns:
        for x in df["schema_bad_fields"].tolist():
            if isinstance(x, list):
                bad_counter.update(x)

    return {
        "n_rows": int(n),
        "schema_ok_pct": schema_ok_pct,
        "error_rate_pct": error_rate_pct,
        "issue_empty_pct": issue_empty_pct,
        "assigned_to_empty_pct": assigned_empty_pct,
        "avg_comments_len_chars": avg_comments_len,
        "median_comments_len_chars": median_comments_len,
        "schema_bad_fields_counts": dict(bad_counter),
    }


def summarize_results_dir(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        # expected: issue_tracking_results_<model>.jsonl inside results root OR store per model separately in analysis
        # Here we support both layouts: direct jsonl in root OR in subdir.
        candidates = list(model_dir.glob("issue_tracking_results_*.jsonl"))
        if not candidates:
            continue
        jsonl_path = candidates[0]
        df = read_results(jsonl_path)
        m = compute_nonexpert_metrics(df)
        m["model_dir"] = model_dir.name
        rows.append(m)
    return pd.DataFrame(rows).sort_values("model_dir")


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]).resolve()
    out_csv = results_dir / "nonexpert_metrics_issue_tracking.csv"
    summary = summarize_results_dir(results_dir)
    summary.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(summary)