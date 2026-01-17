"""
CLI wrapper to run the issue-tracking pipeline.

Usage (from project root, once env is set):

    python notebooks/issue_tracking/cel/run_pipeline.py

Adjust `TEST_MODE` / `MAX_ROWS` below as needed.
"""

from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Import pipeline from the same folder
from pipeline import run_issue_tracking_pipeline  # noqa: E402


def find_project_root(start: Path | None = None) -> Path:
    """Climb upwards until we find the 'data' folder."""
    if start is None:
        start = Path.cwd()
    root = start
    for _ in range(6):
        if (root / "data").exists():
            return root
        root = root.parent
    return start


if __name__ == "__main__":
    load_dotenv()  # load OPENAI_API_KEY if in .env at project root

    ROOT = find_project_root()
    print("Detected project root:", ROOT)

    input_path = ROOT / "data" / "raw" / "synthetic" / "issue_tracking" / "synthetic_whatsapp_client_messages.jsonl"
    output_dir = ROOT / "notebooks" / "issue_tracking" / "cel" / "results"

    TEST_MODE = True
    MAX_ROWS = 20

    run_issue_tracking_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        model="gpt-4.1-mini",
        test_mode=TEST_MODE,
        max_rows=MAX_ROWS,
        sleep_sec=0.0,
    )