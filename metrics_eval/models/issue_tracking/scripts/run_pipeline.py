"""
CLI wrapper to run issue tracking pipeline (single model).

Example:
  python models/issue_tracking/scripts/run_pipeline.py --full --model gpt-5
"""

import argparse
from pathlib import Path

from pipeline import run_issue_tracking_pipeline, INPUT_PATH_DEFAULT, OUTPUT_DIR_DEFAULT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(INPUT_PATH_DEFAULT))
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    p.add_argument("--model", type=str, default="gpt-5")
    p.add_argument("--full", action="store_true")
    p.add_argument("--max_rows", type=int, default=30)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    run_issue_tracking_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model=args.model,
        test_mode=not args.full,
        max_rows=args.max_rows,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()