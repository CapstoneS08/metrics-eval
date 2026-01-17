"""
CLI wrapper for the summarising pipeline.

Default input:
  <root>/data/processed/summarising/summarising_input.jsonl

Usage examples (from project root):
  python notebooks/summarising/cel/run_pipeline.py
  python notebooks/summarising/cel/run_pipeline.py --full
"""

import argparse
from pathlib import Path

import pipeline  # same folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarising model pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(pipeline.INPUT_PATH_DEFAULT),
        help=f"Path to processed input JSONL/Excel (default: {pipeline.INPUT_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(pipeline.OUTPUT_DIR_DEFAULT),
        help=f"Directory to save outputs (default: {pipeline.OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on full dataset (ignore TEST_MODE/max_rows).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=30,
        help="Max rows in test mode (default: 30).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name for OpenAI Responses API.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between calls (seconds).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=== Summarising Pipeline ===")
    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Model:      {args.model}")
    print(f"Full run:   {args.full}")
    print(f"Max rows:   {args.max_rows if not args.full else 'ALL'}")

    pipeline.run_summarising_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        test_mode=not args.full,
        max_rows=args.max_rows,
        model=args.model,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()