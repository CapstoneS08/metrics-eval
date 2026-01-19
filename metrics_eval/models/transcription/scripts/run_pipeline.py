"""
Run transcription over a metadata.csv and save predictions.csv.

Usage examples (from repo root: metrics_eval):
  python -m models.transcription.scripts.run_pipeline --backend openai_whisper --model whisper-1
  python -m models.transcription.scripts.run_pipeline --backend faster_whisper --model base

Outputs:
  models/transcription/results/<backend>__<model>/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from models.transcription.scripts.pipeline import transcribe


def get_repo_root() -> Path:
    # scripts/ -> transcription/ -> models/ -> metrics_eval
    return Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--backend", type=str, default="openai_whisper")
    parser.add_argument("--model", type=str, default="whisper-1")

    # Only used for faster_whisper
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compute_type", type=str, default="int8")
    parser.add_argument("--language", type=str, default="en")

    parser.add_argument("--n_samples", type=int, default=-1, help="Limit rows, -1 = all")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")

    args = parser.parse_args()

    root = get_repo_root()
    metadata_path = root / "data" / "transcription" / args.split / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if args.n_samples != -1:
        df = df.head(args.n_samples)

    out_dir = root / "models" / "transcription" / "results" / f"{args.backend}__{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"

    if out_path.exists() and not args.force:
        print(f"[SKIP] predictions already exist: {out_path}")
        print("Use --force to regenerate.")
        return

    rows = []
    for _, r in df.iterrows():
        sample_id = r["id"]
        audio_path = r["audio_path"]
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"Missing audio_path for id={sample_id}. Did Notebook 1 finish?")

        result = transcribe(
            audio_path=audio_path,
            backend=args.backend,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )

        rows.append(
            {
                "id": sample_id,
                "audio_path": audio_path,
                "backend": result["backend"],
                "model": result["model"],
                "transcript": result["transcript"],
                "latency_s": result["latency_s"],
            }
        )

        print(f"[{sample_id}] done ({result['latency_s']:.2f}s)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()