"""
Mean Overall_Score per language (German, English, French, Italian), one row per run.

Reads Overall_Score per sequence from each run's unified_evaluation_summary_*.json in
evaluation/results/, joins it with the Language column of data/sampling/brackets_statistics.xlsx
on (Filename, Bracketed_Content), and averages by language.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

# evaluation/config.py (this script's folder is on sys.path)
import config
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
LANGUAGE_FILE = Path(__file__).parent.parent / config.BRACKETS_STATISTICS_FILE

# The two Qwen runs share the same model_instance ("qwen3-14B-Q6"); tell them apart by the
# processing timestamp in the filename.
RUN_LABELS = {
    "20260301_021912": "Qwen3-14B, standard",
    "20260301_022635": "Qwen3-14B, thinking",
}


def main() -> None:
    """
    Compute the language table and write it to ``language_scores.xlsx`` next to this script.

    Runs are labelled by model instance, the two Qwen runs by ``RUN_LABELS``.

    Raises:
        ValueError: If a sequence of a run does not match exactly one language row.
    """
    # Some bracketed sequences occur twice in the same letter (e.g. letter2378 "pag[ina]"), so
    # (Filename, Bracketed_Content) is not unique. Keep one language row per key; otherwise the
    # join below duplicates those sequences and they are counted twice in the means.
    languages = pd.read_excel(LANGUAGE_FILE)[
        ["Filename", "Bracketed_Content", "Language"]
    ].drop_duplicates(["Filename", "Bracketed_Content"])

    rows = []
    for summary_file in sorted(RESULTS_DIR.glob("*/unified_evaluation_summary*.json")):
        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)

        timestamp = re.search(r"_(\d{8}_\d{6})\.json$", summary_file.name).group(1)
        label = RUN_LABELS.get(timestamp, summary["summary"]["model_instance"])

        details = pd.DataFrame(summary["details"])[
            ["Filename", "Bracketed_Content", "Overall_Score"]
        ]
        merged = details.merge(
            languages, on=["Filename", "Bracketed_Content"], how="left", validate="many_to_one"
        )
        if len(merged) != len(details) or merged["Language"].isna().any():
            raise ValueError(
                f"{summary_file.name}: language join did not match every sequence exactly once"
            )

        row = merged.groupby("Language")["Overall_Score"].mean()
        row.name = label
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.round(1))

    out_path = Path(__file__).parent / "language_scores.xlsx"
    table.to_excel(out_path)
    print(f"\nSaved to: {out_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the command-line arguments (none besides ``--help``).

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``

    Returns:
        argparse.Namespace: Empty namespace
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parse_args()
    main()
