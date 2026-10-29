"""
Comparison figures of the runs.

Usage (after ``eval.py`` for every run and ``aggregate_results.py``)::

    python comp_vis.py

The five runs are named in ``RUNS``. Reads each run's Excel report and processing metadata, takes
the n values from ``aggregated_results.json``, prints the scores and writes to ``comparisons/``
(file names end with the current time): ``overall_metrics_*.png`` and per dimension (d4a, d4b,
d1) ``*_macro_averaged_*.png`` and ``*_per_category_*.png``. Per-category and macro scores are
recomputed from the exact TP/FP/FN counts of the report sheets, not from their rounded
Precision/Recall/F1 columns.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "comparisons"

# The runs in plotting order: (label, results folder, processing timestamp, name in
# aggregated_results.json)
RUNS = [
    ("claude-sonnet-4-6", "claude-sonnet-4-6", "20260301_020434", "claude-sonnet-4-6"),
    ("gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11", "20260301_015947", "gpt-5.2-2025-12-11"),
    (
        "olmo2-32B-instruct-Q4",
        "olmo2-32B-instruct-Q4",
        "20260927_114615",
        "olmo2-32B-instruct-Q4",
    ),
    ("qwen3-14B-Q6 (thinking)", "qwen3-14B-Q6", "20260301_022635", "qwen3-14B-Q6 (thinking)"),
    (
        "qwen3-14B-Q6 (no-thinking)",
        "qwen3-14B-Q6",
        "20260301_021912",
        "qwen3-14B-Q6 (no thinking)",
    ),
]

# One colour per run, in the order of RUNS; used by all charts and the technical details table
MODEL_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#50C878", "#9B59B6"]
BAR_WIDTH = 0.15

DIMENSION_TITLES = {
    "d1": "D1 - Source Fidelity",
    "d4a": "D4a - Classification",
    "d4b": "D4b - Element Application",
}

# Report sheet with the per-category counts of each dimension
CATEGORY_SHEETS = {
    "d4a": "Classification (D4a)",
    "d4b": "Element Application (D4b)",
    "d1": "Source Fidelity (D1)",
}

# Result key of the n of each dimension: D4a = sequences with a D4a result, D4b = well-formed
# sequences, D1 = sequences with a D4b element match
N_KEYS = {"d4a": "d4a_scored", "d4b": "d0_xml_wellformed", "d1": "d4b_element_match"}


def load_aggregated_summary() -> Dict[str, Dict[str, Any]]:
    """
    Load the ``summary`` rows of ``aggregated_results.json`` next to this script.

    Returns:
        dict: Model name -> summary row
    """
    with open(SCRIPT_DIR / "aggregated_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {row["Model"]: row for row in data["summary"]}


def _prf1_from_counts(row: pd.Series) -> Tuple[float, float, float]:
    """
    Recompute precision/recall/F1 from the exact TP/FP/FN counts rather than trusting the sheet's
    Precision/Recall/F1 columns, which are pre-rounded to 2 decimals and therefore compound
    rounding error when macro-averaged.

    Args:
        row: Row of a category sheet

    Returns:
        tuple: Precision, recall, F1 (0-1)
    """
    tp = float(row["True Positives"]) if pd.notna(row["True Positives"]) else 0.0
    fp = float(row["False Positives"]) if pd.notna(row["False Positives"]) else 0.0
    fn = float(row["False Negatives"]) if pd.notna(row["False Negatives"]) else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


def analyze_run(folder: str, timestamp: str, aggregated_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read one run's evaluation report and processing metadata and collect what the figures need.

    Per-category precision, recall and F1 are recomputed from the TP/FP/FN counts of the category
    sheets; macro values are their unweighted means. The overall score is "Average Overall Score
    (all)" of the Summary sheet, rounded half up to 1 decimal.

    Args:
        folder: Run folder in ``results/``
        timestamp: Processing timestamp in the file names
        aggregated_row: The run's row of ``aggregated_results.json``

    Returns:
        dict: ``category`` (dimension -> category -> (precision, recall, F1)), ``macro``
        (dimension -> (precision, recall, F1)), ``overall_score``, the n values (``N_KEYS`` and
        ``processed``) and the metadata fields of the technical details table
    """
    run_dir = RESULTS_DIR / folder
    file_stem = folder.replace("-", "_").replace(".", "_")
    excel_file = run_dir / f"evaluation_{file_stem}_{timestamp}.xlsx"

    details = pd.read_excel(excel_file, sheet_name="Details")

    category: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    macro: Dict[str, Tuple[float, float, float]] = {}
    for dimension, sheet in CATEGORY_SHEETS.items():
        df = pd.read_excel(excel_file, sheet_name=sheet)
        scores = {
            str(row["Category"]) if pd.notna(row["Category"]) else "": _prf1_from_counts(row)
            for _, row in df.iterrows()
        }
        category[dimension] = scores
        macro[dimension] = tuple(
            sum(s[i] for s in scores.values()) / len(scores) for i in range(3)
        )

    # "Average Overall Score (all)" (mean over all sequences, invalid XML counted as 0)
    overall_score = 0.0
    summary = pd.read_excel(excel_file, sheet_name="Summary")
    for _, row in summary.iterrows():
        if pd.notna(row.iloc[0]) and "Average Overall Score (all)" in str(row.iloc[0]):
            score_str = re.sub(r"[^\d.]", "", str(row.iloc[1]))
            # Summary value is already rounded to 2 decimals; round half up to 1 decimal so 81.85
            # shows as 81.9 (float formatting would give 81.8)
            overall_score = float(Decimal(score_str).quantize(Decimal("0.1"), ROUND_HALF_UP))
            break

    with open(run_dir / f"processing_metadata_{timestamp}.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "category": category,
        "macro": macro,
        "overall_score": overall_score,
        # Invalid XML has no D4a result
        "d4a_scored": int(details["Classification_Match (D4a)"].notna().sum()),
        "d0_xml_wellformed": aggregated_row["D0 XML Wellformed"],
        "d4b_element_match": aggregated_row["D4b Element Match"],
        "processed": aggregated_row["Processed"],
        "processing_time": metadata["processing_time"],
        "total_tokens": metadata["token_count"],
        "estimated_cost": metadata["estimated_cost_usd"],
        "temperature": metadata["temperature"],
    }


def print_scores(model_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print macro F1 per dimension and the overall score of every run.

    Args:
        model_results: Run label -> result of ``analyze_run``
    """
    header = f"{'Run':<30} {'D4a F1':>8} {'D4b F1':>8} {'D1 F1':>8} {'Overall':>8}"
    print(header)
    print("-" * len(header))
    for label, results in model_results.items():
        f1 = [results["macro"][d][2] * 100 for d in ("d4a", "d4b", "d1")]
        print(
            f"{label:<30} {f1[0]:>8.1f} {f1[1]:>8.1f} {f1[2]:>8.1f} "
            f"{results['overall_score']:>8.1f}"
        )


def _bar_offset(i: int, n_runs: int) -> float:
    """Offset of run ``i``'s bar within a group of ``n_runs`` bars."""
    return (i - n_runs / 2 + 0.5) * BAR_WIDTH


def create_technical_details_panel(ax: Any, model_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Draw a table with model, processing time, tokens, cost and temperature per run, each row in
    the run's colour.

    Args:
        ax: Matplotlib axes to draw on
        model_results: Run label -> result of ``analyze_run``
    """
    ax.axis("off")

    headers = ["Model", "Time", "Tokens", "Cost", "Temp"]
    table_data = [
        [
            label,
            results["processing_time"],
            f"{int(results['total_tokens']):,}",
            f"${results['estimated_cost']:.2f}",
            str(results["temperature"]),
        ]
        for label, results in model_results.items()
    ]

    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(len(headers))))

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("black")
        table[(0, j)].set_text_props(weight="bold", color="white")
    for i in range(len(model_results)):
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(MODEL_COLORS[i])
            table[(i + 1, j)].set_text_props(color="white", weight="bold")


def create_overall_metrics_dashboard(
    model_results: Dict[str, Dict[str, Any]], output_file: Path
) -> None:
    """
    Overall chart: macro F1 of D4a, D4b and D1 plus the overall score, one bar per run, with the
    technical details table inset in the top-right corner.

    Each bar is labelled with its value and with n (D4a: sequences with a D4a result, D4b:
    well-formed sequences, D1: sequences with a D4b match, overall: processed sequences).

    Args:
        model_results: Run label -> result of ``analyze_run``
        output_file: Path of the PNG
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = [
        "Intervention\nCategorization\n(D4a)",
        "Element\nApplication\n(D4b)",
        "Source\nFidelity\n(D1)",
        "TEI Encoding\nScore",
    ]
    x = np.arange(len(metrics))

    for i, results in enumerate(model_results.values()):
        data = [results["macro"][d][2] * 100 for d in ("d4a", "d4b", "d1")]
        data.append(results["overall_score"])
        n_values = [results[N_KEYS[d]] for d in ("d4a", "d4b", "d1")]
        n_values.append(results["processed"])

        bars = ax.bar(
            x + _bar_offset(i, len(model_results)),
            data,
            BAR_WIDTH,
            color=MODEL_COLORS[i],
            alpha=0.8,
        )
        for bar, score, n_val in zip(bars, data, n_values):
            center = bar.get_x() + bar.get_width() / 2.0
            ax.text(
                center,
                bar.get_height() + 1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                center, 2, f"n({int(n_val)})", ha="center", va="bottom", fontsize=9, color="#444"
            )

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100", "", ""])

    inset_ax = ax.inset_axes([0.52, 0.66, 0.46, 0.32])
    create_technical_details_panel(inset_ax, model_results)

    fig.suptitle("Cross-Model Comparison Dashboard", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def create_macro_averaged_dashboard(
    model_results: Dict[str, Dict[str, Any]], dimension: str, output_file: Path
) -> None:
    """
    Macro-averaged precision, recall and F1 of one dimension, one group per run.

    Args:
        model_results: Run label -> result of ``analyze_run``
        dimension: One of 'd1', 'd4a', 'd4b'
        output_file: Path of the PNG
    """
    n_explanations = {
        "d1": "n = sequences with correct element application (D4b = TRUE)",
        "d4a": "n = processed sequences",
        "d4b": "n = well-formed sequences",
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(len(model_results))
    width = 0.25

    precision_data, recall_data, f1_data = (
        [results["macro"][dimension][i] * 100 for results in model_results.values()]
        for i in range(3)
    )
    model_labels = [
        f"{label}\n(n={results[N_KEYS[dimension]]})" for label, results in model_results.items()
    ]

    bars1 = ax.bar(x - width, precision_data, width, label="Precision", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x, recall_data, width, label="Recall", color="#e74c3c", alpha=0.8)
    bars3 = ax.bar(x + width, f1_data, width, label="F1 Score", color="#2ecc71", alpha=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Score (%)", fontweight="bold", fontsize=11)
    ax.set_title(
        f"{DIMENSION_TITLES[dimension]} - Macro-Averaged Metrics",
        fontweight="bold",
        fontsize=16,
        pad=18,
    )
    ax.text(
        0.5,
        0.99,
        n_explanations[dimension],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        style="italic",
    ).set_clip_on(False)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def create_category_dashboard(
    model_results: Dict[str, Dict[str, Any]], dimension: str, output_file: Path
) -> None:
    """
    F1 per category of one dimension, one bar per run.

    Args:
        model_results: Run label -> result of ``analyze_run``
        dimension: One of 'd1', 'd4a', 'd4b'
        output_file: Path of the PNG
    """
    n_explanations = {
        "d1": "n = element-matched sequences (correct D4b)",
        "d4a": "n = processed sequences",
        "d4b": "n = well-formed XML sequences",
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    categories = sorted(
        {cat for results in model_results.values() for cat in results["category"][dimension]}
    )
    x = np.arange(len(categories))

    for i, (label, results) in enumerate(model_results.items()):
        scores = results["category"][dimension]
        data = [scores[cat][2] * 100 if cat in scores else 0.0 for cat in categories]

        bars = ax.bar(
            x + _bar_offset(i, len(model_results)),
            data,
            BAR_WIDTH,
            label=f"{label} (n={results[N_KEYS[dimension]]})",
            color=MODEL_COLORS[i],
            alpha=0.8,
        )
        for bar, score in zip(bars, data):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{score:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

    ax.set_ylabel("F1 Score (%)", fontweight="bold", fontsize=14)
    ax.set_title(
        f"{DIMENSION_TITLES[dimension]} - Per Category F1 Scores",
        fontweight="bold",
        fontsize=19,
        pad=18,
    )
    ax.text(
        0.5,
        0.99,
        n_explanations[dimension],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        style="italic",
    ).set_clip_on(False)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13, rotation=45, ha="right")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 125])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100", "", ""], fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main() -> int:
    """
    Analyse the five runs and write the comparison figures to ``comparisons/``.

    Returns:
        int: Exit code, 0 on success
    """
    aggregated = load_aggregated_summary()
    model_results = {
        label: analyze_run(folder, timestamp, aggregated[aggregated_name])
        for label, folder, timestamp, aggregated_name in RUNS
    }

    print_scores(model_results)
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_overall_metrics_dashboard(model_results, OUTPUT_DIR / f"overall_metrics_{timestamp}.png")
    for dimension in ["d1", "d4a", "d4b"]:
        create_macro_averaged_dashboard(
            model_results, dimension, OUTPUT_DIR / f"{dimension}_macro_averaged_{timestamp}.png"
        )
    for dimension in ["d1", "d4a", "d4b"]:
        create_category_dashboard(
            model_results, dimension, OUTPUT_DIR / f"{dimension}_per_category_{timestamp}.png"
        )
    return 0


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
    sys.exit(main())
