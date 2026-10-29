"""
Compare the runs sequence by sequence and list where they fail.

Usage (after ``eval.py`` for every run)::

    python compare_models.py

The five summary files are named in the code. Prints, per dimension (D1, D4a, D2 project
validity), the sequences that at least one run failed, and a failure count per run; writes the
sequences that all five runs failed, with the runs' outputs, to
``results/model_comparison_failures.xlsx``. Sequences are keyed by (Filename,
Bracketed_Content, occurrence), so the 4 pairs that occur twice in the sample are kept apart (the
runs list the sequences in the same order).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def load_results(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load the per-sequence records of one summary file.

    Args:
        json_path: Path of a ``unified_evaluation_summary_*.json``

    Returns:
        list: The ``details`` records
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("details", [])


def main() -> None:
    """Compare the five runs and write ``results/model_comparison_failures.xlsx``."""
    results_dir = Path(__file__).parent / "results"

    # Define model files (using the latest for each model)
    model_files = {
        "Claude": results_dir
        / "claude-sonnet-4-6"
        / "unified_evaluation_summary_claude_sonnet_4_6_20260301_020434.json",
        "GPT": results_dir
        / "gpt-5.2-2025-12-11"
        / "unified_evaluation_summary_gpt_5_2_2025_12_11_20260301_015947.json",
        "Qwen-Std": results_dir
        / "qwen3-14B-Q6"
        / "unified_evaluation_summary_qwen3_14B_Q6_20260301_021912.json",
        "Qwen-Think": results_dir
        / "qwen3-14B-Q6"
        / "unified_evaluation_summary_qwen3_14B_Q6_20260301_022635.json",
        "OLMo": results_dir
        / "olmo2-32B-instruct-Q4"
        / "unified_evaluation_summary_olmo2_32B_instruct_Q4_20260927_114615.json",
    }

    # Load all results
    all_results = {}
    for model_name, file_path in model_files.items():
        if file_path.exists():
            all_results[model_name] = load_results(file_path)
            print(f"Loaded {len(all_results[model_name])} sequences for {model_name}")

    # Create a unified view keyed by (Filename, Bracketed_Content, occurrence)
    unified = {}
    for model_name, sequences in all_results.items():
        occurrences = {}
        for seq in sequences:
            pair = (seq.get("Filename", ""), seq.get("Bracketed_Content", ""))
            occurrences[pair] = occurrences.get(pair, 0) + 1
            key = pair + (occurrences[pair],)
            if key not in unified:
                unified[key] = {
                    "Filename": seq.get("Filename", ""),
                    "Bracketed_Content": seq.get("Bracketed_Content", ""),
                    "Gold_Classification": seq.get("Gold_Classification", ""),
                    "Bracketed_Normalized": seq.get("Bracketed_Content_Normalized", ""),
                }
            # Add model-specific columns
            unified[key][f"{model_name}_D1"] = seq.get("Source_Fidelity (D1)")
            unified[key][f"{model_name}_D4a"] = seq.get("Classification_Match (D4a)")
            unified[key][f"{model_name}_D4b"] = seq.get("Element_Match (D4b)")
            unified[key][f"{model_name}_Project"] = seq.get("Project_valid (D2)")
            unified[key][f"{model_name}_Class"] = seq.get("LLM_Classification", "")
            unified[key][f"{model_name}_Output"] = seq.get("LLM_Output_Normalized", "")

    # Convert to DataFrame
    df = pd.DataFrame(list(unified.values()))

    # Filter to only show rows where at least one model failed D1 or D4a
    d1_cols = [c for c in df.columns if c.endswith("_D1")]
    d4a_cols = [c for c in df.columns if c.endswith("_D4a")]

    # Check for any failure (False) in D1 columns
    d1_any_fail = df[d1_cols].apply(lambda row: any(v == False for v in row), axis=1)
    d4a_any_fail = df[d4a_cols].apply(lambda row: any(v == False for v in row), axis=1)

    # Source Fidelity failures
    print("\n" + "=" * 100)
    print("SOURCE FIDELITY (D1) FAILURES - Where at least one model failed")
    print("=" * 100)

    d1_failures = df[d1_any_fail][
        ["Filename", "Bracketed_Content", "Gold_Classification"] + d1_cols
    ].copy()
    if not d1_failures.empty:
        # Replace True/False/None with symbols for readability
        for col in d1_cols:
            d1_failures[col] = d1_failures[col].map({True: "OK", False: "FAIL", None: "-"})
        print(d1_failures.to_string(index=False))
        print(f"\nTotal: {len(d1_failures)} sequences with D1 failures")
    else:
        print("No D1 failures found.")

    # Classification failures
    print("\n" + "=" * 100)
    print("CLASSIFICATION (D4a) FAILURES - Where at least one model failed")
    print("=" * 100)

    d4a_failures = df[d4a_any_fail][
        ["Filename", "Bracketed_Content", "Gold_Classification"] + d4a_cols
    ].copy()
    if not d4a_failures.empty:
        for col in d4a_cols:
            d4a_failures[col] = d4a_failures[col].map({True: "OK", False: "FAIL", None: "-"})
        print(d4a_failures.to_string(index=False))
        print(f"\nTotal: {len(d4a_failures)} sequences with D4a failures")
    else:
        print("No D4a failures found.")

    # Project validity failures
    print("\n" + "=" * 100)
    print("PROJECT VALIDITY (D2) FAILURES - Where at least one model failed")
    print("=" * 100)

    project_cols = [c for c in df.columns if c.endswith("_Project")]
    project_any_fail = df[project_cols].apply(lambda row: any(v == False for v in row), axis=1)

    project_failures = df[project_any_fail][
        ["Filename", "Bracketed_Content", "Gold_Classification"] + project_cols
    ].copy()
    if not project_failures.empty:
        for col in project_cols:
            project_failures[col] = project_failures[col].map(
                {True: "OK", False: "FAIL", None: "-"}
            )
        print(project_failures.to_string(index=False))
        print(f"\nTotal: {len(project_failures)} sequences with Project validity failures")
    else:
        print("No Project validity failures found.")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY - Failure counts per model")
    print("=" * 100)

    summary = []
    for model in ["Claude", "GPT", "Qwen-Think", "Qwen-Std", "OLMo"]:
        d1_col = f"{model}_D1"
        d4a_col = f"{model}_D4a"
        proj_col = f"{model}_Project"

        if d1_col in df.columns:
            d1_fails = (df[d1_col] == False).sum()
            d4a_fails = (df[d4a_col] == False).sum()
            proj_fails = (df[proj_col] == False).sum()
            total = df[d1_col].notna().sum()

            summary.append(
                {
                    "Model": model,
                    "Total": total,
                    "D1_Fails": d1_fails,
                    "D4a_Fails": d4a_fails,
                    "Project_Fails": proj_fails,
                }
            )

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # Export to Excel with LLM outputs for failures
    print("\n" + "=" * 100)
    print("Exporting to Excel with LLM outputs...")
    print("=" * 100)

    models = ["Claude", "GPT", "Qwen-Think", "Qwen-Std", "OLMo"]

    # Filter to only rows where ALL 5 models failed
    d1_all_fail = df[d1_cols].apply(lambda row: sum(v == False for v in row) == 5, axis=1)
    d4a_all_fail = df[d4a_cols].apply(lambda row: sum(v == False for v in row) == 5, axis=1)
    project_all_fail = df[project_cols].apply(lambda row: sum(v == False for v in row) == 5, axis=1)

    print(f"D1: {d1_all_fail.sum()} sequences where ALL 5 models failed")
    print(f"D4a: {d4a_all_fail.sum()} sequences where ALL 5 models failed")
    print(f"Project: {project_all_fail.sum()} sequences where ALL 5 models failed")

    # Create Excel writer
    excel_path = results_dir / "model_comparison_failures.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # D1 failures with outputs (ALL 5 failed)
        d1_export = df[d1_all_fail][
            ["Filename", "Bracketed_Content", "Gold_Classification", "Bracketed_Normalized"]
            + [f"{m}_D1" for m in models]
            + [f"{m}_Output" for m in models]
        ].copy()
        for col in [f"{m}_D1" for m in models]:
            d1_export[col] = d1_export[col].map({True: "OK", False: "FAIL", None: "-"})
        d1_export.to_excel(writer, sheet_name="D1_All5Fail", index=False)

        # D4a failures with outputs (ALL 5 failed)
        d4a_export = df[d4a_all_fail][
            ["Filename", "Bracketed_Content", "Gold_Classification", "Bracketed_Normalized"]
            + [f"{m}_D4a" for m in models]
            + [f"{m}_Class" for m in models]
        ].copy()
        for col in [f"{m}_D4a" for m in models]:
            d4a_export[col] = d4a_export[col].map({True: "OK", False: "FAIL", None: "-"})
        d4a_export.to_excel(writer, sheet_name="D4a_All5Fail", index=False)

        # Project validity failures (ALL 5 failed)
        project_export = df[project_all_fail][
            ["Filename", "Bracketed_Content", "Gold_Classification"]
            + [f"{m}_Project" for m in models]
        ].copy()
        for col in [f"{m}_Project" for m in models]:
            project_export[col] = project_export[col].map({True: "OK", False: "FAIL", None: "-"})
        project_export.to_excel(writer, sheet_name="Project_All5Fail", index=False)

        # Summary
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Saved to: {excel_path}")


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
