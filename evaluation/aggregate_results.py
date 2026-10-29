"""
Aggregate the evaluation results of all runs into one summary table.

Usage (after ``eval.py`` for every run)::

    python aggregate_results.py

Reads every ``results/<model>/unified_evaluation_summary_*.json`` and writes
``aggregated_results.xlsx`` and ``aggregated_results.json`` next to this script: a summary row per
run (D0, D1, D2, D4a, D4b counts and percentages of processed sequences, macro F1), the per-category
F1 scores per run, and the sequences that every run failed (D4a, D4b, D1). Runs of the same model
are told apart by the ``thinking_mode`` of their processing metadata.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from openpyxl.utils import get_column_letter


def extract_processing_timestamp(filename: str) -> Optional[str]:
    """
    Extract the processing timestamp from a summary file name.

    Args:
        filename: File name like ``unified_evaluation_summary_model_20260301_021912.json``

    Returns:
        str: ``YYYYMMDD_HHMMSS``, or None if the name does not end in a timestamp
    """
    match = re.search(r"_(\d{8}_\d{6})\.json$", filename)
    return match.group(1) if match else None


def get_thinking_mode(summary_path: Union[str, Path]) -> Optional[bool]:
    """
    Look up ``thinking_mode`` in the processing metadata that belongs to a summary file.

    Uses ``processing_metadata_<timestamp>.json`` in the same folder, else a
    ``processing_metadata.json`` whose ``timestamp`` matches.

    Args:
        summary_path: Path of a ``unified_evaluation_summary_*.json``

    Returns:
        The ``thinking_mode`` value (True, False or None), or None if no metadata is found
    """
    summary_file = Path(summary_path)
    model_dir = summary_file.parent

    # Extract timestamp from summary filename
    proc_timestamp = extract_processing_timestamp(summary_file.name)
    if not proc_timestamp:
        return None

    # Look for matching processing_metadata file
    metadata_file = model_dir / f"processing_metadata_{proc_timestamp}.json"
    if metadata_file.exists():
        with open(metadata_file, encoding="utf-8") as fp:
            metadata = json.load(fp)
            return metadata.get("thinking_mode")

    # Fallback: check generic processing_metadata.json
    generic_metadata = model_dir / "processing_metadata.json"
    if generic_metadata.exists():
        with open(generic_metadata, encoding="utf-8") as fp:
            metadata = json.load(fp)
            if metadata.get("timestamp") == proc_timestamp:
                return metadata.get("thinking_mode")

    return None


def load_summary_files() -> List[Dict[str, Any]]:
    """
    Load every ``results/<model>/unified_evaluation_summary*.json`` (one folder level only).

    Returns:
        list: The parsed files, each with ``_file_path`` and ``_thinking_mode`` added, sorted by
        path (this order sets the column order of the per-category tables)
    """
    results_dir = Path(__file__).parent / "results"
    # One level only: results/<run>/unified_evaluation_summary*.json
    summary_files = sorted(results_dir.glob("*/unified_evaluation_summary*.json"))

    data = []
    for f in summary_files:
        with open(f, encoding="utf-8") as fp:
            content = json.load(fp)
            content["_file_path"] = str(f)
            content["_thinking_mode"] = get_thinking_mode(f)
            data.append(content)

    return data


def calculate_d1_from_d4b(details: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """
    Count source-fidelity passes among the sequences whose D4b element match is True.

    Args:
        details: Per-sequence records of one run

    Returns:
        tuple: ``(d1_true_count, d4b_true_count, percentage)``; ``(0, 0, 0.0)`` if no D4b match
    """
    d4b_true = [d for d in details if d.get("Element_Match (D4b)") == True]
    d4b_count = len(d4b_true)

    if d4b_count == 0:
        return 0, 0, 0.0

    d1_true = sum(1 for d in d4b_true if d.get("Source_Fidelity (D1)") == True)
    pct = (d1_true / d4b_count) * 100

    return d1_true, d4b_count, pct


def get_model_name(data: Dict[str, Any]) -> str:
    """
    Return the model instance name, with " (thinking)" or " (no thinking)" if the run has a mode.

    Args:
        data: A loaded summary file (with ``_thinking_mode``)

    Returns:
        str: The display name of the run
    """
    model = data["summary"].get("model_instance", "Unknown")
    thinking_mode = data.get("_thinking_mode")
    if thinking_mode is True:
        model = f"{model} (thinking)"
    elif thinking_mode is False:
        model = f"{model} (no thinking)"
    return model


def extract_category_f1(all_data: List[Dict[str, Any]], category_key: str) -> pd.DataFrame:
    """
    Build a table of per-category F1 scores (in percent, 1 decimal) with one column per run.

    Args:
        all_data: Loaded summary files
        category_key: Key of the category list, e.g. ``"classification_d4a_categories"``

    Returns:
        pandas.DataFrame: Categories (sorted) as rows, runs as columns
    """
    # Collect all categories across all models
    all_categories = set()
    for data in all_data:
        categories = data.get(category_key, [])
        for cat in categories:
            all_categories.add(cat["category"])

    # Build the table
    rows = []
    for category in sorted(all_categories):
        row = {"Category": category}
        for data in all_data:
            model = get_model_name(data)
            categories = data.get(category_key, [])
            f1 = None
            for cat in categories:
                if cat["category"] == category:
                    f1 = round(cat["f1"] * 100, 1)  # Convert to percentage
                    break
            row[model] = f1
        rows.append(row)

    return pd.DataFrame(rows)


def find_all_models_failed(
    all_data: List[Dict[str, Any]], match_field: str, display_fields: List[str]
) -> pd.DataFrame:
    """
    Find the sequences for which the match field is False in every run.

    Sequences are keyed by (Filename, Bracketed_Content, occurrence), so a pair that occurs twice
    in the sample is listed per occurrence (the runs list the sequences in the same order).

    Args:
        all_data: List of loaded model data
        match_field: Field name to check (e.g., 'Classification_Match (D4a)')
        display_fields: Fields copied from the first run's record into the output

    Returns:
        DataFrame with instances where all models failed
    """
    # Group details by (Filename, Bracketed_Content, occurrence)
    instances = defaultdict(dict)

    for data in all_data:
        model = get_model_name(data)
        details = data.get("details", [])
        occurrences = defaultdict(int)

        for d in details:
            pair = (d.get("Filename", ""), d.get("Bracketed_Content", ""))
            occurrences[pair] += 1
            key = pair + (occurrences[pair],)
            if key not in instances:
                instances[key] = {"_info": {f: d.get(f) for f in display_fields}}
            instances[key][model] = d.get(match_field)

    # Find instances where ALL models have FALSE
    model_names = [get_model_name(d) for d in all_data]
    all_failed = []

    for key, results in instances.items():
        # Check if all models have FALSE (not True, not None)
        all_false = all(results.get(m) == False for m in model_names)
        if all_false:
            row = {"Filename": key[0], "Bracketed_Content": key[1]}
            row.update(results["_info"])
            all_failed.append(row)

    return pd.DataFrame(all_failed)


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the summary row of one run.

    Percentages are relative to the processed sequences and rounded to 1 decimal. ``Timestamp`` is
    the time ``eval.py`` was run, not the processing time.

    Args:
        data: A loaded summary file

    Returns:
        dict: One row of the Summary sheet
    """
    summary = data["summary"]
    details = data.get("details", [])

    model = get_model_name(data)
    timestamp = summary.get("timestamp", "")[:16].replace("T", " ")  # e.g., "2026-03-01 10:09"
    processed = summary.get("total_processed", 0)

    # D0: XML wellformedness
    d0 = summary.get("dimension_0_xml_wellformedness", {})
    d0_passed = d0.get("passed", 0)
    d0_pct = (d0_passed / processed * 100) if processed > 0 else 0

    # D2: Schema validation (TEI)
    d2 = summary.get("dimension_2_schema_validation", {})
    d2_tei = d2.get("tei_valid_count", 0)
    d2_tei_pct = (d2_tei / processed * 100) if processed > 0 else 0

    # D2: Project validity
    d2_proj = d2.get("project_valid_count", 0)
    d2_proj_pct = (d2_proj / processed * 100) if processed > 0 else 0

    # D4a: Classification accuracy
    d4a = summary.get("dimension_4a_classification", {})
    d4a_passed = d4a.get("passed", 0)
    d4a_pct = (d4a_passed / processed * 100) if processed > 0 else 0
    d4a_macro_f1 = d4a.get("macro_f1_pct", 0)

    # D4b: Element application
    d4b = summary.get("dimension_4b_element_application", {})
    d4b_passed = d4b.get("passed", 0)
    d4b_pct = (d4b_passed / processed * 100) if processed > 0 else 0
    d4b_macro_f1 = d4b.get("macro_f1_pct", 0)

    # D1: Source fidelity
    d1 = summary.get("dimension_1_source_fidelity", {})
    d1_passed_all = d1.get("passed", 0)
    d1_pct_all = (d1_passed_all / processed * 100) if processed > 0 else 0
    d1_macro_f1 = d1.get("macro_f1_pct", 0)

    # D1: Source fidelity (from D4b TRUE records only - for conditional analysis)
    d1_passed_d4b, d4b_base, d1_pct_d4b = calculate_d1_from_d4b(details)

    return {
        "Model": model,
        "Timestamp": timestamp,
        "Processed": processed,
        "D0 XML Wellformed": d0_passed,
        "D0 %": round(d0_pct, 1),
        "D2 TEI Valid": d2_tei,
        "D2 TEI %": round(d2_tei_pct, 1),
        "D2 Project Valid": d2_proj,
        "D2 Project %": round(d2_proj_pct, 1),
        "D1 Source Fidelity": d1_passed_all,
        "D1 %": round(d1_pct_all, 1),
        "D1 Macro F1": round(d1_macro_f1, 1),
        "D4a Classification": d4a_passed,
        "D4a %": round(d4a_pct, 1),
        "D4a Macro F1": round(d4a_macro_f1, 1),
        "D4b Element Match": d4b_passed,
        "D4b %": round(d4b_pct, 1),
        "D4b Macro F1": round(d4b_macro_f1, 1),
        "D1 (of D4b)": d1_passed_d4b,
        "D1 % (of D4b)": round(d1_pct_d4b, 1),
    }


def main() -> int:
    """
    Aggregate all summary files and write ``aggregated_results.xlsx`` and ``.json``.

    The Summary rows are sorted by model name, then by the evaluation time (``Timestamp``).

    Returns:
        int: Exit code, 0 on success, 1 if there are no summary files
    """
    all_data = load_summary_files()

    if not all_data:
        print("Error: no evaluation summary files found in results/.", file=sys.stderr)
        return 1

    rows = [extract_metrics(d) for d in all_data]
    df = pd.DataFrame(rows)

    # Sort by model name, then timestamp
    df = df.sort_values(["Model", "Timestamp"]).reset_index(drop=True)

    # Display
    print("\n=== Aggregated Evaluation Results ===\n")
    print(df.to_string(index=False))

    # Create category F1 tables
    df_d4a_categories = extract_category_f1(all_data, "classification_d4a_categories")
    df_d4b_categories = extract_category_f1(all_data, "element_application_d4b_categories")
    df_d1_categories = extract_category_f1(all_data, "source_fidelity_d1_categories")
    df_d4b_relaxed = extract_category_f1(all_data, "element_application_d4b_relaxed_categories")

    # Find instances where all models failed
    df_all_failed_d4a = find_all_models_failed(
        all_data, "Classification_Match (D4a)", ["Gold_Classification", "LLM_Classification"]
    )
    df_all_failed_d4b = find_all_models_failed(
        all_data, "Element_Match (D4b)", ["Gold_Classification", "Gold_element", "LLM_element"]
    )
    df_all_failed_d1 = find_all_models_failed(
        all_data,
        "Source_Fidelity (D1)",
        ["Gold_Classification", "Bracketed_Content_Normalized", "LLM_Output_Normalized"],
    )

    # Save to Excel with wider columns
    output_path = Path(__file__).parent / "aggregated_results.xlsx"

    def auto_fit_columns(worksheet: Any, dataframe: pd.DataFrame) -> None:
        """Set each column's width to its longest value or header, plus 3 characters."""
        for i, col in enumerate(dataframe.columns):
            max_len = max(dataframe[col].astype(str).map(len).max(), len(str(col))) + 3
            worksheet.column_dimensions[get_column_letter(i + 1)].width = max_len

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Summary")
        auto_fit_columns(writer.sheets["Summary"], df)

        df_d4a_categories.to_excel(writer, index=False, sheet_name="D4a Classification F1")
        auto_fit_columns(writer.sheets["D4a Classification F1"], df_d4a_categories)

        df_d4b_categories.to_excel(writer, index=False, sheet_name="D4b Element F1")
        auto_fit_columns(writer.sheets["D4b Element F1"], df_d4b_categories)

        df_d1_categories.to_excel(writer, index=False, sheet_name="D1 Source Fidelity F1")
        auto_fit_columns(writer.sheets["D1 Source Fidelity F1"], df_d1_categories)

        df_d4b_relaxed.to_excel(writer, index=False, sheet_name="D4b Relaxed F1")
        auto_fit_columns(writer.sheets["D4b Relaxed F1"], df_d4b_relaxed)

        # All models failed sheets
        if not df_all_failed_d4a.empty:
            df_all_failed_d4a.to_excel(writer, index=False, sheet_name="All Failed D4a")
            auto_fit_columns(writer.sheets["All Failed D4a"], df_all_failed_d4a)

        if not df_all_failed_d4b.empty:
            df_all_failed_d4b.to_excel(writer, index=False, sheet_name="All Failed D4b")
            auto_fit_columns(writer.sheets["All Failed D4b"], df_all_failed_d4b)

        if not df_all_failed_d1.empty:
            df_all_failed_d1.to_excel(writer, index=False, sheet_name="All Failed D1")
            auto_fit_columns(writer.sheets["All Failed D1"], df_all_failed_d1)

    print(f"\nAll models failed - D4a Classification: {len(df_all_failed_d4a)} instances")
    print(f"All models failed - D4b Element: {len(df_all_failed_d4b)} instances")
    print(f"All models failed - D1 Source Fidelity: {len(df_all_failed_d1)} instances")
    print(f"\nSaved to: {output_path}")

    # Save to JSON
    json_output = {
        "summary": df.to_dict(orient="records"),
        "d4a_classification_f1": df_d4a_categories.to_dict(orient="records"),
        "d4b_element_f1": df_d4b_categories.to_dict(orient="records"),
        "d1_source_fidelity_f1": df_d1_categories.to_dict(orient="records"),
        "d4b_relaxed_f1": df_d4b_relaxed.to_dict(orient="records"),
        "all_failed_d4a": df_all_failed_d4a.to_dict(orient="records"),
        "all_failed_d4b": df_all_failed_d4b.to_dict(orient="records"),
        "all_failed_d1": df_all_failed_d1.to_dict(orient="records"),
    }

    json_path = Path(__file__).parent / "aggregated_results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(json_output, fp, indent=2, ensure_ascii=False)
    print(f"Saved to: {json_path}")
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
