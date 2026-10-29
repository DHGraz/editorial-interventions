"""
Evaluate one LLM run's TEI encodings of bracketed sequences against the gold standard.

Usage (from ``evaluation/``)::

    python eval.py <model_instance>      # e.g. qwen3-14B-Q6; asks which processing run to use
    python eval.py                       # asks for the model instance first

For every sampled bracketed sequence (``data/input/brackets_sample.xlsx``), the model's encoding
is taken from ``data/output/<model_instance>/processing_<timestamp>/xml_update_mapping.json`` and
scored on D0 (well-formed XML), D1 (source fidelity), D2 (TEI schema validity with Jing, project
rules), D4a (intervention type) and D4b (TEI elements); the gold-standard types come from
``data/sampling/brackets_statistics.xlsx``. Writes to ``results/<model_instance>/``: the Excel
report ``evaluation_<model>_<timestamp>.xlsx``, the summary
``unified_evaluation_summary_<model>_<timestamp>.json``, two heatmaps, and a copy of the run's
``processing_metadata.json``.

Contains hard-coded corrections for a few sequences (search for ``HACK``), documented in
``evaluation/results/README.md``.
"""

import argparse
import difflib
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from lxml import etree

# Repository root on sys.path, for the root-level utils package. These imports have to come after
# it. "config" is evaluation/config.py (the script's folder is on sys.path; there is no config.py at
# the root).
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import config  # noqa: E402

from utils.console_logging import configure_console_logging  # noqa: E402
from utils.gold_standard_stats import categorize_intervention_type  # noqa: E402
from utils.xml_utils import (  # noqa: E402
    compare_xml_elements,
    extract_all_text_content,
    extract_element_content_by_xpath,
    get_latest_xml_mapping_file,
    get_tei_encoding_for_element,
    load_xml_mapping,
    normalize_text_for_editorial_interventions,
)

# evaluation/utils/evaluation_utils.py, loaded by path: "utils" already names the root package
_eval_utils_file = Path(__file__).parent / "utils" / "evaluation_utils.py"
_spec = importlib.util.spec_from_file_location("evaluation_utils", _eval_utils_file)
_eval_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_utils)

create_empty_evaluation_result = _eval_utils.create_empty_evaluation_result
create_parent_xpath = _eval_utils.create_parent_xpath
normalize_apostrophes = _eval_utils.normalize_apostrophes
remove_note_elements = _eval_utils.remove_note_elements

try:
    from utils.xml_utils import JingValidator  # noqa: E402
except ImportError:
    JingValidator = None

# Intervention type categories, in the order of all per-category outputs
INTERVENTION_CATEGORIES = [
    "Abbreviation",
    "Correction",
    "Supplement",
    "Omission",
    "Uncertainty",
    "Redundancy",
]

# TEI elements counted as editorial interventions in the gold-standard element
INTERVENTION_ELEMENTS = [
    "supplied",
    "gap",
    "unclear",
    "choice",
    "abbr",
    "expan",
    "sic",
    "corr",
    "add",
    "del",
    "subst",
    "surplus",
]

# Overall score factor for an encoding that breaks the project rules (D2 project validity)
SCHEMA_FACTOR_RULES_BROKEN = 0.75


def count_brackets_and_types(bracketed_content: str, gold_classification: str) -> str:
    """
    Count bracket pairs and intervention types.

    Args:
        bracketed_content: Text with brackets like "Febr[uar] [17]97"
        gold_classification: Intervention types like "Abbreviation; Supplement"

    Returns:
        String in format "bracket_count;type_count" (e.g., "2;2")
    """
    bracket_count = bracketed_content.count("[") if bracketed_content else 0

    if not gold_classification or gold_classification in ["Unknown", None]:
        type_count = 0
    else:
        # Gold_Classification uses "; " as separator (e.g., "Abbreviation; Supplement")
        type_count = len([t.strip() for t in gold_classification.split(";") if t.strip()])

    return f"{bracket_count};{type_count}"


def normalize_bracketed_content_by_gold_types(
    bracketed_content: str, gold_classification: str, filename: Optional[str] = None
) -> str:
    """
    Build the text a correct encoding of a bracketed sequence should contain (for D1).

    The gold-standard types are assigned to the bracket pairs from left to right (type *i* to pair
    *i*); the pairs are replaced from right to left so that positions stay valid. Per type:

    - Abbreviation: abbreviated form followed by expanded form, e.g. ``Capt[ai]n`` ->
      ``CaptnCaptain``
      (prefix and suffix up to the next space or bracket; no duplicated prefix if the content
      already starts with it)
    - Supplement: content without brackets, ``[17]97`` -> ``1797``
    - Redundancy, Correction: ``[sic]``/``[sic!]`` removed, the word before it kept; other content
      kept without brackets
    - Omission: bracket and content removed
    - Uncertainty: ``[?]`` removed; otherwise the content without question marks
    - unknown types: content kept without brackets

    With more bracket pairs than types, a single type is applied to all pairs (e.g.
    ``L[ieu]t[ena]nd`` -> ``LtndLieutenand``); with several types the last type is applied to the
    last pair first, then the remaining types as above (a single remaining type to all remaining
    pairs). Six letters have hard-coded results or a trailing ``)``/``,`` handled separately (see
    the HACK block at the top).

    Args:
        bracketed_content: Text with brackets like "Febr[uar] [17]97"
        gold_classification: Intervention types like "Abbreviation; Supplement"
        filename: Letter file name, for the hard-coded special cases

    Returns:
        str: The normalised text; ``bracketed_content`` unchanged if it is empty, has no brackets,
        or the classification is missing or "Unknown"
    """
    if not bracketed_content:
        return bracketed_content

    # =====================================================================
    # HARDCODED HACKS for specific files
    # =====================================================================

    # letter174.xml: [sic] should be treated as supplied
    if filename == "letter174.xml" and "[sic]" in bracketed_content:
        return "Personne dans toute l'armée ne l'aime"

    # letter31.xml: 4;2 case with abbreviation + uncertainty
    if filename == "letter31.xml" and "A[llgemeinen]" in bracketed_content:
        return "A LitZAllgemeinen Literatur-Zeitung gemacht"

    # letter7726.xml: affezi[ionatissi]mo has overlapping "i" - remove duplicate
    if filename == "letter7726.xml" and "affezi[ionatissi]mo" in bracketed_content:
        return "affezimoaffezionatissimo"

    # letter2864.xml: My selection[?] should normalize to sic | corr format
    if filename == "letter2864.xml" and "My selection[?]" in bracketed_content:
        return "My selection | election"

    # letter732.xml and letter7514.xml: handle trailing ) or , separately
    trailing_char = None
    if filename in ["letter732.xml", "letter7514.xml"]:
        if bracketed_content.endswith(")"):
            trailing_char = ")"
            bracketed_content = bracketed_content[:-1]
        elif bracketed_content.endswith(","):
            trailing_char = ","
            bracketed_content = bracketed_content[:-1]

    # =====================================================================

    if not gold_classification or gold_classification in ["Unknown", None]:
        result = bracketed_content
        # Add trailing char back if we removed it
        if trailing_char:
            result = result + trailing_char
        return result

    # Parse intervention types (they are separated by "; ")
    intervention_types = [t.strip().lower() for t in gold_classification.split(";") if t.strip()]

    if not intervention_types:
        return bracketed_content + (trailing_char or "")

    # Find all bracket pairs in the text (with their positions)
    bracket_pattern = r"\[([^\]]*)\]"
    matches = list(re.finditer(bracket_pattern, bracketed_content))

    if not matches:
        return bracketed_content

    bracket_count = len(matches)
    type_count = len(intervention_types)

    # Special case: more brackets than types
    if bracket_count > type_count:
        # Case: multiple brackets with multiple types (e.g., 4;2 for "A[llgemeinen]
        # Lit[eratur-]Z[eitung] gemacht[?]")
        # Process from back: last type gets last bracket, remaining brackets get remaining types
        if type_count > 1:
            # Process from back: apply last type to last bracket
            last_type = intervention_types[-1].lower()
            last_match = matches[-1]
            last_bracket_content = last_match.group(1)

            # Apply normalization to last bracket based on last type
            if last_type == "uncertainty":
                # Remove [?] or [content?] - keep text without ? and brackets
                if last_bracket_content == "?":
                    replacement = ""
                else:
                    replacement = last_bracket_content.replace("?", "")
            elif last_type == "supplement":
                replacement = last_bracket_content
            elif last_type == "omission":
                replacement = ""
            elif last_type in ["redundancy", "correction"]:
                if last_bracket_content.lower() in ["sic", "sic!"]:
                    # Remove only [sic], keep the word before it
                    replacement = ""
                else:
                    replacement = last_bracket_content
            else:
                replacement = last_bracket_content

            # Replace the last bracket pair in the string
            result = (
                bracketed_content[: last_match.start()]
                + replacement
                + bracketed_content[last_match.end() :]
            )

            # Now handle remaining brackets with remaining types (n-1 brackets, type_count-1 types)
            remaining_matches = list(re.finditer(bracket_pattern, result))
            remaining_types = intervention_types[:-1]  # All types except the last

            # If remaining is n;1 (e.g., 3;1), handle as single type
            if len(remaining_types) == 1:
                remaining_type = remaining_types[0].lower()
                if remaining_type == "abbreviation":
                    abbr_form = re.sub(bracket_pattern, "", result)
                    expan_form = re.sub(bracket_pattern, r"\1", result)
                    return abbr_form + expan_form + (trailing_char or "")
                elif remaining_type == "supplement":
                    return re.sub(bracket_pattern, r"\1", result) + (trailing_char or "")
                elif remaining_type == "omission":
                    return re.sub(bracket_pattern, "", result) + (trailing_char or "")
                elif remaining_type == "uncertainty":

                    def replace_unclear(m: re.Match) -> str:
                        """Content of the matched bracket pair, without question marks."""
                        return m.group(1).replace("?", "")

                    return re.sub(bracket_pattern, replace_unclear, result) + (trailing_char or "")
                else:
                    return re.sub(bracket_pattern, r"\1", result) + (trailing_char or "")

            # Otherwise, continue with normal processing on the modified result
            bracketed_content = result
            matches = remaining_matches
            intervention_types = remaining_types
            bracket_count = len(matches)
            type_count = len(intervention_types)

        # Case: multiple brackets but single type (e.g., "2;1" for L[ieu]t[ena]nd)
        # All brackets belong to the same intervention
        elif type_count == 1:
            single_type = intervention_types[0].lower()

            if single_type == "abbreviation":
                # L[ieu]t[ena]nd -> LtndLieutenand
                # Abbreviated form: all text without bracket contents (L + t + nd = Ltnd)
                # Expanded form: all text with bracket contents (L + ieu + t + ena + nd =
                # Lieutenand)
                abbr_form = re.sub(bracket_pattern, "", bracketed_content)
                expan_form = re.sub(bracket_pattern, r"\1", bracketed_content)
                return abbr_form + expan_form + (trailing_char or "")

            elif single_type == "supplement":
                # Just remove all brackets, keep all content
                return re.sub(bracket_pattern, r"\1", bracketed_content) + (trailing_char or "")

            elif single_type == "omission":
                # Remove all brackets and their content
                return re.sub(bracket_pattern, "", bracketed_content) + (trailing_char or "")

            elif single_type == "uncertainty":
                # Remove all brackets and question marks, keep other content
                def replace_unclear(m: re.Match) -> str:
                    """Content of the matched bracket pair, without question marks."""
                    return m.group(1).replace("?", "")

                return re.sub(bracket_pattern, replace_unclear, bracketed_content) + (
                    trailing_char or ""
                )

            elif single_type in ["redundancy", "correction"]:
                # Remove [sic] patterns AND the word before them
                # Use a regex that matches: optional_space + word + optional_space + [sic]
                sic_pattern = r"(\s*\S+)?\s*\[(sic!?)\]"

                def replace_sic_with_word(m: re.Match) -> str:
                    """Remove ``[sic]``/``[sic!]`` and the word before it (keep other matches)."""
                    content_in_bracket = m.group(2)
                    if content_in_bracket and content_in_bracket.lower() in ["sic", "sic!"]:
                        return ""  # Remove word before + [sic]
                    return m.group(0)  # Keep as is

                result = re.sub(
                    sic_pattern, replace_sic_with_word, bracketed_content, flags=re.IGNORECASE
                )
                # Also handle other bracket patterns (non-sic)
                result = re.sub(
                    bracket_pattern,
                    lambda m: m.group(1) if m.group(1).lower() not in ["sic", "sic!"] else "",
                    result,
                )
                # Clean up double spaces
                return re.sub(r" +", " ", result).strip() + (trailing_char or "")

            else:
                # Unknown type with multiple brackets: just remove brackets
                return re.sub(bracket_pattern, r"\1", bracketed_content) + (trailing_char or "")

    # Normal case: type count matches or exceeds bracket count
    # Mapping: type[i] corresponds to bracket[i] (left-to-right matching)
    # But we PROCESS from right to left to avoid index shifting issues
    result = bracketed_content

    # Determine how many pairs we can process (minimum of brackets and types)
    pairs_to_process = min(bracket_count, type_count)

    # Process from right to left (highest index first)
    for pair_idx in range(pairs_to_process - 1, -1, -1):
        # Re-find matches in current result string (positions may have shifted)
        matches = list(re.finditer(bracket_pattern, result))

        if pair_idx >= len(matches):
            continue

        match = matches[pair_idx]
        bracket_content = match.group(1)  # Content inside brackets
        bracket_start = match.start()
        bracket_end = match.end()

        intervention_type = intervention_types[pair_idx]

        # Find prefix (text before bracket, bounded by space/newline or previous bracket)
        prefix_start = bracket_start
        for pos in range(bracket_start - 1, -1, -1):
            if result[pos] in " \t\n":
                prefix_start = pos + 1
                break
            if result[pos] == "]":
                prefix_start = pos + 1
                break
            if pos == 0:
                prefix_start = 0
        prefix = result[prefix_start:bracket_start]

        # Find suffix (text after bracket, bounded by space/newline or next bracket)
        suffix_end = bracket_end
        for pos in range(bracket_end, len(result)):
            if result[pos] in " \t\n":
                suffix_end = pos
                break
            if result[pos] == "[":
                suffix_end = pos
                break
            if pos == len(result) - 1:
                suffix_end = len(result)
        suffix = result[bracket_end:suffix_end]

        # Apply normalization based on intervention type
        if intervention_type == "abbreviation":
            # Capt[ai]n -> CaptnCaptain (abbr_form + expan_form)
            # abbr_form = prefix + suffix (e.g., Captn)
            # expan_form = prefix + bracket_content + suffix (e.g., Captain)
            # Check if bracket_content already starts with the prefix (case-insensitive)
            if prefix and bracket_content.lower().startswith(prefix.lower()):
                # Content already includes the prefix, e.g., E[Exzellenz]
                # abbr_form = prefix + suffix, expan_form = bracket_content + suffix
                abbr_form = prefix + suffix
                expan_form = bracket_content + suffix
            else:
                # Normal case: Capt[ai]n
                # abbr_form = Captn, expan_form = Captain
                abbr_form = prefix + suffix
                expan_form = prefix + bracket_content + suffix
            replacement = abbr_form + expan_form
            # Replace from prefix_start to suffix_end
            result = result[:prefix_start] + replacement + result[suffix_end:]

        elif intervention_type == "supplement":
            # Just remove brackets, keep content: [17]97 -> 1797
            replacement = bracket_content
            result = result[:bracket_start] + replacement + result[bracket_end:]

        elif intervention_type == "redundancy":
            # [sic] or [sic!] -> remove only [sic], keep the word before it
            if bracket_content.lower() in ["sic", "sic!"]:
                # Just remove [sic], keep the word before it
                result = result[:bracket_start] + result[bracket_end:]
            else:
                # Just remove brackets, keep content
                result = result[:bracket_start] + bracket_content + result[bracket_end:]

        elif intervention_type == "omission":
            # [...] or [xxx] -> remove completely
            result = result[:bracket_start] + result[bracket_end:]

        elif intervention_type == "uncertainty":
            # 3 options:
            # 1) characters + ? like in un[cer?]tain: remove ? and brackets, keep content
            # 2) only question mark [?]: remove completely
            # 3) no question mark: just remove brackets
            if bracket_content == "?":
                # [?] -> remove completely
                result = result[:bracket_start] + result[bracket_end:]
            elif "?" in bracket_content:
                # Has question mark: remove ? and brackets, keep rest
                cleaned_content = bracket_content.replace("?", "")
                result = result[:bracket_start] + cleaned_content + result[bracket_end:]
            else:
                # No question mark: just remove brackets
                result = result[:bracket_start] + bracket_content + result[bracket_end:]

        elif intervention_type == "correction":
            # [sic] or [sic!] -> remove only [sic], keep the word before it
            if bracket_content.lower() in ["sic", "sic!"]:
                # Just remove [sic], keep the word before it
                result = result[:bracket_start] + result[bracket_end:]
            else:
                # For other correction patterns, just remove brackets
                result = result[:bracket_start] + bracket_content + result[bracket_end:]
        else:
            # Unknown type: just remove brackets, keep content
            result = result[:bracket_start] + bracket_content + result[bracket_end:]

    # Add trailing char back if we removed it (for letter732.xml and letter7514.xml)
    if trailing_char:
        result = result + trailing_char

    return result


def ensure_evaluation_directories(model_instance: str) -> str:
    """
    Create ``evaluation/results/<model_instance>/`` (next to this script).

    Args:
        model_instance: Model instance name

    Returns:
        str: Path of the folder
    """
    evaluation_base = str(Path(__file__).resolve().parent / "results" / model_instance)
    os.makedirs(evaluation_base, exist_ok=True)
    return evaluation_base


def extract_timestamp_from_mapping_file(mapping_file: Optional[str]) -> str:
    """
    Take the run timestamp from the mapping file's folder name ``processing_YYYYMMDD_HHMMSS``.

    Args:
        mapping_file: Path of an ``xml_update_mapping.json``, or None

    Returns:
        str: ``YYYYMMDD_HHMMSS``; the current time if there is no file or the folder name does not
        match
    """
    if not mapping_file:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get parent folder name (e.g., processing_20260228_192607)
    parent_folder = Path(mapping_file).parent.name

    # Look for timestamp pattern in folder name
    timestamp_match = re.search(r"processing_(\d{8}_\d{6})", parent_folder)

    if timestamp_match:
        return timestamp_match.group(1)
    else:
        # Fallback to current timestamp if pattern not found
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamped_evaluation_filename(
    model_instance: str, base_name: str, mapping_file: Optional[str] = None
) -> str:
    """
    Build the output path ``evaluation/results/<model_instance>/<base_name>_<model>_<timestamp>``.

    In ``<model>``, ``-`` and ``.`` of the model instance are replaced by ``_``. Creates the
    results folder.

    Args:
        model_instance: Model instance name
        base_name: File name prefix, e.g. "evaluation"
        mapping_file: Mapping file whose folder gives the timestamp

    Returns:
        str: Path without extension
    """
    timestamp = extract_timestamp_from_mapping_file(mapping_file)
    sanitized_model_instance = model_instance.replace("-", "_").replace(".", "_")
    filename = f"{base_name}_{sanitized_model_instance}_{timestamp}"

    # Ensure the directory structure exists
    evaluation_dir = ensure_evaluation_directories(model_instance)

    return os.path.join(evaluation_dir, filename)


def load_brackets_extraction_data(
    input_file: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load the sampled text elements and expand them to one row per bracketed sequence.

    Args:
        input_file: Optional path to input Excel file (relative paths are resolved against the
            repository root). If None, uses config.EVALUATION_INPUT_FILE.

    Returns:
        DataFrame with Filename, Full_XPath, ID, Bracketed_Content and Parent_Element_Content
        columns, or None if the file is missing or lacks a required column (the error is printed).
    """
    # Get project root and resolve input file path
    project_root = Path(__file__).parent.parent
    if input_file is None:
        input_file = project_root / config.EVALUATION_INPUT_FILE
    elif not Path(input_file).is_absolute():
        # If relative path, resolve relative to project root
        input_file = project_root / input_file

    try:
        # Read Excel file
        df_excel = pd.read_excel(input_file, engine="openpyxl")

        # Check required columns
        required_columns = ["Filename", "XPath", "Bracketed_Content"]
        missing_columns = [col for col in required_columns if col not in df_excel.columns]
        if missing_columns:
            print(f"Error: {input_file} lacks required columns: {missing_columns}", file=sys.stderr)
            return None

        # Transform Excel data to DataFrame format with one row per bracketed sequence
        rows = []
        for _, row in df_excel.iterrows():
            filename = row.get("Filename", "")
            xpath = row.get("XPath", "")
            element_id = row.get("ID", "")
            bracketed_content_str = str(row.get("Bracketed_Content", ""))
            parent_element_content = row.get("Parent_Element_Content", "")

            # Split bracketed content by pipe separator to get individual sequences
            if pd.isna(bracketed_content_str) or bracketed_content_str == "":
                bracketed_sequences = []
            else:
                # Split by pipe and strip whitespace
                bracketed_sequences = [
                    seq.strip() for seq in bracketed_content_str.split("|") if seq.strip()
                ]

            # Create one row for each bracketed sequence
            for bracketed_content in bracketed_sequences:
                rows.append(
                    {
                        "Filename": filename,
                        "Full_XPath": xpath,
                        "ID": element_id,
                        "Bracketed_Content": bracketed_content,
                        "Parent_Element_Content": parent_element_content,
                    }
                )

        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} bracketed sequences from {len(df_excel)} elements in {input_file}")
        return df

    except FileNotFoundError:
        print(
            f"Error: {input_file} not found. Please ensure the Excel input file exists.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Error: could not load {input_file}: {e}", file=sys.stderr)
        return None


def reconstruct_edited_element(
    parent_element_xml: str, mapping_data: Dict, filename: str, element_id: str
) -> Optional[str]:
    """
    Rebuild the model's version of an element by inserting its encodings into the source XML.

    Each replacement whose ``original_item`` occurs in the element's text is applied once per
    occurrence, as a plain string replacement of the first remaining occurrence in the XML string
    (the position sorting does not change the result). The result is parsed with lxml's recovering
    parser, so a malformed encoding is repaired by lxml rather than rejected.

    Args:
        parent_element_xml: The full parent element XML as string (from Parent_Element_Content)
        mapping_data: The loaded XML update mapping data
        filename: The XML filename
        element_id: The element ID to find in mapping

    Returns:
        The reconstructed element XML as string; the original XML if there is no mapping entry or
        replacement or an error occurs; None if ``parent_element_xml`` is empty
    """
    if not parent_element_xml or pd.isna(parent_element_xml):
        return None

    if filename not in mapping_data:
        return parent_element_xml  # No changes, return original

    # Find the element entry in the mapping
    element_entries = [
        entry for entry in mapping_data[filename] if entry.get("element_id") == element_id
    ]

    if not element_entries:
        return parent_element_xml  # No mapping for this element, return original

    element_entry = element_entries[0]
    replacements = element_entry.get("replacements", [])

    if not replacements:
        return parent_element_xml  # No replacements, return original

    try:
        # Parse the parent element XML
        parser = etree.XMLParser(recover=True)
        element = etree.fromstring(parent_element_xml.encode("utf-8"), parser)

        # Get all text content from the element (including text in child elements)
        # We'll work with the element's text and tail text
        def get_all_text_content(elem: etree._Element) -> List[str]:
            """Recursively get all text content from an element"""
            text_parts = []
            if elem.text:
                text_parts.append(elem.text)
            for child in elem:
                text_parts.extend(get_all_text_content(child))
                if child.tail:
                    text_parts.append(child.tail)
            return text_parts

        # For simplicity, we'll do string replacement on the XML string
        # This works because the original_item appears in the text content
        updated_xml = parent_element_xml

        # Sort replacements by position in text (process from end to start to avoid position shifts)
        # Get the full element text to find positions
        full_text = "".join(get_all_text_content(element))

        # Create list of (position, replacement) tuples
        replacement_positions = []
        for replacement in replacements:
            original_item = replacement.get("original_item", "")
            if original_item in full_text:
                # Find all occurrences
                start = 0
                while True:
                    pos = full_text.find(original_item, start)
                    if pos == -1:
                        break
                    replacement_positions.append((pos, replacement))
                    start = pos + 1

        # Sort by position descending to process from end to start
        replacement_positions.sort(key=lambda x: x[0], reverse=True)

        # Apply replacements (from end to start to avoid position shifts)
        for _pos, replacement in replacement_positions:
            original_item = replacement.get("original_item", "")
            tei_encoding = replacement.get("tei_encoding", "")

            if original_item and tei_encoding:
                # Find and replace in the XML string
                # We need to be careful - the original_item might appear in text nodes
                # For evaluation purposes, we'll do a simple string replacement
                # This should work because the bracketed sequences are unique in the context
                updated_xml = updated_xml.replace(
                    original_item, tei_encoding, 1
                )  # Replace first occurrence

        # Validate the result is well-formed XML
        try:
            reconstructed = etree.fromstring(updated_xml.encode("utf-8"), parser)
            # Return as string
            return etree.tostring(reconstructed, encoding="unicode", method="xml")
        except etree.XMLSyntaxError:
            # If reconstruction fails, return original
            print(
                f"Warning: could not rebuild element {element_id}, using the source element",
                file=sys.stderr,
            )
            return parent_element_xml

    except Exception as e:
        print(
            f"Warning: could not rebuild element {element_id}, using the source element: {e}",
            file=sys.stderr,
        )
        return parent_element_xml


@lru_cache(maxsize=None)
def _unchanged_in_gold(source_xml: str, gold_xml: str) -> List[bool]:
    """
    For every character of source_xml, whether it survives unchanged in gold_xml.

    The gold standard encodes only the sampled bracketed sequences and leaves all
    other brackets in the element as they are, so a character-level alignment of
    the two elements tells sampled from unsampled bracket occurrences. Results are cached.

    Args:
        source_xml: Source element as XML string
        gold_xml: Gold-standard element as XML string

    Returns:
        list: One bool per character of ``source_xml``
    """
    unchanged = [False] * len(source_xml)
    matcher = difflib.SequenceMatcher(None, source_xml, gold_xml, autojunk=False)
    for a, _, size in matcher.get_matching_blocks():
        unchanged[a : a + size] = [True] * size
    return unchanged


def find_sampled_occurrences(source_xml: str, gold_xml: str, bracketed_content: str) -> List[int]:
    """
    Return the start offsets of bracketed_content in source_xml that the gold standard encoded.

    An element can contain the same bracketed text more than once, e.g. an unsampled
    H[er]r before the sampled [er]. Searching for the first occurrence then picks the
    wrong one and shifts the order of all sequences in the element. Occurrences that
    appear unchanged in the gold standard were not encoded and are skipped.

    Args:
        source_xml: Source element as XML string
        gold_xml: Gold-standard element as XML string
        bracketed_content: The bracketed sequence

    Returns:
        list: Offsets in ``source_xml``, in ascending order; empty if an argument is empty
    """
    if not source_xml or not gold_xml or not bracketed_content:
        return []
    unchanged = _unchanged_in_gold(source_xml, gold_xml)
    positions = []
    start = source_xml.find(bracketed_content)
    while start != -1:
        if not all(unchanged[start : start + len(bracketed_content)]):
            positions.append(start)
        start = source_xml.find(bracketed_content, start + 1)
    return positions


def get_goldstandard_elements_for_position(
    brackets_data: pd.DataFrame,
    current_row_index: int,
    gold_element_diff_summary: Optional[str],
    gold_element_xml: Optional[str] = None,
) -> str:
    """
    Get the goldstandard elements that correspond to the current bracketed sequence position.

    Args:
        brackets_data: DataFrame with all bracketed sequences
        current_row_index: Index of the current row being processed
        gold_element_diff_summary: Comma-separated string of all goldstandard elements
        gold_element_xml: Gold standard parent element (XML string), used to locate
            the sampled occurrence of each sequence in the parent element

    Returns:
        str: Comma-separated elements for this position, or "No elements" if none
    """
    if not gold_element_diff_summary or gold_element_diff_summary == "No changes":
        return "No elements"

    # Get current row info
    current_row = brackets_data.iloc[current_row_index]
    current_filename = current_row["Filename"]
    current_xpath = current_row["Full_XPath"]
    current_bracketed_content = current_row["Bracketed_Content"]
    current_row_label = brackets_data.index[current_row_index]

    # Find all rows with the same PARENT element (group by parent xpath, not full xpath with text
    # node index)
    # This ensures all text nodes within the same parent element are processed together
    current_parent_xpath = create_parent_xpath(current_xpath)
    same_element_rows = brackets_data[
        (brackets_data["Filename"] == current_filename)
        & (brackets_data["Full_XPath"].apply(create_parent_xpath) == current_parent_xpath)
    ]

    # Parse the goldstandard elements
    if gold_element_diff_summary == "No changes":
        return "No elements"

    gold_elements = [elem.strip() for elem in gold_element_diff_summary.split(",")]

    # Check if this is a single bracketed sequence or multiple sequences
    # Get all bracketed sequences for this element, sorted by their position in the parent element
    # text
    # This ensures the order matches the text order, not the DataFrame row order
    if len(same_element_rows) > 1:
        # Multiple brackets in same element - sort by position in parent element
        parent_content = same_element_rows.iloc[0].get("Parent_Element_Content", "")
        if parent_content:
            # Create list of (position, row label, bracketed_content) tuples
            bracket_positions = []
            claimed_positions = set()
            for idx, row in same_element_rows.iterrows():
                bracketed_content = row["Bracketed_Content"]
                # Find position of bracketed content in parent element
                # First try the occurrence the gold standard encoded (each occurrence is
                # claimed once, so identical sequences in one element get their own positions)
                pos = -1
                for candidate in find_sampled_occurrences(
                    parent_content, gold_element_xml, bracketed_content
                ):
                    if candidate not in claimed_positions:
                        pos = candidate
                        claimed_positions.add(candidate)
                        break
                # Otherwise try to find the exact bracketed content (with brackets)
                if pos == -1:
                    pos = parent_content.find(bracketed_content)
                if pos == -1:
                    # If not found, try finding the text without brackets
                    text_without_brackets = re.sub(r"\[.*?\]", "", bracketed_content).strip()
                    if text_without_brackets:
                        # Find position of this text in parent content
                        pos = parent_content.find(text_without_brackets)
                        if pos == -1:
                            # Fallback: try finding just the first few characters
                            first_chars = (
                                text_without_brackets[:10]
                                if len(text_without_brackets) > 10
                                else text_without_brackets
                            )
                            pos = parent_content.find(first_chars)
                    else:
                        # If no text without brackets, try to find any part of the content
                        # Extract content from first bracket
                        bracket_match = re.search(r"\[([^\]]+)\]", bracketed_content)
                        if bracket_match:
                            bracket_text = bracket_match.group(1)
                            pos = parent_content.find(bracket_text) if bracket_text else 999999
                        else:
                            pos = 999999
                bracket_positions.append((pos if pos != -1 else 999999, idx, bracketed_content))

            # Sort by position
            bracket_positions.sort(key=lambda x: x[0])
            ordered_row_labels = [label for _, label, _ in bracket_positions]
            all_bracketed_sequences = [bc for _, _, bc in bracket_positions]
        else:
            # No parent content available, use DataFrame order as fallback
            ordered_row_labels = same_element_rows.index.tolist()
            all_bracketed_sequences = same_element_rows["Bracketed_Content"].tolist()
    else:
        ordered_row_labels = same_element_rows.index.tolist()
        all_bracketed_sequences = same_element_rows["Bracketed_Content"].tolist()

    # If only one bracketed sequence, need to determine how many interventions it contains
    # Count the number of bracket pairs in the bracketed content to match elements accordingly
    if len(all_bracketed_sequences) == 1:
        # Count bracket pairs in the current bracketed content
        bracket_count = current_bracketed_content.count("[")

        # Apply the same grouping logic as gold_standard_stats.py group_single
        def group_elements_by_editorial_interventions_single(
            elements: List[str], expected_count: int
        ) -> str:
            """Group elements for a single sequence with multiple brackets.

            Processes elements in document order, treating choice patterns as
            single interventions.
            """
            grouped = []
            rest = elements.copy()
            interventions = 0

            while interventions < expected_count and rest:
                elem = rest[0]

                # Check if this starts a choice pattern (counts as 1 intervention)
                if elem == "choice":
                    # Check if the next two elements form a recognized choice pattern
                    if len(rest) >= 3 and rest[1] == "abbr" and rest[2] == "expan":
                        grouped.append(["choice", "abbr", "expan"])
                        rest.pop(0)  # choice
                        rest.pop(0)  # abbr
                        rest.pop(0)  # expan
                    elif len(rest) >= 3 and rest[1] == "sic" and rest[2] == "corr":
                        grouped.append(["choice", "sic", "corr"])
                        rest.pop(0)  # choice
                        rest.pop(0)  # sic
                        rest.pop(0)  # corr
                    else:
                        grouped.append([rest.pop(0)])
                else:
                    grouped.append([rest.pop(0)])

                interventions += 1

            if not grouped:
                return "No elements"
            return ", ".join(x for g in grouped for x in g)

        return group_elements_by_editorial_interventions_single(gold_elements, bracket_count)

    # Multiple bracketed sequences - need to split elements by position
    # Use bracket count per sequence (same logic as gold_standard_stats.py group_multi)
    # Look up the current row itself, not its text: identical sequences can occur twice
    current_position = ordered_row_labels.index(current_row_label)

    def group_elements_by_sequences(elements: List[str], sequences: List[str]) -> List[List[str]]:
        """Group elements to match sequences in document order.

        First assigns one intervention per sequence, then distributes remaining
        based on extra brackets. This ensures each sequence gets at least one
        intervention when available, even if bracket counts don't match.
        """

        def take_one_intervention(rest: List[str]) -> List[str]:
            """Take one intervention from the element list."""
            if not rest:
                return []
            elem = rest[0]
            if elem == "choice":
                if len(rest) >= 3 and rest[1] == "abbr" and rest[2] == "expan":
                    result = ["choice", "abbr", "expan"]
                    rest.pop(0)
                    rest.pop(0)
                    rest.pop(0)
                    return result
                elif len(rest) >= 3 and rest[1] == "sic" and rest[2] == "corr":
                    result = ["choice", "sic", "corr"]
                    rest.pop(0)
                    rest.pop(0)
                    rest.pop(0)
                    return result
            return [rest.pop(0)]

        # Count total interventions available
        rest = elements.copy()
        total_interventions = 0
        temp = elements.copy()
        while temp:
            elem = temp[0]
            if elem == "choice" and len(temp) >= 3:
                if (temp[1] == "abbr" and temp[2] == "expan") or (
                    temp[1] == "sic" and temp[2] == "corr"
                ):
                    temp.pop(0)
                    temp.pop(0)
                    temp.pop(0)
                else:
                    temp.pop(0)
            else:
                temp.pop(0)
            total_interventions += 1

        # Calculate bracket counts per sequence
        bracket_counts = [seq.count("[") for seq in sequences]
        total_brackets = sum(bracket_counts)

        # Determine how many interventions each sequence should get
        interventions_per_seq = []
        if total_interventions <= len(sequences):
            # Not enough interventions for all sequences - give one each until exhausted
            interventions_per_seq = [
                1 if i < total_interventions else 0 for i in range(len(sequences))
            ]
        elif total_interventions >= total_brackets:
            # Enough interventions for all brackets - use bracket counts
            interventions_per_seq = bracket_counts
        else:
            # More sequences than interventions but fewer than brackets
            # First give one to each, then distribute remaining by extra brackets
            interventions_per_seq = [1] * len(sequences)
            remaining = total_interventions - len(sequences)
            for i, count in enumerate(bracket_counts):
                extra_needed = count - 1
                if extra_needed > 0 and remaining > 0:
                    give = min(extra_needed, remaining)
                    interventions_per_seq[i] += give
                    remaining -= give

        # Now assign interventions based on calculated counts
        grouped = []
        rest = elements.copy()
        for n_interventions in interventions_per_seq:
            seq_group = []
            for _ in range(n_interventions):
                seq_group.extend(take_one_intervention(rest))
            grouped.append(seq_group) if seq_group else grouped.append([])

        return grouped

    # Group the goldstandard elements based on bracket counts per sequence
    element_groups = group_elements_by_sequences(gold_elements, all_bracketed_sequences)

    # Return the group that corresponds to the current position
    if current_position < len(element_groups):
        result = ", ".join(element_groups[current_position])
        return result
    else:
        return "No elements"


def validate_xml_wellformed(tei_encoding: Optional[str]) -> bool:
    """
    Check whether an encoding is well-formed XML (D0), alone or as a fragment.

    Args:
        tei_encoding: The model's encoding of one sequence

    Returns:
        bool: True if it parses as XML, or as content of a wrapper element; False if it is empty or
        does not parse
    """
    if not tei_encoding or tei_encoding.strip() == "":
        return False

    try:
        # Try parsing as-is first
        etree.fromstring(tei_encoding)
        return True
    except etree.XMLSyntaxError:
        # Try wrapping in root element for XML fragments
        try:
            wrapped_xml = f"<root>{tei_encoding}</root>"
            etree.fromstring(wrapped_xml)
            return True
        except etree.XMLSyntaxError:
            return False
    except Exception:
        return False


def validate_tei_schema(tei_encoding: Optional[str]) -> Union[bool, str]:
    """
    Check an encoding against the TEI schema (D2) with Jing.

    The encoding is placed in the ``<p>`` of a minimal TEI document, written to a temporary file and
    validated against ``resources/schema/tei_all.rng``. If a folder ``java_portable`` exists in the
    current directory, ``JAVA_HOME`` and ``PATH`` of this process are set to a Java found there.

    Args:
        tei_encoding: The model's encoding of one sequence

    Returns:
        True: Valid according to TEI schema
        False: Invalid according to TEI schema (validation failed), or empty encoding
        "N/A": Technical error (JingValidator not available, Java issues, etc.)
    """
    if not tei_encoding or tei_encoding.strip() == "":
        return False

    try:
        # Check if JingValidator is available
        if JingValidator is None:
            return "N/A"

        # Initialize Jing validator with correct path
        # Get project root for resource paths
        project_root = Path(__file__).parent.parent
        jing_jar_path = project_root / config.JING_JAR_FILE
        jing_validator = JingValidator(str(jing_jar_path))

        # Set up portable Java path if available
        portable_java_path = Path("java_portable")
        if portable_java_path.exists():
            # Look for java.exe in the jPortable installation
            java_exe_candidates = [
                portable_java_path / "CommonFiles" / "Java" / "bin" / "java.exe",
                portable_java_path / "bin" / "java.exe",
                portable_java_path / "App" / "Java" / "bin" / "java.exe",
            ]

            # Also search recursively for java.exe
            java_dirs = sorted(portable_java_path.glob("**/java.exe"))
            java_exe_candidates.extend(java_dirs)

            java_exe_path = None
            for candidate in java_exe_candidates:
                if candidate.exists():
                    java_exe_path = candidate
                    break

            if java_exe_path:
                # Set the Java path for the validator

                # For jPortable, set up the environment correctly

                # Set JAVA_HOME to the main Java directory (not JRE subdirectory)
                java_home = java_exe_path.parent.parent
                os.environ["JAVA_HOME"] = str(java_home)

                # Add both the bin directory and the main Java directory to PATH
                current_path = os.environ.get("PATH", "")
                java_bin_path = str(java_exe_path.parent)
                if java_bin_path not in current_path:
                    os.environ["PATH"] = java_bin_path + os.pathsep + current_path

                # For jPortable, we may need to create a symbolic link or copy jvm.cfg
                # Let's check if we need to fix the jvm.cfg issue
                main_lib_path = java_home / "lib" / "i386"
                jre_lib_path = java_home / "jre1.8.0_461" / "lib" / "i386"

                if jre_lib_path.exists() and not main_lib_path.exists():
                    try:
                        main_lib_path.mkdir(parents=True, exist_ok=True)
                        # Copy jvm.cfg to the expected location

                        jvm_cfg_source = jre_lib_path / "jvm.cfg"
                        jvm_cfg_target = main_lib_path / "jvm.cfg"
                        if jvm_cfg_source.exists() and not jvm_cfg_target.exists():
                            shutil.copy2(jvm_cfg_source, jvm_cfg_target)
                    except Exception as e:
                        # Not fatal: Java may still start without the copied jvm.cfg
                        print(
                            f"Warning: could not copy jvm.cfg to {main_lib_path}: {e}",
                            file=sys.stderr,
                        )
            else:
                print(f"Warning: no java.exe found in {portable_java_path}", file=sys.stderr)

        # Check if Jing validator is available
        if not jing_validator.is_available():
            print(
                f"Warning: Jing not found at {jing_jar_path}, TEI validation result is N/A",
                file=sys.stderr,
            )
            return "N/A"

        # Create minimal TEI document with the encoding inside <p>
        tei_document = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
      <fileDesc>
         <titleStmt><title>Validation Test</title></titleStmt>
         <publicationStmt><p>Test</p></publicationStmt>
         <sourceDesc><p>Test</p></sourceDesc>
      </fileDesc>
  </teiHeader>
  <text>
      <body>
         <p>{tei_encoding}</p>
      </body>
  </text>
</TEI>"""

        # Create temporary file for validation
        import tempfile
        import time

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(tei_document)
                temp_file.flush()

            # Validate using Jing
            schema_path = project_root / config.TEI_SCHEMA_FILE
            if not schema_path.exists():
                print(
                    f"Warning: TEI schema not found at {schema_path}, TEI validation result is N/A",
                    file=sys.stderr,
                )
                return "N/A"

            is_valid, error_message = jing_validator.validate_quick(
                temp_file_path, str(schema_path)
            )

            # If Java has initialization issues, return N/A (technical error)
            if not is_valid and (
                "initialization of VM" in error_message or "could not open" in error_message
            ):
                return "N/A"

            # Return validation result: True if valid, False if invalid
            return is_valid

        finally:
            # Clean up temporary file with retry mechanism for Windows
            if temp_file_path and Path(temp_file_path).exists():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        Path(temp_file_path).unlink()
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.1)  # Wait a bit before retrying
                        else:
                            print(
                                f"Warning: could not delete temporary file {temp_file_path}: {e}",
                                file=sys.stderr,
                            )
                            # Don't fail the validation just because of cleanup issues

    except Exception as e:
        # If any error occurs during validation, return N/A (technical error)
        print(
            f"Warning: TEI validation failed with a technical error, result is N/A: {e}",
            file=sys.stderr,
        )
        return "N/A"


def extract_tei_elements_from_encoding(tei_encoding: Optional[str]) -> str:
    """
    Extract TEI element names from a TEI encoding string.
    Returns a comma-separated string of element names found in the encoding.
    PRESERVES DUPLICATE ELEMENTS to handle multiple instances of same intervention type.

    Args:
        tei_encoding: String containing TEI XML encoding

    Returns:
        str: Comma-separated list of TEI element names; "No elements" if there are none or the
        encoding is only an XML comment; "Parse error - not valid XML" if it parses neither alone
        nor inside a wrapper element
    """
    if not tei_encoding:
        return "No elements"

    # Check if this is an XML comment - if so, it's not a proper encoding
    if tei_encoding.strip().startswith("<!--") and tei_encoding.strip().endswith("-->"):
        return "No elements"

    try:
        # Parse the TEI encoding as XML
        tree = etree.fromstring(tei_encoding)

        # Extract all element tags (including duplicates)
        element_tags = [tree.tag]

        # Also get child element tags
        for child in tree.iter():
            if child != tree:  # Don't include the root element twice
                element_tags.append(child.tag)

        # Return all tags preserving duplicates and order
        return ", ".join(element_tags) if element_tags else "No elements"

    except Exception:
        # If parsing fails, try wrapping in a root element (for XML fragments)
        try:
            wrapped_xml = f"<root>{tei_encoding}</root>"
            tree = etree.fromstring(wrapped_xml)

            # Extract child element tags (skip the wrapper root)
            element_tags = []
            for child in tree:
                # Add all descendants including the child itself
                for element in child.iter():
                    element_tags.append(element.tag)

            # Return all tags preserving duplicates and order
            return ", ".join(element_tags) if element_tags else "No elements"

        except Exception:
            # If still fails, return the original text (might not be XML)
            return "Parse error - not valid XML"


def calculate_element_accuracy_percentage(
    llm_diff: Dict[str, Any], gold_diff: Dict[str, Any]
) -> float:
    """
    Calculate the percentage of correctly added elements by comparing LLM output with gold standard.
    Includes partial credit for child elements when parent structure is missing.

    Args:
        llm_diff: Dictionary from compare_xml_elements for input vs LLM output
        gold_diff: Dictionary from compare_xml_elements for input vs gold standard

    Returns:
        float: Percentage of correctly added elements (0-100)
    """
    # Get added elements from both comparisons
    llm_added = llm_diff.get("added_elements", [])
    gold_added = gold_diff.get("added_elements", [])

    # If no elements were added in gold standard, LLM should also add no elements
    if len(gold_added) == 0:
        if len(llm_added) == 0:
            return 100.0  # Perfect match - both add nothing
        else:
            return 0.0  # LLM added elements when it shouldn't have

    # If LLM added no elements but gold standard did, accuracy is 0
    if len(llm_added) == 0:
        return 0.0

    # Create normalized element signatures for comparison
    def get_element_signature(elem: Dict[str, str]) -> str:
        """Get a normalized signature of an element for comparison (ignoring attributes)"""
        try:
            # Parse the element content
            tree = etree.fromstring(elem["content"])

            # Get tag only (ignore attributes)
            tag = tree.tag

            # Get child element tags (for nested structures like choice/abbr/expan)
            child_tags = [child.tag for child in tree]
            child_tags.sort()

            # Create signature: tag + child tags (no attributes)
            signature = tag
            if child_tags:
                signature += f" [{','.join(child_tags)}]"

            return signature
        except Exception:
            # Fallback to just the tag if parsing fails
            return elem["tag"]

    def get_child_elements(elem: Dict[str, str]) -> List[str]:
        """Extract child element tags from an element"""
        try:
            tree = etree.fromstring(elem["content"])
            child_tags = [child.tag for child in tree]
            return sorted(child_tags)
        except Exception:
            return []

    # Create lists of element signatures (preserving duplicates)
    llm_signatures = [get_element_signature(elem) for elem in llm_added]
    gold_signatures = [get_element_signature(elem) for elem in gold_added]

    # Count direct matches by finding how many LLM elements match gold standard elements
    direct_matches = 0
    gold_signatures_copy = gold_signatures.copy()  # Make a copy to modify

    for llm_sig in llm_signatures:
        if llm_sig in gold_signatures_copy:
            direct_matches += 1
            gold_signatures_copy.remove(
                llm_sig
            )  # Remove the matched element to avoid double counting

    # Calculate partial credit for child elements when parent structure is missing
    partial_credit = 0.0

    # Only calculate partial credit if there are no direct matches
    # (to avoid double counting)
    if direct_matches == 0:
        # Get all child elements from LLM and Gold Standard
        llm_child_elements = []
        gold_child_elements = []

        # For LLM: get child elements + top-level element tags
        for elem in llm_added:
            llm_child_elements.extend(get_child_elements(elem))
            # Also add the element tag itself (for cases where LLM has individual elements)
            llm_child_elements.append(elem["tag"])

        # For Gold Standard: get child elements
        for elem in gold_added:
            gold_child_elements.extend(get_child_elements(elem))

        # Count matching child elements
        if gold_child_elements:
            llm_child_copy = llm_child_elements.copy()
            gold_child_copy = gold_child_elements.copy()

            child_matches = 0
            for gold_child in gold_child_copy:
                if gold_child in llm_child_copy:
                    child_matches += 1
                    llm_child_copy.remove(gold_child)

            # Calculate partial credit: 50% of the child element matches
            partial_credit = (child_matches / len(gold_child_elements)) * 50.0

    # Calculate total accuracy
    if len(gold_signatures) == 0:
        return 100.0 if len(llm_signatures) == 0 else 0.0

    # Direct matches get full credit, partial matches get 50% credit
    direct_credit = (direct_matches / len(gold_signatures)) * 100.0
    total_accuracy = direct_credit + partial_credit

    # Cap at 100%
    total_accuracy = min(total_accuracy, 100.0)

    return round(total_accuracy, 1)


def compare_intervention_types(llm_type: Optional[str], gold_type: Optional[str]) -> bool:
    """
    Compare the model's and the gold-standard intervention type (D4a).

    Types are compared by their first 7 letters, case-insensitively ("Correction" equals
    "Corrections"). For labels with several types separated by ";", the number of types and each
    type in the same order must match. Missing values and "None" only match themselves.

    Args:
        llm_type: The model's type label
        gold_type: The gold-standard type label

    Returns:
        bool: True if the types match
    """
    if not llm_type or not gold_type or llm_type == "None" or gold_type == "None":
        return llm_type == gold_type

    # Handle multiple types separated by semicolons - require exact match
    if ";" in gold_type or ";" in llm_type:
        # Split both types and normalize
        gold_types = [t.strip() for t in gold_type.split(";")]
        llm_types = [t.strip() for t in llm_type.split(";")]

        # Must have same number of types
        if len(gold_types) != len(llm_types):
            return False

        # Check each corresponding type matches
        for gold_single_type, llm_single_type in zip(gold_types, llm_types):
            gold_prefix = gold_single_type.lower()[:7]
            llm_prefix = llm_single_type.lower()[:7]
            if gold_prefix != llm_prefix:
                return False
        return True
    else:
        # Single type comparison - existing logic
        llm_prefix = llm_type.lower()[:7]
        gold_prefix = gold_type.lower()[:7]
        return llm_prefix == gold_prefix


def _check_single_intervention_type_valid(intervention_type: str, elements: List[str]) -> bool:
    """
    Check if elements satisfy the requirements for a single intervention type.

    Args:
        intervention_type: Single intervention type (e.g., "Abbreviation")
        elements: List of element names in lowercase

    Returns:
        bool: True if valid for this intervention type, False otherwise
    """
    if intervention_type.lower().startswith("abbreviation"):
        # Must have choice element with abbr and expan
        return "choice" in elements and "abbr" in elements and "expan" in elements

    elif intervention_type.lower().startswith("supplement"):
        # Must be a supplied tag with text content
        return "supplied" in elements

    elif intervention_type.lower().startswith("correction"):
        # Must have either only sic element, or choice + sic and corr
        if "sic" in elements:
            if len(elements) == 1:
                return True  # only sic
            elif "choice" in elements and "corr" in elements and len(elements) == 3:
                return True  # choice + sic + corr
        return False

    elif intervention_type.lower().startswith("omission"):
        # Must use gap element
        return "gap" in elements

    elif intervention_type.lower().startswith(
        "uncertainty"
    ) or intervention_type.lower().startswith("unclear"):
        # Must use unclear element
        return "unclear" in elements

    elif intervention_type.lower().startswith("redundan"):
        # Redundancy/Redundant Text must use <surplus> element (del is not acceptable)
        return "surplus" in elements

    # For other intervention types, return False
    return False


def _check_intervention_counts_valid(
    intervention_counts: Dict[str, int], elements: List[str]
) -> bool:
    """
    Check if elements can satisfy the required counts for each intervention type.

    Args:
        intervention_counts: Dictionary mapping intervention type to required count
        elements: List of element names in lowercase

    Returns:
        bool: True if elements can satisfy all required intervention types, False otherwise
    """
    # Count available elements
    element_counts = {}
    for elem in elements:
        element_counts[elem] = element_counts.get(elem, 0) + 1

    # Check each intervention type requirement
    for intervention_type, required_count in intervention_counts.items():
        if intervention_type.lower().startswith("abbreviation"):
            # Each abbreviation needs choice, abbr, expan
            choice_available = element_counts.get("choice", 0)
            abbr_available = element_counts.get("abbr", 0)
            expan_available = element_counts.get("expan", 0)

            # Can we form enough abbreviation groups?
            possible_groups = min(choice_available, abbr_available, expan_available)
            if possible_groups < required_count:
                return False

        elif intervention_type.lower().startswith("supplement"):
            # Each supplement needs one supplied element
            supplied_available = element_counts.get("supplied", 0)
            if supplied_available < required_count:
                return False

        elif intervention_type.lower().startswith("correction"):
            # Each correction needs either sic OR (choice + sic + corr)
            sic_available = element_counts.get("sic", 0)
            choice_available = element_counts.get("choice", 0)
            corr_available = element_counts.get("corr", 0)

            # Count how many corrections we can form
            # Option 1: sic-only corrections
            sic_only_corrections = sic_available

            # Option 2: choice+sic+corr corrections (need to account for choices already used in
            # abbreviations)
            # For now, assume all choices are available for corrections (simplified)
            choice_sic_corr_corrections = min(choice_available, sic_available, corr_available)

            total_possible_corrections = sic_only_corrections + choice_sic_corr_corrections
            if total_possible_corrections < required_count:
                return False

        elif intervention_type.lower().startswith("omission"):
            # Each omission needs one gap element
            gap_available = element_counts.get("gap", 0)
            if gap_available < required_count:
                return False

        elif intervention_type.lower().startswith(
            "uncertainty"
        ) or intervention_type.lower().startswith("unclear"):
            # Each uncertainty needs one unclear element
            unclear_available = element_counts.get("unclear", 0)
            if unclear_available < required_count:
                return False

        elif intervention_type.lower().startswith("redundan"):
            # Each Redundancy/Redundant Text needs one surplus element (del is not acceptable)
            surplus_available = element_counts.get("surplus", 0)
            if surplus_available < required_count:
                return False

    return True


def check_project_valid(
    intervention_type: Optional[str], element_str: str, tei_encoding: Optional[str]
) -> Optional[bool]:
    """
    Check if the LLM followed project-specific instruction rules for editorial interventions.
    Supports multiple intervention types separated by semicolons.

    The check uses the intervention type the model assigned, not the gold-standard type. Rules:

    - No element may have attributes
    - The elements required by each assigned type (abbr, expan, supplied, sic, corr, unclear,
      surplus) must contain text; gap may be empty
    - Abbreviation: choice, abbr and expan present
    - Supplement: supplied present
    - Correction: only sic, or exactly choice, sic and corr
    - Omission: gap present
    - Uncertainty: unclear present
    - Redundancy: surplus present

    With several types, enough elements for each type must be present (counted per type).

    Args:
        intervention_type: The intervention type(s) (e.g., "Abbreviation", "Correction; Supplement",
            etc.)
        element_str: The element string (e.g., "choice, abbr, expan", "unclear", etc.)
        tei_encoding: The TEI encoding string to check for attributes

    Returns:
        bool: True if valid, False if invalid, None if not categorized
    """
    # If not categorized or categorized as None/unknown, return N/A
    if (
        not intervention_type
        or intervention_type == "None"
        or intervention_type.lower() == "unknown"
    ):
        return None

    # Normalize element string to lowercase and split by comma
    elements = [elem.strip().lower() for elem in element_str.split(",")]

    # Check if any elements have attributes by parsing the TEI encoding
    if has_attributes_in_tei(tei_encoding):
        return False

    # Check if required elements have text content (except gap which can be empty)
    if not check_elements_have_text_content(tei_encoding, intervention_type):
        return False

    # Handle multiple intervention types separated by semicolons
    if ";" in intervention_type:
        intervention_types = [t.strip() for t in intervention_type.split(";")]
        # Count how many of each intervention type we need
        intervention_counts = {}
        for single_type in intervention_types:
            intervention_counts[single_type] = intervention_counts.get(single_type, 0) + 1
        # Check if we have enough elements to satisfy all required intervention types
        if _check_intervention_counts_valid(intervention_counts, elements):
            return True
        else:
            return False
    else:
        # Single intervention type - use existing logic
        if _check_single_intervention_type_valid(intervention_type, elements):
            return True
        else:
            return False


def has_attributes_in_tei(tei_encoding: Optional[str]) -> bool:
    """
    Check if any elements in the TEI encoding have attributes.

    Args:
        tei_encoding: The TEI encoding string

    Returns:
        bool: True if any element has attributes; False otherwise, also if the encoding is empty
        or cannot be parsed inside a ``<root>`` wrapper
    """
    if not tei_encoding:
        return False

    try:
        # Wrap in a root element to ensure valid XML for parsing
        wrapped_xml = f"<root>{tei_encoding}</root>"
        root = etree.fromstring(wrapped_xml)

        # Check all elements for attributes
        for elem in root.iter():
            if elem.tag != "root" and elem.attrib:  # Skip our wrapper root
                return True
        return False
    except Exception:
        # If parsing fails, conservatively return False
        return False


def check_elements_have_text_content(
    tei_encoding: Optional[str], intervention_type: Optional[str]
) -> bool:
    """
    Check if required TEI elements have text content based on intervention type.

    Elements that must have text content:
    - Abbreviation: abbr and expan
    - Supplement: supplied
    - Correction: sic and/or corr
    - Uncertainty: unclear
    - Redundancy: surplus

    Exception: gap element can be empty (self-closing)

    Args:
        tei_encoding: The TEI encoding string
        intervention_type: The intervention type (e.g., "Abbreviation", "Correction")

    Returns:
        bool: False if a required element is empty; True otherwise, also if the encoding or the
        type is missing or the encoding cannot be parsed inside a ``<root>`` wrapper
    """
    if not tei_encoding or not intervention_type:
        return True  # Can't check, assume valid

    try:
        wrapped_xml = f"<root>{tei_encoding}</root>"
        root = etree.fromstring(wrapped_xml)

        intervention_lower = intervention_type.lower()

        # Handle multiple intervention types
        if ";" in intervention_type:
            intervention_types = [t.strip().lower() for t in intervention_type.split(";")]
        else:
            intervention_types = [intervention_lower]

        for int_type in intervention_types:
            if int_type.startswith("abbreviation"):
                # Check abbr and expan have text
                for tag in ["abbr", "expan"]:
                    elements = root.xpath(f".//{tag}")
                    for elem in elements:
                        text = (elem.text or "").strip()
                        if not text:
                            return False

            elif int_type.startswith("supplement"):
                # Check supplied has text
                elements = root.xpath(".//supplied")
                for elem in elements:
                    text = (elem.text or "").strip()
                    if not text:
                        return False

            elif int_type.startswith("correction"):
                # Check sic and corr have text (at least one must have content)
                sic_elements = root.xpath(".//sic")
                corr_elements = root.xpath(".//corr")

                for elem in sic_elements:
                    text = (elem.text or "").strip()
                    if not text:
                        return False

                for elem in corr_elements:
                    text = (elem.text or "").strip()
                    if not text:
                        return False

            elif int_type.startswith("uncertainty") or int_type.startswith("unclear"):
                # Check unclear has text
                elements = root.xpath(".//unclear")
                for elem in elements:
                    text = (elem.text or "").strip()
                    if not text:
                        return False

            elif int_type.startswith("redundan"):
                # Check surplus has text (Redundancy or Redundant Text)
                elements = root.xpath(".//surplus")
                for elem in elements:
                    text = (elem.text or "").strip()
                    if not text:
                        return False

            # gap is allowed to be empty (self-closing), so no check needed for omission

        return True

    except Exception:
        # If parsing fails, assume valid
        return True


def extract_corr_content(tei_encoding: Optional[str]) -> Optional[str]:
    """
    Extract the content from the FIRST <corr> element in a TEI encoding.

    Args:
        tei_encoding: The TEI encoding string

    Returns:
        str: Content of the first <corr> element; None if there is none or the encoding cannot be
        parsed inside a ``<root>`` wrapper
    """
    if not tei_encoding:
        return None

    try:
        # Wrap in a root element to ensure valid XML for parsing
        wrapped_xml = f"<root>{tei_encoding}</root>"
        root = etree.fromstring(wrapped_xml)

        # Find the first corr element only (to match the specific bracketed sequence)
        corr_elements = root.xpath(".//corr")
        if corr_elements:
            return (corr_elements[0].text or "").strip()

        return None
    except Exception:
        # If parsing fails, return None
        return None


def normalize_tei_encoding_for_intervention(
    tei_encoding: Optional[str],
    intervention_type: Optional[str],
    llm_elements: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize TEI encoding by extracting text content, with special handling for correction types.

    For correction intervention types:
    - sic only: Returns text content without sic tags
    - choice + sic + corr: Returns text without corr + " | " + corr content

    Args:
        tei_encoding: The TEI encoding string
        intervention_type: The intervention type (e.g., "Correction")
        llm_elements: The LLM elements string (e.g., "choice, sic, corr" or "sic")

    Returns:
        str: Normalized text content with special formatting for corrections; the encoding
        unchanged if it parses neither alone nor inside a wrapper element; None if it is empty
    """
    if not tei_encoding:
        return None

    try:
        # Check if this is a correction type that needs special handling
        is_correction = intervention_type and intervention_type.lower().startswith("correction")

        # Determine if this is choice_sic_corr or sic_only based on llm_elements
        has_choice_sic_corr = False
        if llm_elements:
            elements_list = [e.strip().lower() for e in llm_elements.split(",")]
            has_choice_sic_corr = (
                "choice" in elements_list and "sic" in elements_list and "corr" in elements_list
            )

        if is_correction:
            # For corrections, extract text and handle corr elements
            try:
                # Parse the TEI encoding
                tei_tree = etree.fromstring(tei_encoding)
            except Exception:
                # If parsing fails, try wrapping in a root element
                wrapped_xml = f"<root>{tei_encoding}</root>"
                tei_tree = etree.fromstring(wrapped_xml)

            if has_choice_sic_corr:
                # Extract corr content before removing it
                corr_elements = tei_tree.xpath(".//corr")
                corr_content = ""
                if corr_elements:
                    corr_content = etree.tostring(
                        corr_elements[0], encoding="unicode", method="text"
                    ).strip()

                # Remove all corr elements (but keep sic elements)
                for corr_elem in corr_elements:
                    parent = corr_elem.getparent()
                    if parent is not None:
                        parent.remove(corr_elem)

                # Extract text content (now without corr elements)
                text_without_corr = etree.tostring(tei_tree, encoding="unicode", method="text")

                # Return format: text without corr | corr content
                if corr_content:
                    return f"{text_without_corr} | {corr_content}"
                else:
                    return text_without_corr
            else:
                # sic only - just extract text content
                return etree.tostring(tei_tree, encoding="unicode", method="text")

        else:
            # For non-correction types, use standard text extraction
            try:
                tei_tree = etree.fromstring(tei_encoding)
            except Exception:
                # If parsing fails, try wrapping in a root element
                wrapped_xml = f"<root>{tei_encoding}</root>"
                tei_tree = etree.fromstring(wrapped_xml)

            return etree.tostring(tei_tree, encoding="unicode", method="text")

    except Exception:
        # If all parsing fails, return the original text
        return tei_encoding


def check_type_element_match(intervention_type: Optional[str], element_str: Optional[str]) -> bool:
    """
    Check if the intervention type matches the element that was chosen.
    Supports multiple intervention types separated by semicolons.

    Mapping rules:
    - Abbreviation: choice + abbr + expan OR only expan
    - Correction: choice + sic + corr OR only sic
    - Uncertainty: unclear
    - Omission: gap
    - Supplement: supplied

    Args:
        intervention_type: The intervention type(s) (e.g., "Abbreviation",
            "Correction; Supplement", etc.)
        element_str: The element string (e.g., "choice, abbr, expan", "unclear", etc.)

    Returns:
        bool: True if the intervention type matches the element, False otherwise
    """
    if not intervention_type or not element_str or intervention_type == "None":
        return False

    # Normalize element string to lowercase and split by comma
    elements = [elem.strip().lower() for elem in element_str.split(",")]

    # Handle multiple intervention types separated by semicolons
    if ";" in intervention_type:
        intervention_types = [t.strip() for t in intervention_type.split(";")]

        # Check if we can satisfy all intervention types with the available elements
        return _check_multiple_intervention_types(intervention_types, elements)
    else:
        # Single intervention type - use existing logic
        return _check_single_type_element_match(intervention_type, elements)


def _check_multiple_intervention_types(intervention_types: List[str], elements: List[str]) -> bool:
    """
    Check if multiple intervention types can be satisfied with the available elements.
    This function tries to find a valid combination where each intervention type
    can be satisfied with a subset of the available elements.

    Args:
        intervention_types: List of intervention types (e.g., ["Correction", "Supplement"])
        elements: List of element names in lowercase

    Returns:
        bool: True if all intervention types can be satisfied, False otherwise
    """
    # For each intervention type, check if it can be satisfied with available elements
    for intervention_type in intervention_types:
        if not _check_single_type_element_match(intervention_type, elements):
            return False
    return True


def _compare_elements_with_correction_equivalence(
    llm_elements: Optional[str], gold_elements: Optional[str]
) -> bool:
    """
    Compare element strings with special equivalence rules for corrections.

    Special case: choice+sic+corr in goldstandard should match single sic in LLM
    because both represent correction interventions.

    Args:
        llm_elements: LLM element string (e.g., "sic")
        gold_elements: Goldstandard element string (e.g., "choice, sic, corr")

    Returns:
        bool: True if elements match (including correction equivalence), False otherwise
    """
    # First check exact match
    if llm_elements == gold_elements:
        return True

    # Handle "No elements" cases
    if (
        not llm_elements
        or not gold_elements
        or llm_elements == "No elements"
        or gold_elements == "No elements"
    ):
        return llm_elements == gold_elements

    # Parse both strings into lists
    llm_tags = [tag.strip().lower() for tag in llm_elements.split(",")]
    gold_tags = [tag.strip().lower() for tag in gold_elements.split(",")]

    # Special case: choice+sic+corr in goldstandard should match single sic in LLM
    if (
        len(llm_tags) == 1
        and llm_tags[0] == "sic"
        and len(gold_tags) == 3
        and "choice" in gold_tags
        and "sic" in gold_tags
        and "corr" in gold_tags
    ):
        return True

    # Reverse case: single sic in goldstandard should match choice+sic+corr in LLM
    if (
        len(gold_tags) == 1
        and gold_tags[0] == "sic"
        and len(llm_tags) == 3
        and "choice" in llm_tags
        and "sic" in llm_tags
        and "corr" in llm_tags
    ):
        return True

    # No special equivalence applies, return False
    return False


def _check_single_type_element_match(intervention_type: str, elements: List[str]) -> bool:
    """
    Check if a single intervention type matches the available elements.

    Args:
        intervention_type: Single intervention type (e.g., "Abbreviation")
        elements: List of element names in lowercase

    Returns:
        bool: True if the intervention type matches the element, False otherwise
    """
    # Check mapping rules
    if intervention_type.lower().startswith("abbreviation"):
        # Abbreviation: choice + abbr + expan OR only expan
        if "expan" in elements:
            if "choice" in elements and "abbr" in elements:
                return True  # choice + abbr + expan
            elif len(elements) == 1 and "expan" in elements:
                return True  # only expan
        return False

    elif intervention_type.lower().startswith("correction"):
        # Correction: choice + sic + corr OR sic (with or without other elements)
        if "sic" in elements:
            if "choice" in elements and "corr" in elements:
                return True  # choice + sic + corr
            else:
                return True  # sic present (with or without other elements)
        return False

    elif intervention_type.lower().startswith("uncertainty"):
        # Uncertainty: unclear
        return "unclear" in elements

    elif intervention_type.lower().startswith("omission"):
        # Omission: gap
        return "gap" in elements

    elif intervention_type.lower().startswith("supplement"):
        # Supplement: supplied
        return "supplied" in elements

    # For other intervention types, return False
    return False


def _json_serial(val: Any) -> Any:
    """
    Convert numpy/pandas values to JSON-serializable Python types.

    Args:
        val: A value from the details DataFrame (also dicts and lists of such values)

    Returns:
        The value as a plain Python type; None for None and NaN
    """
    if val is None:
        return None
    if pd.isna(val):
        return None
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    if isinstance(val, (dict,)):
        return {k: _json_serial(v) for k, v in val.items()}
    if isinstance(val, (list,)):
        return [_json_serial(v) for v in val]
    return val


def _write_evaluation_summary_json(
    output_path: str,
    model_instance: str,
    total_sequences: int,
    total_encoded: int,
    xml_wellformed_count: int,
    xml_wellformed_percentage: float,
    tei_valid_count: int,
    tei_valid_percentage: float,
    project_valid_count: int,
    project_valid_percentage: float,
    d1_evaluated: int,
    d1_passed: int,
    content_precision: float,
    content_recall: float,
    content_f1_score: float,
    content_macro_f1: float,
    d4a_evaluated: int,
    d4a_passed: int,
    intervention_precision: float,
    intervention_recall: float,
    intervention_f1_score: float,
    intervention_macro_f1: float,
    d4b_evaluated: int,
    d4b_passed: int,
    precision: float,
    recall: float,
    f1_score: float,
    element_macro_f1: float,
    content_category_metrics: Dict[str, Dict[str, float]],
    intervention_category_metrics: Dict[str, Dict[str, float]],
    element_category_metrics: Dict[str, Dict[str, float]],
    element_category_metrics_relaxed: Dict[str, Dict[str, float]],
    intervention_categories: List[str],
    valid_count: int,
    invalid_count: int,
    validity_rate: float,
    avg_overall_valid: float,
    min_overall_valid: float,
    max_overall_valid: float,
    avg_overall_all: float,
    min_overall_all: float,
    max_overall_all: float,
    avg_core_quality: float,
    min_core_quality: float,
    max_core_quality: float,
    avg_schema_factor: float,
    min_schema_factor: float,
    max_schema_factor: float,
    df: pd.DataFrame,
) -> None:
    """
    Write evaluation summary and category results to JSON (mirrors Excel Summary + D1/D4a/D4b).

    The file has ``summary`` (with the current time as ``timestamp``), one list per category
    metric (D1, D4a, D4b, D4b relaxed) and ``details`` (one record per evaluated sequence).
    Percentages are rounded to 2 decimals, category precision/recall/F1 to 4, schema factors to 3.

    Args:
        output_path: Path of the JSON file
        model_instance: Model instance name
        total_sequences, total_encoded: Number of evaluated and of processed sequences
        xml_wellformed_count, xml_wellformed_percentage, tei_valid_count, tei_valid_percentage,
            project_valid_count, project_valid_percentage: D0 and D2 results
        d1_evaluated, d1_passed, content_precision, content_recall, content_f1_score,
            content_macro_f1: D1 results (fractions, not percentages)
        d4a_evaluated, d4a_passed, intervention_precision, intervention_recall,
            intervention_f1_score, intervention_macro_f1: D4a results
        d4b_evaluated, d4b_passed, precision, recall, f1_score, element_macro_f1: D4b results
        content_category_metrics, intervention_category_metrics, element_category_metrics,
            element_category_metrics_relaxed: Per-category tp/fp/fn/precision/recall/f1
        intervention_categories: Category names, in output order
        valid_count, invalid_count, validity_rate, avg_overall_valid, min_overall_valid,
            max_overall_valid, avg_overall_all, min_overall_all, max_overall_all, avg_core_quality,
            min_core_quality, max_core_quality, avg_schema_factor, min_schema_factor,
            max_schema_factor: Overall tiered scoring
        df: Details DataFrame (one row per evaluated sequence)
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "report_type": "Evaluation Summary (Bracketed Sequences)",
        "model_instance": model_instance,
        "total_sequences": total_sequences,
        "total_processed": total_encoded,
        "dimension_0_xml_wellformedness": {
            "processed": total_encoded,
            "passed": xml_wellformed_count,
            "failed": total_encoded - xml_wellformed_count,
            "pass_rate_pct": round(xml_wellformed_percentage, 2),
        },
        "dimension_1_source_fidelity": {
            "note": "Only evaluated when D4b (Element Match) is TRUE",
            "evaluated": d1_evaluated,
            "passed": d1_passed,
            "failed": d1_evaluated - d1_passed,
            "precision_pct": round(content_precision * 100, 2),
            "recall_pct": round(content_recall * 100, 2),
            "f1_score_pct": round(content_f1_score * 100, 2),
            "macro_f1_pct": round(content_macro_f1 * 100, 2),
        },
        "dimension_2_schema_validation": {
            "tei_valid_count": tei_valid_count,
            "tei_valid_pass_rate_pct": round(tei_valid_percentage, 2),
            "project_valid_count": project_valid_count,
            "project_valid_pass_rate_pct": round(project_valid_percentage, 2),
        },
        "dimension_4a_classification": {
            "evaluated": d4a_evaluated,
            "passed": d4a_passed,
            "failed": d4a_evaluated - d4a_passed,
            "precision_pct": round(intervention_precision * 100, 2),
            "recall_pct": round(intervention_recall * 100, 2),
            "f1_score_pct": round(intervention_f1_score * 100, 2),
            "macro_f1_pct": round(intervention_macro_f1 * 100, 2),
        },
        "dimension_4b_element_application": {
            "evaluated": d4b_evaluated,
            "passed": d4b_passed,
            "failed": d4b_evaluated - d4b_passed,
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "f1_score_pct": round(f1_score * 100, 2),
            "macro_f1_pct": round(element_macro_f1 * 100, 2),
        },
        "overall_tiered_scoring": {
            "valid_instances": valid_count,
            "invalid_instances": invalid_count,
            "validity_rate_pct": round(validity_rate, 2),
            "average_overall_score_valid_xml": round(avg_overall_valid, 2),
            "min_overall_score_valid_xml": round(min_overall_valid, 2),
            "max_overall_score_valid_xml": round(max_overall_valid, 2),
            "average_overall_score_all": round(avg_overall_all, 2),
            "min_overall_score_all": round(min_overall_all, 2),
            "max_overall_score_all": round(max_overall_all, 2),
            "average_core_quality_score": round(avg_core_quality, 2),
            "min_core_quality_score": round(min_core_quality, 2),
            "max_core_quality_score": round(max_core_quality, 2),
            "average_schema_factor": round(avg_schema_factor, 3),
            "min_schema_factor": round(min_schema_factor, 3),
            "max_schema_factor": round(max_schema_factor, 3),
        },
    }
    # Category-based evaluations (D1, D4a, D4b worksheets)
    source_fidelity_d1_categories = []
    for cat in intervention_categories:
        m = content_category_metrics[cat]
        source_fidelity_d1_categories.append(
            {
                "category": cat,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
        )
    classification_d4a_categories = []
    for cat in intervention_categories:
        m = intervention_category_metrics[cat]
        classification_d4a_categories.append(
            {
                "category": cat,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
        )
    element_application_d4b_categories = []
    for cat in intervention_categories:
        m = element_category_metrics[cat]
        element_application_d4b_categories.append(
            {
                "category": cat,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
        )
    element_application_d4b_relaxed_categories = []
    for cat in intervention_categories:
        m = element_category_metrics_relaxed[cat]
        element_application_d4b_relaxed_categories.append(
            {
                "category": cat,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
        )
    payload = {
        "summary": summary,
        "source_fidelity_d1_categories": source_fidelity_d1_categories,
        "classification_d4a_categories": classification_d4a_categories,
        "element_application_d4b_categories": element_application_d4b_categories,
        "element_application_d4b_relaxed_categories": element_application_d4b_relaxed_categories,
        "details": [],
    }
    # Per-instance details from DataFrame (for visualization)
    details_records = df.to_dict(orient="records")
    for rec in details_records:
        payload["details"].append(_json_serial(rec))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _create_unified_evaluation_heatmap(
    intervention_category_metrics: Dict[str, Dict[str, float]],
    element_category_metrics: Dict[str, Dict[str, float]],
    element_category_metrics_relaxed: Dict[str, Dict[str, float]],
    intervention_categories: List[str],
    save_path: str,
) -> None:
    """
    Create a 3-panel heatmap (Classification D4a, Element D4b standard, Element D4b relaxed).

    Does nothing if matplotlib or seaborn cannot be imported.

    Args:
        intervention_category_metrics: D4a metrics per category
        element_category_metrics: D4b metrics per category
        element_category_metrics_relaxed: Relaxed D4b metrics per category
        intervention_categories: Category names (columns)
        save_path: PNG path
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    categories = list(intervention_categories)

    # Build matrices: rows = Precision, Recall, F1; columns = categories
    def to_matrix(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Precision, recall and F1 of each category as rows x categories."""
        data = []
        for cat in categories:
            m = metrics_dict[cat]
            data.append([m["precision"], m["recall"], m["f1"]])
        return pd.DataFrame(data, index=categories, columns=["Precision", "Recall", "F1 Score"]).T

    intervention_matrix = to_matrix(intervention_category_metrics)
    standard_matrix = to_matrix(element_category_metrics)
    relaxed_matrix = to_matrix(element_category_metrics_relaxed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Performance Heatmap Across Categories and Matching Types", fontsize=14, fontweight="bold"
    )

    sns.heatmap(
        intervention_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=axes[0],
        cbar_kws={"label": "Score"},
    )
    axes[0].set_title("Intervention Type\nCategorization (D4a)", fontweight="bold")
    axes[0].set_ylabel("Metric")

    sns.heatmap(
        standard_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=axes[1],
        cbar_kws={"label": "Score"},
    )
    axes[1].set_title("Element Matching\nStandard (D4b)", fontweight="bold")
    axes[1].set_ylabel("")

    sns.heatmap(
        relaxed_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=axes[2],
        cbar_kws={"label": "Score"},
    )
    axes[2].set_title("Element Matching\nRelaxed (D4b)", fontweight="bold")
    axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _create_sequence_dimension_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """
    Create a heatmap: one row per bracketed sequence, columns = D0, D1, D2 (project), D4a, D4b.

    A cell is 1 if the value is True, else 0 (False, None and N/A all count as 0). Does nothing if
    matplotlib or seaborn cannot be imported or there are no rows.

    Args:
        df: Details DataFrame
        save_path: PNG path
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    # Build matrix: rows = sequences, columns = dimensions (1 = pass, 0 = fail/N/A)
    col_map = {
        "XML_wellformed (D0)": "D0",
        "Source_Fidelity (D1)": "D1",
        "Project_valid (D2)": "D2 (project)",
        "Classification_Match (D4a)": "D4a",
        "Element_Match (D4b)": "D4b",
    }
    matrix = df[list(col_map.keys())].copy()
    matrix = matrix.rename(columns=col_map)
    # True -> 1, False / None / NaN -> 0
    matrix = matrix.apply(lambda s: s.map(lambda x: 1 if x is True else 0))
    n_rows, n_cols = matrix.shape
    if n_rows == 0:
        return
    fig_h = max(6, n_rows * 0.12)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    annot = n_rows <= 80
    show_yticklabels = n_rows <= 100
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="d",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Pass (1) / Fail (0)"},
        vmin=0,
        vmax=1,
        yticklabels=show_yticklabels,
    )
    ax.set_title(
        "Per-sequence dimension pass/fail (D0, D1, D2 project, D4a, D4b)", fontweight="bold"
    )
    ax.set_ylabel("Bracketed sequence (row index)" if show_yticklabels else "Sequence index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def estimate_local_cost_usd(processing_time_seconds: float, power_w: float) -> float:
    """
    Estimated cost of a local run: electricity plus amortised GPU cost.

    Electricity: power (kW) x runtime (h) x electricity rate. Hardware: GPU price spread over
    ``AMORTIZATION_YEARS`` x ``USAGE_HOURS_PER_YEAR`` hours of use, times the runtime. The
    parameters are in ``config.py``.

    Args:
        processing_time_seconds: Runtime of the run
        power_w: Measured average GPU power draw during the run, in watts

    Returns:
        float: Cost in USD, rounded to cents
    """
    hours = processing_time_seconds / 3600
    electricity = power_w / 1000 * hours * config.ELECTRICITY_RATE_USD_PER_KWH
    hourly_hardware_rate = config.GPU_PRICE_USD / (
        config.AMORTIZATION_YEARS * config.USAGE_HOURS_PER_YEAR
    )
    return round(electricity + hourly_hardware_rate * hours, 2)


def _copy_processing_metadata_to_results(processing_dir: Path, results_dir: Path) -> None:
    """
    Copy the run's ``processing_metadata.json`` to the results folder.

    Takes it from the processing folder, or from its ``log/`` subfolder, and names the copy
    ``processing_metadata_<timestamp>.json`` after the processing folder. An existing copy is
    overwritten. For local runs listed in ``config.MEASURED_POWER_W``, the copy gets
    ``estimated_cost_usd`` (see ``estimate_local_cost_usd``).

    Args:
        processing_dir: ``data/output/<model>/processing_<timestamp>/``
        results_dir: The model's results folder
    """
    if not processing_dir.exists():
        return

    # Extract timestamp from processing folder name
    timestamp_match = re.search(r"processing_(\d{8}_\d{6})", processing_dir.name)
    timestamp_suffix = f"_{timestamp_match.group(1)}" if timestamp_match else ""

    # Copy processing_metadata.json (may be at processing root or in log/)
    for src in [
        processing_dir / "processing_metadata.json",
        processing_dir / "log" / "processing_metadata.json",
    ]:
        if src.exists():
            dest = results_dir / f"processing_metadata{timestamp_suffix}.json"
            run_key = f"{processing_dir.parent.name}/{processing_dir.name}"
            power_w = config.MEASURED_POWER_W.get(run_key)
            if power_w is None:
                shutil.copy2(src, dest)
            else:
                with open(src, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata["estimated_cost_usd"] = estimate_local_cost_usd(
                    metadata["processing_time_seconds"], power_w
                )
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            break


def create_evaluation_excel(
    brackets_data: pd.DataFrame,
    xml_edited_dir: Union[str, Path],
    references_dir: Union[str, Path],
    xml_input_dir: Union[str, Path],
    model_instance: str,
    output_file: Optional[str] = None,
    mapping_file: Optional[str] = None,
    intervention_type_lookup: Optional[Dict[Tuple[str, str], str]] = None,
) -> str:
    """
    Evaluate every bracketed sequence of one run and write the reports.

    For each sequence: the model's encoding and type are looked up in the mapping file, the model's
    version of the parent element is rebuilt, and D0, D1, D2 (TEI and project), D4a and D4b are
    computed against the gold standard. Sequences without gold-standard elements or with type
    "Unknown" are left out. Not processed sequences get N/A for all later dimensions, malformed
    ones for D2-D4b. The overall score per sequence is the mean of D1, D4a and D4b (D1 counts
    only where D4b is true), times 0.75 if the project rules are broken, and 0 for malformed XML.

    Writes the Excel report (sheets Summary, Details, Source Fidelity (D1), Classification (D4a),
    Element Application (D4b), Element Categories (Relaxed)), the summary JSON next to it, the two
    heatmaps, and copies the run's processing metadata into the results folder.

    Args:
        brackets_data: DataFrame with bracketed sequences and parent element content
        xml_edited_dir: Not used
        references_dir: Directory containing gold standard reference files
        xml_input_dir: Directory containing the source letters (input element text, and fallback
            if Parent_Element_Content is missing)
        model_instance: Required model instance name (e.g., 'qwen3-14B-Q6')
        output_file: Optional output file path (if None, generates timestamped filename)
        mapping_file: Optional specific mapping file to use (if None, the newest by modification
            time)
        intervention_type_lookup: Optional dict mapping (Filename, Bracketed_Content) to the
            gold-standard intervention type; sequences not in it get a computed type

    Returns:
        str: Path of the Excel report
    """
    # Load XML mapping data
    xml_mapping_data = None
    if mapping_file is None:
        mapping_file = get_latest_xml_mapping_file(
            model_instance, Path(__file__).parent.parent / config.LLM_OUTPUT_DIR
        )

    # Load processing metadata from the same directory as the mapping file
    processing_metadata = None
    if mapping_file:
        xml_mapping_data = load_xml_mapping(mapping_file)
        print(f"Loaded TEI encodings from {mapping_file}")
        # Load processing metadata
        metadata_path = Path(mapping_file).parent / "processing_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                processing_metadata = json.load(f)
            print(f"Loaded processing metadata from {metadata_path}")
    else:
        print("No mapping file found or specified")

    # Generate timestamped filename with model information using mapping file timestamp
    if output_file is None:
        base_path = get_timestamped_evaluation_filename(model_instance, "evaluation", mapping_file)
        output_file = f"{base_path}.xlsx"

    evaluation_data = []

    # Counters for filtering statistics
    total_instances = 0
    filtered_instances = 0

    # Process each entry from the brackets extraction data
    for index, row in brackets_data.iterrows():
        total_instances += 1
        filename = row["Filename"]
        xpath = row["Full_XPath"]
        element_id = row.get("ID", "")  # Get the element ID if available
        bracketed_content = row["Bracketed_Content"]
        parent_element_content = row.get(
            "Parent_Element_Content", ""
        )  # Get parent element from Excel

        # Get TEI encoding and LLM intervention type from mapping if available
        tei_encoding = None
        llm_intervention_type = "None"
        if xml_mapping_data and element_id:
            # Get the TEI encoding for this specific bracketed sequence
            tei_encoding = get_tei_encoding_for_element(
                xml_mapping_data, filename, element_id, bracketed_content
            )

            # Get the LLM intervention type for this specific bracketed sequence
            if filename in xml_mapping_data:
                # Normalize apostrophe types for comparison (using imported utility)
                normalized_bracketed_content = normalize_apostrophes(bracketed_content)

                # Search for the exact match for this specific bracketed sequence
                found_match = False
                for entry in xml_mapping_data[filename]:
                    if entry.get("element_id") == element_id:
                        for replacement in entry.get("replacements", []):
                            replacement_sequence = replacement.get("original_item", "")
                            normalized_replacement = normalize_apostrophes(replacement_sequence)

                            # Try exact match first (most precise)
                            if replacement_sequence == bracketed_content:
                                llm_intervention_type = replacement.get("intervention_type", "None")
                                found_match = True
                                break
                            # Try normalized exact match
                            elif normalized_replacement == normalized_bracketed_content:
                                llm_intervention_type = replacement.get("intervention_type", "None")
                                found_match = True
                                break

                        if found_match:
                            break

                # Only use fallback matching if no exact match was found
                if not found_match:
                    for entry in xml_mapping_data[filename]:
                        if entry.get("element_id") == element_id:
                            for replacement in entry.get("replacements", []):
                                replacement_sequence = replacement.get("original_item", "")
                                normalized_replacement = normalize_apostrophes(replacement_sequence)

                                # Fall back to substring matching (bidirectional) - less precise
                                if (
                                    bracketed_content in replacement_sequence
                                    or replacement_sequence in bracketed_content
                                ):
                                    llm_intervention_type = replacement.get(
                                        "intervention_type", "None"
                                    )
                                    found_match = True
                                    break
                                # Fall back to normalized substring matching
                                elif (
                                    normalized_bracketed_content in normalized_replacement
                                    or normalized_replacement in normalized_bracketed_content
                                ):
                                    llm_intervention_type = replacement.get(
                                        "intervention_type", "None"
                                    )
                                    found_match = True
                                    break

                            if found_match:
                                break

        # Extract TEI elements from the LLM encoding early (needed for normalization)
        llm_elements = extract_tei_elements_from_encoding(tei_encoding)

        # Create tei_encoding_normalized using specialized function for intervention types
        tei_encoding_normalized = normalize_tei_encoding_for_intervention(
            tei_encoding, llm_intervention_type, llm_elements
        )

        # Normalize whitespace for TEI encoding text (apostrophe normalization happens later)
        if tei_encoding_normalized:
            tei_encoding_normalized = re.sub(r"\s+", " ", tei_encoding_normalized.strip())

        print(f"Processing: {filename} - {xpath}")

        # Define file paths
        gold_file = Path(references_dir) / filename
        input_file = Path(xml_input_dir) / filename

        # Check if files exist
        gold_exists = gold_file.exists()

        # Check if we have parent element content from Excel
        has_parent_element = (
            parent_element_content
            and not pd.isna(parent_element_content)
            and str(parent_element_content).strip() != ""
        )

        if not has_parent_element:
            print(
                f"Warning: no Parent_Element_Content for {filename} - {element_id}", file=sys.stderr
            )
            # Try to extract from input file as fallback
            parent_xpath = create_parent_xpath(xpath)
            parent_element_content = extract_element_content_by_xpath(input_file, parent_xpath)
            if not parent_element_content:
                evaluation_data.append(
                    create_empty_evaluation_result(
                        filename=filename,
                        bracketed_content=bracketed_content,
                        xpath=xpath,
                        parent_xpath=parent_xpath,
                        gold_exists=gold_exists,
                    )
                )
                continue

        if not gold_exists:
            print(f"Warning: gold standard file not found: {gold_file}", file=sys.stderr)
            evaluation_data.append(
                create_empty_evaluation_result(
                    filename=filename,
                    bracketed_content=bracketed_content,
                    xpath=xpath,
                    gold_exists=False,
                )
            )
            continue

        # Create parent element XPath by removing /text() or text()[n] suffix
        parent_xpath = create_parent_xpath(xpath)

        # Use parent element content from Excel (or extract as fallback)
        if (
            parent_element_content
            and not pd.isna(parent_element_content)
            and str(parent_element_content).strip() != ""
        ):
            input_element = str(parent_element_content)
        else:
            # Fallback: extract from input file
            input_element = extract_element_content_by_xpath(input_file, parent_xpath)

        # Reconstruct output element by replacing bracketed sequences with TEI encodings
        if xml_mapping_data:
            output_element = reconstruct_edited_element(
                input_element, xml_mapping_data, filename, element_id
            )
        else:
            output_element = input_element  # No mapping data, output same as input

        # Extract gold standard element content (full XML with tags) using parent XPath
        gold_standard_element = extract_element_content_by_xpath(gold_file, parent_xpath)
        # Kept with notes, like Parent_Element_Content, for locating sequences in the element
        gold_standard_element_with_notes = gold_standard_element

        # Removing note elements for better readibility in the evaluation file
        input_element = remove_note_elements(input_element)
        output_element = remove_note_elements(output_element)
        gold_standard_element = remove_note_elements(gold_standard_element)

        # Compare input and output elements to determine what editorial interventions were added
        # (after removing note elements so we only compare actual editorial content)
        element_diff = compare_xml_elements(input_element, output_element, filename, parent_xpath)

        # Extract added elements for Excel column (including child elements) - Input vs LLM Output
        all_tags = []
        for elem in element_diff["added_elements"]:
            # Parse the XML content to extract all element tags (including the element itself and
            # children)
            try:
                child_tree = etree.fromstring(elem["content"])
                # Use iter() to get all elements including the root and its children
                for element in child_tree.iter():
                    # Ensure we get the string representation of the tag, handling namespaces
                    tag_str = str(element.tag) if element.tag is not None else "unknown"
                    # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]
                    all_tags.append(tag_str)
            except Exception:
                # If parsing fails, just use the parent tag
                parent_tag = str(elem["tag"]) if elem["tag"] is not None else "unknown"
                if "}" in parent_tag:
                    parent_tag = parent_tag.split("}")[1]
                all_tags.append(parent_tag)

        element_diff_summary = ", ".join(all_tags) if all_tags else "No changes"

        # Extract added elements for Excel column (including child elements) - Input vs Gold
        # Standard
        # (using note-removed elements for consistent comparison)
        gold_element_diff = compare_xml_elements(
            input_element, gold_standard_element, filename, parent_xpath
        )

        # Extract elements in document order directly from the gold standard element
        gold_all_tags = []
        if gold_standard_element:
            try:
                # Parse the gold standard element and extract all editorial intervention elements in
                # document order
                gold_tree = etree.fromstring(gold_standard_element)
                for element in gold_tree.iter():
                    # Only include editorial intervention elements (not structural elements like p,
                    # div, etc.)
                    tag_str = str(element.tag) if element.tag is not None else "unknown"
                    # Remove namespace prefix if present
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]

                    # Only include editorial intervention elements
                    if tag_str in INTERVENTION_ELEMENTS:
                        gold_all_tags.append(tag_str)
            except Exception as e:
                print(
                    f"Warning: could not parse gold standard element of {filename}: {e}",
                    file=sys.stderr,
                )
                # Fallback to original method
                for elem in gold_element_diff["added_elements"]:
                    try:
                        child_tree = etree.fromstring(elem["content"])
                        for element in child_tree.iter():
                            tag_str = str(element.tag) if element.tag is not None else "unknown"
                            if "}" in tag_str:
                                tag_str = tag_str.split("}")[1]
                            gold_all_tags.append(tag_str)
                    except Exception:
                        parent_tag = str(elem["tag"]) if elem["tag"] is not None else "unknown"
                        if "}" in parent_tag:
                            parent_tag = parent_tag.split("}")[1]
                        gold_all_tags.append(parent_tag)

        gold_element_diff_summary = ", ".join(gold_all_tags) if gold_all_tags else "No changes"

        # Check if LLM and gold standard added the same element types
        # Extract all element tags from LLM output (including child elements)
        llm_element_tags = []
        for elem in element_diff["added_elements"]:
            # Parse the XML content to extract all element tags (including the element itself and
            # children)
            try:
                child_tree = etree.fromstring(elem["content"])
                # Use iter() to get all elements including the root and its children
                for element in child_tree.iter():
                    # Ensure we get the string representation of the tag, handling namespaces
                    tag_str = str(element.tag) if element.tag is not None else "unknown"
                    # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]
                    llm_element_tags.append(tag_str)
            except Exception as e:
                print(
                    f"Warning: could not parse an element added by the model: {e}", file=sys.stderr
                )
                # If parsing fails, just use the parent tag
                parent_tag = str(elem["tag"]) if elem["tag"] is not None else "unknown"
                if "}" in parent_tag:
                    parent_tag = parent_tag.split("}")[1]
                llm_element_tags.append(parent_tag)

        # Extract all element tags from gold standard (including child elements)
        gold_element_tags = []
        for elem in gold_element_diff["added_elements"]:
            # Parse the XML content to extract all element tags (including the element itself and
            # children)
            try:
                child_tree = etree.fromstring(elem["content"])
                # Use iter() to get all elements including the root and its children
                for element in child_tree.iter():
                    # Ensure we get the string representation of the tag, handling namespaces
                    tag_str = str(element.tag) if element.tag is not None else "unknown"
                    # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]
                    gold_element_tags.append(tag_str)
            except Exception as e:
                print(
                    f"Warning: could not parse an element added in the gold standard: {e}",
                    file=sys.stderr,
                )
                # If parsing fails, just use the parent tag
                parent_tag = str(elem["tag"]) if elem["tag"] is not None else "unknown"
                if "}" in parent_tag:
                    parent_tag = parent_tag.split("}")[1]
                gold_element_tags.append(parent_tag)

        # Sort both lists for comparison (order doesn't matter, only element types)
        llm_element_tags.sort()
        gold_element_tags.sort()

        # Element_Match is True if both lists contain exactly the same element types
        element_match = llm_element_tags == gold_element_tags

        # Flexible_Element_Match considers certain editorial interventions as equivalent
        def normalize_element_groups(elements: List[str]) -> List[str]:
            """Group equivalent editorial interventions together"""
            # Count elements for different groupings
            choice_count = elements.count("choice")
            abbr_count = elements.count("abbr")
            expan_count = elements.count("expan")
            sic_count = elements.count("sic")
            corr_count = elements.count("corr")
            supplied_count = elements.count("supplied")

            # Calculate how many complete choice+abbr+expan groups
            abbr_expan_groups = min(choice_count, abbr_count, expan_count)

            # Calculate how many complete choice+sic+corr groups
            # (after accounting for choices used in abbr+expan groups)
            remaining_choice = choice_count - abbr_expan_groups
            sic_corr_groups = min(remaining_choice, sic_count, corr_count)

            # Calculate how many supplied elements (each counts as one group)
            supplied_groups = supplied_count

            # Total equivalent groups

            # Create the normalized list
            normalized = []

            # Add the equivalent groups
            for _ in range(abbr_expan_groups):
                normalized.append("choice_abbr_expan")
            for _ in range(sic_corr_groups):
                normalized.append("choice_sic_corr")
            for _ in range(supplied_groups):
                normalized.append(
                    "choice_abbr_expan"
                )  # Treat supplied as equivalent to choice_abbr_expan

            # Add any remaining elements that aren't part of these groups
            for elem in elements:
                if elem not in ["choice", "abbr", "expan", "sic", "corr", "supplied"]:
                    normalized.append(elem)

            return normalized

        # Apply flexible matching logic
        llm_flexible = normalize_element_groups(llm_element_tags.copy())
        gold_flexible = normalize_element_groups(gold_element_tags.copy())

        # Sort for comparison
        llm_flexible.sort()
        gold_flexible.sort()

        flexible_element_match = llm_flexible == gold_flexible

        # Extract all text content from input element and normalize it
        # Use parent_xpath to get the full element, remove notes, then extract text
        input_element_raw = extract_element_content_by_xpath(input_file, parent_xpath)
        input_element_cleaned = remove_note_elements(input_element_raw)

        # Extract text content from the cleaned element
        input_text_content = None
        if input_element_cleaned:
            try:
                # Parse the cleaned XML and extract text content
                input_tree = etree.fromstring(input_element_cleaned)
                input_text_content = etree.tostring(input_tree, encoding="unicode", method="text")
            except Exception as e:
                print(
                    f"Warning: could not extract text from the input element: {e}", file=sys.stderr
                )
                # Fallback to original method
                input_text_content = extract_all_text_content(input_file, xpath)

        input_content_normalized = (
            normalize_text_for_editorial_interventions(
                input_text_content, bracketed_content, element_diff_summary
            )
            if input_text_content
            else None
        )

        # Normalize whitespace for input content (remove indentation and normalize spaces)
        if input_content_normalized:
            input_content_normalized = re.sub(r"\s+", " ", input_content_normalized.strip())

        # Extract output content using the same logic as input: get element, remove notes, extract
        # text
        # output_element is already reconstructed and cleaned (note elements removed earlier)
        output_element_cleaned = output_element  # Already cleaned in earlier step

        # Extract text content from the cleaned output element
        output_text_content = None
        if output_element_cleaned:
            try:
                # Parse the cleaned XML and extract text content
                output_tree = etree.fromstring(output_element_cleaned)
                output_text_content = etree.tostring(output_tree, encoding="unicode", method="text")
            except Exception as e:
                print(
                    f"Warning: could not extract text from the output element: {e}", file=sys.stderr
                )
                # Fallback: extract text from the element string directly
                if output_element_cleaned:
                    # Simple text extraction from XML string
                    try:
                        output_tree = etree.fromstring(output_element_cleaned)
                        output_text_content = "".join(output_tree.itertext())
                    except Exception:
                        output_text_content = None

        # Normalize whitespace for output content (remove indentation and normalize spaces)
        if output_text_content:
            output_text_content = re.sub(r"\s+", " ", output_text_content.strip())

        # Perform string comparison between normalized input and output
        string_comparison = False
        if input_content_normalized is not None and output_text_content is not None:
            # Both are already text content (not XML), so direct comparison
            input_normalized = input_content_normalized.strip()
            output_normalized = output_text_content.strip()
            string_comparison = input_normalized == output_normalized

        # Perform string comparison without any whitespace characters
        string_comparison_no_ws = False
        if input_content_normalized is not None and output_text_content is not None:
            # Remove all whitespace characters (spaces, tabs, newlines, etc.)
            input_no_ws = re.sub(r"\s+", "", input_content_normalized)
            output_no_ws = re.sub(r"\s+", "", output_text_content)
            string_comparison_no_ws = input_no_ws == output_no_ws

        # Initialize bracketed_content_normalized placeholder (will be calculated later after
        # llm_elements)
        bracketed_content_normalized = None

        # Perform string comparison between bracketed content and TEI encoding (both normalized)
        string_comparison_content = False

        # Calculate element accuracy percentage
        element_accuracy_percentage = calculate_element_accuracy_percentage(
            element_diff, gold_element_diff
        )

        # Validate XML and TEI
        xml_wellformed = validate_xml_wellformed(tei_encoding)
        tei_valid = validate_tei_schema(tei_encoding)

        # Get goldstandard elements for this specific bracketed sequence position (needed for
        # normalization)
        goldstandard_elements = get_goldstandard_elements_for_position(
            brackets_data, index, gold_element_diff_summary, gold_standard_element_with_notes
        )

        # Get Gold_Classification (intervention types) from brackets_statistics.xlsx lookup
        # Fallback to computed value if not found in lookup
        lookup_key = (filename, bracketed_content)
        if intervention_type_lookup and lookup_key in intervention_type_lookup:
            intervention_type = intervention_type_lookup[lookup_key]
        else:
            # Fallback: the same rules that built brackets_statistics.xlsx; only used for
            # sequences missing from it
            intervention_type = categorize_intervention_type(goldstandard_elements)

        # Compute Brackets_and_Types column: "bracket_count;type_count"
        brackets_and_types = count_brackets_and_types(bracketed_content, intervention_type)

        # Create bracketed_content_normalized using Gold_Classification types
        # Processes brackets from right to left, matching each to its corresponding type
        bracketed_content_normalized = (
            normalize_bracketed_content_by_gold_types(
                bracketed_content, intervention_type, filename
            )
            if bracketed_content
            else None
        )

        # Normalize whitespace and apostrophes for bracketed content (same as input content)
        if bracketed_content_normalized:
            bracketed_content_normalized = re.sub(r"\s+", " ", bracketed_content_normalized.strip())
            # Replace straight apostrophe (U+0027) with curly apostrophe (U+2019)
            bracketed_content_normalized = bracketed_content_normalized.replace("'", "\u2019")

        # Match apostrophe type in LLM output to match bracketed content
        if tei_encoding_normalized and bracketed_content_normalized:
            if "\u2019" in bracketed_content_normalized:
                # Bracketed content uses curly apostrophe, replace straight with curly in LLM output
                tei_encoding_normalized = tei_encoding_normalized.replace("'", "\u2019")
            elif "'" in bracketed_content_normalized:
                # Bracketed content uses straight apostrophe, replace curly with straight in LLM
                # output
                tei_encoding_normalized = tei_encoding_normalized.replace("\u2019", "'")

        # Now calculate string comparison between bracketed content and TEI encoding (both
        # normalized)
        if bracketed_content_normalized is not None and tei_encoding_normalized is not None:
            string_comparison_content = (
                bracketed_content_normalized.strip() == tei_encoding_normalized.strip()
            )

        # Calculate new element match based on LLM_element vs Goldstandard_element columns
        # Include special case: choice+sic+corr in goldstandard should match single sic in LLM
        sequence_element_match = _compare_elements_with_correction_equivalence(
            llm_elements, goldstandard_elements
        )

        # Calculate new flexible element match based on LLM_element vs Goldstandard_element columns
        def compare_element_strings_flexibly(
            llm_str: Optional[str], gold_str: Optional[str]
        ) -> bool:
            """Compare two comma-separated element strings using flexible matching logic"""
            if not llm_str or not gold_str or llm_str == "No elements" or gold_str == "No elements":
                return llm_str == gold_str

            # Parse both strings into lists
            llm_tags = [tag.strip() for tag in llm_str.split(",")]
            gold_tags = [tag.strip() for tag in gold_str.split(",")]

            # Apply the same flexible normalization logic as before
            def normalize_tag_list(tags: List[str]) -> List[str]:
                """Apply flexible normalization to a list of tags"""
                # Count elements for different groupings
                choice_count = tags.count("choice")
                abbr_count = tags.count("abbr")
                expan_count = tags.count("expan")
                sic_count = tags.count("sic")
                corr_count = tags.count("corr")
                supplied_count = tags.count("supplied")

                # Calculate complete groups
                abbr_expan_groups = min(choice_count, abbr_count, expan_count)
                remaining_choice = choice_count - abbr_expan_groups
                sic_corr_groups = min(remaining_choice, sic_count, corr_count)

                # Create normalized list
                normalized = []
                for _ in range(abbr_expan_groups):
                    normalized.append("choice_abbr_expan")
                for _ in range(sic_corr_groups):
                    normalized.append("choice_sic_corr")
                for _ in range(supplied_count):
                    normalized.append("choice_abbr_expan")  # Treat supplied as equivalent

                # Add remaining elements
                for elem in tags:
                    if elem not in ["choice", "abbr", "expan", "sic", "corr", "supplied"]:
                        normalized.append(elem)

                return sorted(normalized)

            # Compare normalized versions
            llm_normalized = normalize_tag_list(llm_tags)
            gold_normalized = normalize_tag_list(gold_tags)

            return llm_normalized == gold_normalized

        sequence_flexible_match = compare_element_strings_flexibly(
            llm_elements, goldstandard_elements
        )

        # Calculate element accuracy percentage based on sequence-specific comparison
        def calculate_sequence_element_accuracy(
            llm_str: Optional[str], gold_str: Optional[str]
        ) -> float:
            """
            Calculate accuracy percentage for sequence-specific element comparison:
            - Exactly the same = 100%
            - No overlaps = 0%
            - Choice element in gold + only child element in LLM = 50%
            """
            if not llm_str or not gold_str or llm_str == "No elements" or gold_str == "No elements":
                if llm_str == gold_str:
                    return 100.0
                else:
                    return 0.0

            # Exact match = 100%
            if llm_str == gold_str:
                return 100.0

            # Parse both strings into lists
            llm_tags = [tag.strip() for tag in llm_str.split(",")]
            gold_tags = [tag.strip() for tag in gold_str.split(",")]

            # Special logic for choice elements
            if "choice" in gold_tags:
                # Get the choice child elements from gold standard
                gold_choice_children = set()
                if "abbr" in gold_tags and "expan" in gold_tags:
                    gold_choice_children = {"abbr", "expan"}
                elif "sic" in gold_tags and "corr" in gold_tags:
                    gold_choice_children = {"sic", "corr"}

                if gold_choice_children:
                    # Case 1: LLM has choice + same two child elements = 100%
                    if "choice" in llm_tags:
                        llm_choice_children = {
                            tag for tag in llm_tags if tag in ["abbr", "expan", "sic", "corr"]
                        }
                        if llm_choice_children == gold_choice_children:
                            return 100.0
                        # Case 2: LLM has choice + only one child match = 50%
                        elif len(llm_choice_children & gold_choice_children) == 1:
                            return 50.0
                        # Case 2b: LLM has choice + wrong child elements (no overlap) = 0%
                        elif (
                            len(llm_choice_children & gold_choice_children) == 0
                            and len(llm_choice_children) > 0
                        ):
                            return 0.0

                    # Case 3: No choice in LLM but has child elements
                    else:
                        llm_choice_children = {
                            tag for tag in llm_tags if tag in ["abbr", "expan", "sic", "corr"]
                        }
                        overlap = llm_choice_children & gold_choice_children

                        # Case 3a: One child element match = 50%
                        if len(overlap) == 1:
                            return 50.0
                        # Case 3b: Both child elements but no choice wrapper = 0%
                        elif len(overlap) == 2:
                            return 0.0

            # Check for any overlaps (common elements)
            common_elements = set(llm_tags) & set(gold_tags)
            if common_elements:
                # There are some overlaps but not exact match
                # Calculate partial percentage based on overlap
                overlap_ratio = len(common_elements) / len(set(gold_tags))
                return round(overlap_ratio * 100.0, 1)

            # No overlaps = 0%
            return 0.0

        sequence_element_accuracy = calculate_sequence_element_accuracy(
            llm_elements, goldstandard_elements
        )

        # Add to evaluation data
        # Determine if the bracketed content was encoded by the LLM
        # Check if the TEI encoding looks like XML content (has angle brackets with tags)
        # This should be True even for malformed XML so we can report xml_wellformed=False
        is_xml_comment = (
            tei_encoding
            and tei_encoding.strip().startswith("<!--")
            and tei_encoding.strip().endswith("-->")
        )
        has_xml_tags = (
            tei_encoding and "<" in tei_encoding and ">" in tei_encoding and not is_xml_comment
        )
        encoded = has_xml_tags

        # Determine if the LLM categorised the intervention (assigned a type other than "None" or
        # "unknown")

        # intervention_type (Gold_Classification) was already computed earlier for normalization

        # Apply filtering criteria: only evaluate instances with Goldstandard_element
        # and exclude "Unknown" or None intervention types
        should_evaluate = (
            goldstandard_elements
            and goldstandard_elements != "No elements"
            and intervention_type not in ["Unknown", None]
        )

        if not should_evaluate:
            # Skip this instance - don't add to evaluation data
            filtered_instances += 1
            continue

        # Compare intervention types using first 7 letters (to handle variations like "Correction"
        # vs "Corrections")
        intervention_match = compare_intervention_types(llm_intervention_type, intervention_type)

        # Check if intervention type matches the element that was chosen
        type_element_match = check_type_element_match(llm_intervention_type, llm_elements)

        # Calculate project_valid based on project-specific rules
        project_valid = check_project_valid(llm_intervention_type, llm_elements, tei_encoding)

        # Calculate corr content comparison for Correction intervention types
        llm_corr_content = None
        gold_corr_content = None
        corr_match = None

        if llm_intervention_type and llm_intervention_type.lower().startswith("correction"):
            # Extract corr content from LLM TEI encoding
            llm_corr_content = extract_corr_content(tei_encoding)

            # Extract corr content from gold standard (S) element
            if gold_standard_element:
                gold_corr_content = extract_corr_content(gold_standard_element)

            # Compare corr contents
            if llm_corr_content is not None and gold_corr_content is not None:
                corr_match = llm_corr_content.strip() == gold_corr_content.strip()
            else:
                corr_match = False

            # For choice_sic_corr: update bracketed_content_normalized to include gold_corr
            if llm_elements:
                elements_list = [e.strip().lower() for e in llm_elements.split(",")]
                has_choice_sic_corr = (
                    "choice" in elements_list and "sic" in elements_list and "corr" in elements_list
                )
                if has_choice_sic_corr and gold_corr_content and bracketed_content_normalized:
                    # Append " | " + gold_corr_content to match the LLM_Output_Normalized format
                    bracketed_content_normalized = (
                        f"{bracketed_content_normalized} | {gold_corr_content}"
                    )
                    # Recalculate string comparison with updated normalized values
                    if tei_encoding_normalized is not None:
                        string_comparison_content = (
                            bracketed_content_normalized.strip() == tei_encoding_normalized.strip()
                        )

        # Cascading N/A logic: if not processed or not XML wellformed, downstream metrics are N/A
        # This ensures they are not counted in F1 score calculations
        if not encoded:
            # Not processed at all - all metrics should be N/A
            xml_wellformed = None
            tei_valid = "N/A"
            project_valid = None
            string_comparison_content = None
            intervention_match = None
            sequence_element_match = None
            sequence_element_accuracy = None
            sequence_flexible_match = None
            type_element_match = None
            corr_match = None
        elif not xml_wellformed:
            # Processed but not XML wellformed - downstream metrics should be N/A
            tei_valid = "N/A"
            project_valid = None
            string_comparison_content = None
            intervention_match = None
            sequence_element_match = None
            sequence_element_accuracy = None
            sequence_flexible_match = None
            type_element_match = None
            corr_match = None

        evaluation_data.append(
            {
                "Filename": filename,
                "Bracketed_Content": bracketed_content,
                "Brackets_and_Types": brackets_and_types,
                "Processed": encoded,
                "XML_wellformed (D0)": xml_wellformed,
                "Bracketed_Content_Normalized": bracketed_content_normalized,
                "LLM_Output_Normalized": tei_encoding_normalized,
                "Source_Fidelity (D1)": string_comparison_content,
                "LLM_corr": llm_corr_content,
                "Gold_corr": gold_corr_content,
                "Corr_Match": corr_match,
                "TEI_valid (D2)": tei_valid,
                "Project_valid (D2)": project_valid,
                "LLM_Classification": llm_intervention_type,
                "Gold_Classification": intervention_type,
                "Classification_Match (D4a)": intervention_match,
                "LLM_encoding": tei_encoding,
                "LLM_element": llm_elements,
                "Gold_element": goldstandard_elements,
                "Element_Match (D4b)": sequence_element_match,
                "Element_Accuracy_Percentage (D4b)": sequence_element_accuracy,
                "Flexible_Element_Match (D4b)": sequence_flexible_match,
                "Type-Element_Match": type_element_match,
                "Input_Element": input_element,
                "Output_Element": output_element,
                "Gold_Standard_Element": gold_standard_element,
                "Full_Element_Diff (input - llm output)": element_diff_summary,
                "Full_Element_Diff (input - goldstandard)": gold_element_diff_summary,
                "Full_Element_Match": element_match,
                "Full_Element_Accuracy_Percentage": element_accuracy_percentage,
                "Full_Flexible_Element_Match": flexible_element_match,
                "Input_Content_Normalized": input_content_normalized,
                "Output_Content_Normalized": output_text_content,
                "String_Comparison": string_comparison,
                "String_Comparison (no WS)": string_comparison_no_ws,
                "LLM_File_Exists": True,
                "Gold_File_Exists": True,
                "Xpath (Text)": xpath,
                "Xpath (Parent Element)": parent_xpath,
            }
        )

    # Create DataFrame and export to Excel
    df = pd.DataFrame(evaluation_data)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Details")

        # Get the worksheet
        worksheet = writer.sheets["Details"]

        # Freeze the first row (header row)
        worksheet.freeze_panes = "A2"

        # Add auto-filter to header row
        worksheet.auto_filter.ref = worksheet.dimensions

        # Set column widths. Columns are looked up by name, so adding or removing a column
        # does not shift the formatting onto the wrong column.
        from openpyxl.styles import Alignment, PatternFill
        from openpyxl.utils import get_column_letter

        column_index = {name: i + 1 for i, name in enumerate(df.columns)}
        column_widths = {
            "Filename": 15,
            "Bracketed_Content": 20,
            "Brackets_and_Types": 12,
            "Processed": 10,
            "XML_wellformed (D0)": 15,
            "Bracketed_Content_Normalized": 25,
            "LLM_Output_Normalized": 25,
            "Source_Fidelity (D1)": 20,
            "LLM_corr": 15,
            "Gold_corr": 15,
            "Corr_Match": 10,
            "TEI_valid (D2)": 12,
            "Project_valid (D2)": 14,
            "LLM_Classification": 18,
            "Gold_Classification": 18,
            "Classification_Match (D4a)": 18,
            "LLM_encoding": 30,
            "LLM_element": 15,
            "Gold_element": 18,
            "Element_Match (D4b)": 15,
            "Element_Accuracy_Percentage (D4b)": 20,
            "Flexible_Element_Match (D4b)": 18,
            "Type-Element_Match": 18,
            "Input_Element": 80,
            "Output_Element": 80,
            "Gold_Standard_Element": 80,
            "Full_Element_Diff (input - llm output)": 20,
            "Full_Element_Diff (input - goldstandard)": 20,
            "Full_Element_Match": 10,
            "Full_Element_Accuracy_Percentage": 10,
            "Full_Flexible_Element_Match": 10,
            "Input_Content_Normalized": 80,
            "Output_Content_Normalized": 80,
            "String_Comparison": 15,
            "String_Comparison (no WS)": 15,
            "LLM_File_Exists": 10,
            "Gold_File_Exists": 10,
            "Xpath (Text)": 50,
            "Xpath (Parent Element)": 50,
        }

        for name, width in column_widths.items():
            if name in column_index:
                worksheet.column_dimensions[get_column_letter(column_index[name])].width = width

        # Apply text wrapping to all cells
        num_rows = len(df) + 1
        num_cols = len(df.columns)

        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                cell = worksheet.cell(row=row, column=col)
                cell.alignment = Alignment(wrap_text=False, vertical="top")

        # Apply color coding to comparison columns
        green_fill = PatternFill(
            start_color="90EE90", end_color="90EE90", fill_type="solid"
        )  # Light green
        red_fill = PatternFill(
            start_color="FFB6C1", end_color="FFB6C1", fill_type="solid"
        )  # Light red
        yellow_fill = PatternFill(
            start_color="FFFF99", end_color="FFFF99", fill_type="solid"
        )  # Light yellow for partial matches
        light_yellow = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")

        # True = green, False = red; None values remain uncolored
        boolean_columns = [
            "Processed",
            "XML_wellformed (D0)",
            "Source_Fidelity (D1)",
            "Corr_Match",
            "TEI_valid (D2)",
            "Project_valid (D2)",
            "Classification_Match (D4a)",
            "Element_Match (D4b)",
            "Flexible_Element_Match (D4b)",
            "Type-Element_Match",
            "Full_Element_Match",
            "Full_Flexible_Element_Match",
            "String_Comparison",
            "String_Comparison (no WS)",
        ]

        for row in range(2, num_rows + 1):  # Start from row 2 (skip header)
            for name in boolean_columns:
                if name not in column_index:
                    continue
                cell = worksheet.cell(row=row, column=column_index[name])
                if cell.value == True:
                    cell.fill = green_fill
                elif cell.value == False:
                    cell.fill = red_fill

            # Element_Accuracy_Percentage (D4b): 100 green, >=50 yellow, >0 light yellow, 0 red
            if "Element_Accuracy_Percentage (D4b)" in column_index:
                cell = worksheet.cell(
                    row=row, column=column_index["Element_Accuracy_Percentage (D4b)"]
                )
                if cell.value is not None and isinstance(cell.value, (int, float)):
                    if cell.value == 100.0:
                        cell.fill = green_fill
                    elif cell.value >= 50.0:
                        cell.fill = yellow_fill
                    elif cell.value > 0.0:
                        cell.fill = light_yellow
                    else:
                        cell.fill = red_fill

            # Full_Element_Accuracy_Percentage: 100 green, >=50 yellow, below red
            if "Full_Element_Accuracy_Percentage" in column_index:
                cell = worksheet.cell(
                    row=row, column=column_index["Full_Element_Accuracy_Percentage"]
                )
                if cell.value is not None and isinstance(cell.value, (int, float)):
                    if cell.value == 100.0:
                        cell.fill = green_fill
                    elif cell.value >= 50.0:
                        cell.fill = yellow_fill
                    else:
                        cell.fill = red_fill

    print(f"\nEvaluation complete! Results saved to: {output_file}")

    # Calculate F1 Score Metrics
    df = pd.DataFrame(evaluation_data)

    # Count True Positives, False Positives, False Negatives for Intervention Type Categorization
    # Note: Rows with None/NaN in the match column (due to not processed or invalid XML) are
    # excluded
    # TP = Processed = True + XML_wellformed = True + Classification Match = True
    # FP = Processed = True + XML_wellformed = True + Classification Match = False
    # FN = Processed = False OR XML_wellformed = False (excluded from evaluation)
    # Explicitly filter for non-null values to ensure None/NaN are excluded
    valid_classification = df["Classification_Match (D4a)"].notna()
    intervention_true_positives = len(
        df[valid_classification & (df["Classification_Match (D4a)"] == True)]
    )
    intervention_false_positives = len(
        df[valid_classification & (df["Classification_Match (D4a)"] == False)]
    )
    # FN = sequences that couldn't be evaluated (not processed OR invalid XML)
    intervention_false_negatives = len(df[~valid_classification])

    # Calculate Intervention Type F1 Score
    total_intervention_attempts = intervention_true_positives + intervention_false_positives
    total_intervention_sequences = (
        intervention_true_positives + intervention_false_negatives
    )  # All bracketed sequences

    intervention_precision = (
        intervention_true_positives / total_intervention_attempts
        if total_intervention_attempts > 0
        else 0
    )
    intervention_recall = (
        intervention_true_positives / total_intervention_sequences
        if total_intervention_sequences > 0
        else 0
    )
    intervention_f1_score = (
        2
        * (intervention_precision * intervention_recall)
        / (intervention_precision + intervention_recall)
        if (intervention_precision + intervention_recall) > 0
        else 0
    )

    # Count True Positives, False Positives, False Negatives (Standard F1 - Element Application)
    # Explicitly filter for non-null values to ensure None/NaN are excluded
    valid_element = df["Element_Match (D4b)"].notna()
    true_positives = len(df[valid_element & (df["Element_Match (D4b)"] == True)])
    false_positives = len(df[valid_element & (df["Element_Match (D4b)"] == False)])
    # FN = sequences that couldn't be evaluated (not processed OR invalid XML)
    false_negatives = len(df[~valid_element])

    # Calculate Standard Precision, Recall, and F1 Score
    total_encoding_attempts = true_positives + false_positives
    total_bracketed_sequences = true_positives + false_negatives

    precision = true_positives / total_encoding_attempts if total_encoding_attempts > 0 else 0
    recall = true_positives / total_bracketed_sequences if total_bracketed_sequences > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Per-category and macro F1 scores

    # Define intervention type categories (excluding None and Other)
    intervention_categories = INTERVENTION_CATEGORIES

    # 1. INTERVENTION TYPE (D4a) MACRO F1
    intervention_category_metrics = {}

    # Multi-label counting: a label such as "Abbreviation; Supplement" contributes once per type
    # it contains (repeated types count repeatedly). Comparing the whole label with a single
    # category would never match a two-type label, so correct two-type answers would earn no
    # credit while partial answers would still be penalised.
    def _count_type_in_label(label: Any, cat: str) -> int:
        """Number of ``;``-separated types in ``label`` that match ``cat``; 0 for no label."""
        if not isinstance(label, str) or not label.strip():
            return 0
        return sum(
            1
            for part in label.split(";")
            if part.strip() and compare_intervention_types(part.strip(), cat)
        )

    for category in intervention_categories:
        tp = fp = fn = 0
        for _, row in df.iterrows():
            gold_n = _count_type_in_label(row["Gold_Classification"], category)
            if pd.isna(row["Classification_Match (D4a)"]):
                # Not processed / invalid XML: every gold occurrence of this type is missed
                fn += gold_n
                continue
            llm_n = _count_type_in_label(row["LLM_Classification"], category)
            matched = min(gold_n, llm_n)
            tp += matched
            fp += llm_n - matched
            fn += gold_n - matched

        # Calculate per-category metrics
        cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cat_f1 = (
            2 * (cat_precision * cat_recall) / (cat_precision + cat_recall)
            if (cat_precision + cat_recall) > 0
            else 0
        )

        intervention_category_metrics[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": cat_precision,
            "recall": cat_recall,
            "f1": cat_f1,
        }

    # Macro F1 for Intervention Types (average individual F1 scores - including all categories)
    all_f1_scores = [metrics["f1"] for metrics in intervention_category_metrics.values()]
    intervention_macro_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

    # 2. ELEMENT APPLICATION (D4b) MACRO F1, STANDARD AND RELAXED
    # Create element-based categories using intervention-based groupings
    def categorize_elements(elements_str: Optional[str]) -> Optional[str]:
        """Categorize elements using intervention-based groupings"""
        if not elements_str or elements_str == "No elements":
            return None  # Excluded from analysis
        elements = [elem.strip().lower() for elem in elements_str.split(",")]
        if "choice" in elements:
            if "abbr" in elements and "expan" in elements:
                return "Abbreviation"
            elif "sic" in elements and "corr" in elements:
                return "Correction"
            else:
                return None  # Unrecognized choice combination, excluded from analysis
        elif "supplied" in elements:
            return "Supplement"
        elif "gap" in elements:
            return "Omission"
        elif "unclear" in elements:
            return "Uncertainty"
        elif "del" in elements or "surplus" in elements:
            return "Redundancy"
        elif "sic" in elements:
            # A bare <sic> (no choice/corr) is a valid correction encoding under the project
            # rules, and Element_Match already treats it as equal to choice+sic+corr.
            return "Correction"
        # Note: unrecognized elements are excluded from analysis
        else:
            return None  # Excluded from category analysis

    # Add element category columns to dataframe for analysis
    df["LLM_Element_Category"] = df["LLM_element"].apply(categorize_elements)
    df["Gold_Element_Category"] = df["Gold_element"].apply(categorize_elements)

    # Calculate element application metrics for standard and relaxed
    element_category_metrics = {}
    element_category_metrics_relaxed = {}

    for category in intervention_categories:
        # Standard Element Application
        # Only consider sequences where LLM attempted encoding for this category
        category_attempts = df[df["LLM_Element_Category"] == category]

        if len(category_attempts) > 0:
            tp_standard = len(category_attempts[category_attempts["Element_Match (D4b)"] == True])
            fp_standard = len(category_attempts[category_attempts["Element_Match (D4b)"] == False])
        else:
            tp_standard = fp_standard = 0

        # FN: Gold standard is this category but EITHER:
        # 1. LLM encoded as different category (misclassification)
        # 2. LLM couldn't be evaluated (Element_Match is N/A)
        # Note: Don't count Element_Match = False here as those are already in FP if LLM = category
        fn_misclassified = len(
            df[
                (df["Gold_Element_Category"] == category)
                & (df["LLM_Element_Category"] != category)
                & (df["LLM_Element_Category"].notna())
            ]
        )
        fn_not_evaluated = len(
            df[(df["Gold_Element_Category"] == category) & (df["Element_Match (D4b)"].isna())]
        )
        fn_standard = fn_misclassified + fn_not_evaluated

        # Calculate per-category metrics
        precision_std = (
            tp_standard / (tp_standard + fp_standard) if (tp_standard + fp_standard) > 0 else 0
        )
        recall_std = (
            tp_standard / (tp_standard + fn_standard) if (tp_standard + fn_standard) > 0 else 0
        )
        f1_std = (
            2 * (precision_std * recall_std) / (precision_std + recall_std)
            if (precision_std + recall_std) > 0
            else 0
        )

        element_category_metrics[category] = {
            "tp": tp_standard,
            "fp": fp_standard,
            "fn": fn_standard,
            "precision": precision_std,
            "recall": recall_std,
            "f1": f1_std,
        }

        # Relaxed Element Application
        # Only consider sequences where LLM attempted encoding for this category
        category_attempts_relaxed = df[df["LLM_Element_Category"] == category]
        if len(category_attempts_relaxed) > 0:
            tp_relaxed = len(
                category_attempts_relaxed[
                    category_attempts_relaxed["Flexible_Element_Match (D4b)"] == True
                ]
            )
            fp_relaxed = len(
                category_attempts_relaxed[
                    category_attempts_relaxed["Flexible_Element_Match (D4b)"] == False
                ]
            )
        else:
            tp_relaxed = fp_relaxed = 0

        fn_misclassified_rel = len(
            df[
                (df["Gold_Element_Category"] == category)
                & (df["LLM_Element_Category"] != category)
                & (df["LLM_Element_Category"].notna())
            ]
        )
        fn_not_evaluated_rel = len(
            df[
                (df["Gold_Element_Category"] == category)
                & (df["Flexible_Element_Match (D4b)"].isna())
            ]
        )
        fn_relaxed = fn_misclassified_rel + fn_not_evaluated_rel

        precision_rel = (
            tp_relaxed / (tp_relaxed + fp_relaxed) if (tp_relaxed + fp_relaxed) > 0 else 0
        )
        recall_rel = tp_relaxed / (tp_relaxed + fn_relaxed) if (tp_relaxed + fn_relaxed) > 0 else 0
        f1_rel = (
            2 * (precision_rel * recall_rel) / (precision_rel + recall_rel)
            if (precision_rel + recall_rel) > 0
            else 0
        )

        element_category_metrics_relaxed[category] = {
            "tp": tp_relaxed,
            "fp": fp_relaxed,
            "fn": fn_relaxed,
            "precision": precision_rel,
            "recall": recall_rel,
            "f1": f1_rel,
        }

    # Macro F1 for Element Application (Standard) - including all categories
    all_element_f1_scores = [metrics["f1"] for metrics in element_category_metrics.values()]
    element_macro_f1 = (
        sum(all_element_f1_scores) / len(all_element_f1_scores) if all_element_f1_scores else 0
    )

    # 3. CONTENT PRESERVATION F1 SCORE
    # Count True Positives, False Positives, False Negatives for Content Preservation
    # D1 is only evaluated when D4b (Element_Match) is TRUE
    valid_source_fidelity = df["Source_Fidelity (D1)"].notna() & (df["Element_Match (D4b)"] == True)
    content_true_positives = len(df[valid_source_fidelity & (df["Source_Fidelity (D1)"] == True)])
    content_false_positives = len(df[valid_source_fidelity & (df["Source_Fidelity (D1)"] == False)])
    # FN = sequences where D4b is TRUE but D1 couldn't be evaluated OR D4b is not TRUE
    content_false_negatives = len(
        df[(df["Element_Match (D4b)"] == True) & (df["Source_Fidelity (D1)"].isna())]
    ) + len(df[df["Element_Match (D4b)"] != True])

    # Calculate Content Preservation F1 Score
    total_content_attempts = content_true_positives + content_false_positives
    total_content_sequences = content_true_positives + content_false_negatives

    content_precision = (
        content_true_positives / total_content_attempts if total_content_attempts > 0 else 0
    )
    content_recall = (
        content_true_positives / total_content_sequences if total_content_sequences > 0 else 0
    )
    content_f1_score = (
        2 * (content_precision * content_recall) / (content_precision + content_recall)
        if (content_precision + content_recall) > 0
        else 0
    )

    # Content Preservation (D1) F1 by Category
    content_category_metrics = {}

    for category in intervention_categories:
        # Only consider sequences where LLM attempted encoding for this category AND D4b is TRUE
        category_attempts = df[
            (df["LLM_Element_Category"] == category) & (df["Element_Match (D4b)"] == True)
        ]

        if len(category_attempts) > 0:
            tp_content = len(category_attempts[category_attempts["Source_Fidelity (D1)"] == True])
            fp_content = len(category_attempts[category_attempts["Source_Fidelity (D1)"] == False])
        else:
            tp_content = fp_content = 0

        # FN: Gold standard is this category but EITHER:
        # 1. LLM encoded as different category (misclassification) OR D4b is not TRUE
        # 2. LLM couldn't be evaluated (Source_Fidelity is N/A) when D4b is TRUE
        fn_misclassified_or_wrong_element = len(
            df[
                (df["Gold_Element_Category"] == category)
                & ((df["LLM_Element_Category"] != category) | (df["Element_Match (D4b)"] != True))
            ]
        )
        fn_not_evaluated_content = len(
            df[
                (df["Gold_Element_Category"] == category)
                & (df["Element_Match (D4b)"] == True)
                & (df["Source_Fidelity (D1)"].isna())
            ]
        )
        fn_content = fn_misclassified_or_wrong_element + fn_not_evaluated_content

        # Calculate per-category metrics
        precision_content = (
            tp_content / (tp_content + fp_content) if (tp_content + fp_content) > 0 else 0
        )
        recall_content = (
            tp_content / (tp_content + fn_content) if (tp_content + fn_content) > 0 else 0
        )
        f1_content = (
            2 * (precision_content * recall_content) / (precision_content + recall_content)
            if (precision_content + recall_content) > 0
            else 0
        )

        content_category_metrics[category] = {
            "tp": tp_content,
            "fp": fp_content,
            "fn": fn_content,
            "precision": precision_content,
            "recall": recall_content,
            "f1": f1_content,
        }

    # Macro F1 for Content Preservation - including all categories
    all_content_f1_scores = [metrics["f1"] for metrics in content_category_metrics.values()]
    content_macro_f1 = (
        sum(all_content_f1_scores) / len(all_content_f1_scores) if all_content_f1_scores else 0
    )

    # Calculate XML Wellformedness and TEI Validity Metrics (only for encoded instances)
    encoded_instances = df[df["Processed"] == True]
    total_encoded = len(encoded_instances)

    if total_encoded > 0:
        xml_wellformed_count = len(
            encoded_instances[encoded_instances["XML_wellformed (D0)"] == True]
        )
        # Count only True values (exclude False and "N/A")
        tei_valid_count = len(encoded_instances[encoded_instances["TEI_valid (D2)"] == True])
        project_valid_count = len(
            encoded_instances[encoded_instances["Project_valid (D2)"] == True]
        )

        xml_wellformed_percentage = (xml_wellformed_count / total_encoded) * 100
        tei_valid_percentage = (tei_valid_count / total_encoded) * 100
        project_valid_percentage = (project_valid_count / total_encoded) * 100
    else:
        xml_wellformed_count = 0
        tei_valid_count = 0
        project_valid_count = 0
        xml_wellformed_percentage = 0
        tei_valid_percentage = 0
        project_valid_percentage = 0

    # Calculate pass rates for semantic dimensions (D1, D4a, D4b)
    # D1 Source Fidelity pass rate - only evaluated when D4b (Element_Match) is TRUE
    d1_evaluated = len(
        df[(df["Source_Fidelity (D1)"].notna()) & (df["Element_Match (D4b)"] == True)]
    )
    d1_passed = len(df[(df["Source_Fidelity (D1)"] == True) & (df["Element_Match (D4b)"] == True)])
    d1_pass_rate = (d1_passed / d1_evaluated * 100) if d1_evaluated > 0 else 0

    # D4a Classification pass rate
    d4a_evaluated = len(df[df["Classification_Match (D4a)"].notna()])
    d4a_passed = len(df[df["Classification_Match (D4a)"] == True])
    d4a_pass_rate = (d4a_passed / d4a_evaluated * 100) if d4a_evaluated > 0 else 0

    # D4b Element Match pass rate
    d4b_evaluated = len(df[df["Element_Match (D4b)"].notna()])
    d4b_passed = len(df[df["Element_Match (D4b)"] == True])
    d4b_pass_rate = (d4b_passed / d4b_evaluated * 100) if d4b_evaluated > 0 else 0

    # Debug information
    print(
        f"Validation metrics: Total encoded: {total_encoded}, "
        f"XML wellformed: {xml_wellformed_count}, TEI valid: {tei_valid_count}, "
        f"Project valid: {project_valid_count}"
    )
    print(
        f"Percentages: XML {xml_wellformed_percentage:.2f}%, "
        f"TEI valid {tei_valid_percentage:.2f}%, Project valid {project_valid_percentage:.2f}%"
    )
    print(
        f"Semantic pass rates: D1 {d1_pass_rate:.2f}%, D4a {d4a_pass_rate:.2f}%, "
        f"D4b {d4b_pass_rate:.2f}%"
    )

    # Add metrics to Excel as additional sheets
    print("Adding metrics sheets to Excel file...")

    # Read the existing Excel file and add new sheets
    from openpyxl.styles import Alignment, Font, PatternFill

    # Calculate Overall Tiered Scoring
    # Tier 0: XML Wellformedness (blocking condition)
    # Tier 1: Core Quality = (D1 + D4a + D4b) / 3
    # Tier 2: Schema Factor = 1.0 if Project_valid else 0.75
    # Overall Score = Core Quality × Schema Factor (0 if not XML wellformed)

    def calculate_instance_score(row: pd.Series) -> Tuple[float, float, float]:
        """Calculate tiered score for a single instance"""
        # Tier 0: Validity Gate
        if row["XML_wellformed (D0)"] != True:
            return 0.0, 0.0, 0.0  # overall, core_quality, schema_factor

        # Tier 1: Core Quality (binary 0/100 for each metric)
        # D1 is only evaluated when D4b is TRUE; if D4b is not TRUE, D1 contributes 0
        if row["Element_Match (D4b)"] == True:
            d1_score = 100.0 if row["Source_Fidelity (D1)"] == True else 0.0
        else:
            d1_score = 0.0  # D1 not evaluated when D4b is not TRUE

        d4a_score = 100.0 if row["Classification_Match (D4a)"] == True else 0.0
        d4b_score = 100.0 if row["Element_Match (D4b)"] == True else 0.0
        core_quality = (d1_score + d4a_score + d4b_score) / 3

        # Tier 2: Schema Factor
        schema_factor = 1.0 if row["Project_valid (D2)"] == True else SCHEMA_FACTOR_RULES_BROKEN

        # Overall Score
        overall_score = core_quality * schema_factor

        return overall_score, core_quality, schema_factor

    # Apply to all rows
    scores = df.apply(calculate_instance_score, axis=1)
    df["Overall_Score"] = [s[0] for s in scores]
    df["Core_Quality_Score"] = [s[1] for s in scores]
    df["Schema_Factor"] = [s[2] for s in scores]

    # Calculate aggregate statistics
    valid_xml_df = df[df["XML_wellformed (D0)"] == True]

    # Valid XML only
    valid_count = len(valid_xml_df)
    invalid_count = len(df) - valid_count
    validity_rate = (valid_count / len(df) * 100) if len(df) > 0 else 0

    avg_overall_valid = valid_xml_df["Overall_Score"].mean() if len(valid_xml_df) > 0 else 0
    min_overall_valid = valid_xml_df["Overall_Score"].min() if len(valid_xml_df) > 0 else 0
    max_overall_valid = valid_xml_df["Overall_Score"].max() if len(valid_xml_df) > 0 else 0

    # All instances
    avg_overall_all = df["Overall_Score"].mean() if len(df) > 0 else 0
    min_overall_all = df["Overall_Score"].min() if len(df) > 0 else 0
    max_overall_all = df["Overall_Score"].max() if len(df) > 0 else 0

    # Core Quality (valid XML only)
    avg_core_quality = valid_xml_df["Core_Quality_Score"].mean() if len(valid_xml_df) > 0 else 0
    min_core_quality = valid_xml_df["Core_Quality_Score"].min() if len(valid_xml_df) > 0 else 0
    max_core_quality = valid_xml_df["Core_Quality_Score"].max() if len(valid_xml_df) > 0 else 0

    # Schema Factor
    avg_schema_factor = valid_xml_df["Schema_Factor"].mean() if len(valid_xml_df) > 0 else 0
    min_schema_factor = valid_xml_df["Schema_Factor"].min() if len(valid_xml_df) > 0 else 0
    max_schema_factor = valid_xml_df["Schema_Factor"].max() if len(valid_xml_df) > 0 else 0

    with pd.ExcelWriter(
        output_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        # Sheet 2: Metrics Summary - Dimension-based structure
        total_sequences = len(df)

        metrics_summary_data = [
            ["EVALUATION SUMMARY", ""],
            ["", ""],
            ["Model Instance:", model_instance],
            [
                "Processing Time:",
                processing_metadata.get("processing_time", "N/A") if processing_metadata else "N/A",
            ],
            [
                "Date Processed:",
                processing_metadata.get("date_processed", "N/A") if processing_metadata else "N/A",
            ],
            ["", ""],
            ["Total Bracketed Sequences:", total_sequences],
            ["Total Processed:", total_encoded],
            ["", ""],
            ["DIMENSION 0: XML WELLFORMEDNESS", ""],
            ["Processed:", total_encoded],
            ["Passed:", xml_wellformed_count],
            ["Failed:", total_encoded - xml_wellformed_count],
            ["Pass Rate:", f"{xml_wellformed_percentage:.2f}%"],
            ["", ""],
            ["DIMENSION 1: SOURCE FIDELITY", ""],
            ["Note: Only evaluated when D4b (Element Match) is TRUE", ""],
            ["Evaluated:", d1_evaluated],
            ["Passed:", d1_passed],
            ["Failed:", d1_evaluated - d1_passed],
            ["Precision:", f"{content_precision * 100:.2f}%"],
            ["Recall:", f"{content_recall * 100:.2f}%"],
            ["F1 Score:", f"{content_f1_score * 100:.2f}%"],
            ["Macro F1:", f"{content_macro_f1 * 100:.2f}%"],
            ["", ""],
            ["DIMENSION 2: SCHEMA VALIDATION", ""],
            ["TEI Schema Valid:", tei_valid_count],
            ["TEI Schema Pass Rate:", f"{tei_valid_percentage:.2f}%"],
            ["Project Valid:", project_valid_count],
            ["Project Pass Rate:", f"{project_valid_percentage:.2f}%"],
            ["", ""],
            ["DIMENSION 4a: CLASSIFICATION ACCURACY", ""],
            ["Evaluated:", d4a_evaluated],
            ["Passed:", d4a_passed],
            ["Failed:", d4a_evaluated - d4a_passed],
            ["Precision:", f"{intervention_precision * 100:.2f}%"],
            ["Recall:", f"{intervention_recall * 100:.2f}%"],
            ["F1 Score:", f"{intervention_f1_score * 100:.2f}%"],
            ["Macro F1:", f"{intervention_macro_f1 * 100:.2f}%"],
            ["", ""],
            ["DIMENSION 4b: ELEMENT APPLICATION", ""],
            ["Evaluated:", d4b_evaluated],
            ["Passed:", d4b_passed],
            ["Failed:", d4b_evaluated - d4b_passed],
            ["Precision:", f"{precision * 100:.2f}%"],
            ["Recall:", f"{recall * 100:.2f}%"],
            ["F1 Score:", f"{f1_score * 100:.2f}%"],
            ["Macro F1:", f"{element_macro_f1 * 100:.2f}%"],
            ["", ""],
            ["OVERALL TIERED SCORING", ""],
            ["Valid Instances:", valid_count],
            ["Invalid Instances:", invalid_count],
            ["Validity Rate:", f"{validity_rate:.2f}%"],
            ["", ""],
            ["Average Overall Score (valid XML):", f"{avg_overall_valid:.2f}"],
            ["Min Overall Score (valid XML):", f"{min_overall_valid:.2f}"],
            ["Max Overall Score (valid XML):", f"{max_overall_valid:.2f}"],
            ["", ""],
            ["Average Overall Score (all):", f"{avg_overall_all:.2f}"],
            ["Min Overall Score (all):", f"{min_overall_all:.2f}"],
            ["Max Overall Score (all):", f"{max_overall_all:.2f}"],
            ["", ""],
            ["Average Core Quality Score:", f"{avg_core_quality:.2f}"],
            ["Min Core Quality Score:", f"{min_core_quality:.2f}"],
            ["Max Core Quality Score:", f"{max_core_quality:.2f}"],
            ["", ""],
            ["Average Schema Factor:", f"{avg_schema_factor:.3f}"],
            ["Min Schema Factor:", f"{min_schema_factor:.3f}"],
            ["Max Schema Factor:", f"{max_schema_factor:.3f}"],
        ]

        # Write metrics summary
        metrics_summary_df = pd.DataFrame(metrics_summary_data, columns=["Metric", "Value"])
        metrics_summary_df.to_excel(writer, index=False, header=False, sheet_name="Summary")

        # Format Summary sheet
        worksheet = writer.sheets["Summary"]
        worksheet.freeze_panes = "A2"

        # Set column widths
        worksheet.column_dimensions["A"].width = 35
        worksheet.column_dimensions["B"].width = 15

        # Apply formatting
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        section_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")

        # Style the main header (row 1)
        cell = worksheet.cell(row=1, column=1)
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True, size=14)

        # Style dimension section headers (D0, D1, D2, D4a, D4b, Overall), found by their labels
        dimension_rows = [
            row_num
            for row_num, (metric, _) in enumerate(metrics_summary_data, start=1)
            if metric.startswith(("DIMENSION", "OVERALL"))
        ]
        for row_num in dimension_rows:
            for col in (1, 2):
                cell = worksheet.cell(row=row_num, column=col)
                cell.fill = section_fill
                cell.font = Font(bold=True)

        # Color fills for F1 score highlighting
        green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        yellow_fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
        red_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

        # Sheet 3: Source Fidelity (D1) - was Content Preservation Categories
        source_fidelity_data = [
            [
                "Category",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        ]

        for category in intervention_categories:
            metrics = content_category_metrics[category]
            source_fidelity_data.append(
                [
                    category,
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    round(metrics["precision"], 2),
                    round(metrics["recall"], 2),
                    round(metrics["f1"], 2),
                ]
            )

        source_fidelity_df = pd.DataFrame(source_fidelity_data[1:], columns=source_fidelity_data[0])
        source_fidelity_df.to_excel(writer, index=False, sheet_name="Source Fidelity (D1)")

        # Format Source Fidelity (D1) sheet
        worksheet = writer.sheets["Source Fidelity (D1)"]
        worksheet.freeze_panes = "A2"
        worksheet.column_dimensions["A"].width = 25
        for col in ["B", "C", "D"]:
            worksheet.column_dimensions[col].width = 15
        for col in ["E", "F", "G"]:
            worksheet.column_dimensions[col].width = 12

        # Apply header formatting
        for col in range(1, 8):  # A to G
            cell = worksheet.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Apply color coding based on F1 scores
        for row in range(2, len(intervention_categories) + 2):
            f1_cell = worksheet.cell(row=row, column=7)  # F1 Score column
            if f1_cell.value and f1_cell.value >= 0.8:
                f1_cell.fill = green_fill
            elif f1_cell.value and f1_cell.value >= 0.5:
                f1_cell.fill = yellow_fill
            elif f1_cell.value is not None:
                f1_cell.fill = red_fill

        # Sheet 4: Classification (D4a) - was Intervention Categories
        classification_data = [
            [
                "Category",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        ]

        for category in intervention_categories:
            metrics = intervention_category_metrics[category]
            classification_data.append(
                [
                    category,
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    round(metrics["precision"], 2),
                    round(metrics["recall"], 2),
                    round(metrics["f1"], 2),
                ]
            )

        classification_df = pd.DataFrame(classification_data[1:], columns=classification_data[0])
        classification_df.to_excel(writer, index=False, sheet_name="Classification (D4a)")

        # Format Classification (D4a) sheet
        worksheet = writer.sheets["Classification (D4a)"]
        worksheet.freeze_panes = "A2"
        worksheet.column_dimensions["A"].width = 25
        for col in ["B", "C", "D"]:
            worksheet.column_dimensions[col].width = 15
        for col in ["E", "F", "G"]:
            worksheet.column_dimensions[col].width = 12

        # Apply header formatting
        for col in range(1, 8):  # A to G
            cell = worksheet.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Apply color coding based on F1 scores
        for row in range(2, len(intervention_categories) + 2):
            f1_cell = worksheet.cell(row=row, column=7)  # F1 Score column
            if f1_cell.value and f1_cell.value >= 0.8:
                f1_cell.fill = green_fill
            elif f1_cell.value and f1_cell.value >= 0.5:
                f1_cell.fill = yellow_fill
            elif f1_cell.value is not None:
                f1_cell.fill = red_fill

        # Sheet 5: Element Application (D4b) - was Element Categories (Standard)
        element_application_data = [
            [
                "Category",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        ]

        for category in intervention_categories:
            metrics = element_category_metrics[category]
            element_application_data.append(
                [
                    category,
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    round(metrics["precision"], 2),
                    round(metrics["recall"], 2),
                    round(metrics["f1"], 2),
                ]
            )

        element_application_df = pd.DataFrame(
            element_application_data[1:], columns=element_application_data[0]
        )
        element_application_df.to_excel(writer, index=False, sheet_name="Element Application (D4b)")

        # Format Element Application (D4b) sheet
        worksheet = writer.sheets["Element Application (D4b)"]
        worksheet.freeze_panes = "A2"
        worksheet.column_dimensions["A"].width = 25
        for col in ["B", "C", "D"]:
            worksheet.column_dimensions[col].width = 15
        for col in ["E", "F", "G"]:
            worksheet.column_dimensions[col].width = 12

        # Apply header formatting
        for col in range(1, 8):  # A to G
            cell = worksheet.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Apply color coding based on F1 scores
        for row in range(2, len(intervention_categories) + 2):
            f1_cell = worksheet.cell(row=row, column=7)  # F1 Score column
            if f1_cell.value and f1_cell.value >= 0.8:
                f1_cell.fill = green_fill
            elif f1_cell.value and f1_cell.value >= 0.5:
                f1_cell.fill = yellow_fill
            elif f1_cell.value is not None:
                f1_cell.fill = red_fill

        # Sheet 6: Element Categories (Relaxed)
        element_relaxed_data = [
            [
                "Category",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        ]

        for category in intervention_categories:
            metrics = element_category_metrics_relaxed[category]
            element_relaxed_data.append(
                [
                    category,
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    round(metrics["precision"], 2),
                    round(metrics["recall"], 2),
                    round(metrics["f1"], 2),
                ]
            )

        element_relaxed_df = pd.DataFrame(element_relaxed_data[1:], columns=element_relaxed_data[0])
        element_relaxed_df.to_excel(writer, index=False, sheet_name="Element Categories (Relaxed)")

        # Format Element Categories (Relaxed) sheet
        worksheet = writer.sheets["Element Categories (Relaxed)"]
        worksheet.freeze_panes = "A2"
        worksheet.column_dimensions["A"].width = 25
        for col in ["B", "C", "D"]:
            worksheet.column_dimensions[col].width = 15
        for col in ["E", "F", "G"]:
            worksheet.column_dimensions[col].width = 12

        # Apply header formatting
        for col in range(1, 8):  # A to G
            cell = worksheet.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Apply color coding based on F1 scores
        for row in range(2, len(intervention_categories) + 2):
            f1_cell = worksheet.cell(row=row, column=7)  # F1 Score column
            if f1_cell.value and f1_cell.value >= 0.8:
                f1_cell.fill = green_fill
            elif f1_cell.value and f1_cell.value >= 0.5:
                f1_cell.fill = yellow_fill
            elif f1_cell.value is not None:
                f1_cell.fill = red_fill

    # Reorder sheets to put Summary first
    from openpyxl import load_workbook

    wb = load_workbook(output_file)
    # Move Summary sheet to first position (index 0)
    if "Summary" in wb.sheetnames:
        # Get current index of Summary sheet
        current_index = wb.sheetnames.index("Summary")
        # Move it to position 0 (first)
        wb.move_sheet("Summary", offset=-current_index)
    wb.save(output_file)
    wb.close()

    # Export evaluation summary and category results to JSON (same folder as Excel:
    # results/model/processing)
    # Use same timestamp as Excel file for consistency
    results_dir = Path(output_file).parent
    excel_stem = Path(output_file).stem  # e.g., "evaluation_qwen3_14B_Q6_20260228_184931"
    json_filename = excel_stem.replace("evaluation_", "unified_evaluation_summary_") + ".json"
    json_path = str(results_dir / json_filename)
    _write_evaluation_summary_json(
        output_path=json_path,
        model_instance=model_instance,
        total_sequences=total_sequences,
        total_encoded=total_encoded,
        xml_wellformed_count=xml_wellformed_count,
        xml_wellformed_percentage=xml_wellformed_percentage,
        tei_valid_count=tei_valid_count,
        tei_valid_percentage=tei_valid_percentage,
        project_valid_count=project_valid_count,
        project_valid_percentage=project_valid_percentage,
        d1_evaluated=d1_evaluated,
        d1_passed=d1_passed,
        content_precision=content_precision,
        content_recall=content_recall,
        content_f1_score=content_f1_score,
        content_macro_f1=content_macro_f1,
        d4a_evaluated=d4a_evaluated,
        d4a_passed=d4a_passed,
        intervention_precision=intervention_precision,
        intervention_recall=intervention_recall,
        intervention_f1_score=intervention_f1_score,
        intervention_macro_f1=intervention_macro_f1,
        d4b_evaluated=d4b_evaluated,
        d4b_passed=d4b_passed,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        element_macro_f1=element_macro_f1,
        content_category_metrics=content_category_metrics,
        intervention_category_metrics=intervention_category_metrics,
        element_category_metrics=element_category_metrics,
        element_category_metrics_relaxed=element_category_metrics_relaxed,
        intervention_categories=intervention_categories,
        valid_count=valid_count,
        invalid_count=invalid_count,
        validity_rate=validity_rate,
        avg_overall_valid=avg_overall_valid,
        min_overall_valid=min_overall_valid,
        max_overall_valid=max_overall_valid,
        avg_overall_all=avg_overall_all,
        min_overall_all=min_overall_all,
        max_overall_all=max_overall_all,
        avg_core_quality=avg_core_quality,
        min_core_quality=min_core_quality,
        max_core_quality=max_core_quality,
        avg_schema_factor=avg_schema_factor,
        min_schema_factor=min_schema_factor,
        max_schema_factor=max_schema_factor,
        df=df,
    )
    print(f"Evaluation summary JSON saved to: {json_path}")

    # Generate unified evaluation heatmap (same style as visualization.py
    # create_performance_heatmap)
    heatmap_path = results_dir / "unified_evaluation_heatmap.png"
    _create_unified_evaluation_heatmap(
        intervention_category_metrics=intervention_category_metrics,
        element_category_metrics=element_category_metrics,
        element_category_metrics_relaxed=element_category_metrics_relaxed,
        intervention_categories=intervention_categories,
        save_path=str(heatmap_path),
    )
    print(f"Unified evaluation heatmap saved to: {heatmap_path}")

    sequence_heatmap_path = results_dir / "unified_evaluation_sequence_heatmap.png"
    _create_sequence_dimension_heatmap(df=df, save_path=str(sequence_heatmap_path))
    print(f"Per-sequence dimension heatmap saved to: {sequence_heatmap_path}")

    # Copy processing metadata from data/output/{model}/processing_{timestamp}/ to results folder
    if mapping_file:
        processing_dir = Path(mapping_file).resolve().parent
        _copy_processing_metadata_to_results(processing_dir, results_dir)
        print(f"Processing metadata copied to: {results_dir}")

    print("Metrics sheets added to Excel file successfully!")

    # Print summary statistics
    total_entries = len(evaluation_data)
    successful_comparisons = sum(
        1 for entry in evaluation_data if entry["LLM_File_Exists"] and entry["Gold_File_Exists"]
    )

    print("\nSummary:")
    print(f"Total entries: {total_entries}")
    print(f"Successful comparisons: {successful_comparisons}")

    # Print filtering statistics
    print("\nFiltering Statistics:")
    print(f"Total instances processed: {total_instances}")
    print(f"Instances filtered out: {filtered_instances}")
    print(f"Instances evaluated: {total_instances - filtered_instances}")
    print(
        "Filtering criteria: Only instances with Gold_element and excluding "
        "'Unknown'/'Other' intervention types"
    )

    return output_file


def discover_available_models() -> List[str]:
    """
    Find the model instances in ``data/output/`` that have a run with a mapping file.

    Returns:
        list: Model instance names, sorted
    """
    available_models = []

    # Get project root (parent of evaluation directory)
    project_root = Path(__file__).parent.parent

    # Check data/output/{model_instance}/processing_{timestamp}/
    output_base = project_root / config.LLM_OUTPUT_DIR
    if output_base.exists():
        for model_dir in output_base.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                model_instance = model_dir.name
                # Find all processing directories with mapping files
                processing_dirs = [
                    d
                    for d in model_dir.glob("processing_*")
                    if d.is_dir() and (d / config.MAPPING_FILENAME).exists()
                ]
                if processing_dirs:
                    # Only add once per model instance (not per processing directory)
                    if model_instance not in available_models:
                        available_models.append(model_instance)

    return sorted(available_models)


def get_model_selection_from_user() -> Optional[str]:
    """
    Ask on the console which model instance to evaluate.

    Returns:
        str: The chosen model instance, or None if there is none or the user cancels
    """
    available_models = discover_available_models()

    if not available_models:
        print("No model results found in data/output directory.")
        return None

    print("Available model instances with results:")
    print("=" * 50)

    for i, model_instance in enumerate(available_models, 1):
        print(f"{i:2d}. {model_instance}")

    print()

    while True:
        try:
            choice = input(f"Please select an option (1-{len(available_models)}): ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(available_models):
                selected_model_instance = available_models[choice_num - 1]
                print(f"\nSelected: {selected_model_instance}")
                return selected_model_instance
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(available_models)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``

    Returns:
        argparse.Namespace: ``model_instance`` (None if not given)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The menus read their answers from stdin, so they can be piped in, e.g.\n"
        "  printf '\\n' | python eval.py gpt-5.2-2025-12-11   (Enter = default processing run)",
    )
    parser.add_argument(
        "model_instance",
        nargs="?",
        help="model instance folder in data/output/ (e.g. qwen3-14B-Q6); if omitted, a menu lists "
        "the available ones",
    )
    return parser.parse_args(argv)


def main(model_instance: str) -> int:
    """
    Run the evaluation for one model instance.

    Loads the sampled sequences and the gold-standard types, asks on the console which processing
    run (mapping file) to use, newest first by modification time (Enter = newest), and calls
    ``create_evaluation_excel``.

    Args:
        model_instance: Required model instance name (e.g., 'qwen3-14B-Q6')

    Returns:
        int: Exit code, 0 on success, 1 if an input file or folder is missing
    """
    print("Editorial Interventions Evaluation Script")
    print("=" * 50)
    print(f"Model Instance: {model_instance}")
    print()

    # Load brackets extraction data
    brackets_data = load_brackets_extraction_data()
    if brackets_data is None:
        return 1

    # Load brackets_statistics.xlsx to get pre-computed intervention types
    project_root_for_stats = Path(__file__).parent.parent
    brackets_stats_file = project_root_for_stats / config.BRACKETS_STATISTICS_FILE
    intervention_type_lookup = {}
    if brackets_stats_file.exists():
        try:
            stats_df = pd.read_excel(brackets_stats_file, engine="openpyxl")
            # Create lookup dictionary: (Filename, Bracketed_Content) -> Intervention_Type (Gold)
            for _, row in stats_df.iterrows():
                key = (row["Filename"], row["Bracketed_Content"])
                intervention_type_lookup[key] = row.get("Intervention_Type (Gold)", "Unknown")
            print(
                "Loaded intervention types from brackets_statistics.xlsx: "
                f"{len(intervention_type_lookup)} entries"
            )
        except Exception as e:
            print(f"Warning: could not load brackets_statistics.xlsx: {e}", file=sys.stderr)
    else:
        print(
            f"Warning: brackets_statistics.xlsx not found at {brackets_stats_file}", file=sys.stderr
        )

    # Get project root (parent of evaluation directory)
    project_root = Path(__file__).parent.parent

    # Define directories - resolve relative to project root
    xml_edited_dir = project_root / config.XML_EDITED_DIR
    references_dir = project_root / config.REFERENCES_DIR
    xml_input_dir = project_root / config.XML_INPUT_DIR

    # Check if directories exist
    if not references_dir.exists():
        print(f"Error: {references_dir} directory not found.", file=sys.stderr)
        return 1

    if not xml_input_dir.exists():
        print(f"Error: {xml_input_dir} directory not found.", file=sys.stderr)
        return 1

    # Ask user which mapping file to use
    mapping_file = None
    # Get project root
    project_root = Path(__file__).parent.parent

    # Check data/output/{model_instance}/processing_*/xml_update_mapping.json
    output_base = project_root / config.LLM_OUTPUT_DIR / model_instance
    mapping_files = []
    if output_base.exists():
        for processing_dir in sorted(
            sorted(output_base.glob("processing_*")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            mapping_file_path = processing_dir / config.MAPPING_FILENAME
            if mapping_file_path.exists():
                mapping_files.append(mapping_file_path)

    if mapping_files:
        # Sort by modification time (most recent first)
        mapping_files.sort(
            key=lambda p: p.stat().st_mtime if isinstance(p, Path) else Path(p).stat().st_mtime,
            reverse=True,
        )
        print(f"\nAvailable mapping files for {model_instance}:")
        for i, file in enumerate(mapping_files, 1):
            file_path = Path(file) if not isinstance(file, Path) else file
            print(f"{i}. {file_path.name} (from {file_path.parent.name})")
        print(f"{len(mapping_files) + 1}. Use latest automatically (default)")

        while True:
            try:
                choice = input(
                    f"\nSelect mapping file (1-{len(mapping_files) + 1}, "
                    "or press Enter for default): "
                ).strip()
                if choice == "":
                    choice = len(mapping_files) + 1
                choice = int(choice)

                if 1 <= choice <= len(mapping_files):
                    selected_file = mapping_files[choice - 1]
                    mapping_file = (
                        str(selected_file) if isinstance(selected_file, Path) else selected_file
                    )
                    file_path = Path(mapping_file)
                    print(f"Using mapping file: {file_path.name} (from {file_path.parent.name})")
                    break
                elif choice == len(mapping_files) + 1:
                    print("Using latest mapping file automatically")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(mapping_files) + 1}")
            except ValueError:
                print("Please enter a valid number")

    # Create evaluation Excel file
    output_file = create_evaluation_excel(
        brackets_data,
        xml_edited_dir,
        references_dir,
        xml_input_dir,
        model_instance,
        mapping_file=mapping_file,
        intervention_type_lookup=intervention_type_lookup,
    )

    print("\nEvaluation completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Files saved to: results/{model_instance}/")
    return 0


if __name__ == "__main__":
    args = parse_args()
    configure_console_logging()
    model_instance = args.model_instance or get_model_selection_from_user()
    if not model_instance:
        print("No model selected or no models available. Exiting.")
        print(
            "Please ensure you have model results in the data/output directory or provide "
            "model instance via command line."
        )
        sys.exit(1)
    sys.exit(main(model_instance))
