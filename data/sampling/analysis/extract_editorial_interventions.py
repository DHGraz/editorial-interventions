"""
Extract the bracketed sequences (editorial interventions) of TEI letters into an Excel file.

Usage (see ``--help``; the full corpus in ``letters-all/`` is not part of this repository)::

    python extract_editorial_interventions.py
    python extract_editorial_interventions.py --letters-dir ../../input/bracket_sample \\
        --metadata ../../input/corresp-metadata_sample.json --output demo.xlsx

Reference script for this corpus: the XPath, the bracket-parsing rules and the metadata format
are specific to it. Its output is not the corpus analysis file
(``editorial_interventions_brackets_analysis.xlsx``) from which the sample was selected: on the
full corpus it gives 7,610 rows, that file has 7,894. The published sample is defined by that
file's ``Final_Selection`` column (see ``data/sampling/sampling.py``).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from lxml import etree
from openpyxl.styles import Alignment

# Configuration constants
LETTERS_BASE_DIR = "letters-all"
OUTPUT_FILENAME = "editorial_interventions_brackets.xlsx"
METADATA_FILENAME = "corresp-metadata.json"
EXCEL_SHEET_NAME = "Editorial Interventions"
BRACKETED_SEQUENCE_COLUMN = "Bracketed_Sequence"

# Repository root on sys.path, for the root-level utils package (four levels up from this file).
# The utils import has to come after it.
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.console_logging import configure_console_logging  # noqa: E402
from utils.xml_utils import (  # noqa: E402
    TEI_NSMAP,
    extract_letter_number,
    generate_proper_xpath_with_text_index,
    normalize_text,
)


def extract_bracketed_sequences(text: str) -> List[str]:
    """Extract all bracketed sequences according to specific rules.

    For regular brackets: extract tokens/words containing brackets, combining adjacent
    bracketed tokens (e.g., "E[uer] H[ochwürden]" becomes one sequence:
    "E[uer] H[ochwürden]"). For [sic], [?] and [...]: extract from last punctuation
    dot to each bracket sequence.

    Args:
        text: Input text string to extract bracketed sequences from.

    Returns:
        List of bracketed sequences found in the text.
    """
    if not text or "[" not in text or "]" not in text:
        return []

    # Normalize the text first
    text = normalize_text(text)

    bracketed_sequences = []

    # Check for [sic], [?] or [...] patterns - extract all occurrences
    special_patterns = [r"\[sic\]", r"\[\?\]", r"\[\.\.\.+\]"]

    for pattern in special_patterns:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            # Find the last punctuation mark before this match
            text_before = text[: match.start()]
            # Look for the last occurrence of sentence-ending punctuation
            last_punct_pos = -1
            for punct in [". ", "? ", "! ", ": ", "; "]:
                pos = text_before.rfind(punct)
                if pos > last_punct_pos:
                    last_punct_pos = pos

            if last_punct_pos >= 0:
                # Extract from after the punctuation to the end of the target
                start_pos = last_punct_pos + 2  # Skip punctuation and space
                end_pos = match.end()
                sequence = text[start_pos:end_pos].strip()
            else:
                # No punctuation found, return the whole text up to and including the target
                sequence = text[: match.end()].strip()

            if sequence and sequence not in bracketed_sequences:
                bracketed_sequences.append(sequence)

    # For regular brackets, find all tokens with brackets and group adjacent ones
    # Split by whitespace and find tokens containing both [ and ]
    tokens = text.split()

    # Find all bracketed token positions first
    bracketed_token_indices = []
    for i, token in enumerate(tokens):
        if "[" in token and "]" in token:
            # Check if this token is not already covered by special patterns
            is_special = any(
                pattern.replace("\\", "").replace("+", "") in token
                for pattern in ["[sic]", "[?]", "[...]"]
            )
            if not is_special:
                bracketed_token_indices.append(i)

    # Group adjacent bracketed tokens into sequences
    if bracketed_token_indices:
        visited = set()

        for idx in bracketed_token_indices:
            if idx in visited:
                continue

            # Start a new sequence from this token
            sequence_tokens = [tokens[idx]]
            visited.add(idx)

            # Look for adjacent bracketed tokens (both before and after)
            # Check tokens after
            next_idx = idx + 1
            while next_idx in bracketed_token_indices:
                sequence_tokens.append(tokens[next_idx])
                visited.add(next_idx)
                next_idx += 1

            # Check tokens before (only if we haven't processed them yet)
            prev_idx = idx - 1
            while prev_idx in bracketed_token_indices and prev_idx not in visited:
                sequence_tokens.insert(0, tokens[prev_idx])
                visited.add(prev_idx)
                prev_idx -= 1

            # Join the sequence tokens with spaces
            combined_sequence = " ".join(sequence_tokens)

            if combined_sequence and combined_sequence not in bracketed_sequences:
                bracketed_sequences.append(combined_sequence)

    return bracketed_sequences


def load_metadata_languages(metadata_path: Path) -> Optional[Dict[int, Optional[str]]]:
    """Load letter numbers and their corresponding languages from correspondence metadata JSON.

    Args:
        metadata_path: Path to JSON file containing correspondence metadata.

    Returns:
        Dictionary mapping letter number to language, or None if file is missing
        or parsing fails.
    """
    if not metadata_path.exists():
        print(
            f"Warning: correspondence metadata not found at {metadata_path}; "
            "Language_Metadata stays empty.",
            file=sys.stderr,
        )
        return None

    print(f"Loading correspondence metadata from {metadata_path}...")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        language_map = {}
        for entry in metadata:
            letter_num = entry.get("letter")
            language = entry.get("language", "")
            if letter_num is not None:
                language_map[letter_num] = language if language else None

        print(f"Loaded {len(language_map)} letter-language mappings from correspondence metadata")
        return language_map

    except (json.JSONDecodeError, OSError) as e:
        print(
            f"Warning: could not load correspondence metadata {metadata_path}: {e}; "
            "Language_Metadata stays empty.",
            file=sys.stderr,
        )
        return None


def extract_editorial_interventions(
    letters_base_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract all bracketed sequences of a folder of TEI letters and write them to Excel.

    Reads every ``*.xml`` directly in ``letters_base_dir`` (no subfolders), in name order, and
    takes the text nodes with ``[`` and ``]`` from the original-language ``div`` (outside notes and
    placeholder paragraphs). Rows whose bracketed sequence already occurred (anywhere in the corpus)
    are dropped, keeping the first.

    Args:
        letters_base_dir: Directory containing the TEI letter XML files. Defaults to
            LETTERS_BASE_DIR next to this script (the full corpus, not part of this repository).
        metadata_path: Path to the correspondence metadata JSON file. Defaults to
            METADATA_FILENAME next to this script.
        output_file: Path of the Excel file to write. Defaults to OUTPUT_FILENAME next to this
            script.

    Returns:
        DataFrame with columns Filename, Xpath, Content, Bracketed_Sequence, Language_Metadata;
        empty if no interventions were found.
    """

    # Define paths - processing all XML files in letters_base_dir
    script_dir = Path(__file__).parent
    letters_base_dir = Path(letters_base_dir) if letters_base_dir else script_dir / LETTERS_BASE_DIR
    output_file = output_file or str(script_dir / OUTPUT_FILENAME)
    metadata_path = (
        Path(metadata_path) if metadata_path else Path(__file__).parent / METADATA_FILENAME
    )

    language_map = load_metadata_languages(metadata_path)

    # List to store all extracted data
    extracted_data = []

    # Get all XML files in letters-all
    xml_files = list(letters_base_dir.glob("*.xml"))
    xml_files.sort()  # Sort for consistent processing order

    print(f"Found {len(xml_files)} XML files in {letters_base_dir}")

    total_files_processed = 0
    total_interventions_found = 0

    # Process each XML file
    for file_idx, xml_file in enumerate(xml_files, 1):
        if file_idx % 50 == 0:  # Progress indicator every 50 files
            print(f"    Processing file {file_idx}/{len(xml_files)}...")

        try:
            # Parse XML
            tree = etree.parse(xml_file)
            root = tree.getroot()

            # Define namespace map for TEI
            nsmap = TEI_NSMAP

            # Use the specified XPath to find text nodes with brackets
            # Note: We need to handle the namespace properly
            xpath_query = (
                "//tei:text/tei:body/tei:div[@subtype='original']"
                "//*[not(ancestor::tei:note) and not(self::tei:note) "
                "and not(starts-with(., '[In Bearbeitung') or starts-with(.,'[Noch nicht'))]"
                "/text()[contains(.,'[') and contains(., ']')]"
            )

            text_nodes = root.xpath(xpath_query, namespaces=nsmap)

            if not text_nodes:
                continue

            total_interventions_found += len(text_nodes)

            for text_node in text_nodes:
                text_content = text_node.strip()
                if text_content and "[" in text_content and "]" in text_content:
                    # Normalize text content
                    text_content = normalize_text(text_content)

                    # Extract all bracketed sequences from this text node
                    bracketed_sequences = extract_bracketed_sequences(text_content)

                    # Get the parent element
                    parent = text_node.getparent()

                    # Generate proper XPath with element names and text node index
                    full_xpath = generate_proper_xpath_with_text_index(parent, text_node, root)

                    # Language (letter metadata) from correspondence metadata
                    lang_meta = None
                    if language_map is not None:
                        letter_num = extract_letter_number(xml_file.name)
                        lang_meta = language_map.get(letter_num) if letter_num is not None else None

                    # Create one row for each bracketed sequence found in this text node
                    for bracketed_sequence in bracketed_sequences:
                        row = {
                            "Filename": xml_file.name,
                            "Xpath": full_xpath,
                            "Content": text_content,
                            "Bracketed_Sequence": bracketed_sequence,
                            "Language_Metadata": lang_meta,
                        }
                        extracted_data.append(row)

        except (etree.XMLSyntaxError, etree.ParseError) as e:
            print(f"Warning: could not parse {xml_file.name}, file skipped: {e}", file=sys.stderr)
            continue
        except OSError as e:
            print(f"Warning: could not read {xml_file.name}, file skipped: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(
                f"Warning: unexpected error in {xml_file.name}, file skipped: {e}", file=sys.stderr
            )
            continue

        total_files_processed += 1

    print(f"\nTotal files processed: {total_files_processed}")

    # Create DataFrame and remove duplicates
    if extracted_data:
        df = pd.DataFrame(extracted_data)

        print(f"\nBefore deduplication: {len(df)} editorial interventions")

        # Remove duplicates based on Bracketed_Sequence column
        df_deduplicated = df.drop_duplicates(subset=[BRACKETED_SEQUENCE_COLUMN], keep="first")

        print(f"After deduplication: {len(df_deduplicated)} unique editorial interventions")
        print(f"Removed {len(df) - len(df_deduplicated)} duplicates")

        # Export to Excel with formatting
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df_deduplicated.to_excel(writer, index=False, sheet_name=EXCEL_SHEET_NAME)

            # Get the worksheet
            worksheet = writer.sheets[EXCEL_SHEET_NAME]

            # Set column widths (adjusted for removed Subdirectory column)
            worksheet.column_dimensions["A"].width = 20  # filename
            worksheet.column_dimensions["B"].width = 30  # xpath
            worksheet.column_dimensions["C"].width = 80  # content
            worksheet.column_dimensions["D"].width = 30  # bracketed_sequence
            worksheet.column_dimensions["E"].width = 20  # language (letter metadata)

            # Apply text wrapping to all cells
            num_rows = len(df_deduplicated) + 1
            num_cols = len(df_deduplicated.columns)

            for row in range(1, num_rows + 1):
                for col in range(1, num_cols + 1):
                    cell = worksheet.cell(row=row, column=col)
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

        print(
            f"\nExtraction complete! Found {len(df_deduplicated)} unique editorial interventions."
        )
        print(f"Results saved to: {output_file}")

        # Display first few rows
        print("\nFirst few entries:")
        print(df_deduplicated.head())

        return df_deduplicated
    else:
        print("No editorial interventions found.")
        return pd.DataFrame()


def main() -> int:
    """
    Parse the command-line arguments and run the extraction.

    Returns:
        int: Exit code, 0 on success, 1 if nothing was extracted
    """
    parser = argparse.ArgumentParser(
        description="Extract editorial interventions (bracketed sequences) from TEI XML files "
        "and export them to an Excel file for analysis. The XPath, bracket-parsing "
        "rules, and metadata schema are specific to the Hammer-Purgstall TEI corpus "
        "and would need adapting for a different corpus or bracket convention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
        "  python extract_editorial_interventions.py\n"
        "  python extract_editorial_interventions.py --version\n\n"
        "  # Demo run against the published 90-letter sample (not the full corpus,\n"
        "  # which is not part of this repository):\n"
        "  python extract_editorial_interventions.py \\\n"
        "      --letters-dir ../../input/bracket_sample \\\n"
        "      --metadata ../../input/corresp-metadata_sample.json \\\n"
        "      --output editorial_interventions_brackets_demo.xlsx",
    )
    parser.add_argument(
        "--letters-dir",
        type=Path,
        default=None,
        help=f"Directory containing TEI letter XML files (default: {LETTERS_BASE_DIR}, "
        "the full corpus, not part of this repository)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=f"Path to correspondence metadata JSON (default: {METADATA_FILENAME} next to this "
        "script)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Path of the Excel file to write (default: {OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--version", action="version", version="extract_editorial_interventions 1.0.0"
    )

    args = parser.parse_args()
    configure_console_logging()

    letters_base_dir = args.letters_dir or Path(__file__).parent / LETTERS_BASE_DIR

    print("Starting Editorial Interventions Extraction")
    print("=" * 50)
    print("This script extracts editorial interventions from XML files using the specified XPath.")
    print(f"Processing all subdirectories in {letters_base_dir}")
    print()

    # Extract editorial interventions
    df = extract_editorial_interventions(
        letters_base_dir=args.letters_dir,
        metadata_path=args.metadata,
        output_file=args.output,
    )

    if not df.empty:
        print(f"\nSuccessfully extracted {len(df)} unique editorial interventions.")
        print(f"Output file: {args.output or Path(__file__).parent / OUTPUT_FILENAME}")
    else:
        print("Error: no editorial interventions found or extraction failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
