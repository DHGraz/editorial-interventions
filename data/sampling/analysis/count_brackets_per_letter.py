"""
Count opening square brackets per letter of the full corpus, with the letter's language.

Reads every ``*.xml`` directly in ``letters-all/`` next to this script, counts ``[`` in the text
nodes of the original-language ``div`` (outside notes and placeholder paragraphs; same scope as
``extract_editorial_interventions.py``), takes the language from the letter list
``Briefliste GESAMT für online_2025.xlsx`` next to this script, and writes
``brackets_per_letter.xlsx`` (counts per letter and a summary per language).

Needs the full corpus and the letter list, which are not part of this repository.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from lxml import etree

# Same config as extract_editorial_interventions
LETTERS_BASE_DIR = Path(__file__).parent / "letters-all"
METADATA_FILENAME = "Briefliste GESAMT für online_2025.xlsx"
METADATA_SHEET_NAME = "BL chrono"
LETTER_NUM_COLUMN = "Lfde Nr NEU"
LANGUAGE_COLUMN = "Sprache"
OUTPUT_FILENAME = "brackets_per_letter.xlsx"

# XPath: all text nodes in <text>/body/div[@subtype='original'], excluding notes
# and paragraphs that start with "[In Bearbeitung" / "[Noch nicht"
XPATH_TEXT_NODES = (
    "//tei:text/tei:body/tei:div[@subtype='original']"
    "//*[not(ancestor::tei:note) and not(self::tei:note) "
    "and not(starts-with(., '[In Bearbeitung') or starts-with(., '[Noch nicht'))]"
    "/text()"
)

# Repository root on sys.path, for the root-level utils package
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.xml_utils import extract_letter_number  # noqa: E402


def load_metadata_languages(metadata_path: Path) -> Optional[Dict[int, Optional[str]]]:
    """
    Map letter numbers to languages from the letter list.

    Only the first of several languages (separated by ";") is kept.

    Args:
        metadata_path: Path of the letter list Excel file (sheet "BL chrono")

    Returns:
        dict: Letter number -> language (or None); None if the file or a column is missing
    """
    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}; Language column will be empty.")
        return None
    df = pd.read_excel(metadata_path, sheet_name=METADATA_SHEET_NAME)
    if LETTER_NUM_COLUMN not in df.columns or LANGUAGE_COLUMN not in df.columns:
        print(f"Metadata missing '{LETTER_NUM_COLUMN}' or '{LANGUAGE_COLUMN}'.")
        return None
    language_map = {}
    for _, row in df.iterrows():
        letter_num = row[LETTER_NUM_COLUMN]
        language = row[LANGUAGE_COLUMN]
        if pd.notna(letter_num):
            try:
                letter_num = int(float(letter_num))
                lang = str(language).strip() if pd.notna(language) else None
                language_map[letter_num] = lang.split(";")[0].strip() if lang else None
            except (ValueError, TypeError):
                continue
    print(f"Loaded {len(language_map)} letter-language mappings.")
    return language_map


def count_opening_brackets_in_file(xml_path: Path, nsmap: dict) -> int:
    """
    Count all '[' in text nodes under <text> (same scope as extraction script).

    Args:
        xml_path: Path of a TEI letter
        nsmap: Namespace map with the ``tei`` prefix

    Returns:
        int: Number of opening brackets
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    text_nodes = root.xpath(XPATH_TEXT_NODES, namespaces=nsmap)
    total = 0
    for node in text_nodes:
        if node is not None and isinstance(node, str):
            total += node.count("[")
        else:
            total += (node or "").count("[")
    return total


def main() -> None:
    """Count the brackets of all letters and write ``brackets_per_letter.xlsx``."""
    metadata_path = Path(__file__).parent / METADATA_FILENAME
    language_map = load_metadata_languages(metadata_path)
    nsmap = {"tei": "http://www.tei-c.org/ns/1.0"}

    rows = []
    xml_files = sorted([f for f in LETTERS_BASE_DIR.glob("*.xml") if f.is_file()])
    print(f"Processing {len(xml_files)} XML files in {LETTERS_BASE_DIR}")

    for xml_file in xml_files:
        try:
            count = count_opening_brackets_in_file(xml_file, nsmap)
            letter_num = extract_letter_number(xml_file.name)
            language = (
                language_map.get(letter_num) if language_map and letter_num is not None else None
            )
            rows.append(
                {
                    "Filename": xml_file.name,
                    "Language": language,
                    "Opening_Bracket_Count": count,
                }
            )
        except (etree.XMLSyntaxError, etree.ParseError, OSError) as e:
            print(f"  Error {xml_file}: {e}")
            continue

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("Language", dropna=False)
        .agg(
            Letters_Per_Language=("Filename", "count"),
            Brackets_Per_Language=("Opening_Bracket_Count", "sum"),
        )
        .reset_index()
    )
    total_letters = summary["Letters_Per_Language"].sum()
    total_brackets = summary["Brackets_Per_Language"].sum()
    summary["Letters_Pct"] = (
        (summary["Letters_Per_Language"] / total_letters * 100).round(1) if total_letters else 0
    )
    summary["Brackets_Pct"] = (
        (summary["Brackets_Per_Language"] / total_brackets * 100).round(1) if total_brackets else 0
    )
    out_path = Path(__file__).parent / OUTPUT_FILENAME
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Bracket counts")
        summary.to_excel(writer, index=False, sheet_name="Summary by language")
    print(f"Wrote {len(df)} rows + summary to {out_path}")


if __name__ == "__main__":
    main()
