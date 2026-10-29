"""
Gold-standard intervention types for the sampled bracketed sequences.

Used by ``data/sampling/sampling.py`` to build ``brackets_statistics.xlsx`` (one row per bracketed
sequence with Filename, Bracketed_Content, Language and Intervention_Type (Gold)). The type of a
sequence is derived from the TEI intervention elements in the gold-standard version of its parent
element; when a parent contains several sequences, the elements are distributed over them in
document order. A few sequences have hard-coded overrides (search for ``letter278`` etc.).
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from lxml import etree

from .xml_utils import extract_element_content_by_xpath

logger = logging.getLogger(__name__)

TEI_INTERVENTION_TAGS = {
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
}


def _parent_xpath(xpath: str) -> str:
    """
    Remove a trailing ``/text()`` or ``/text()[n]`` from an XPath.

    Args:
        xpath: XPath of a text node

    Returns:
        str: XPath of the containing element; ``xpath`` unchanged if it has no text() step
    """
    if "/text()[" in xpath:
        return re.sub(r"/text\(\)\[\d+\]$", "", xpath)
    if xpath.endswith("/text()"):
        return xpath[:-7]
    return xpath


def _remove_note_elements(xml_content: Optional[str]) -> Optional[str]:
    """
    Remove ``note`` elements (without namespace) from an XML string, keeping their tail text
    where the note was (appended to the previous sibling's tail, or to the parent's text if the
    note is the first child). Callers only use the element names of the result.

    Args:
        xml_content: XML string, or None

    Returns:
        str: The XML without notes; ``xml_content`` unchanged if it cannot be parsed (a warning is
        logged); None for None
    """
    if xml_content is None:
        return None
    try:
        tree = etree.fromstring(xml_content)
        for note in tree.xpath(".//note"):
            parent = note.getparent()
            if parent is None:
                continue
            tail_text = note.tail if note.tail else ""
            prev = note.getprevious()
            parent.remove(note)
            if tail_text:
                if prev is not None:
                    prev.tail = (prev.tail or "") + tail_text
                else:
                    parent.text = (parent.text or "") + tail_text
        return etree.tostring(tree, encoding="unicode", method="xml")
    except Exception as e:
        logger.warning(f"Could not remove note elements, keeping the XML unchanged: {e}")
        return xml_content


def _apply_hardcoded_goldstandard_hacks(
    filename: str, bracketed_content: str, gold_all_tags: list
) -> list:
    # letter278: doubled letters are corrections (sic/corr), not supplements
    """
    Replace the gold-standard element list for a few hard-coded sequences.

    Args:
        filename: Letter file name
        bracketed_content: The bracketed sequence
        gold_all_tags: Intervention element names found in the gold-standard parent element

    Returns:
        list: The override for the listed sequences, otherwise ``gold_all_tags`` unchanged
    """
    if filename == "letter278.xml":
        if bracketed_content in ("m'accom[m]ode", "Pic[c]adilly", "étouf[f]ée,"):
            return ["choice", "sic", "corr"]
    # letter299: Sand[?] is unclear, not part of the abbreviation in same parent
    if filename == "letter299.xml" and bracketed_content == "Sand[?].":
        return ["unclear"]
    # letter2378: adjacent brackets map to single choice elements
    if filename == "letter2378.xml":
        if bracketed_content in ("S[ua] E[ccellenza]", "Sig[nor] Co[nte]", "S[ua] E[ccallenza]"):
            return ["choice", "abbr", "expan"]
    # letter3185: [Lücke] is a gap marker
    if filename == "letter3185.xml" and bracketed_content == "[Lücke]":
        return ["gap"]
    # letter6534: encreased [sic] is a correction
    if filename == "letter6534.xml" and bracketed_content == "encreased [sic]":
        return ["choice", "sic", "corr"]
    return gold_all_tags


def _get_goldstandard_elements_for_position(
    brackets_data: pd.DataFrame,
    current_row_index: int,
    gold_element_diff_summary: str,
) -> str:
    """
    Pick the gold-standard elements that belong to one bracketed sequence.

    All sequences of the same file and parent element share the parent's gold-standard element list.
    With a single sequence, as many interventions are taken as the sequence has bracket pairs (a
    ``choice`` with ``abbr``+``expan`` or ``sic``+``corr`` counts as one). With several sequences,
    the interventions are distributed in document order: one per sequence first, the rest by extra
    bracket pairs, or by bracket counts if there are enough for all. The hard-coded overrides are
    checked first.

    Args:
        brackets_data: One row per bracketed sequence, with Filename, Full_XPath, ID and
            Bracketed_Content
        current_row_index: Position of the current row in ``brackets_data``
        gold_element_diff_summary: Comma-separated element names of the gold-standard parent, or
            "No changes"

    Returns:
        str: Comma-separated element names for this sequence, or "No elements"
    """
    row = brackets_data.iloc[current_row_index]
    fn = row["Filename"]
    xp = row["Full_XPath"]
    bc = row["Bracketed_Content"]

    # Check hardcoded hacks first (before checking if summary is empty)
    # letter278: doubled letters are corrections (sic/corr), not supplements
    if fn == "letter278.xml" and bc in ("m'accom[m]ode", "Pic[c]adilly", "étouf[f]ée,"):
        return "choice, sic, corr"
    # letter299: Sand[?] is unclear, not part of the abbreviation in same parent
    if fn == "letter299.xml" and bc == "Sand[?].":
        return "unclear"
    # letter2378: adjacent brackets map to single choice elements
    if fn == "letter2378.xml" and bc in (
        "S[ua] E[ccellenza]",
        "Sig[nor] Co[nte]",
        "S[ua] E[ccallenza]",
    ):
        return "choice, abbr, expan"
    # letter3185: [Lücke] is a gap marker
    if fn == "letter3185.xml" and bc == "[Lücke]":
        return "gap"
    # letter6534: encreased [sic] is a correction
    if fn == "letter6534.xml" and bc == "encreased [sic]":
        return "choice, sic, corr"
    if not gold_element_diff_summary or gold_element_diff_summary == "No changes":
        return "No elements"

    # Group by parent XPath (without text node index) so all text nodes in the
    # same parent element are processed together with shared gold elements
    parent_xp = _parent_xpath(xp)
    same_parent = brackets_data[
        (brackets_data["Filename"] == fn)
        & (brackets_data["Full_XPath"].apply(_parent_xpath) == parent_xp)
    ]
    gold_elements = [e.strip() for e in gold_element_diff_summary.split(",")]
    all_seqs = same_parent["Bracketed_Content"].tolist()

    # Position of the current row within the parent group (by index: content can be duplicated)
    same_parent_indices = same_parent.index.tolist()
    current_pos_in_parent = same_parent_indices.index(brackets_data.index[current_row_index])

    def group_single(elements: list, expected_count: int) -> str:
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

    if len(all_seqs) == 1:
        n = bc.count("[")
        return group_single(gold_elements, n)

    def group_multi(elements: list, sequences: list) -> list:
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

    groups = group_multi(gold_elements, all_seqs)
    if current_pos_in_parent < len(groups):
        return ", ".join(groups[current_pos_in_parent])
    return "No elements"


def categorize_intervention_type(goldstandard_str: Optional[str]) -> Optional[str]:
    """
    Map a list of TEI element names to intervention types.

    ``choice``+``abbr``+``expan`` -> Abbreviation, ``choice``+``sic``+``corr`` -> Correction (each
    complete pattern once), then per element: supplied -> Supplement, gap -> Omission, unclear ->
    Uncertainty, surplus -> Redundancy, and each abbr/expan/sic/corr, if there is no ``choice`` at
    all -> Abbreviation or Correction. A ``choice`` without any recognised pattern or other type
    gives "Other".

    Args:
        goldstandard_str: Comma-separated element names, or "No elements"

    Returns:
        str: Types joined with "; " (repeated types are repeated), "Unknown" for an empty input, or
        None if no type was recognised
    """
    if not goldstandard_str or goldstandard_str == "No elements":
        return "Unknown"
    elements = [e.strip().lower() for e in goldstandard_str.split(",")]
    types = []

    # Count occurrences of each element
    choice_count = elements.count("choice")
    abbr_count = elements.count("abbr")
    expan_count = elements.count("expan")
    sic_count = elements.count("sic")
    corr_count = elements.count("corr")

    # Handle choice-based patterns by counting how many complete patterns exist
    if choice_count > 0:
        # Count abbreviation patterns (choice+abbr+expan)
        abbr_pattern_count = min(choice_count, abbr_count, expan_count)
        for _ in range(abbr_pattern_count):
            types.append("Abbreviation")
            choice_count -= 1
            abbr_count -= 1
            expan_count -= 1

        # Count correction patterns (choice+sic+corr)
        corr_pattern_count = min(choice_count, sic_count, corr_count)
        for _ in range(corr_pattern_count):
            types.append("Correction")
            choice_count -= 1
            sic_count -= 1
            corr_count -= 1

    # Process individual elements (count each occurrence)
    for e in elements:
        if e == "supplied":
            types.append("Supplement")
        elif e == "gap":
            types.append("Omission")
        elif e == "unclear":
            types.append("Uncertainty")
        elif e == "surplus":
            types.append("Redundancy")
        elif e == "sic" and "choice" not in elements:
            types.append("Correction")
        elif e == "corr" and "choice" not in elements:
            types.append("Correction")
        elif e == "abbr" and "choice" not in elements:
            types.append("Abbreviation")
        elif e == "expan" and "choice" not in elements:
            types.append("Abbreviation")

    # If we have choice but no recognized pattern, and no other types, mark as Other
    if "choice" in elements and not types:
        types.append("Other")

    if types:
        return "; ".join(types)
    return None


def _gold_element_summary(gold_xml: Optional[str], filename: str, bracketed_content: str) -> str:
    """
    List the TEI intervention elements of a gold-standard element, in document order.

    Args:
        gold_xml: XML string of the gold-standard parent element, or None
        filename: Letter file name (for the hard-coded overrides)
        bracketed_content: The bracketed sequence (for the hard-coded overrides)

    Returns:
        str: Comma-separated element names, or "No changes" if there are none or the XML cannot be
        parsed
    """
    if not gold_xml:
        return "No changes"
    try:
        tree = etree.fromstring(gold_xml)
        tags = []
        for el in tree.iter():
            tag = el.tag
            if "}" in tag:
                tag = tag.split("}")[1]
            if tag in TEI_INTERVENTION_TAGS:
                tags.append(tag)
    except Exception:
        return "No changes"
    tags = _apply_hardcoded_goldstandard_hacks(filename, bracketed_content, tags)
    return ", ".join(tags) if tags else "No changes"


def create_bracket_statistics_df(
    brackets_df: pd.DataFrame,
    xml_input_dir: Path,
    ref_dir: Path,
) -> pd.DataFrame:
    """
    Build the gold-standard intervention type for every bracketed sequence.

    Each row of ``brackets_df`` is expanded to one row per ``|``-separated sequence. For each
    sequence, the gold-standard version of its parent element is read from ``ref_dir`` and its
    intervention elements are mapped to types (see ``_get_goldstandard_elements_for_position`` and
    ``categorize_intervention_type``).

    Args:
        brackets_df: Sampled text elements with Filename, XPath, ID, Bracketed_Content (sequences
            separated by ``|``) and Language
        xml_input_dir: Folder of the source letters; not used
        ref_dir: Folder of the gold-standard letters

    Returns:
        pandas.DataFrame: Filename, Bracketed_Content, Language, Intervention_Type (Gold); one row
        per sequence, in input order
    """
    rows = []
    for _, r in brackets_df.iterrows():
        seqs = [s.strip() for s in str(r["Bracketed_Content"] or "").split("|") if s.strip()]
        lang = r.get("Language", "") or ""
        for bc in seqs:
            rows.append(
                {
                    "Filename": r["Filename"],
                    "XPath": r["XPath"],
                    "ID": r["ID"],
                    "Bracketed_Content": bc,
                    "Language": lang,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["Filename", "Bracketed_Content", "Language", "Intervention_Type (Gold)"]
        )

    expanded = pd.DataFrame(rows)
    expanded["Full_XPath"] = expanded["XPath"]

    out = []
    for idx, (_, row) in enumerate(expanded.iterrows(), start=0):
        fn = row["Filename"]
        xp = row["Full_XPath"]
        bc = row["Bracketed_Content"]
        lang = row["Language"]
        parent = _parent_xpath(xp)

        gold_file = ref_dir / fn
        gold_el = (
            extract_element_content_by_xpath(gold_file, parent) if gold_file.exists() else None
        )

        gold_el = _remove_note_elements(gold_el)

        summary = _gold_element_summary(gold_el, fn, bc)
        elems = _get_goldstandard_elements_for_position(expanded, idx, summary)
        intervention_type = categorize_intervention_type(elems) or "Unknown"

        out.append(
            {
                "Filename": fn,
                "Bracketed_Content": bc,
                "Language": lang,
                "Intervention_Type (Gold)": intervention_type,
            }
        )

    return pd.DataFrame(out)
