"""
Shared XML helpers for the evaluation and sampling scripts.

Contains:

- XPath-based extraction from TEI letter files (element XML, full text with notes removed)
- normalisation of bracketed source text into the text an encoding is expected to produce
  (``normalize_text_for_editorial_interventions``), used for the source-fidelity check (D1)
- a structural comparison of two XML fragments (``compare_xml_elements``)
- access to the LLM output mapping (``xml_update_mapping.json``)
- ``JingValidator``, a wrapper around the bundled Jing RelaxNG validator
- small helpers for XPath construction and text normalisation used by the sampling scripts

XPaths stored in the data files have no namespace prefixes; the extraction functions add the
``tei:`` prefix for a fixed list of element names before querying.
"""

import inspect
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from lxml import etree

logger = logging.getLogger(__name__)

TEI_NS = "http://www.tei-c.org/ns/1.0"
# Namespace map for XPath queries with the "tei:" prefix
TEI_NSMAP = {"tei": TEI_NS}
# Namespace part of an lxml tag name in Clark notation, "{namespace}local"
TEI_TAG_PREFIX = "{" + TEI_NS + "}"
# Placeholder for a line break that survives whitespace normalisation
PARAGRAPH_BREAK_MARKER = "<<<PARAGRAPH_BREAK>>>"

# Bundled Jing RelaxNG validator and the time limits for running it
DEFAULT_JING_JAR = Path(__file__).parent.parent / "resources" / "tools" / "jing-RELEASE220.jar"
JING_TIMEOUT_SECONDS = 30
JAVA_CHECK_TIMEOUT_SECONDS = 5


def extract_element_content_by_xpath(xml_file_path: Union[str, Path], xpath: str) -> Optional[str]:
    """
    Extract the first element matching an XPath, as an XML string.

    The ``tei:`` prefix is added to a fixed list of TEI element names in ``xpath`` before the
    query; namespace declarations are removed from the returned string. Any exception (unreadable
    file, invalid XPath) is logged and treated as "not found".

    Args:
        xml_file_path: Path to the XML file
        xpath: XPath without namespace prefixes (an element XPath, not a text node XPath)

    Returns:
        str: The element including its tags (without tail text), or None if nothing matches or an
        error occurred
    """
    try:
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        # Define namespace mapping for TEI
        nsmap = TEI_NSMAP

        # Convert the XPath to work with namespaces
        # The XPath from Excel is already clean (no namespace prefixes)
        # We need to add namespace prefixes for the actual XML query

        # Add namespace prefix to each element name
        element_xpath_with_ns = xpath
        for element_name in [
            "TEI",
            "text",
            "body",
            "div",
            "p",
            "opener",
            "dateline",
            "closer",
            "salute",
            "signed",
            "postscript",
            "address",
            "addrLine",
            "persName",
            "hi",
            "supplied",
            "gap",
            "choice",
            "unclear",
            "sic",
            "corr",
            "abbr",
            "expan",
            "note",
        ]:
            # Replace element names with namespace prefix
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}[", f"/tei:{element_name}["
            )
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}/", f"/tei:{element_name}/"
            )
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}", f"/tei:{element_name}"
            )

        # Execute XPath query with namespace
        elements = root.xpath(element_xpath_with_ns, namespaces=nsmap)

        if elements:
            # Get the entire element content including the element tags
            element = elements[0]

            # Get the XML content as string, preserving structure and markup
            xml_content = etree.tostring(element, encoding="unicode", method="xml", with_tail=False)

            # Clean namespace declarations for better readability
            xml_content = xml_content.replace(f' xmlns="{TEI_NS}"', "")
            xml_content = xml_content.replace(f' xmlns:tei="{TEI_NS}"', "")

            return xml_content
        else:
            return None
    except Exception as e:
        logger.error(f"Could not extract element from {xml_file_path} with XPath {xpath}: {e}")
        return None


def extract_all_text_content(
    xml_file_path: Union[str, Path],
    xpath: str,
    remove_notes: bool = True,
    add_newlines_after: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Extract all text of an element and its descendants, optionally without notes.

    ``/text()`` or ``/text()[n]`` is removed from ``xpath`` to select the containing element. The
    element is copied before notes are removed; the tail text of a removed note is kept (appended
    to the previous sibling's tail, or to the parent's text).

    Args:
        xml_file_path: Path to the XML file
        xpath: XPath without namespace prefixes
        remove_notes: If True, ``note`` elements and their content are left out
        add_newlines_after: Element names after which the marker ``<<<PARAGRAPH_BREAK>>>`` is
            inserted (it replaces leading whitespace of the element's tail), e.g.
            ``{"salute", "signed", "dateline"}``. The marker survives later whitespace
            normalisation; callers turn it into a line break

    Returns:
        str: The concatenated text, or None if nothing matches or an error occurred (the error
        is logged)
    """
    try:
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        # Define namespace mapping for TEI
        nsmap = TEI_NSMAP

        # Remove /text() and /text()[n] from XPath to get the element

        element_xpath = re.sub(r"/text\(\)(\[\d+\])?", "", xpath)

        # Convert the XPath to work with namespaces
        element_xpath_with_ns = element_xpath
        for element_name in [
            "TEI",
            "text",
            "body",
            "div",
            "p",
            "opener",
            "dateline",
            "closer",
            "salute",
            "signed",
            "postscript",
            "address",
            "addrLine",
            "persName",
            "hi",
            "supplied",
            "gap",
            "choice",
            "unclear",
            "sic",
            "corr",
        ]:
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}[", f"/tei:{element_name}["
            )
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}/", f"/tei:{element_name}/"
            )
            element_xpath_with_ns = element_xpath_with_ns.replace(
                f"/{element_name}", f"/tei:{element_name}"
            )

        # Execute XPath query with namespace
        elements = root.xpath(element_xpath_with_ns, namespaces=nsmap)

        if elements:
            element = elements[0]

            # Create a deep copy of the element to avoid modifying the original
            element_copy = etree.fromstring(
                etree.tostring(element, encoding="unicode", method="xml")
            )

            # Remove note elements if requested
            if remove_notes:
                # Find all note elements (handle namespaced elements by checking local name)
                notes_to_remove = []
                for elem in element_copy.iter():
                    # Get tag name without namespace
                    tag = elem.tag
                    if "}" in tag:
                        tag = tag.split("}")[1]
                    if tag == "note":
                        notes_to_remove.append(elem)

                for note in notes_to_remove:
                    parent = note.getparent()
                    if parent is not None:
                        # Get the tail text (text that comes after the note element)
                        tail_text = note.tail if note.tail else ""
                        # Store the previous sibling BEFORE removing the note
                        prev_sibling = note.getprevious()
                        # Remove the note element
                        parent.remove(note)
                        # Preserve tail text by adding it to the previous sibling or parent
                        if tail_text:
                            if prev_sibling is not None:
                                # Add the tail text to the previous sibling's tail
                                if prev_sibling.tail:
                                    prev_sibling.tail += tail_text
                                else:
                                    prev_sibling.tail = tail_text
                            else:
                                # No previous sibling, add to parent's text
                                if parent.text:
                                    parent.text += tail_text
                                else:
                                    parent.text = tail_text

            # Add newline markers after specified elements (use marker to survive normalization)
            if add_newlines_after:
                for elem in element_copy.iter():
                    # Get tag name without namespace
                    tag = elem.tag
                    if "}" in tag:
                        tag = tag.split("}")[1]

                    if tag in add_newlines_after:
                        # Add marker to the element's tail
                        if elem.tail:
                            elem.tail = PARAGRAPH_BREAK_MARKER + elem.tail.lstrip()
                        else:
                            elem.tail = PARAGRAPH_BREAK_MARKER

            # Get all text content recursively (including child elements)
            return etree.tostring(element_copy, encoding="unicode", method="text")
        else:
            return None
    except Exception as e:
        logger.error(f"Could not extract text from {xml_file_path} with XPath {xpath}: {e}")
        return None


def normalize_text_for_editorial_interventions(
    text: Optional[str], bracketed_content: Optional[str], element_diff_summary: Optional[str]
) -> Optional[str]:
    """
    Turn bracketed source text into the text a correct encoding is expected to contain.

    The tags in ``element_diff_summary`` are grouped into intervention units
    (``choice``+``abbr``+``expan``, ``choice``+``sic``+``corr``, then single elements) and the
    groups are applied to the ``|``-separated sequences in ``bracketed_content`` in order. Per
    type: ``supplied``/``corr``/``expan`` keep the bracket content, ``gap`` drops it, ``abbr``
    keeps the abbreviated form, ``choice_abbr_expan`` produces abbreviated form followed by the
    expanded form, ``[sic]`` markers are removed. Sequences with several bracket pairs are handled
    by special cases (see the inline comments). ``unclear`` keeps the bracket content without
    question marks.

    Args:
        text: Source text containing the bracketed sequences
        bracketed_content: One or more bracketed sequences, separated by ``|``
        element_diff_summary: Comma-separated tags added by the encoding, e.g.
            ``"choice, abbr, expan"``

    Returns:
        str: The normalised text; ``text`` unchanged if any argument is empty
    """
    if not text or not bracketed_content or not element_diff_summary:
        return text

    normalized_text = text

    # Parse the element_diff_summary to get all tags
    all_tags = [tag.strip() for tag in element_diff_summary.split(",")]

    # Group elements into logical editorial intervention units
    def group_elements_by_editorial_interventions(elements: List[str]) -> List[str]:
        """Group elements into logical editorial intervention units"""
        grouped_interventions = []
        remaining_elements = elements.copy()

        # Group choice+abbr+expan combinations (priority handling)
        while (
            "choice" in remaining_elements
            and "abbr" in remaining_elements
            and "expan" in remaining_elements
        ):
            grouped_interventions.append("choice_abbr_expan")
            remaining_elements.remove("choice")
            remaining_elements.remove("abbr")
            remaining_elements.remove("expan")

        # Group choice+sic+corr combinations (priority handling)
        while (
            "choice" in remaining_elements
            and "sic" in remaining_elements
            and "corr" in remaining_elements
        ):
            grouped_interventions.append("choice_sic_corr")
            remaining_elements.remove("choice")
            remaining_elements.remove("sic")
            remaining_elements.remove("corr")

        # Handle individual elements (fallback handling)
        # For multiple individual elements, preserve original order from element_diff_summary
        if len(remaining_elements) > 1:
            # Preserve original order for multiple individual intervention types
            for elem in elements:
                if elem in remaining_elements:
                    if elem == "abbr":
                        grouped_interventions.append("abbr_only")
                    elif elem == "expan":
                        grouped_interventions.append("expan_only")
                    elif elem == "sic":
                        grouped_interventions.append("sic_only")
                    elif elem == "corr":
                        grouped_interventions.append("corr_only")
                    else:
                        grouped_interventions.append(elem)
                    remaining_elements.remove(elem)
        else:
            # Process in order of preference for single elements
            intervention_order = [
                "supplied",
                "gap",
                "unclear",
                "abbr",
                "expan",
                "sic",
                "corr",
                "choice",
            ]

            for intervention in intervention_order:
                while intervention in remaining_elements:
                    if intervention == "abbr":
                        grouped_interventions.append("abbr_only")
                    elif intervention == "expan":
                        grouped_interventions.append("expan_only")
                    elif intervention == "sic":
                        grouped_interventions.append("sic_only")
                    elif intervention == "corr":
                        grouped_interventions.append("corr_only")
                    else:
                        grouped_interventions.append(intervention)
                    remaining_elements.remove(intervention)

        # Any other remaining elements each get their own group
        for elem in remaining_elements:
            grouped_interventions.append(elem)

        return grouped_interventions

    # Group the elements
    intervention_groups = group_elements_by_editorial_interventions(all_tags)

    # Split bracketed content by '|' to get individual sequences
    bracketed_sequences = [seq.strip() for seq in bracketed_content.split("|")]

    # Apply normalization for each sequence in order
    for i, sequence in enumerate(bracketed_sequences):
        if i < len(intervention_groups):
            intervention_type = intervention_groups[i]

            # Count bracket pairs in this sequence
            bracket_pairs_in_sequence = sequence.count("[")

            # Special handling for multiple bracket pairs
            if bracket_pairs_in_sequence > 1:
                # Case 1: Single intervention type with multiple bracket pairs
                # (e.g., choice_abbr_expan for "H[er]r[n]")
                if len(intervention_groups) == 1 and intervention_type == "choice_abbr_expan":
                    # Several bracket pairs in one abbreviation, e.g. "H[er]r[n]" or
                    # "O[ber]K[onsistorial]R[at]"

                    # Find all bracket patterns
                    bracket_pattern = r"\[([^\]]*)\]"
                    matches = list(re.finditer(bracket_pattern, sequence))

                    if len(matches) >= 2:  # We have at least 2 bracket pairs
                        # abbr_form: text before the first bracket, between the brackets and after
                        # the last bracket
                        abbr_form = sequence[: matches[0].start()]  # Text before first bracket

                        # Add text between bracket pairs
                        for i in range(len(matches) - 1):
                            abbr_form += sequence[matches[i].end() : matches[i + 1].start()]

                        # Add text after last bracket (excluding certain characters)
                        after_last = sequence[matches[-1].end() :]
                        abbr_form += after_last if after_last not in [")", ","] else ""

                        # Build expan_form: remove all brackets (show full expanded form)
                        expan_form = re.sub(bracket_pattern, r"\1", sequence)

                        normalized_text = normalized_text.replace(sequence, abbr_form + expan_form)
                        continue

                # Case 1.5: Mixed intervention types with choice_abbr_expan
                # (e.g., choice_abbr_expan + supplied for "Septemb[er] [17]98")
                # Skip if ALL groups are choice_abbr_expan - let Case 2 handle that
                elif (
                    len(intervention_groups) > 1
                    and i == 0
                    and len(bracketed_sequences) == 1
                    and "choice_abbr_expan" in intervention_groups
                    and not all(g == "choice_abbr_expan" for g in intervention_groups)
                ):
                    # choice_abbr_expan combined with other intervention types

                    # Find all bracket patterns
                    bracket_pattern = r"\[([^\]]*)\]"
                    matches = list(re.finditer(bracket_pattern, sequence))

                    # For choice_abbr_expan + other types, we need special handling
                    # The choice_abbr_expan should handle all its bracket pairs as a unit
                    # and other intervention types handle their individual bracket pairs

                    # Find which intervention types are choice_abbr_expan and which are others
                    choice_abbr_expan_indices = [
                        idx
                        for idx, intervention in enumerate(intervention_groups)
                        if intervention == "choice_abbr_expan"
                    ]
                    other_intervention_indices = [
                        idx
                        for idx, intervention in enumerate(intervention_groups)
                        if intervention != "choice_abbr_expan"
                    ]

                    # choice_abbr_expan takes the bracket at its group index, the other
                    # interventions take the brackets at theirs
                    current_sequence = sequence

                    # Process choice_abbr_expan first (if it exists)
                    if choice_abbr_expan_indices:
                        # For choice_abbr_expan, we need to show both abbreviated and expanded forms
                        choice_idx = choice_abbr_expan_indices[0]
                        if choice_idx < len(matches):
                            # Get the bracket match
                            match = matches[choice_idx]
                            before_bracket = current_sequence[: match.start()]
                            bracket_content = match.group(1)

                            # Create both forms: abbr_form + expan_form
                            # abbr_form is the text before the bracket (already in sequence)
                            # expan_form is the text before + content inside the bracket
                            # Only the expanded form is inserted; the abbreviated form is
                            # already there
                            expan_form = before_bracket + bracket_content
                            replacement = expan_form

                            # Replace the bracket with both forms
                            current_sequence = (
                                current_sequence[: match.start()]
                                + replacement
                                + current_sequence[match.end() :]
                            )

                    # Process other intervention types
                    for other_idx in other_intervention_indices:
                        if other_idx < len(matches):
                            current_sequence = apply_single_intervention_normalization(
                                current_sequence, intervention_groups[other_idx], other_idx
                            )

                    normalized_text = normalized_text.replace(sequence, current_sequence)
                    continue

                # Case 2: Multiple intervention types with multiple bracket pairs
                # (e.g., gap + supplied for "fortunatissm[xxx] ia[m]")
                elif len(intervention_groups) > 1 and i == 0 and len(bracketed_sequences) == 1:
                    # One sequence with several bracket pairs and several intervention types
                    # Process each bracket pair with its corresponding intervention type

                    # Find all bracket patterns
                    bracket_pattern = r"\[([^\]]*)\]"
                    matches = list(re.finditer(bracket_pattern, sequence))

                    # Process from last to first to avoid index shifting
                    # Map intervention types to bracket positions correctly
                    current_sequence = sequence
                    for j in reversed(range(min(len(intervention_groups), len(matches)))):
                        # Use the intervention type that corresponds to this bracket position
                        # intervention_groups is already in bracket order
                        single_intervention = intervention_groups[j]
                        current_sequence = apply_single_intervention_normalization(
                            current_sequence, single_intervention, j
                        )
                    normalized_text = normalized_text.replace(sequence, current_sequence)
                    continue

            # Apply specific normalization based on intervention type
            if intervention_type == "supplied":
                # Remove brackets, keep content
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    content_inside = sequence.split("[")[1].split("]")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + content_inside + after_brackets
                    )

            elif intervention_type == "gap":
                # Remove brackets and content completely
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + after_brackets
                    )

            elif intervention_type == "choice_abbr_expan":
                # Handle choice+abbr+expan: show both abbreviated form and expanded form
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    content_inside = sequence.split("[")[1].split("]")[0]
                    after_brackets = sequence.split("]")[1]
                    # Check if content_inside already starts with before_brackets (case-insensitive)
                    # e.g., "v[von]" or "E[Exzellenz]" - don't duplicate the prefix
                    if before_brackets and content_inside.lower().startswith(
                        before_brackets.lower()
                    ):
                        # Content already includes the prefix, use it directly
                        abbr_form = before_brackets + (
                            after_brackets if after_brackets not in [")", ","] else ""
                        )
                        expan_form = content_inside + after_brackets
                    else:
                        # Both forms: abbr_form (before_brackets + after_brackets) followed by
                        # expan_form (before_brackets + content_inside + after_brackets)
                        abbr_form = before_brackets + (
                            after_brackets if after_brackets not in [")", ","] else ""
                        )
                        expan_form = before_brackets + content_inside + after_brackets
                    normalized_text = normalized_text.replace(sequence, abbr_form + expan_form)

            elif intervention_type == "choice_sic_corr":
                # Handle choice+sic+corr: show original text with [sic] marker removed
                if sequence == "[sic]":
                    # Simply remove [sic] marker
                    normalized_text = normalized_text.replace(sequence, "")
                elif sequence.endswith(" [sic]"):
                    # For pattern "text [sic]", remove the " [sic]" part
                    normalized_text = normalized_text.replace(
                        sequence, sequence.replace(" [sic]", "")
                    )
                elif sequence.endswith("[sic]"):
                    # For pattern "text[sic]", remove the "[sic]" part
                    normalized_text = normalized_text.replace(
                        sequence, sequence.replace("[sic]", "")
                    )
                elif "[" in sequence and "]" in sequence:
                    # For other sic/corr patterns, show corrected form
                    before_brackets = sequence.split("[")[0]
                    content_inside = sequence.split("[")[1].split("]")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + content_inside + after_brackets
                    )

            elif intervention_type == "abbr_only":
                # Only abbr: keep the abbreviated form (text before brackets)
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + after_brackets
                    )

            elif intervention_type == "expan_only":
                # Only expan: show expanded form
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    content_inside = sequence.split("[")[1].split("]")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + content_inside + after_brackets
                    )

            elif intervention_type == "sic_only":
                # Only sic: remove [sic] marker, keep original text
                if sequence == "[sic]":
                    normalized_text = normalized_text.replace(sequence, "")
                elif "[" in sequence and "]" in sequence:
                    # For text marked as sic, keep the original form
                    before_brackets = sequence.split("[")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + after_brackets
                    )

            elif intervention_type == "corr_only":
                # Only corr: show corrected form
                if "[" in sequence and "]" in sequence:
                    before_brackets = sequence.split("[")[0]
                    content_inside = sequence.split("[")[1].split("]")[0]
                    after_brackets = sequence.split("]")[1]
                    normalized_text = normalized_text.replace(
                        sequence, before_brackets + content_inside + after_brackets
                    )

            elif intervention_type == "unclear":
                # Handle unclear patterns - remove ALL question marks and brackets, keep content
                if "[" in sequence and "]" in sequence:
                    # Remove all brackets and question marks, keep only the content
                    # Handles patterns like:
                    # [?] -> ""
                    # text[?] -> "text"
                    # [?content] -> "content"
                    # [content?] -> "content"
                    # [?content?] -> "content"
                    # jü[?disch?] -> "jüdisch"

                    # Replace any [content with ?] pattern - remove brackets and all question marks
                    pattern = r"\[([^\]]*)\]"

                    def replace_unclear(match: re.Match) -> str:
                        """Content of the matched bracket pair, without question marks."""
                        content = match.group(1)  # Content inside brackets
                        # Remove all question marks from the content
                        cleaned_content = content.replace("?", "")
                        return cleaned_content

                    normalized_text = normalized_text.replace(
                        sequence, re.sub(pattern, replace_unclear, sequence)
                    )

            # Handle any remaining choice elements (fallback)
            elif intervention_type == "choice":
                # Generic choice handling - keep content as is
                pass

    return normalized_text


def apply_single_intervention_normalization(
    sequence: str, intervention_type: str, bracket_index: int
) -> str:
    """
    Normalise one bracket pair of a sequence according to its intervention type.

    ``supplied`` and unknown types keep the content, ``unclear`` keeps it without question marks,
    ``gap``, ``choice_sic_corr`` and ``sic_only`` drop it, ``choice_abbr_expan`` replaces it with
    the expanded word (prefix up to the previous space or bracket, plus content, unless the
    content already starts with the prefix).

    Args:
        sequence: The sequence containing one or more bracket pairs
        intervention_type: Intervention group name, as produced by
            ``normalize_text_for_editorial_interventions``
        bracket_index: Which bracket pair to process (0-based)

    Returns:
        str: The sequence with that bracket pair replaced; unchanged if the index is out of range
    """

    # Find all bracket patterns in the sequence
    bracket_pattern = r"\[([^\]]*)\]"
    matches = list(re.finditer(bracket_pattern, sequence))

    if bracket_index >= len(matches):
        return sequence

    # Get the specific bracket to process
    match = matches[bracket_index]
    bracket_content = match.group(1)  # Content inside brackets

    # Apply normalization based on intervention type
    if intervention_type == "supplied":
        # Remove brackets, keep content
        replacement = bracket_content
    elif intervention_type == "unclear":
        # Remove ALL question marks, keep other content
        replacement = bracket_content.replace("?", "")
    elif intervention_type == "gap":
        # Remove brackets and content completely
        replacement = ""
    elif intervention_type == "choice_abbr_expan":
        # For choice_abbr_expan: replace the bracket with the expanded word (prefix + content);
        # the abbreviated form stays in the text before it
        # Find the prefix (text before this bracket, bounded by space/newline or previous bracket)
        prefix_start = match.start()
        for pos in range(match.start() - 1, -1, -1):
            if sequence[pos] in " \t\n":
                prefix_start = pos + 1
                break
            # Also stop at a previous bracket's end
            if sequence[pos] == "]":
                prefix_start = pos + 1
                break
            if pos == 0:
                prefix_start = 0
        prefix = sequence[prefix_start : match.start()]
        # Check if bracket_content already starts with the prefix (case-insensitive)
        # e.g., "v[von]" or "E[Exzellenz]" - don't duplicate the prefix
        if prefix and bracket_content.lower().startswith(prefix.lower()):
            # Content already includes the prefix, use it directly
            replacement = bracket_content
        else:
            # Replacement is the expanded form (prefix + bracket_content)
            replacement = prefix + bracket_content
    elif intervention_type in ("choice_sic_corr", "sic_only"):
        # For corrections: remove the [sic] marker entirely
        # The bracket content is typically "sic" so we remove it completely
        replacement = ""
    else:
        # Default: keep content, remove brackets
        replacement = bracket_content

    # Replace only this specific bracket occurrence
    result = sequence[: match.start()] + replacement + sequence[match.end() :]
    return result


def sort_elements_by_text_position(
    added_elements: List[Dict[str, str]], output_xml: str
) -> List[Dict[str, str]]:
    """
    Sort added elements by where their tag first occurs in the output XML.

    The sort key of an element is the position of the first ``<tagname`` in the serialised
    output, so several elements with the same tag share one position and keep their relative
    order (stable sort). Elements whose content cannot be parsed go to the end.

    Args:
        added_elements: Added-element dicts with a ``content`` field (XML string)
        output_xml: The output XML string

    Returns:
        list: The sorted elements; the original list if the output cannot be parsed or sorting
        fails (a warning is logged)
    """

    if not added_elements or len(added_elements) <= 1:
        return added_elements

    try:
        # Parse the output XML to get text positions
        def parse_xml_for_position_sorting(xml_string: str) -> Optional[etree._Element]:
            """Parse ``xml_string``, else wrapped in ``<root>`` (first child); None on failure."""
            try:
                return etree.fromstring(xml_string)
            except etree.XMLSyntaxError:
                try:
                    wrapped_xml = f"<root>{xml_string}</root>"
                    root = etree.fromstring(wrapped_xml)
                    return root[0] if len(root) > 0 else root
                except etree.XMLSyntaxError:
                    return None

        output_tree = parse_xml_for_position_sorting(output_xml)
        if output_tree is None:
            return added_elements

        # Create a mapping of element content to text position
        element_positions = {}

        # Get the full text representation and find positions based on text order
        full_text = etree.tostring(output_tree, encoding="unicode", method="xml")

        # For each added element, find its position in the full text
        for element_dict in added_elements:
            content = element_dict.get("content", "")
            # Find the position of this element in the full XML text
            # We'll use the position of the opening tag as the sort key
            if content:
                # Extract just the opening tag for position finding
                try:
                    elem_tree = etree.fromstring(content)
                    tag_name = elem_tree.tag
                    if "}" in tag_name:
                        tag_name = tag_name.split("}")[1]

                    # Find position of this tag in the full text
                    position = full_text.find(f"<{tag_name}")
                    if position == -1:
                        # If not found, try without namespace
                        position = full_text.find(f"<{tag_name}>")
                    element_positions[content] = position if position != -1 else float("inf")

                except etree.XMLSyntaxError:
                    element_positions[content] = float("inf")

        # Sort added_elements by their position in the document
        def get_position(element_dict: Dict[str, str]) -> float:
            """Sort key: position of the element's tag in the output; infinity if unknown."""
            content = element_dict.get("content", "")
            return element_positions.get(content, float("inf"))  # Unknown elements go to end

        sorted_elements = sorted(added_elements, key=get_position)

        return sorted_elements

    except Exception as e:
        # If sorting fails, return original order
        logger.warning(f"Could not sort added elements by position, keeping their order: {e}")
        return added_elements


def compare_xml_elements(
    input_xml: Optional[str], output_xml: Optional[str], filename: str, xpath: str
) -> Dict[str, Any]:
    """
    Compare the direct child elements of two XML fragments by tag name.

    Children are grouped by tag (attributes and whitespace are ignored); per tag, surplus
    children in the output are reported as added, surplus children in the input as removed.
    Deeper descendants are not compared, and ``modified_elements`` is always empty.
    ``text_changed`` compares only the stripped leading text of the two root elements. Added
    elements are sorted with ``sort_elements_by_text_position``.

    Args:
        input_xml: XML string of the input element
        output_xml: XML string of the output element
        filename: File name, copied into the report
        xpath: XPath, copied into the report

    Returns:
        dict: Report with ``filename``, ``xpath``, ``added_elements``, ``removed_elements``
        (lists of ``{"tag", "content"}``), ``modified_elements``, ``text_changed``,
        ``input_text``, ``output_text``. If an input is None or parsing fails, the report has an
        ``error`` field and empty lists instead of the texts
    """
    # Handle None values
    if input_xml is None or output_xml is None:
        return {
            "filename": filename,
            "xpath": xpath,
            "error": "Input or output XML is None",
            "added_elements": [],
            "removed_elements": [],
            "modified_elements": [],
            "text_changed": False,
        }

    try:
        # Parse XML strings - handle both complete documents and element fragments
        def parse_xml_string(xml_string: Optional[str]) -> etree._Element:
            """Parse XML string, handling both complete documents and element fragments"""
            try:
                # Try parsing as complete document first
                return etree.fromstring(xml_string)
            except etree.XMLSyntaxError:
                # If that fails, try wrapping in a root element
                try:
                    wrapped_xml = f"<root>{xml_string}</root>"
                    root = etree.fromstring(wrapped_xml)
                    # Return the first child (our actual element)
                    return root[0] if len(root) > 0 else root
                except etree.XMLSyntaxError as e:
                    logger.error(f"Could not parse XML fragment {xml_string!r}: {e}")
                    raise

        input_tree = parse_xml_string(input_xml)
        output_tree = parse_xml_string(output_xml)

        # Get all child nodes (both elements and text)
        input_children = list(input_tree)
        output_children = list(output_tree)

        # Get text content (normalized)
        input_text = input_tree.text.strip() if input_tree.text else ""
        output_text = output_tree.text.strip() if output_tree.text else ""

        # Helper function to normalize element for comparison
        def normalize_element_for_comparison(
            elem: Optional[etree._Element],
        ) -> Optional[etree._Element]:
            """Normalize element by removing whitespace and ignoring attributes"""
            # Handle None elements
            if elem is None:
                return None

            # Create a copy to avoid modifying original
            try:
                elem_copy = etree.fromstring(etree.tostring(elem, encoding="unicode", method="xml"))
            except etree.XMLSyntaxError:
                # If parsing fails, try wrapping in a root element
                xml_string = etree.tostring(elem, encoding="unicode", method="xml")
                if xml_string is None:
                    return None
                wrapped_xml = f"<root>{xml_string}</root>"
                root = etree.fromstring(wrapped_xml)
                elem_copy = root[0] if len(root) > 0 else root

            # Remove all whitespace-only text nodes
            # Note: We can't directly remove text nodes, so we'll set them to empty
            for node in elem_copy.xpath(".//text()"):
                if node.strip() == "":
                    # Set the text node to empty string instead of trying to remove it
                    parent = node.getparent()
                    if parent is not None:
                        # Find the text node in the parent's children and set it to empty
                        if parent.text == node:
                            parent.text = ""
                        # Also check tail text
                        for child in parent:
                            if child.tail == node:
                                child.tail = ""

            # Remove all attributes to ignore them in comparison
            elem_copy.attrib.clear()
            # Also remove attributes from all child elements
            for child in elem_copy.xpath(".//*"):
                child.attrib.clear()

            return elem_copy

        # Helper function to get element signature for comparison
        def get_element_signature(elem: Optional[etree._Element]) -> str:
            """Get a normalized signature of an element for comparison (ignoring attributes)"""
            normalized = normalize_element_for_comparison(elem)
            if normalized is None:
                return "None"
            # Get tag only (attributes are ignored)
            signature = f"{normalized.tag}"
            return signature

        # Compare child elements structurally
        added_elements = []
        removed_elements = []
        modified_elements = []

        # Create maps of element signatures to elements
        input_signatures = {}
        output_signatures = {}

        # Process input children - only count actual elements (not text nodes)
        for child in input_children:
            if hasattr(child, "tag"):  # This is an element, not a text node
                sig = get_element_signature(child)
                if sig not in input_signatures:
                    input_signatures[sig] = []
                input_signatures[sig].append(child)

        # Process output children - only count actual elements (not text nodes)
        for child in output_children:
            if hasattr(child, "tag"):  # This is an element, not a text node
                sig = get_element_signature(child)
                if sig not in output_signatures:
                    output_signatures[sig] = []
                output_signatures[sig].append(child)

        # Find added elements (in output but not in input)
        for sig, output_elems in output_signatures.items():
            if sig not in input_signatures:
                # New element type added
                for elem in output_elems:
                    # Tag name as a string; comments and other special nodes become "comment"
                    tag_str = str(elem.tag) if elem.tag is not None else "unknown"
                    # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]
                    # Handle comment and other special nodes
                    if tag_str.startswith("<cyfunction") or "Comment" in tag_str:
                        tag_str = "comment"
                    added_elements.append(
                        {
                            "tag": tag_str,
                            "content": etree.tostring(
                                elem, encoding="unicode", method="xml", with_tail=False
                            ),
                        }
                    )
            else:
                # Element type exists in both, check if we have more instances
                input_count = len(input_signatures[sig])
                output_count = len(output_elems)
                if output_count > input_count:
                    # More instances in output - count the difference as added
                    for i in range(input_count, output_count):
                        # Tag name as a string; comments and other special nodes become "comment"
                        tag_str = (
                            str(output_elems[i].tag)
                            if output_elems[i].tag is not None
                            else "unknown"
                        )
                        # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                        if "}" in tag_str:
                            tag_str = tag_str.split("}")[1]
                        # Handle comment and other special nodes
                        if tag_str.startswith("<cyfunction") or "Comment" in tag_str:
                            tag_str = "comment"
                        added_elements.append(
                            {
                                "tag": tag_str,
                                "content": etree.tostring(
                                    output_elems[i],
                                    encoding="unicode",
                                    method="xml",
                                    with_tail=False,
                                ),
                            }
                        )

        # Find removed elements (in input but not in output)
        for sig, input_elems in input_signatures.items():
            if sig not in output_signatures:
                # Element type removed
                for elem in input_elems:
                    # Tag name as a string; comments and other special nodes become "comment"
                    tag_str = str(elem.tag) if elem.tag is not None else "unknown"
                    # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                    if "}" in tag_str:
                        tag_str = tag_str.split("}")[1]
                    # Handle comment and other special nodes
                    if tag_str.startswith("<cyfunction") or "Comment" in tag_str:
                        tag_str = "comment"
                    removed_elements.append(
                        {
                            "tag": tag_str,
                            "content": etree.tostring(
                                elem, encoding="unicode", method="xml", with_tail=False
                            ),
                        }
                    )
            else:
                # Element type exists in both, check if we have fewer instances
                input_count = len(input_elems)
                output_count = len(output_signatures[sig])
                if input_count > output_count:
                    # More instances in input - count the difference as removed
                    for i in range(output_count, input_count):
                        # Tag name as a string; comments and other special nodes become "comment"
                        tag_str = (
                            str(input_elems[i].tag) if input_elems[i].tag is not None else "unknown"
                        )
                        # Remove namespace prefix if present (e.g., {namespace}tag -> tag)
                        if "}" in tag_str:
                            tag_str = tag_str.split("}")[1]
                        # Handle comment and other special nodes
                        if tag_str.startswith("<cyfunction") or "Comment" in tag_str:
                            tag_str = "comment"
                        removed_elements.append(
                            {
                                "tag": tag_str,
                                "content": etree.tostring(
                                    input_elems[i],
                                    encoding="unicode",
                                    method="xml",
                                    with_tail=False,
                                ),
                            }
                        )

        # Check for text changes (normalized comparison)
        text_changed = input_text != output_text

        # Sort added_elements by their position in the text to preserve spatial order
        added_elements = sort_elements_by_text_position(added_elements, output_xml)

        # Create diff report
        diff_report = {
            "filename": filename,
            "xpath": xpath,
            "added_elements": added_elements,
            "removed_elements": removed_elements,
            "modified_elements": modified_elements,
            "text_changed": text_changed,
            "input_text": input_text,
            "output_text": output_text,
        }

        return diff_report

    except Exception as e:
        logger.error(f"Could not compare XML elements of {filename} ({xpath}): {e}")
        return {
            "filename": filename,
            "xpath": xpath,
            "error": str(e),
            "added_elements": [],
            "removed_elements": [],
            "modified_elements": [],
            "text_changed": False,
        }


def get_latest_xml_mapping_file(
    model_instance: str, output_dir: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """
    Find the newest ``xml_update_mapping.json`` of a model instance.

    Looks in ``<output_dir>/<model_instance>/processing_*/``. Without ``output_dir``,
    ``data/output`` under the project root is used; the project root is then taken as two levels
    above the file of the calling module (e.g. ``evaluation/eval.py`` -> repository root), or the
    current directory if that cannot be determined. "Newest" means the latest file modification
    time; among runs with the same time (e.g. after a fresh clone) the first in name order is
    taken.

    Args:
        model_instance: Model instance name, e.g. ``"qwen3-14B-Q6"``
        output_dir: Folder with the stored LLM output (``eval.py`` passes
            ``config.LLM_OUTPUT_DIR`` under the repository root)

    Returns:
        str: Path of the newest mapping file, or None if there is none
    """
    if output_dir is not None:
        output_base = Path(output_dir) / model_instance
    else:
        # Project root from the calling module's file: two levels up from evaluation/eval.py
        try:
            frame = inspect.currentframe()
            caller_file = frame.f_back.f_globals.get("__file__", "")
            if caller_file:
                project_root = Path(caller_file).parent.parent
            else:
                project_root = Path.cwd()
        except Exception:
            project_root = Path.cwd()

        # data/output/{model_instance}/processing_{timestamp}/xml_update_mapping.json
        output_base = project_root / "data" / "output" / model_instance

    # Find all processing directories
    mapping_files = []
    if output_base.exists():
        for processing_dir in sorted(output_base.glob("processing_*")):
            mapping_file = processing_dir / "xml_update_mapping.json"
            if mapping_file.exists():
                mapping_files.append(mapping_file)

    if not mapping_files:
        return None

    # Sort by modification time (most recent first)
    mapping_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(mapping_files[0])


def load_xml_mapping(mapping_file: Union[str, Path]) -> Dict:
    """
    Load the XML update mapping from JSON file.

    Args:
        mapping_file: Path to the xml_update_mapping JSON file

    Returns:
        Dictionary containing the XML update mapping data
    """
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tei_encoding_for_element(
    mapping_data: Dict, filename: str, element_id: str, bracketed_sequence: str
) -> Optional[str]:
    """
    Look up the model's TEI encoding of one bracketed sequence in the mapping data.

    The replacements of the element are searched in order, and the first one matching the
    sequence is returned. A replacement matches if its ``original_item`` is equal to the
    sequence, equal after normalising apostrophes and ellipses, or contains / is contained in the
    sequence (raw, then normalised).

    Args:
        mapping_data: Loaded ``xml_update_mapping.json``
        filename: Letter file name, e.g. ``"letter101.xml"``
        element_id: Element ID in the mapping, e.g. ``"L101_div1_p2"``
        bracketed_sequence: The bracketed sequence to look up

    Returns:
        str: The ``tei_encoding`` of the first matching replacement, or None
    """
    if filename not in mapping_data:
        return None

    # Find the element entry in the mapping
    element_entries = [
        entry for entry in mapping_data[filename] if entry["element_id"] == element_id
    ]

    if not element_entries:
        return None

    # Normalize text for comparison (apostrophes and ellipsis)
    def normalize_text(text: str) -> str:
        """Normalize different character variants to a standard form"""
        if not text:
            return text
        # Replace various apostrophe types with standard straight apostrophe
        text = text.replace(chr(8217), "'")  # right single quotation mark (8217)
        text = text.replace(chr(8216), "'")  # left single quotation mark (8216)
        text = text.replace(chr(180), "'")  # acute accent (180)
        text = text.replace(chr(96), "'")  # grave accent (96)
        # Normalize ellipsis: Unicode ellipsis (…) to multiple dots
        text = text.replace("…", "..")  # ellipsis to 2 dots (common variant)
        text = text.replace("....", "..")  # 4 dots to 2 dots
        text = text.replace("...", "..")  # 3 dots to 2 dots
        return text

    normalized_sequence = normalize_text(bracketed_sequence)

    # Find the specific replacement for the bracketed sequence
    for entry in element_entries:
        for replacement in entry["replacements"]:
            # Use original_item (the key name in the mapping file)
            replacement_sequence = replacement.get("original_item", "")
            if not replacement_sequence:
                continue
            normalized_replacement = normalize_text(replacement_sequence)

            # Try exact match first (fastest)
            if replacement_sequence == bracketed_sequence:
                return replacement["tei_encoding"]
            # Try normalized exact match
            elif normalized_replacement == normalized_sequence:
                return replacement["tei_encoding"]
            # Fall back to substring matching (bidirectional)
            elif (
                bracketed_sequence in replacement_sequence
                or replacement_sequence in bracketed_sequence
            ):
                return replacement["tei_encoding"]
            # Fall back to normalized substring matching
            elif (
                normalized_sequence in normalized_replacement
                or normalized_replacement in normalized_sequence
            ):
                return replacement["tei_encoding"]

    return None


class JingValidator:
    """Validates XML files against a RelaxNG schema with the Jing validator (needs Java)."""

    def __init__(self, jing_jar_path: Optional[str] = None) -> None:
        """
        Set the path of the Jing jar.

        Args:
            jing_jar_path: Path to the Jing jar; None uses the bundled
                ``resources/tools/jing-RELEASE220.jar``
        """
        if jing_jar_path is None:
            self.jing_jar_path = DEFAULT_JING_JAR
        else:
            self.jing_jar_path = Path(jing_jar_path)

    def validate_quick(self, xml_file: str, schema_file: str) -> Tuple[bool, str]:
        """
        Validate a file and return pass/fail with the first error line.

        Jing's output is decoded as UTF-8; undecodable bytes are replaced, so an unusual
        character in a message cannot make the call fail.

        Args:
            xml_file: Path of the XML file to validate
            schema_file: Path of the RelaxNG schema

        Returns:
            tuple: ``(is_valid, error_message)``; the message is empty if the file is valid
        """
        try:
            # System Java if it runs, otherwise a portable installation
            java_cmd = self._find_java_executable()

            result = subprocess.run(
                [java_cmd, "-jar", str(self.jing_jar_path), str(schema_file), str(xml_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=JING_TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                return True, ""
            else:
                # Get all output for better error reporting
                stdout_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
                stderr_lines = result.stderr.strip().split("\n") if result.stderr.strip() else []
                all_lines = stdout_lines + stderr_lines

                # Filter out empty lines
                error_lines = [line for line in all_lines if line.strip()]

                if error_lines:
                    # Return the first meaningful error line
                    first_error = error_lines[0]
                    return False, self._extract_clean_error_message(first_error)
                else:
                    return False, f"Validation failed (return code: {result.returncode})"

        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except FileNotFoundError:
            return False, "Java not found - please ensure Java is installed and in PATH"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _extract_clean_error_message(self, error_line: str) -> str:
        """
        Shorten a Jing error line to the part after ``error:`` (dropping file and position).

        Args:
            error_line: One line of Jing output

        Returns:
            str: The shortened message, or the stripped line if it has no ``error:`` part
        """
        if not error_line:
            return "Unknown validation error"

        # Clean up common patterns in Jing error messages
        clean_line = error_line.strip()

        # Remove file path prefixes (e.g., "/tmp/tmpXXX.xml:5:10: error: ...")
        if ":" in clean_line and "error:" in clean_line:
            parts = clean_line.split("error:", 1)
            if len(parts) > 1:
                return f"error: {parts[1].strip()}"

        # Handle other common patterns
        if clean_line.startswith("fatal:"):
            return clean_line
        if clean_line.startswith("error:"):
            return clean_line

        return clean_line

    def is_available(self) -> bool:
        """
        Check whether the Jing jar exists (Java itself is not checked).

        Returns:
            bool: True if the jar file exists
        """
        return self.jing_jar_path.exists()

    def _find_java_executable(self) -> str:
        """
        Find a Java executable: ``java`` on the PATH, else one in ``./java_portable``.

        In ``java_portable``, the first ``jdk-*/bin/java.exe`` in name order is taken, then the
        usual jPortable locations.

        Returns:
            str: ``"java"`` if it runs (or as the last resort), otherwise the path of a portable
            ``java.exe`` (looked up relative to the current directory)
        """
        # First try system Java
        try:
            result = subprocess.run(
                ["java", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=JAVA_CHECK_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return "java"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try portable Java installations
        portable_java_path = Path("java_portable")
        if portable_java_path.exists():
            # Look for OpenJDK structure (jdk-*)
            jdk_dirs = sorted(portable_java_path.glob("jdk-*/bin/java.exe"))
            if jdk_dirs:
                return str(jdk_dirs[0])

            # Look for jPortable structure
            jportable_candidates = [
                portable_java_path / "CommonFiles" / "Java" / "bin" / "java.exe",
                portable_java_path / "bin" / "java.exe",
                portable_java_path / "App" / "Java" / "bin" / "java.exe",
            ]

            for candidate in jportable_candidates:
                if candidate.exists():
                    return str(candidate)

        # Fallback to system java
        return "java"


def normalize_text(text: str) -> str:
    """Normalize text content by removing redundant whitespaces and newlines.

    Replaces newlines, tabs, and multiple spaces with single spaces, then strips
    leading and trailing whitespace.

    Args:
        text: Input text string to normalize.

    Returns:
        Normalized text string with redundant whitespaces and newlines removed.
    """
    if not text:
        return ""

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


def extract_letter_number(filename: Union[str, Path]) -> Optional[int]:
    """Extract letter number from filename.

    Examples:
        "letter138.xml" -> 138
        "letter12.xml" -> 12

    Args:
        filename: Filename string or Path object containing letter number.

    Returns:
        Letter number as integer if found, None otherwise.
    """
    if not filename:
        return None
    filename_str = str(filename)
    match = re.search(r"letter(\d+)\.xml", filename_str)
    if match:
        return int(match.group(1))
    return None


def generate_proper_xpath_with_text_index(
    element: etree._Element, text_node: str, root: etree._Element
) -> str:
    """Build an XPath with positions for every step and a text node index.

    If ``text_node`` is the tail of ``element``, the containing element is the parent. Every step
    gets a position (``p[1]`` even for an only child). The text node index is the position of the
    first text node with the same content among the container's text and its children's tails;
    it is omitted (``/text()``) if the container has only one text node or the text is not found.

    Args:
        element: Element whose text or tail is ``text_node``
        text_node: Content of the text node
        root: Root element of the tree

    Returns:
        str: XPath without namespace prefixes, e.g.
            ``/TEI[1]/text[1]/body[1]/div[1]/p[2]/text()[3]``
    """
    # Determine the actual container element
    actual_container = element

    # If this text node is the tail of the element, the actual container is the parent
    if hasattr(element, "tail") and element.tail == text_node:
        actual_container = element.getparent()
        if actual_container is None:
            actual_container = element  # Fallback

    # Build path from actual container up to root
    path_parts = []
    current = actual_container

    while current is not None and current != root:
        # Get element name without namespace
        tag = current.tag
        if tag.startswith(TEI_TAG_PREFIX):
            tag = tag.replace(TEI_TAG_PREFIX, "")

        # Count siblings with same tag name
        parent = current.getparent()
        if parent is not None:
            siblings = [s for s in parent if s.tag == current.tag]
            # Always include position number, even if there's only one sibling
            position = siblings.index(current) + 1
            path_parts.append(f"{tag}[{position}]")
        else:
            path_parts.append(f"{tag}[1]")

        current = parent

    # Add root element
    if current is not None:
        root_tag = current.tag
        if root_tag.startswith(TEI_TAG_PREFIX):
            root_tag = root_tag.replace(TEI_TAG_PREFIX, "")
        path_parts.append(f"{root_tag}[1]")

    # Reverse and join
    path_parts.reverse()
    xpath = "/" + "/".join(path_parts)

    # Now find the correct text node index
    # Collect all text nodes in document order: element text + tail text of all children
    all_text_nodes = []

    # Add element's own text if it exists
    if actual_container.text:
        all_text_nodes.append(actual_container.text)

    # Add tail text of all child elements
    for child in actual_container:
        if child.tail:
            all_text_nodes.append(child.tail)

    # Find our text node in this collection
    if len(all_text_nodes) > 1:
        try:
            text_index = all_text_nodes.index(text_node) + 1
            return f"{xpath}/text()[{text_index}]"
        except ValueError:
            # Fallback if not found
            return f"{xpath}/text()"
    else:
        return f"{xpath}/text()"


def create_simplified_xpath(full_xpath: str) -> str:
    """Create a short element ID part from an XPath, starting at the first ``div``.

    Example: ``/TEI[1]/text[1]/body[1]/div[1]/p[2]/text()`` -> ``div1_p2``. The last step is
    always dropped (it is expected to be ``text()``).

    Args:
        full_xpath: Full XPath string to simplify.

    Returns:
        Simplified XPath string starting from div, or "unknown" if div not found.
    """
    # Split the XPath into parts
    parts = full_xpath.split("/")

    # Find the div element and everything after it
    div_index = None
    for i, part in enumerate(parts):
        if part == "div" or part.startswith("div["):
            div_index = i
            break

    if div_index is not None:
        # Get parts from div onwards (excluding the text() part)
        relevant_parts = parts[div_index:-1]  # Exclude the last text() part

        # Create simplified representation
        simplified_parts = []
        for part in relevant_parts:
            if part:
                # Extract element name and number if present
                match = re.match(r"(\w+)\[(\d+)\]", part)
                if match:
                    element_name = match.group(1)
                    element_number = match.group(2)
                    simplified_parts.append(f"{element_name}{element_number}")
                else:
                    # No number, just use the element name
                    simplified_parts.append(part)

        return "_".join(simplified_parts)

    return "unknown"
