"""
Build the published sample of bracketed sequences from the full letter corpus.

Usage (needs the unpublished full corpus and corpus-level files in ``data/sampling/analysis/``)::

    python data/sampling/sampling.py

Steps:

1. Read the hand-made selection (rows with ``Final_Selection == "x"``) from
   ``data/sampling/analysis/editorial_interventions_brackets_analysis.xlsx``, copy the selected
   letters from ``analysis/letters-all/`` to ``data/sampling/bracket_sample/`` and extract each
   selected text node, its bracketed sequences and its parent element
   (``data/sampling/brackets_sample.xlsx``)
2. Determine the gold-standard intervention type of every sequence
   (``data/sampling/brackets_statistics.xlsx``, via ``utils/gold_standard_stats.py``)
3. Cut a context window around the sequences of each element for the LLM prompt
   (``data/sampling/brackets_sample.json``)

The published copies of ``brackets_sample.*`` and ``bracket_sample/`` are in ``data/input/``
(moved there by hand). Importing this module configures logging and creates a log file
``data/sampling/sampling_<timestamp>.log``.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from lxml import etree

# Repository root on sys.path, for the root-level utils package (three levels up: sampling ->
# data -> root). The utils imports have to come after it.
_parent_dir = Path(__file__).resolve().parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from utils.gold_standard_stats import create_bracket_statistics_df  # noqa: E402
from utils.xml_utils import (  # noqa: E402
    TEI_NSMAP,
    create_simplified_xpath,
    extract_all_text_content,
    extract_element_content_by_xpath,
    normalize_text,
)

__version__ = "1.0.0"

# Set up logging
_log_dir = Path(__file__).parent
_log_file = _log_dir / f"sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging: file handler for all details, console handler for major steps and errors
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

# File handler: all log levels
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_file_handler.setFormatter(_file_formatter)

# Console handler: only INFO and above (major steps, warnings, errors)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_formatter = logging.Formatter("%(message)s")
_console_handler.setFormatter(_console_formatter)

_logger.addHandler(_file_handler)
_logger.addHandler(_console_handler)


# Configuration constants
# Base paths
_DATA_DIR = Path("data")
_REFERENCES_DIR = _DATA_DIR / "references"
_SAMPLING_DIR = _DATA_DIR / "sampling"
_SAMPLING_ANALYSIS_DIR = _SAMPLING_DIR / "analysis"

# Paths for bracket extraction and sampling, relative to the repository root; the functions join
# them with _parent_dir, so they do not depend on the current directory
SAMPLING_XML_DIR = _SAMPLING_DIR / "bracket_sample"
SAMPLING_REFERENCES_GOLD_DIR = _REFERENCES_DIR / "gold"
SAMPLING_OUTPUT_EXCEL = _SAMPLING_DIR / "brackets_sample.xlsx"
SAMPLING_STATS_EXCEL = _SAMPLING_DIR / "brackets_statistics.xlsx"
SAMPLING_METADATA_EXCEL = _SAMPLING_ANALYSIS_DIR / "editorial_interventions_brackets_analysis.xlsx"
SAMPLING_SEGMENTS_JSON = _SAMPLING_DIR / "brackets_sample.json"
SAMPLING_LETTERS_ALL_DIR = _SAMPLING_ANALYSIS_DIR / "letters-all"

# Text window size for bracket context extraction (characters before/after)
SAMPLING_CONTEXT_WINDOW_SIZE = 100

# Elements that always need ancestor context (typically contain only names without surrounding text)
ELEMENTS_NEEDING_ANCESTOR = {"persName", "signed"}

# Context ancestor elements to look for
CONTEXT_ANCESTOR_ELEMENTS = {"p", "closer", "opener"}

# Marker for paragraph breaks (used to survive text normalization)
PARAGRAPH_BREAK_MARKER = "<<<PARAGRAPH_BREAK>>>"


def _extract_sentence_window(text: str, bracketed_sequences: list) -> str:
    """
    Extract the sentence containing the brackets (from preceding boundary to following boundary).

    The window runs from the first ``[`` to the last ``]`` in ``text`` (not only those of
    ``bracketed_sequences``), widened backwards to just after the last ``". "`` or paragraph break
    (two newlines) before it, or to the start of the text, and forwards to the first ``.``
    (included) or paragraph break (excluded) after it, or to the end of the text. The result is
    stripped.

    Args:
        text: The text to window
        bracketed_sequences: Bracketed sequences of the text; only checked for being non-empty

    Returns:
        The stripped window. ``text`` unchanged if ``text`` or ``bracketed_sequences`` is empty,
        if ``text`` has no ``[`` or no ``]``, or if the window is empty.
    """
    if not text or not bracketed_sequences:
        return text

    # Find first opening bracket position
    first_bracket_pos = text.find("[")
    if first_bracket_pos == -1:
        return text

    # Find last closing bracket position
    last_bracket_pos = text.rfind("]")
    if last_bracket_pos == -1:
        return text

    # Find preceding boundary (sentence/paragraph start)
    # Look for ". " or "\n\n" before first bracket
    search_text = text[:first_bracket_pos]
    preceding_boundary_pos = -1
    preceding_boundary_len = 0

    # Search backwards for ". "
    for i in range(len(search_text) - 1, -1, -1):
        if search_text[i] == "." and i + 1 < len(search_text) and search_text[i + 1] == " ":
            if i > preceding_boundary_pos:
                preceding_boundary_pos = i
                preceding_boundary_len = 2  # ". "
            break

    # Also check for "\n\n" (paragraph break)
    newline_pos = search_text.rfind("\n\n")
    if newline_pos != -1 and newline_pos > preceding_boundary_pos:
        preceding_boundary_pos = newline_pos
        preceding_boundary_len = 2  # "\n\n"

    # Find following boundary (sentence/paragraph end)
    # Look for ". " or "\n\n" or "." at end after last bracket
    search_start = last_bracket_pos + 1
    following_boundary_pos = -1
    following_boundary_len = 0

    # Look for period
    for i in range(search_start, len(text)):
        if text[i] == ".":
            following_boundary_pos = i
            following_boundary_len = 1
            break

    # Also check for "\n\n" after last bracket
    newline_pos = text.find("\n\n", search_start)
    if newline_pos != -1 and (following_boundary_pos == -1 or newline_pos < following_boundary_pos):
        following_boundary_pos = newline_pos
        following_boundary_len = 0  # Don't include the newlines

    # Determine window boundaries
    if preceding_boundary_pos != -1:
        # Start after the boundary
        start_pos = preceding_boundary_pos + preceding_boundary_len
    else:
        # No preceding boundary, start from beginning
        start_pos = 0

    if following_boundary_pos != -1:
        # Include the period (if it's a period), exclude newlines
        end_pos = following_boundary_pos + following_boundary_len
    else:
        # No following boundary, go to end
        end_pos = len(text)

    result = text[start_pos:end_pos].strip()
    return result if result else text


def _strip_all_brackets(text: str) -> str:
    """
    Remove all square brackets from text, keeping their content.

    Args:
        text: The text containing brackets

    Returns:
        Text with all brackets removed but content preserved
    """
    if not text:
        return text
    # Remove brackets but keep content (handles empty brackets too)
    return re.sub(r"\[([^\]]*)\]", r"\1", text)


def _has_sufficient_context(text: str, min_chars: int = 100) -> bool:
    """
    Check if text has at least min_chars before first '[' and after last ']'.

    Args:
        text: The text content to check
        min_chars: Minimum characters required before first bracket and after last bracket

    Returns:
        True if text has sufficient context on both sides, False otherwise
    """
    if not text or "[" not in text or "]" not in text:
        return False

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    chars_before = first_bracket
    chars_after = len(text) - last_bracket - 1

    return chars_before >= min_chars and chars_after >= min_chars


def _get_ancestor_xpath(xpath: str, ancestor_elements: set) -> str:
    """
    Find the nearest ancestor element from the given set in the XPath.

    Args:
        xpath: XPath expression
        ancestor_elements: Set of element names to look for (e.g., {"p", "closer", "opener"})

    Returns:
        XPath to the nearest ancestor element, or None if not found
    """
    import re

    # Remove /text() or /text()[n] suffix
    base_xpath = re.sub(r"/text\(\)(\[\d+\])?$", "", xpath)

    # Split xpath into parts and find the nearest matching ancestor
    # e.g., "/TEI[1]/text[1]/body[1]/div[1]/p[1]/persName[6]" -> parts
    parts = base_xpath.split("/")

    # Walk backwards through parts to find nearest ancestor
    for i in range(len(parts) - 1, 0, -1):
        part = parts[i]
        # Extract element name (e.g., "p[1]" -> "p")
        match = re.match(r"([a-zA-Z]+)", part)
        if match:
            element_name = match.group(1)
            if element_name in ancestor_elements:
                # Return xpath up to and including this element
                return "/".join(parts[: i + 1])

    return None


def _is_inside_elements(xpath: str, element_names: set) -> bool:
    """
    Check if the XPath is inside any of the specified elements.

    Args:
        xpath: XPath expression
        element_names: Set of element names to check for (e.g., {"persName", "signed"})

    Returns:
        True if xpath contains any of the elements, False otherwise
    """
    import re

    for element_name in element_names:
        # Check if element appears in the xpath (e.g., "/persName[" or "/persName/")
        pattern = rf"/{element_name}(\[|/|$)"
        if re.search(pattern, xpath):
            return True
    return False


def extract_bracketed_content(text: str) -> str:
    """
    Extract all bracketed content from text and return them separated by pipes.

    Groups adjacent bracketed tokens into single sequences. Special handling for:
    - [sic]: extract from last sentence boundary to [sic]
    - [?]: extract from last punctuation mark to [?]
    - [...]: include word before if whitespace before, or characters until preceding whitespace

    Args:
        text: The input text containing bracketed content.

    Returns:
        A string containing all bracketed sequences separated by " | ", or empty string if none
            found.

    Examples:
        - "d[ieses] M[onats]" -> "d[ieses] M[onats]" (grouped as one sequence)
        - "M[anu]s[krip]t" -> "M[anu]s[krip]t" (single token)
        - "[qu'à]" -> "[qu'à]"
        - "Chev[alier]" -> "Chev[alier]"
        - "...after the last dot. Je n´ai rien souffert personnellement, quoique j´ai réfusé de
            signer la soidisante [sic] Constitution..."
          -> "Je n´ai rien souffert personnellement, quoique j´ai réfusé de signer la soidisante
              [sic]"
        - "word_to_include [...]" -> "word_to_include [...]"
        - "ausgestat[...]" -> "ausgestat[...]"
        - "Sentence ends here. Question mark [?]" -> "Question mark [?]"
    """
    if not text or "[" not in text or "]" not in text:
        return ""

    # Normalize the text first
    text = normalize_text(text)

    bracketed_sequences = []
    processed_positions = set()

    # Find all bracket pairs in order, allowing whitespace inside brackets
    i = 0
    while i < len(text):
        if text[i] == "[":
            # Found opening bracket, now find the corresponding closing bracket
            bracket_start = i
            j = i + 1

            # Look for closing bracket, allowing any content including whitespace
            while j < len(text) and text[j] != "]":
                j += 1

            if j < len(text):  # Found closing bracket
                bracket_end = j + 1
                bracket_content = text[bracket_start:bracket_end]

                # Skip if already processed (shouldn't happen with this approach)
                if bracket_start in processed_positions:
                    i = bracket_end
                    continue

                # Apply special handling rules
                if bracket_content == "[sic]":
                    # Extract sentence context for [sic]
                    sic_context = extract_sic_context(text, bracket_content)
                    if sic_context:
                        bracketed_sequences.append(sic_context)
                        # Mark all positions in this context as processed
                        sic_start = text.find(sic_context)
                        if sic_start != -1:
                            for pos in range(sic_start, sic_start + len(sic_context)):
                                processed_positions.add(pos)

                elif bracket_content == "[?]":
                    # Extract punctuation context for [?]
                    question_context = extract_punctuation_context(
                        text, bracket_content, bracket_start
                    )
                    if question_context:
                        bracketed_sequences.append(question_context)
                        # Mark all positions in this context as processed
                        q_start = text.find(question_context)
                        if q_start != -1:
                            for pos in range(q_start, q_start + len(question_context)):
                                processed_positions.add(pos)

                elif re.match(r"\[\.\.\.+\]", bracket_content):
                    # Extract ellipsis context for [...]
                    ellipsis_context = extract_ellipsis_context(
                        text, bracket_content, bracket_start
                    )
                    if ellipsis_context:
                        bracketed_sequences.append(ellipsis_context)
                        # Mark all positions in this context as processed
                        e_start = text.find(ellipsis_context)
                        if e_start != -1:
                            for pos in range(e_start, e_start + len(ellipsis_context)):
                                processed_positions.add(pos)

                else:
                    # Regular bracket - include the complete word/phrase containing the bracket
                    # Find the start of the word containing this bracket
                    word_start = bracket_start
                    while word_start > 0 and text[word_start - 1] not in " \t\n\r":
                        word_start -= 1

                    # Find the end of the word containing this bracket
                    word_end = bracket_end
                    while word_end < len(text) and text[word_end] not in " \t\n\r":
                        word_end += 1

                    # Extract the complete word/phrase
                    complete_sequence = text[word_start:word_end]

                    # Check for adjacent brackets to group them
                    # Look ahead to see if there are more brackets nearby
                    temp_end = word_end
                    while temp_end < len(text):
                        # Skip whitespace
                        while temp_end < len(text) and text[temp_end] in " \t\n\r":
                            temp_end += 1

                        # Check if next character starts a word with a bracket
                        if temp_end < len(text):
                            # Find the end of the next word
                            next_word_start = temp_end
                            next_word_end = temp_end
                            while (
                                next_word_end < len(text) and text[next_word_end] not in " \t\n\r"
                            ):
                                next_word_end += 1

                            next_word = text[next_word_start:next_word_end]

                            # If next word contains brackets, include it in the sequence
                            if "[" in next_word and "]" in next_word:
                                complete_sequence = text[word_start:next_word_end]
                                temp_end = next_word_end

                                # Update the bracket_end to skip over this grouped content
                                bracket_end = next_word_end
                            else:
                                break
                        else:
                            break

                    bracketed_sequences.append(complete_sequence)

                    # Mark all positions in this sequence as processed
                    for pos in range(word_start, word_end):
                        processed_positions.add(pos)

                i = bracket_end
            else:
                # No closing bracket found, move to next character
                i += 1
        else:
            i += 1

    # Join matches with pipe separator
    if bracketed_sequences:
        return " | ".join(bracketed_sequences)
    else:
        return ""


def extract_sic_context(text: str, sic_match: str) -> str:
    """
    Extract context for [sic] markers.

    Uses the following logic:
    1. First find the last sentence boundary (period+space, or other punctuation like comma,
       semicolon, dash)
    2. Then check if there's a bracket AFTER that boundary and before [sic]
    3. If yes, start after that bracket (to avoid including other editorial interventions)
    4. If no, start after the punctuation
    5. Strip leading whitespace from the result

    Args:
        text: The full text containing the [sic] marker.
        sic_match: The [sic] string to find in the text.

    Returns:
        The extracted context string including the [sic] marker, or the marker itself if not found.
    """
    # Find the position of the [sic] match in the text
    sic_pos = text.find(sic_match)

    if sic_pos == -1:
        return sic_match  # Fallback if not found

    text_before_sic = text[:sic_pos]

    # Step 1: Find the last punctuation boundary
    # Look for period followed by space (sentence boundary), or comma, semicolon, dash
    last_punct_pos = -1
    for punct in [". ", ", ", "; ", "– ", "- "]:  # Punctuation followed by space
        pos = text_before_sic.rfind(punct)
        if pos > last_punct_pos:
            last_punct_pos = pos

    # Also check for punctuation at the very end (no space after)
    for punct in [",", ";", "–", "-"]:
        pos = text_before_sic.rfind(punct)
        if pos > last_punct_pos:
            last_punct_pos = pos

    # Check for period not directly before [sic]
    period_pos = text_before_sic.rfind(".")
    if period_pos > last_punct_pos and period_pos != sic_pos - 1:
        last_punct_pos = period_pos

    # Step 2: Look for brackets AFTER the last punctuation and before [sic]
    search_start = last_punct_pos + 1 if last_punct_pos != -1 else 0
    text_after_punct = text_before_sic[search_start:]

    # Find the last bracket in the text after punctuation
    last_bracket_end = -1
    i = 0
    while i < len(text_after_punct):
        if text_after_punct[i] == "[":
            j = i + 1
            while j < len(text_after_punct) and text_after_punct[j] != "]":
                j += 1
            if j < len(text_after_punct):  # Found closing bracket
                last_bracket_end = search_start + j + 1  # Position in original text
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    # Step 3: Determine context start
    if last_bracket_end != -1:
        # There's a bracket after punctuation, start after that bracket
        context_start = last_bracket_end
    elif last_punct_pos != -1:
        # No bracket after punctuation, start after punctuation
        context_start = last_punct_pos + 1
    else:
        # No punctuation found, extract from start of text (start is also a boundary)
        context_start = 0

    # Skip leading whitespace
    while context_start < sic_pos and text[context_start] in " \t\n\r":
        context_start += 1

    context_end = sic_pos + len(sic_match)
    context = text[context_start:context_end].strip()
    return context if context else sic_match


def extract_punctuation_context(
    text: str, question_match: str, question_pos: Optional[int] = None
) -> str:
    """
    Extract context for [?] markers from the last punctuation mark or bracket.

    Extracts from the last sentence-ending punctuation or closing bracket (whichever
    is closer to [?]) to capture meaningful context without including other interventions.
    If neither is found, includes the word/token containing the [?].

    Args:
        text: The full text containing the [?] marker.
        question_match: The [?] string to find.
        question_pos: Optional position of the [?] in the text. If None, uses find().

    Returns:
        The extracted context string including the [?] marker, or the marker itself if not found.
    """
    # Find the position of the [?] match in the text
    if question_pos is None:
        question_pos = text.find(question_match)

    if question_pos == -1:
        return question_match  # Fallback if not found

    # Look for the last punctuation mark before the [?]
    text_before_question = text[:question_pos]

    # Find the last occurrence of sentence-ending punctuation
    last_punct_pos = -1
    for punct in [". ", "? ", "! ", ": ", "; "]:
        pos = text_before_question.rfind(punct)
        if pos > last_punct_pos:
            last_punct_pos = pos

    # Also check for another bracket before [?] - don't include other interventions
    last_bracket_pos = text_before_question.rfind("]")

    if last_punct_pos != -1 or last_bracket_pos != -1:
        # Use whichever is closer to the [?] (larger position = closer)
        if last_bracket_pos > last_punct_pos:
            # Another bracket is closer - start after it (skip trailing whitespace)
            context_start = last_bracket_pos + 1
            while context_start < question_pos and text[context_start] in " \t\n\r":
                context_start += 1
        else:
            # Punctuation is closer - start after it
            context_start = last_punct_pos + 2  # +2 to skip punctuation and space

        context_end = question_pos + len(question_match)
        context = text[context_start:context_end].strip()
        return context
    else:
        # If no punctuation found, include the word before [?] and the [?] itself
        # Look for the start of the word before [?]
        start_pos = question_pos
        while start_pos > 0 and text[start_pos - 1] not in " \t\n\r":
            start_pos -= 1

        # If [?] is at the start of a word (no space before), include the previous word
        if start_pos == question_pos:
            # [?] is at the start of a word, look for the previous word
            prev_word_start = start_pos - 1
            while prev_word_start > 0 and text[prev_word_start] in " \t\n\r":
                prev_word_start -= 1

            # Find the start of the previous word
            while prev_word_start > 0 and text[prev_word_start - 1] not in " \t\n\r":
                prev_word_start -= 1

            start_pos = prev_word_start

        # Look for the end of the word containing [?]
        end_pos = question_pos + len(question_match)
        while end_pos < len(text) and text[end_pos] not in " \t\n\r":
            end_pos += 1

        # Extract the context including word before [?] and [?] itself
        context = text[start_pos:end_pos].strip()
        return context if context else question_match


def extract_ellipsis_context(text: str, ellipsis_match: str, ellipsis_start: int) -> str:
    """
    Extract context for [...] ellipsis patterns.

    Extraction rules:
    - If there's whitespace before [...], include everything until the next whitespace before it
    - If there's no whitespace, include all characters until the nearest preceding whitespace
    - Also include any characters directly connected to the closing bracket ] until the next
      whitespace

    Args:
        text: The full text containing the ellipsis marker.
        ellipsis_match: The [...] string pattern to find.
        ellipsis_start: The starting position of the ellipsis in the text.

    Returns:
        The extracted context string including the ellipsis marker, or the marker itself if position
            is invalid.
    """
    if ellipsis_start == -1:
        return ellipsis_match  # Fallback if not found

    ellipsis_end = ellipsis_start + len(ellipsis_match)

    # First, handle the prefix (before the [...])
    prefix = ""
    if ellipsis_start > 0 and text[ellipsis_start - 1] == " ":
        # Case 1: whitespace before [...] - include word before
        text_before = text[:ellipsis_start].rstrip()
        # Find the last word (everything after the last whitespace)
        last_space_pos = text_before.rfind(" ")
        if last_space_pos != -1:
            prefix = text_before[last_space_pos + 1 :] + " "
        else:
            # No space found, use the whole text before
            prefix = text_before + " " if text_before else ""
    else:
        # Case 2: no whitespace before [...] - include characters until preceding whitespace
        text_before = text[:ellipsis_start]
        # Find the last whitespace before the [...]
        last_space_pos = text_before.rfind(" ")
        if last_space_pos != -1:
            prefix = text_before[last_space_pos + 1 :]
        else:
            # No space found, use everything from the beginning
            prefix = text_before

    # Now handle the suffix (after the [...])
    suffix = ""
    if ellipsis_end < len(text):
        # Check if there are characters directly connected to the closing bracket
        char_after_bracket = ellipsis_end
        while char_after_bracket < len(text) and text[char_after_bracket] not in " \t\n\r":
            char_after_bracket += 1

        if char_after_bracket > ellipsis_end:
            # Found characters directly connected to the closing bracket
            suffix = text[ellipsis_end:char_after_bracket]

    # Combine prefix, ellipsis, and suffix
    context = prefix + ellipsis_match + suffix

    return context.strip()


def extract_element_content(filename: str, xpath: str, xml_input_dir: Path) -> str:
    """
    Read the parent element of a text node from a source letter, as whitespace-normalised XML.

    Args:
        filename: The XML filename to process.
        xpath: XPath of the text node (``/text()`` or ``/text()[n]`` is removed to get the parent).
        xml_input_dir: The directory containing the source letters.

    Returns:
        The element as an XML string with normalised whitespace. If the file or element is missing,
        or an error occurs, a message string instead ("Reference file not found", "Element not found
        in reference", or the error), which then ends up in Parent_Element_Content.
    """
    try:
        # Construct the path to the reference file
        input_file = xml_input_dir / filename

        if not input_file.exists():
            return "Reference file not found"

        # Create parent element XPath by removing /text() or text()[n] suffix
        # Same rule as create_parent_xpath in evaluation/utils/evaluation_utils.py
        parent_xpath = xpath
        if "/text()[" in xpath:
            # Handle text()[n] pattern first (more specific)
            parent_xpath = re.sub(r"/text\(\)\[\d+\]$", "", xpath)
        elif xpath.endswith("/text()"):
            # Handle simple /text() pattern
            parent_xpath = xpath[:-7]  # Remove '/text()'

        # Use element extraction function with parent XPath
        content = extract_element_content_by_xpath(input_file, parent_xpath)

        if content:
            # Normalize the content (but preserve XML structure for elements)
            return normalize_text(content)
        else:
            return "Element not found in reference"

    except OSError as e:
        return f"File I/O error: {str(e)}"
    except etree.XMLSyntaxError as e:
        return f"XML parsing error: {str(e)}"
    except (ValueError, KeyError) as e:
        return f"Data extraction error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def load_metadata_dataframe(metadata_path: Path) -> Optional[pd.DataFrame]:
    """
    Load the hand-made selection: rows of the corpus analysis file with Final_Selection == "x".

    Column names are matched case-insensitively.

    Args:
        metadata_path: Path to ``editorial_interventions_brackets_analysis.xlsx``.

    Returns:
        DataFrame with columns Filename, XPath and Language (from Language_Text_Node), filtered to
        the selected rows with a ``.xml`` file name, or None if the file or a column is missing or
        nothing is selected.
    """
    if not metadata_path.exists():
        _logger.warning(
            f"Editorial interventions not found at {metadata_path}; skipping metadata loading."
        )
        return None

    try:
        df = pd.read_excel(metadata_path)

        # Handle case-insensitive column name matching
        col_mapping = {}
        for required_col in ["Filename", "Xpath", "Final_Selection"]:
            if required_col in df.columns:
                col_mapping[required_col] = required_col
            else:
                matching_cols = [c for c in df.columns if c.lower() == required_col.lower()]
                if matching_cols:
                    col_mapping[required_col] = matching_cols[0]
                else:
                    _logger.error(
                        f'Editorial interventions missing column "{required_col}"; skipping '
                        "metadata loading."
                    )
                    _logger.debug(f"Available columns: {list(df.columns)}")
                    return None

        # Filter to rows with Final_Selection == "x"
        selected = df[
            df[col_mapping["Final_Selection"]].astype(str).str.strip().str.lower() == "x"
        ].copy()

        if selected.empty:
            _logger.warning("No rows found with Final_Selection='x' in metadata.")
            return None

        # Get language column name (case-insensitive)
        lang_col = "Language_Text_Node"
        if lang_col not in df.columns:
            lang_col_lower = lang_col.lower()
            matching_lang_cols = [c for c in df.columns if c.lower() == lang_col_lower]
            if matching_lang_cols:
                lang_col = matching_lang_cols[0]

        # Create clean DataFrame with standardized column names
        metadata_df = pd.DataFrame(
            {
                "Filename": selected[col_mapping["Filename"]].astype(str).str.strip(),
                "XPath": selected[col_mapping["Xpath"]].astype(str).str.strip(),
                "Language": selected[lang_col].astype(str).str.strip()
                if lang_col in selected.columns
                else "",
            }
        )

        # Filter out rows with NaN filenames or XPaths
        metadata_df = metadata_df[
            (metadata_df["Filename"].str.lower() != "nan")
            & (metadata_df["XPath"].str.lower() != "nan")
            & (metadata_df["Filename"].str.endswith(".xml"))
        ].copy()

        # Replace "nan" strings with empty strings for Language
        metadata_df["Language"] = metadata_df["Language"].replace("nan", "")

        _logger.info(f"Loaded {len(metadata_df)} rows from metadata with Final_Selection='x'.")
        _logger.debug(f"Unique filenames: {metadata_df['Filename'].nunique()}")

        return metadata_df

    except (FileNotFoundError, PermissionError, OSError) as e:
        _logger.error(f"Error reading metadata file: {e}; skipping metadata loading.")
        return None
    except (ValueError, KeyError, pd.errors.EmptyDataError) as e:
        _logger.error(f"Error parsing metadata file: {e}; skipping metadata loading.")
        return None


def copy_selected_xml_files(
    metadata_df: pd.DataFrame, letters_all_dir: Path, xml_input_dir: Path
) -> Set[str]:
    """
    Copy the letters named in the selection from letters-all to the sample folder.

    Only files directly in ``letters_all_dir`` are found (no subfolders). Existing files are
    overwritten.

    Args:
        metadata_df: DataFrame with selected metadata rows (from load_metadata_dataframe).
        letters_all_dir: Path to the folder with all letters.
        xml_input_dir: Destination folder (``data/sampling/bracket_sample/``).

    Returns:
        Set of filenames that were successfully copied.
    """
    if metadata_df is None or metadata_df.empty:
        _logger.warning("No metadata DataFrame provided; skipping file copy.")
        return set()

    if not letters_all_dir.exists():
        _logger.warning(
            f"Letters-all directory not found at {letters_all_dir}; skipping file copy."
        )
        return set()

    # Get unique filenames from metadata DataFrame
    unique_filenames = set(metadata_df["Filename"].unique())
    unique_filenames = {f for f in unique_filenames if f and f.endswith(".xml")}

    if not unique_filenames:
        _logger.warning("No valid XML filenames found in metadata; skipping file copy.")
        return set()

    _logger.debug(f"Found {len(unique_filenames)} unique XML files to copy from metadata.")

    # Ensure destination directory exists
    xml_input_dir.mkdir(parents=True, exist_ok=True)

    # Search for files in letters-all directory and copy them
    copied_files = set()
    not_found_files = []

    for filename in unique_filenames:
        # Check if file exists directly in letters-all directory
        source_file = letters_all_dir / filename
        if source_file.exists():
            dest_file = xml_input_dir / filename
            try:
                shutil.copy2(source_file, dest_file)
                copied_files.add(filename)
                _logger.debug(f"Copied {filename}")
            except (OSError, PermissionError, shutil.Error) as e:
                _logger.error(f"Error copying {filename} from {source_file}: {e}")
                not_found_files.append(filename)
        else:
            not_found_files.append(filename)

    if copied_files:
        _logger.info(f"Successfully copied {len(copied_files)} XML files to {xml_input_dir}")
        _logger.debug(f"Copied files: {sorted(copied_files)}")
    if not_found_files:
        _logger.warning(
            f"{len(not_found_files)} files not found in letters-all: "
            f"{', '.join(sorted(not_found_files)[:10])}"
        )
        if len(not_found_files) > 10:
            _logger.debug(f"... and {len(not_found_files) - 10} more files")
        _logger.debug(f"All not found files: {sorted(not_found_files)}")

    return copied_files


def extract_brackets_to_excel() -> pd.DataFrame:
    """
    Extract the selected text nodes and their bracketed sequences and write them to Excel.

    Loads the selection (``load_metadata_dataframe``), copies the letters
    (``copy_selected_xml_files``), and for every selected row reads the text node, its bracketed
    sequences and its parent element from the copied letter. Rows whose text node is missing or
    has no ``]`` are skipped. Writes ``data/sampling/brackets_sample.xlsx``.

    Returns:
        A DataFrame with columns ID, Filename, XPath, Text_Content, Bracketed_Content (sequences
        separated by `` | ``), Parent_Element_Content and Language; an empty DataFrame if the
        selection cannot be loaded or nothing was extracted.
    """
    # Define paths from configuration constants
    xml_input_dir = _parent_dir / SAMPLING_XML_DIR
    output_file = _parent_dir / SAMPLING_OUTPUT_EXCEL
    metadata_path = _parent_dir / SAMPLING_METADATA_EXCEL
    letters_all_dir = _parent_dir / SAMPLING_LETTERS_ALL_DIR

    # Step 1: Load metadata DataFrame first (single source of truth)
    _logger.info(
        "Step 1: Loading the selection from editorial_interventions_brackets_analysis.xlsx..."
    )
    metadata_df = load_metadata_dataframe(metadata_path)

    if metadata_df is None or metadata_df.empty:
        _logger.error("Metadata file not found or invalid. Cannot proceed without metadata.")
        return pd.DataFrame()

    # Step 2: Copy selected XML files from letters-all to data/sampling/bracket_sample
    _logger.info(
        "Step 2: Copying selected XML files from letters-all to data/sampling/bracket_sample..."
    )
    copied_files = copy_selected_xml_files(metadata_df, letters_all_dir, xml_input_dir)
    if copied_files:
        _logger.info(f"Ready to process {len(copied_files)} copied files.")
        _logger.debug(f"Copied files: {sorted(copied_files)}")
    else:
        _logger.info(
            "No files were copied. Proceeding with existing files in data/sampling/bracket_sample."
        )

    # Step 3: Extract content directly from metadata rows
    _logger.info(f"Step 3: Extracting content for {len(metadata_df)} metadata rows...")
    extracted_data = []

    # Track which files we've already parsed (to avoid re-parsing)
    parsed_files: Dict[str, Tuple[etree._ElementTree, etree._Element]] = {}

    for idx, row in metadata_df.iterrows():
        filename = row["Filename"]
        xpath = row["XPath"]
        language = row.get("Language", "")

        # Get or parse XML file
        xml_file = xml_input_dir / filename
        if not xml_file.exists():
            _logger.warning(f"File {filename} not found at {xml_file}, skipping row {idx}")
            continue

        if filename not in parsed_files:
            try:
                tree = etree.parse(xml_file)
                root = tree.getroot()
                parsed_files[filename] = (tree, root)
                _logger.debug(f"Parsed XML file: {filename}")
            except (OSError, etree.XMLSyntaxError, etree.ParseError) as e:
                _logger.error(f"Error parsing {filename}: {e}, skipping row {idx}")
                continue
        else:
            tree, root = parsed_files[filename]

        # Evaluate XPath to get text content directly using cached root
        try:
            # Add namespace prefixes to XPath
            nsmap = TEI_NSMAP
            element_names = [
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
                "surplus",
            ]

            xpath_with_ns = xpath
            for element_name in element_names:
                xpath_with_ns = xpath_with_ns.replace(f"/{element_name}[", f"/tei:{element_name}[")
                xpath_with_ns = xpath_with_ns.replace(f"/{element_name}/", f"/tei:{element_name}/")
                xpath_with_ns = xpath_with_ns.replace(f"/{element_name}", f"/tei:{element_name}")

            # Ensure /text() is preserved correctly
            xpath_with_ns = xpath_with_ns.replace("/tei:text()", "/text()")

            # Evaluate XPath on the cached root
            text_nodes = root.xpath(xpath_with_ns, namespaces=nsmap)

            if not text_nodes:
                _logger.warning(f"No text node found for {filename} at {xpath}, skipping row {idx}")
                continue

            # Get text content from the first matching text node
            text_content = str(text_nodes[0]).strip() if text_nodes else ""
            _logger.debug(f"Extracted text for {filename} at {xpath}: {text_content[:100]}...")

            if not text_content or "]" not in text_content:
                _logger.warning(
                    f"No brackets found in text for {filename} at {xpath}, skipping row {idx}"
                )
                continue

            # Normalize text content
            text_content = normalize_text(text_content)

            # Extract letter number from filename
            letter_number = filename.replace("letter", "").replace(".xml", "")

            # Create simplified XPath starting from div
            simplified_xpath = create_simplified_xpath(xpath)

            # Create ID: L + letter_number + _ + simplified_xpath
            element_id = f"L{letter_number}_{simplified_xpath}"

            # Parent element from the copied source letter (not the gold standard)
            element_content = extract_element_content(filename, xpath, xml_input_dir)
            _logger.debug(
                f"Element content for {filename} at {xpath}: "
                f"{element_content[:100] if element_content else 'None'}..."
            )

            # Add to data list
            extracted_data.append(
                {
                    "ID": element_id,
                    "Filename": filename,
                    "XPath": xpath,
                    "Text_Content": text_content,
                    "Bracketed_Content": extract_bracketed_content(text_content),
                    "Parent_Element_Content": element_content,
                    "Language": language,
                }
            )
            _logger.debug(f"Added row {len(extracted_data)}: {element_id} from {filename}")

        except (ValueError, KeyError, IndexError, etree.XPathError) as e:
            _logger.error(
                f"Error extracting content for {filename} at {xpath}: {e}, skipping row {idx}"
            )
            continue

    # Create DataFrame
    if not extracted_data:
        _logger.warning("No data extracted. Nothing to save.")
        return pd.DataFrame()

    df = pd.DataFrame(extracted_data)
    _logger.info(f"Extracted {len(df)} rows from {len(metadata_df)} metadata rows.")

    # Export to Excel with formatting
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Brackets Extraction")

        # Get the worksheet
        worksheet = writer.sheets["Brackets Extraction"]

        # Set column widths (A=ID, B=Filename, C=Full_XPath, D=Text_Content, E=Bracketed_Content,
        # F=Parent_Element_Content, G=Language if filtered)
        max_id_length = max(len(str(id_val)) for id_val in df["ID"]) if df["ID"].any() else 20
        worksheet.column_dimensions["A"].width = min(max_id_length + 2, 24)
        worksheet.column_dimensions["B"].width = 15
        worksheet.column_dimensions["C"].width = 50
        worksheet.column_dimensions["D"].width = 50
        worksheet.column_dimensions["E"].width = 25
        worksheet.column_dimensions["F"].width = 80
        if "Language" in df.columns:
            worksheet.column_dimensions["G"].width = 20

        from openpyxl.styles import Alignment

        num_rows = len(df) + 1
        num_cols = len(df.columns)

        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                cell = worksheet.cell(row=row, column=col)
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    _logger.info(f"Extraction complete! Found {len(df)} text elements with brackets.")
    _logger.info(f"Results saved to: {output_file}")
    _logger.debug(f"First few entries:\n{df.head().to_string()}")

    return df


class TextSegmentExtractor:
    """Builds the text segments (context plus bracketed sequences) sent to the LLM."""

    def __init__(self) -> None:
        """Create an extractor; it holds no state."""
        pass

    def load_excel_data(self, excel_file: str) -> pd.DataFrame:
        """
        Load text segments from the Excel file with bracketed content.

        Args:
            excel_file: Path to the Excel file containing bracket data.

        Returns:
            A DataFrame filtered to rows with non-empty bracketed content, or empty DataFrame on
                error.
        """
        try:
            df = pd.read_excel(excel_file)
            # Filter for rows that have bracketed content
            df_with_brackets = df[df["Bracketed_Content"].notna() & (df["Bracketed_Content"] != "")]
            return df_with_brackets
        except FileNotFoundError as e:
            _logger.error(f"Excel file not found: {e}")
            return pd.DataFrame()
        except PermissionError as e:
            _logger.error(f"Permission denied accessing Excel file: {e}")
            return pd.DataFrame()
        except (ValueError, KeyError) as e:
            _logger.error(f"Error reading Excel file data: {e}")
            return pd.DataFrame()
        except Exception as e:
            _logger.error(f"Unexpected error loading Excel file: {e}")
            return pd.DataFrame()

    def extract_text_segments(
        self, excel_file: Optional[str] = None, df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text segments from Excel file or DataFrame for LLM analysis.

        Args:
            excel_file: Optional path to Excel file. If provided, loads data from file.
            df: Optional DataFrame. If provided, uses this DataFrame directly.
                At least one of excel_file or df must be provided.

        Returns:
            A list of dictionaries, each containing:
            - element_id: The element identifier
            - filename: The source XML filename
            - xpath: The XPath to the element
            - context: The windowed text content (configurable chars before/after brackets)
            - bracketed_sequences: List of bracketed sequences found
            - index: Sequential index (1-based)

        Raises:
            ValueError: If neither excel_file nor df is provided.
        """
        if df is not None:
            # Use provided DataFrame (e.g. from extract_brackets_to_excel)
            _logger.debug("Using provided DataFrame...")
        elif excel_file is not None:
            _logger.debug(f"Loading data from {excel_file}...")
            df = self.load_excel_data(excel_file)
        else:
            raise ValueError(
                "Provide df or excel_file. The Excel file is generated by "
                "extract_brackets_to_excel; do not expect it as input."
            )

        if df.empty:
            _logger.warning("No data found or error loading file.")
            return []

        _logger.info(f"Found {len(df)} text elements with bracketed content.")

        text_segments = []

        # Get XML input directory for parent context extraction
        xml_input_dir = _parent_dir / SAMPLING_XML_DIR

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            text_content = row["Text_Content"]
            bracketed_content = row["Bracketed_Content"]
            element_id = row["ID"]
            filename = row["Filename"]
            xpath = row.get("XPath", "unknown")

            # Split bracketed content by pipe separator
            bracketed_sequences = [
                seq.strip() for seq in bracketed_content.split("|") if seq.strip()
            ]

            _logger.debug(f"Processing element {i}/{len(df)}: {element_id}")
            _logger.debug(
                f"Found {len(bracketed_sequences)} bracketed sequences: {bracketed_sequences}"
            )

            # Determine context extraction strategy:
            # 1. persName/signed: look for ancestor p/closer/opener, then apply windowing
            # 2. dateline: use text node as-is (no parent lookup, no windowing)
            # 3. Insufficient context: use FULL parent element text (no windowing)
            # 4. Sufficient context: apply windowing to text node
            context_text = text_content
            use_full_text = False  # Flag to skip windowing

            if _is_inside_elements(xpath, ELEMENTS_NEEDING_ANCESTOR):
                # Inside persName or signed - look for ancestor p/closer/opener
                ancestor_xpath = _get_ancestor_xpath(xpath, CONTEXT_ANCESTOR_ELEMENTS)

                if ancestor_xpath:
                    xml_file = xml_input_dir / filename
                    if xml_file.exists():
                        # For closer/opener ancestors, add newlines after salute/signed/dateline
                        add_newlines = None
                        is_closer_opener = (
                            "/closer" in ancestor_xpath or "/opener" in ancestor_xpath
                        )
                        if is_closer_opener:
                            add_newlines = {"salute", "signed", "dateline"}

                        ancestor_text = extract_all_text_content(
                            xml_file,
                            ancestor_xpath,
                            remove_notes=True,
                            add_newlines_after=add_newlines,
                        )
                        if ancestor_text:
                            normalized_ancestor = normalize_text(ancestor_text)
                            # Replace paragraph break markers with actual newlines
                            normalized_ancestor = normalized_ancestor.replace(
                                PARAGRAPH_BREAK_MARKER, "\n\n"
                            )
                            normalized_text_node = normalize_text(text_content)

                            # Find where the text node content appears in ancestor
                            text_node_pos = normalized_ancestor.find(normalized_text_node)

                            if text_node_pos >= 0:
                                # Strip brackets from before/after, keep text node as-is
                                before_text = normalized_ancestor[:text_node_pos]
                                after_text = normalized_ancestor[
                                    text_node_pos + len(normalized_text_node) :
                                ]
                                before_stripped = _strip_all_brackets(before_text)
                                after_stripped = _strip_all_brackets(after_text)
                                context_text = (
                                    before_stripped + normalized_text_node + after_stripped
                                )
                                _logger.debug(
                                    f"Built ancestor context for {element_id}: "
                                    f"before={len(before_stripped)} chars, "
                                    f"after={len(after_stripped)} chars"
                                )
                            else:
                                # Text node not found - fall back to stripping all brackets
                                _logger.warning(
                                    f"Text node not found in ancestor for {element_id}, stripping "
                                    "all brackets"
                                )
                                context_text = _strip_all_brackets(normalized_ancestor)

                            # For closer/opener, use full text (let sentence windowing handle it)
                            if is_closer_opener:
                                use_full_text = True
                            _logger.debug(
                                f"Using ancestor context for {element_id}: {context_text[:100]}..."
                            )
                        else:
                            _logger.warning(
                                f"Could not extract ancestor context for {element_id}, using "
                                "original text"
                            )
                    else:
                        _logger.warning(f"XML file not found for {filename}, using original text")

            elif _is_inside_elements(xpath, {"dateline"}):
                # Dateline: use text node as-is, no windowing
                use_full_text = True
                _logger.debug(f"Dateline element {element_id}, using text as-is")

            elif not _has_sufficient_context(text_content, SAMPLING_CONTEXT_WINDOW_SIZE):
                # Insufficient context - get before/after from parent, strip their brackets
                parent_xpath = re.sub(r"/text\(\)(\[\d+\])?$", "", xpath)

                xml_file = xml_input_dir / filename
                if xml_file.exists():
                    parent_text = extract_all_text_content(
                        xml_file, parent_xpath, remove_notes=True
                    )
                    if parent_text:
                        normalized_parent = normalize_text(parent_text)
                        normalized_text_node = normalize_text(text_content)

                        # Find where the text node content appears in parent
                        text_node_pos = normalized_parent.find(normalized_text_node)

                        if text_node_pos >= 0:
                            # Split into before and after parts
                            before_text = normalized_parent[:text_node_pos]
                            after_text = normalized_parent[
                                text_node_pos + len(normalized_text_node) :
                            ]

                            # Strip ALL brackets from before/after (keep text node as-is)
                            before_stripped = _strip_all_brackets(before_text)
                            after_stripped = _strip_all_brackets(after_text)

                            # Concatenate: stripped_before + original_text_node + stripped_after
                            context_text = before_stripped + normalized_text_node + after_stripped
                            _logger.debug(
                                f"Built context for {element_id}: before={len(before_stripped)} "
                                f"chars, after={len(after_stripped)} chars"
                            )
                        else:
                            # Text node not found in parent - fall back to stripping all brackets
                            _logger.warning(
                                f"Text node not found in parent for {element_id}, stripping all "
                                "brackets"
                            )
                            context_text = _strip_all_brackets(normalized_parent)

                        use_full_text = True  # Use full parent text, no windowing
                        _logger.debug(
                            f"Using parent context for {element_id}: {context_text[:100]}..."
                        )
                    else:
                        _logger.warning(
                            f"Could not extract parent context for {element_id}, using original "
                            "text"
                        )
                else:
                    _logger.warning(f"XML file not found for {filename}, using original text")

            # Apply windowing only if we have sufficient context in text node or ancestor
            if use_full_text:
                # Use full text without windowing
                windowed_text = context_text
            elif bracketed_sequences and context_text:
                first_seq = bracketed_sequences[0]
                last_seq = bracketed_sequences[-1]

                # Find positions of first and last sequences
                first_pos = context_text.find(first_seq)
                last_pos = context_text.find(last_seq)

                if first_pos >= 0 and last_pos >= 0:
                    context_window = SAMPLING_CONTEXT_WINDOW_SIZE
                    start_pos = max(0, first_pos - context_window)
                    end_pos = min(len(context_text), last_pos + len(last_seq) + context_window)
                    windowed_text = context_text[start_pos:end_pos]
                else:
                    # Sequences not found in context - use full context
                    _logger.warning(
                        f"Bracketed sequences not found in context for {element_id}, using full "
                        "context"
                    )
                    windowed_text = context_text
            else:
                windowed_text = context_text

            _logger.debug(f"Windowed text for {element_id}: {windowed_text[:200]}...")

            # Final step: extract sentence window (from preceding dot to following dot)
            final_text = _extract_sentence_window(windowed_text, bracketed_sequences)
            _logger.debug(f"Sentence window for {element_id}: {final_text[:200]}...")

            # Create text segment data (index 1-based, sequential)
            text_segment = {
                "element_id": element_id,
                "filename": filename,
                "xpath": xpath,
                "context": final_text,
                "bracketed_sequences": bracketed_sequences,
                "index": i,
            }

            text_segments.append(text_segment)

        _logger.info("Text segment extraction complete!")
        _logger.info(f"Total text segments extracted: {len(text_segments)}")

        return text_segments


def _build_bracket_statistics_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary sheet from Bracket Statistics DataFrame.

    Creates a summary with columns: Language, Abbreviation, Correction, Omission,
    Supplement, Uncertainty, Redundancy, Total, Language Distribution. Structure:
    - One row per language (counts + language %)
    - "All" row (counts + 100%)
    - "(% of total)" row with category percentages (each / grand total) and Total 100%

    Args:
        stats_df: DataFrame containing bracket statistics with "Intervention_Type (Gold)" column.

    Returns:
        A DataFrame with summary statistics, or empty DataFrame if input is empty or missing
            required columns.
    """
    # Column is created by create_bracket_statistics_df
    intervention_col = "Intervention_Type (Gold)"
    cat_keys = ["Abbreviation", "Correction", "Omission", "Supplement", "Uncertainty", "Redundancy"]
    cols = ["Language"] + cat_keys + ["Total Sequences", "Language Distribution"]
    if stats_df.empty or intervention_col not in stats_df.columns:
        return pd.DataFrame(columns=cols)

    lang_counts = {}
    for _, row in stats_df.iterrows():
        lang = row.get("Language", "") or ""
        if lang not in lang_counts:
            lang_counts[lang] = {c: 0 for c in cat_keys}
            lang_counts[lang]["_total"] = 0
        lang_counts[lang]["_total"] += 1
        raw = str(row.get(intervention_col, "") or "")
        for part in (p.strip() for p in raw.split(";")):
            if part in cat_keys:
                lang_counts[lang][part] += 1

    langs = sorted(lang_counts.keys(), key=lambda x: (x == "", x))
    rows = []
    for lang in langs:
        d = lang_counts[lang]
        rows.append({**{c: d[c] for c in cat_keys}, "_total": d["_total"]})

    totals = {c: sum(r[c] for r in rows) for c in cat_keys}
    grand = sum(r["_total"] for r in rows)

    def pct(x: float, n: float) -> float:
        """Percentage of ``x`` in ``n``, rounded to 1 decimal; 0.0 if ``n`` is 0."""
        return round(100.0 * x / n, 1) if n else 0.0

    def pct_str(x: float, n: float) -> str:
        """``pct(x, n)`` with a ``%`` sign, e.g. ``"12.5%"``; ``"0%"`` for 0."""
        val = pct(x, n)
        return f"{val}%" if val else "0%"

    out_rows = []
    for lang, r in zip(langs, rows):
        out_rows.append(
            {
                "Language": lang if lang else "(empty)",
                **{c: r[c] for c in cat_keys},
                "Total Sequences": r["_total"],
                "Language Distribution": pct_str(r["_total"], grand),
            }
        )
    out_rows.append(
        {
            "Language": "All",
            **{c: totals[c] for c in cat_keys},
            "Total Sequences": grand,
            "Language Distribution": "100%",
        }
    )
    out_rows.append(
        {
            "Language": "(% of total)",
            **{c: pct_str(totals[c], grand) for c in cat_keys},
            "Total Sequences": "100%",
            "Language Distribution": "",
        }
    )
    return pd.DataFrame(out_rows, columns=cols)


def write_bracket_statistics_excel(
    stats_df: pd.DataFrame, summary_df: pd.DataFrame, stats_file: Path
) -> None:
    """Write the bracket statistics and their summary to bracket_statistics.xlsx.

    Args:
        stats_df: DataFrame from create_bracket_statistics_df.
        summary_df: DataFrame from _build_bracket_statistics_summary.
        stats_file: Path of the Excel file to write.
    """
    with pd.ExcelWriter(stats_file, engine="openpyxl") as writer:
        stats_df.to_excel(writer, index=False, sheet_name="Bracket Statistics")
        ws = writer.sheets["Bracket Statistics"]
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 50
        ws.column_dimensions["C"].width = 15
        ws.column_dimensions["D"].width = 22
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        from openpyxl.styles import Alignment

        for r in range(1, len(stats_df) + 2):
            for c in range(1, len(stats_df.columns) + 1):
                ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        ws2 = writer.sheets["Summary"]
        for col, w in [
            ("A", 14),
            ("B", 14),
            ("C", 12),
            ("D", 12),
            ("E", 14),
            ("F", 14),
            ("G", 10),
            ("H", 18),
        ]:
            ws2.column_dimensions[col].width = w
        ws2.freeze_panes = "A2"
        ws2.auto_filter.ref = ws2.dimensions
        from openpyxl.styles import PatternFill

        gray_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
        for r in range(1, len(summary_df) + 2):
            for c in range(1, 9):
                cell = ws2.cell(row=r, column=c)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                # Find the row index in the DataFrame (r-1 because Excel is 1-indexed, DataFrame is
                # 0-indexed)
                df_row_idx = r - 2  # r is 1-indexed Excel, -1 for header, so -2 for DataFrame index
                if df_row_idx >= 0 and df_row_idx < len(summary_df):
                    lang_val = summary_df.iloc[df_row_idx]["Language"]
                    # Gray background for percentage cells in "(% of total)" row (columns B-G, which
                    # are 2-6)
                    if lang_val == "(% of total)" and 2 <= c <= 7:
                        cell.fill = gray_fill
                    # Gray background for Language Distribution (column H, which is 8) for language
                    # rows (not "All")
                    elif lang_val not in ("All", "(% of total)") and c == 8:
                        cell.fill = gray_fill


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments (only ``--version``; the script takes no other options).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract bracketed editorial interventions from XML files and prepare data "
            "for LLM analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    Run the complete preprocessing workflow with default paths.

  %(prog)s --version
    Display version information.

The script performs three main steps:
  1. Extract bracketed content from XML files to Excel
  2. Create bracket statistics Excel with gold-standard intervention types
  3. Extract text segments for LLM analysis and save to JSON

Output files (in data/sampling/):
  - bracket_sample/: copies of the selected letters
  - brackets_sample.xlsx: Extracted bracket data
  - brackets_statistics.xlsx: Gold-standard intervention type per sequence, with a summary
  - brackets_sample.json: Text segments for LLM processing
        """.strip(),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version and exit.",
    )

    return parser.parse_args()


def main() -> int:
    """
    Run the three steps (see the module docstring) and write the outputs to ``data/sampling/``.

    Step 2 is skipped (with a warning) if the sample letters or the gold standard are missing.

    Returns:
        int: Exit code, 0 on success, 1 if no bracketed content was found
    """
    _logger.info("Starting Preprocessing Workflow")
    _logger.info("=" * 50)
    _logger.info(
        "This script extracts bracketed content from XML files and prepares data for LLM analysis."
    )
    _logger.info(f"Log file: {_log_file}")

    # Step 1: Extract brackets to Excel
    _logger.info("Step 1: Extracting bracketed content from XML files...")
    df = extract_brackets_to_excel()

    if df.empty:
        _logger.error("No bracketed content found. Exiting.")
        return 1

    _logger.info("=" * 50)

    # Step 2: Create bracket statistics Excel
    _logger.info(
        "Step 2: Creating bracket_statistics.xlsx with gold-standard intervention types..."
    )
    xml_input_dir = _parent_dir / SAMPLING_XML_DIR
    ref_dir = _parent_dir / SAMPLING_REFERENCES_GOLD_DIR
    stats_file = _parent_dir / SAMPLING_STATS_EXCEL
    if not xml_input_dir.exists():
        _logger.warning(f"Skip: {xml_input_dir} not found.")
    elif not ref_dir.exists():
        _logger.warning(f"Skip: {ref_dir} not found.")
    else:
        stats_df = create_bracket_statistics_df(df, xml_input_dir, ref_dir)
        summary_df = _build_bracket_statistics_summary(stats_df)
        write_bracket_statistics_excel(stats_df, summary_df, stats_file)
        _logger.info(f"Saved {stats_file} ({len(stats_df)} rows, summary: {len(summary_df)} rows).")

    _logger.info("=" * 50)

    # Step 3: Extract text segments for LLM
    _logger.info("Step 3: Creating text segments for LLM analysis...")

    # Initialize the extractor
    extractor = TextSegmentExtractor()

    # Extract text segments using the DataFrame from step 1
    text_segments = extractor.extract_text_segments(df=df)

    # Save text segments to a JSON file for the next step
    segments_file = _parent_dir / SAMPLING_SEGMENTS_JSON
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(text_segments, f, indent=2, ensure_ascii=False)

    _logger.info(f"Text segments saved to {segments_file}")
    _logger.info("=" * 50)
    _logger.info("Preprocessing complete!")
    _logger.info(f"Excel: {SAMPLING_OUTPUT_EXCEL}")
    _logger.info(f"Excel: {SAMPLING_STATS_EXCEL}")
    _logger.info(f"JSON: {SAMPLING_SEGMENTS_JSON}")
    _logger.info("Ready for LLM processing.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main())
