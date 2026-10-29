"""
Helper functions for ``evaluation/eval.py``.

``eval.py`` loads this file directly by its path (``importlib``), not through the package, because
the root ``utils`` package has the same name as ``evaluation/utils``.
"""

import logging
import re
from typing import Any, Dict, Optional

from lxml import etree

logger = logging.getLogger(__name__)


def normalize_apostrophes(text: str) -> str:
    """
    Normalize different character variants to a standard form.

    Replaces various Unicode apostrophe and ellipsis characters with standard
    forms for consistent text comparison.

    Args:
        text: Input text that may contain various apostrophe/ellipsis types.

    Returns:
        Text with all character variants replaced with standard forms.
    """
    if not text:
        return text
    # Replace various apostrophe types with standard straight apostrophe
    text = text.replace(chr(8217), "'")  # right single quotation mark (')
    text = text.replace(chr(8216), "'")  # left single quotation mark (')
    text = text.replace(chr(180), "'")  # acute accent (´)
    text = text.replace(chr(96), "'")  # grave accent (`)
    # Normalize ellipsis: Unicode ellipsis (…) to multiple dots
    text = text.replace("…", "..")  # ellipsis to 2 dots
    text = text.replace("....", "..")  # 4 dots to 2 dots
    text = text.replace("...", "..")  # 3 dots to 2 dots
    return text


def create_parent_xpath(xpath: str) -> str:
    """
    Create parent element XPath by removing /text() or text()[n] suffix.

    Args:
        xpath: Full XPath that may end with /text() or /text()[n].

    Returns:
        Parent element XPath with text node suffix removed.
    """
    parent_xpath = xpath
    if "/text()[" in xpath:
        # Handle text()[n] pattern first (more specific)
        parent_xpath = re.sub(r"/text\(\)\[\d+\]$", "", xpath)
    elif xpath.endswith("/text()"):
        # Handle simple /text() pattern
        parent_xpath = xpath[:-7]  # Remove '/text()'
    return parent_xpath


def remove_note_elements(xml_content: Optional[str]) -> Optional[str]:
    """
    Remove ``note`` elements (without namespace) from an XML string, keeping their tail text
    where the note was (appended to the previous sibling's tail, or to the parent's text if the
    note is the first child).

    Args:
        xml_content: XML string that may contain note elements, or None.

    Returns:
        The XML without notes; the original content if it cannot be parsed (a warning is logged);
        None if the input is None.
    """
    if xml_content is None:
        return None
    try:
        # Parse the XML content
        tree = etree.fromstring(xml_content)

        # Find all note elements and remove them
        notes_to_remove = tree.xpath(".//note")
        for note in notes_to_remove:
            parent = note.getparent()
            if parent is not None:
                # Get the tail text (text that comes after the note element)
                tail_text = note.tail if note.tail else ""
                # Find the previous sibling while the note is still in the tree
                prev_sibling = note.getprevious()
                # Remove the note element
                parent.remove(note)
                # If there was tail text, we need to preserve it
                # Add it to the previous sibling's tail or to the parent's text
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

        # Return the cleaned XML as string
        return etree.tostring(tree, encoding="unicode", method="xml")
    except Exception as e:
        logger.warning(f"Could not remove note elements, keeping the XML unchanged: {e}")
        return xml_content


def create_empty_evaluation_result(
    filename: str,
    bracketed_content: str,
    xpath: str,
    parent_xpath: Optional[str] = None,
    gold_exists: bool = False,
    llm_file_exists: bool = True,
) -> Dict[str, Any]:
    """
    Create the result row for a sequence that cannot be evaluated (e.g. missing data).

    Args:
        filename: The XML filename being evaluated.
        bracketed_content: The bracketed sequence being evaluated.
        xpath: The full XPath to the text node.
        parent_xpath: The XPath to the parent element (defaults to xpath if not provided).
        gold_exists: Whether the gold standard file exists.
        llm_file_exists: Whether the LLM output exists (default True: it is rebuilt from the
            mapping).

    Returns:
        Dictionary with all evaluation fields set to default/empty values.
    """
    if parent_xpath is None:
        parent_xpath = xpath

    return {
        "Filename": filename,
        "Bracketed_Content": bracketed_content,
        "Processed": False,
        "XML_wellformed (D0)": False,
        "Bracketed_Content_Normalized": None,
        "LLM_Output_Normalized": None,
        "Source_Fidelity (D1)": None,
        "LLM_corr": None,
        "Gold_corr": None,
        "Corr_Match": None,
        "TEI_valid (D2)": False,
        "Project_valid (D2)": "N/A",
        "LLM_Classification": "None",
        "Gold_Classification": "None",
        "Classification_Match (D4a)": True,  # Both are "None" so they match
        "LLM_encoding": None,
        "LLM_element": "No elements",
        "Gold_element": "No elements",
        "Element_Match (D4b)": False,
        "Element_Accuracy_Percentage (D4b)": 0.0,
        "Flexible_Element_Match (D4b)": False,
        "Type-Element_Match": False,
        "Input_Element": None,
        "Output_Element": None,
        "Gold_Standard_Element": None,
        "Full_Element_Diff (input - llm output)": None,
        "Full_Element_Diff (input - goldstandard)": None,
        "Full_Element_Match": None,
        "Full_Element_Accuracy_Percentage": None,
        "Full_Flexible_Element_Match": None,
        "Input_Content_Normalized": None,
        "Output_Content_Normalized": None,
        "String_Comparison": False,
        "String_Comparison (no WS)": False,
        "LLM_File_Exists": llm_file_exists,
        "Gold_File_Exists": gold_exists,
        "Xpath (Text)": xpath,
        "Xpath (Parent Element)": parent_xpath,
    }
