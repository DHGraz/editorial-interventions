"""
Write LLM-generated TEI encodings back into the original TEI letter files.

Usage (from ``evaluation/``)::

    python update_xml_files.py <run>     # e.g. qwen3-14B-Q6; asks which processing run to use
    python update_xml_files.py           # asks for the run first

Reads ``data/output/<run>/processing_<timestamp>/xml_update_mapping.json`` and writes the edited
letters to ``xml_edited/<run>/`` (the folder is emptied first).

Replacements within a text node are applied in descending text-node-index order so that an
earlier replacement never shifts the position of a later one.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from lxml import etree

# Repository root on sys.path, as in eval.py. "config" is evaluation/config.py (this script's
# folder is on sys.path; there is no config.py at the root).
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import config  # noqa: E402

from utils.console_logging import configure_console_logging  # noqa: E402
from utils.xml_utils import TEI_NS, TEI_NSMAP, load_xml_mapping  # noqa: E402


def discover_available_runs() -> List[str]:
    """
    List the runs in ``data/output/`` with a ``processing_*/xml_update_mapping.json``.

    Returns:
        list: Run folder names, sorted
    """
    project_root = Path(__file__).parent.parent
    output_base = project_root / config.LLM_OUTPUT_DIR
    runs = []
    if output_base.exists():
        for run_dir in output_base.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                if any(
                    (p / config.MAPPING_FILENAME).exists() for p in run_dir.glob("processing_*")
                ):
                    runs.append(run_dir.name)
    return sorted(runs)


def select_run() -> Optional[str]:
    """
    Ask on the console which run to write back.

    Returns:
        str: The chosen run folder name; None if there is none or the user cancels
    """
    runs = discover_available_runs()
    if not runs:
        print("Error: no runs with xml_update_mapping.json found in data/output/.", file=sys.stderr)
        return None

    print("Available runs:")
    for i, run in enumerate(runs, 1):
        print(f"{i:2d}. {run}")

    while True:
        try:
            choice = input(f"\nSelect a run (1-{len(runs)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(runs):
                return runs[choice_num - 1]
            print(f"Invalid choice. Please enter a number between 1 and {len(runs)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def select_mapping_file(run: str) -> Optional[Path]:
    """
    Ask which ``processing_<timestamp>/xml_update_mapping.json`` of the run to use.

    The files are listed newest first by modification time; Enter or the last option takes the
    newest.

    Args:
        run: Run folder name in ``data/output/``

    Returns:
        Path: The chosen mapping file; None if there is none or the user cancels
    """
    project_root = Path(__file__).parent.parent
    run_dir = project_root / config.LLM_OUTPUT_DIR / run
    mapping_files = sorted(
        (
            p / config.MAPPING_FILENAME
            for p in sorted(run_dir.glob("processing_*"))
            if (p / config.MAPPING_FILENAME).exists()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mapping_files:
        print(f"Error: no xml_update_mapping.json found for run '{run}'.", file=sys.stderr)
        return None

    print(f"\nAvailable mapping files for {run}:")
    for i, path in enumerate(mapping_files, 1):
        print(f"{i:2d}. {path.parent.name}")
    print(f"{len(mapping_files) + 1}. Use latest automatically (default)")

    while True:
        try:
            choice = input(
                f"\nSelect a mapping file (1-{len(mapping_files) + 1}, "
                f"or press Enter for default): "
            ).strip()
            if choice == "":
                return mapping_files[0]
            choice_num = int(choice)
            if 1 <= choice_num <= len(mapping_files):
                return mapping_files[choice_num - 1]
            if choice_num == len(mapping_files) + 1:
                return mapping_files[0]
            print(f"Please enter a number between 1 and {len(mapping_files) + 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def create_xml_edited_folder(edited_folder: Path) -> Path:
    """
    Create the run's output folder if it does not exist.

    Args:
        edited_folder: ``xml_edited/<run>/``

    Returns:
        Path: The folder
    """
    if not edited_folder.exists():
        edited_folder.mkdir(parents=True)
        print(f"Created directory: {edited_folder}")
    return edited_folder


def clear_xml_edited_folder(edited_folder: Path) -> Path:
    """
    Delete all files in the run's output folder before writing new ones.

    Args:
        edited_folder: ``xml_edited/<run>/``

    Returns:
        Path: The folder
    """
    if edited_folder.exists():
        files_removed = 0
        for file_path in edited_folder.iterdir():
            file_path.unlink()
            files_removed += 1
        if files_removed:
            print(f"Cleared {files_removed} file(s) from {edited_folder}")
    return edited_folder


def convert_xpath_to_namespaced(xpath: str) -> str:
    """
    Add the tei: prefix to every element step in an XPath expression.

    ``/text()`` steps are removed first.

    Args:
        xpath: XPath without namespace prefixes

    Returns:
        str: The prefixed XPath
    """
    element_xpath = xpath.replace("/text()", "")
    return re.sub(r"/([A-Za-z][\w-]*)", r"/tei:\1", element_xpath)


def _get_text_node_index(element_info: Dict) -> int:
    """
    Return n of the ``text()[n]`` step in an element's XPath (0 if there is none).

    Args:
        element_info: A mapping entry with an ``xpath``

    Returns:
        int: The text node index
    """
    match = re.search(r"text\(\)\[(\d+)\]", element_info["xpath"])
    return int(match.group(1)) if match else 0


def _fuzzy_pattern(seq: str) -> "re.Pattern":
    """
    Build a regex for seq that tolerates the source XML's line-wrap whitespace and
    typographic quotes where the mapping's original_item has a plain space/apostrophe.

    Args:
        seq: The bracketed sequence from the mapping

    Returns:
        re.Pattern: Pattern in which each space matches any whitespace and ``'`` also matches ``’``
    """
    parts = []
    for ch in seq:
        if ch.isspace():
            parts.append(r"\s+")
        elif ch == "'":
            parts.append("['’]")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts))


def replace_element_text_with_mixed_content(element: etree._Element, new_text: str) -> None:
    """
    Set element.text, parsing new_text as TEI markup if it contains any.

    If ``new_text`` contains ``<`` and ``>`` but does not parse as XML, it is set as plain text, so
    the markup is escaped in the output file.

    Args:
        element: The element whose leading text is replaced
        new_text: Text with the encodings inserted
    """
    if "<" in new_text and ">" in new_text:
        try:
            temp_element = etree.fromstring(f"<temp xmlns='{TEI_NS}'>{new_text}</temp>")
            element.text = temp_element.text
            for i, child in enumerate(temp_element):
                element.insert(i, child)
            return
        except etree.XMLSyntaxError:
            pass
    element.text = new_text


def replace_tail_text_with_mixed_content(
    parent_element: etree._Element, child_element: etree._Element, new_text: str
) -> None:
    """
    Set child_element.tail, parsing new_text as TEI markup if it contains any.

    New child elements are inserted after ``child_element``. As in
    ``replace_element_text_with_mixed_content``, text that does not parse is set as plain text.

    Args:
        parent_element: Parent of ``child_element``
        child_element: The element whose tail text is replaced
        new_text: Text with the encodings inserted
    """
    if "<" in new_text and ">" in new_text:
        try:
            temp_element = etree.fromstring(f"<temp xmlns='{TEI_NS}'>{new_text}</temp>")
            child_element.tail = temp_element.text
            child_index = list(parent_element).index(child_element)
            for i, new_child in enumerate(temp_element):
                parent_element.insert(child_index + 1 + i, new_child)
            return
        except etree.XMLSyntaxError:
            pass
    child_element.tail = new_text


def update_xml_file(
    filename: str, elements_to_update: List[Dict], xml_input_dir: Path, xml_edited_dir: Path
) -> Optional[int]:
    """
    Apply all replacements for one TEI file and write the result to xml_edited_dir.

    The file is parsed with lxml's recovering parser. Elements are processed in descending order of
    their text node index. In each text node, each replacement is applied to the first occurrence of
    its ``original_item``, or, if there is none, to a fuzzy match (see ``_fuzzy_pattern``). The file
    is written (pretty-printed, with XML declaration) even if no replacement was applied.

    Args:
        filename: Letter file name
        elements_to_update: The file's entries of the mapping
        xml_input_dir: Folder of the source letters
        xml_edited_dir: Output folder

    Returns:
        int: Number of replacements applied; None if the input file is missing
    """
    input_file = xml_input_dir / filename
    output_file = xml_edited_dir / filename

    if not input_file.exists():
        print(f"Warning: input file {input_file} not found, skipped", file=sys.stderr)
        return None

    parser = etree.XMLParser(recover=True)
    tree = etree.parse(str(input_file), parser)
    root = tree.getroot()

    total_replacements_made = 0

    # Process text nodes from the end of the file backwards, so a replacement never shifts
    # the position of a text node still to be processed.
    sorted_elements = sorted(elements_to_update, key=_get_text_node_index, reverse=True)

    for element_info in sorted_elements:
        xpath = element_info["xpath"]
        element_id = element_info["element_id"]
        print(
            f"  Processing element {element_id} with {len(element_info['replacements'])} "
            "replacements"
        )

        if "/text()" not in xpath:
            print(f"Warning: skipping non-text-node xpath: {xpath}", file=sys.stderr)
            continue

        element_xpath = re.sub(r"/text\(\)(\[\d+\])?", "", xpath)
        namespace_xpath = convert_xpath_to_namespaced(element_xpath)
        elements = root.xpath(namespace_xpath, namespaces=TEI_NSMAP)
        if not elements:
            print(f"Warning: parent element not found for xpath: {xpath}", file=sys.stderr)
            continue
        element = elements[0]

        match = re.search(r"text\(\)\[(\d+)\]", xpath)
        text_index = int(match.group(1)) if match else 1

        all_text_nodes = []
        if element.text:
            all_text_nodes.append(("element_text", element.text, None))
        for child in element:
            if child.tail:
                all_text_nodes.append(("tail_text", child.tail, child))

        if text_index > len(all_text_nodes):
            print(f"Warning: text node {text_index} not found for xpath: {xpath}", file=sys.stderr)
            continue

        node_kind, original_text, tail_owner = all_text_nodes[text_index - 1]
        updated_text = original_text
        element_replacements_made = 0

        for replacement in element_info["replacements"]:
            bracketed_seq = replacement["original_item"]
            tei_encoding = replacement["tei_encoding"]
            if bracketed_seq in updated_text:
                updated_text = updated_text.replace(bracketed_seq, tei_encoding, 1)
                element_replacements_made += 1
                print(f"    Replaced: {bracketed_seq} -> {tei_encoding}")
            else:
                loose_match = _fuzzy_pattern(bracketed_seq).search(updated_text)
                if loose_match:
                    updated_text = (
                        updated_text[: loose_match.start()]
                        + tei_encoding
                        + updated_text[loose_match.end() :]
                    )
                    element_replacements_made += 1
                    print(f"    Replaced (fuzzy match): {bracketed_seq} -> {tei_encoding}")
                else:
                    print(
                        f"Warning: '{bracketed_seq}' not found in the text of {xpath}, skipped",
                        file=sys.stderr,
                    )

        if element_replacements_made:
            if node_kind == "element_text":
                replace_element_text_with_mixed_content(element, updated_text)
            else:
                replace_tail_text_with_mixed_content(element, tail_owner, updated_text)
            total_replacements_made += element_replacements_made

    tree.write(str(output_file), encoding="utf-8", xml_declaration=True, pretty_print=True)
    print(f"  Successfully updated {filename} with {total_replacements_made} replacements")
    return total_replacements_made


def update_all_xml_files(run: Optional[str] = None) -> int:
    """
    Write back all encodings of one run.

    Args:
        run: Run folder name in ``data/output/``; None asks on the console

    Returns:
        int: Exit code, 0 on success, 1 if no run or mapping file was chosen or an input is missing
    """
    print("XML Write-Back Tool")
    print("=" * 50)

    if run is None:
        run = select_run()
        if run is None:
            return 1

    mapping_file = select_mapping_file(run)
    if mapping_file is None:
        return 1

    xml_update_map = load_xml_mapping(mapping_file)
    if not xml_update_map:
        print(
            f"Error: the mapping file {mapping_file} is empty or could not be loaded.",
            file=sys.stderr,
        )
        return 1

    project_root = Path(__file__).parent.parent
    xml_input_dir = project_root / config.XML_INPUT_DIR
    xml_edited_dir = project_root / config.XML_EDITED_DIR / run

    if not xml_input_dir.exists():
        print(f"Error: {xml_input_dir} directory not found.", file=sys.stderr)
        return 1

    clear_xml_edited_folder(xml_edited_dir)
    create_xml_edited_folder(xml_edited_dir)

    total_files = len(xml_update_map)
    successful_updates = 0
    total_replacements = 0

    print(f"\nMapping file: {mapping_file}")
    print(f"Files to process: {total_files}")
    print("=" * 50)

    for filename, elements_to_update in xml_update_map.items():
        print(f"\nProcessing {filename}:")
        applied = update_xml_file(filename, elements_to_update, xml_input_dir, xml_edited_dir)
        if applied is not None:
            successful_updates += 1
            total_replacements += applied

    print("\n" + "=" * 50)
    print("XML Write-Back Summary:")
    print(f"Files processed: {total_files}")
    print(f"Files successfully updated: {successful_updates}")
    print(f"Total replacements made: {total_replacements}")
    print(f"Updated files saved to: {xml_edited_dir}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``

    Returns:
        argparse.Namespace: ``run`` (None if not given)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The menus read their answers from stdin, so they can be piped in, e.g.\n"
        "  printf '\\n' | python update_xml_files.py gpt-5.2-2025-12-11   "
        "(Enter = default processing run)",
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="run folder in data/output/ (e.g. qwen3-14B-Q6); if omitted, a menu lists the "
        "available ones",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    configure_console_logging()
    sys.exit(update_all_xml_files(args.run))
