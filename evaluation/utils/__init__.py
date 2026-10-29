"""
Utility modules for evaluation processing.
"""

from .evaluation_utils import (
    create_empty_evaluation_result,
    create_parent_xpath,
    normalize_apostrophes,
    remove_note_elements,
)

__all__ = [
    "normalize_apostrophes",
    "create_empty_evaluation_result",
    "create_parent_xpath",
    "remove_note_elements",
]
