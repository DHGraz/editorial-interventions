"""
Utility modules for editorial interventions processing.

Shared by the sampling scripts (``data/sampling/``) and the evaluation (``evaluation/``); import
the submodules directly (``from utils.xml_utils import ...``). They are not imported here, so
importing ``utils`` stays free of side effects.

Submodules:
    xml_utils: XML/XPath helpers, text normalisation, the TEI namespace, Jing validation
    gold_standard_stats: Gold-standard intervention type per sampled bracketed sequence
    console_logging: Logging set-up for the entry-point scripts (messages to stderr)
"""

__all__ = ["console_logging", "gold_standard_stats", "xml_utils"]
