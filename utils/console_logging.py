"""
Console logging for the entry-point scripts.

The modules in ``utils/`` and ``evaluation/utils/`` report problems through ``logging`` and do not
configure it. Each entry-point script calls ``configure_console_logging()`` once at start-up, so
these messages reach stderr with the same ``Warning:``/``Error:`` prefixes as the scripts' own
messages. ``data/sampling/sampling.py`` configures its own logging instead.
"""

import logging
import sys

_PREFIXES = {logging.WARNING: "Warning: ", logging.ERROR: "Error: ", logging.CRITICAL: "Error: "}


class _PrefixFormatter(logging.Formatter):
    """Format a record as its message, prefixed with ``Warning:`` or ``Error:``."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format one record.

        Args:
            record: The log record

        Returns:
            str: The message with its level prefix (no prefix below WARNING)
        """
        return _PREFIXES.get(record.levelno, "") + super().format(record)


def configure_console_logging(level: int = logging.WARNING) -> None:
    """
    Send log messages of ``level`` and above to stderr.

    Configures the root logger (``logging.basicConfig``), so it has no effect if logging was
    already configured.

    Args:
        level: Lowest level shown (default: warnings and errors)
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_PrefixFormatter("%(message)s"))
    logging.basicConfig(level=level, handlers=[handler])
