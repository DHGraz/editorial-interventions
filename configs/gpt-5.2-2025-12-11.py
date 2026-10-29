"""
Config of the GPT-5.2 run
(`data/output/gpt-5.2-2025-12-11/`, `processing_20260301_015947`)

This is a standalone config: llm_processing/config.py is NOT read when --config is used, so
every setting the engine needs for this run is defined here.

Usage (from the repo root or from llm_processing/ -- paths below are resolved relative to this
file, not to the working directory):
    python llm_processing/llm_processing.py --config configs/gpt-5.2-2025-12-11.py --workflow json

Dry run (no API calls, no cost): set ENABLE_API_CALLS = False below, run the command above, and
compare the resulting log/prompts.txt with
data/output/gpt-5.2-2025-12-11/processing_20260301_015947/log/prompts.txt -- system message and
all user messages must be identical.
"""

import os
import sys

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CONFIG_DIR)
_ENGINE_DIR = os.path.join(_REPO_ROOT, "llm_processing")

# =============================================================================
# API KEYS
# =============================================================================
if _ENGINE_DIR not in sys.path:
    sys.path.append(_ENGINE_DIR)
try:
    import keys

    OPENAI_API_KEY = keys.OPENAI
except (ImportError, AttributeError):
    OPENAI_API_KEY = None

# =============================================================================
# GENERAL
# =============================================================================
ENABLE_API_CALLS = True  # False = dry run (prompts only, no API calls, no cost)

# =============================================================================
# MODEL
# =============================================================================
MODEL_NAME = "gpt-5.2-2025-12-11"
TEMPERATURE = 0.3

MAX_TOKENS = 7000
USE_GPU = False

# GPT-5.2 list prices in USD per million tokens, used for estimated_cost_usd; for new runs,
# check the provider's current prices
INPUT_PRICE_USD_PER_MILLION = 1.75
OUTPUT_PRICE_USD_PER_MILLION = 14.00

# =============================================================================
# PROMPT
# =============================================================================
PROMPT_VERSION = "editorial_interventions"
PROMPT_DIR = os.path.join(_REPO_ROOT, "data", "prompts", PROMPT_VERSION)
USER_MESSAGE = "user_message.txt"

# =============================================================================
# INPUT / OUTPUT
# =============================================================================
INPUT_TYPE = "json"
INPUT_PATH = os.path.join(_REPO_ROOT, "data", "input", "brackets_sample.json")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "output")  # writes to data/output/gpt-5.2-2025-12-11/
OUTPUT_EXTENSION = ".json"
JSON_OUTPUT_MODE = "json-array"

JSON_PROCESSING_MODE = "key_extraction"
JSON_EXTRACTION_TYPE = "tei_encoding"
KEY_EXTRACTION_OUTPUT_FORMAT = "xml_mapping"

JSON_CONTEXT_KEY = "context"
JSON_ITEMS_KEY = "bracketed_sequences"
JSON_CONTEXT_LABEL = "Context"
JSON_ITEMS_LABEL = "Bracketed Sequences"

JSON_OUTPUT_TEI_FIELD = "tei"
JSON_OUTPUT_TYPE_FIELD = "type"
JSON_OUTPUT_EXPLANATION_FIELD = None

XML_MAPPING_FILENAME_KEY = "filename"
XML_MAPPING_ELEMENT_ID_KEY = "element_id"
XML_MAPPING_XPATH_KEY = "xpath"
