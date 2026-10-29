"""
Config of the OLMo2-32B-Instruct run
(`data/output/olmo2-32B-instruct-Q4/processing_20260927_114615/`).

This is a standalone config: llm_processing/config.py is NOT read when --config is used, so
every setting the engine needs for this run is defined here.

Usage (from the repo root or from llm_processing/ -- paths below are resolved relative to this
file, not to the working directory):
  python llm_processing/llm_processing.py --config configs/olmo2-32B-instruct-Q4.py --workflow json

Dry run (no GPU needed): set ENABLE_API_CALLS = False below, run the command above, and compare
the resulting log/prompts.txt with
data/output/olmo2-32B-instruct-Q4/processing_20260927_114615/log/prompts.txt.
"""

import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CONFIG_DIR)
_ENGINE_DIR = os.path.join(_REPO_ROOT, "llm_processing")

# =============================================================================
# GENERAL
# =============================================================================
ENABLE_API_CALLS = True  # False = dry run (prompts only, no model load, no GPU needed)

# =============================================================================
# MODEL
# =============================================================================
# AI2 OLMo2 32B (local) -- must contain "olmo" (case-insensitive)
MODEL_NAME = "olmo2-32B-instruct-Q4"
TEMPERATURE = 0.6
MAX_TOKENS = 3000
USE_GPU = True

# ADJUST THIS if the model file lives elsewhere.
MODEL_PATH_OLMO2 = os.path.join(
    _ENGINE_DIR, "..", "..", "..", "models", "OLMo-2-0325-32B-Instruct-Q4_K_S.gguf"
)

# llama.cpp renders the chat template stored in the model file (separate system / user /
# assistant turns).
OLMO_PROMPT_FORMAT = "chat_template"

# Fixed sampling seed, so the run is repeatable.
SEED = 42

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
# writes to data/output/olmo2-32B-instruct-Q4/
OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "output")
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
