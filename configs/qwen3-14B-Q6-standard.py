"""
Config of the Qwen3-14B (standard mode) run
(`data/output/qwen3-14B-Q6/processing_20260301_021912/`).

This is a standalone config: llm_processing/config.py is NOT read when --config is used, so
every setting the engine needs for this run is defined here.

Usage (from the repo root or from llm_processing/ -- paths below are resolved relative to this
file, not to the working directory):
  python llm_processing/llm_processing.py --config configs/qwen3-14B-Q6-standard.py --workflow json

Dry run (no GPU needed): set ENABLE_API_CALLS = False below, run the command above, and compare
the resulting log/prompts.txt with
data/output/qwen3-14B-Q6/processing_20260301_021912/log/prompts.txt -- system message and all
user messages must be identical.
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
MODEL_NAME = "qwen3-14B-Q6"  # Alibaba Qwen3 (local) -- must contain "qwen"
TEMPERATURE = 0.6
# Not recorded in the run's metadata; the responses averaged ~51 output tokens/sequence
# (8687 output tokens / 169 sequences) in standard mode, so this leaves generous headroom
MAX_TOKENS = 3000
USE_GPU = True

QWEN_USE_THINKING = False  # standard mode (see qwen3-14B-Q6-thinking.py for thinking mode)

# ADJUST THIS if the model file lives elsewhere.
MODEL_PATH_QWEN3 = os.path.join(_ENGINE_DIR, "..", "..", "..", "models", "Qwen_Qwen3-14B-Q6_K.gguf")

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
OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "output")  # writes to data/output/qwen3-14B-Q6/
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
