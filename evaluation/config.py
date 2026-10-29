"""
Paths used by ``evaluation/eval.py``, ``evaluation/update_xml_files.py`` and
``evaluation/language_scores.py``.

Each path is relative to the repository root (the scripts join it with the root), or absolute.
The scripts import this file as the module ``config``.
"""

# Stored output of the LLM runs (made with the engine ../llm-processing):
# <run>/processing_<timestamp>/ per processing run; read by eval.py and update_xml_files.py
LLM_OUTPUT_DIR = "data/output"

# File name, in each processing_<timestamp>/ folder, of the model's TEI encoding per bracketed
# sequence
MAPPING_FILENAME = "xml_update_mapping.json"

# Sampled text elements with their bracketed sequences (from data/sampling/sampling.py), read by
# eval.py
EVALUATION_INPUT_FILE = "data/input/brackets_sample.xlsx"

# Directory containing gold-standard reference XML files
REFERENCES_DIR = "data/references/gold"

# Source letters: eval.py reads the input elements from here, update_xml_files.py writes the model
# encodings into copies of them
XML_INPUT_DIR = "data/input/bracket_sample"

# Output of update_xml_files.py (xml_edited/<run>/); eval.py does not need it, it rebuilds the
# model output from the mapping file
XML_EDITED_DIR = "xml_edited"

# Gold-standard intervention type per bracketed sequence (from data/sampling/sampling.py)
BRACKETS_STATISTICS_FILE = "data/sampling/brackets_statistics.xlsx"

# TEI schema and Jing validator for the TEI validity check (D2)
TEI_SCHEMA_FILE = "resources/schema/tei_all.rng"
JING_JAR_FILE = "resources/tools/jing-RELEASE220.jar"

# Estimated cost of local inference (electricity plus amortised GPU cost), which eval.py adds as
# estimated_cost_usd to the metadata copy of every run listed in MEASURED_POWER_W
# (see estimate_local_cost_usd in eval.py). The institutional figures are in EUR and converted at
# 1.08 USD/EUR. They are electricity and hardware costs from Austria in 2025; for your own runs, use
# your own costs and measured power.
GPU_PRICE_USD = 1944.0  # EUR 1,800
ELECTRICITY_RATE_USD_PER_KWH = 0.13  # EUR 0.12
AMORTIZATION_YEARS = 3
USAGE_HOURS_PER_YEAR = 876  # measured annual utilisation of the GPU

# Measured average GPU power draw during inference, in watts, per processing run
# (<model instance>/processing_<timestamp> in LLM_OUTPUT_DIR)
MEASURED_POWER_W = {
    "qwen3-14B-Q6/processing_20260301_021912": 180,  # standard mode
    "qwen3-14B-Q6/processing_20260301_022635": 200,  # thinking mode
    "olmo2-32B-instruct-Q4/processing_20260927_114615": 180,
}
