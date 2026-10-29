# LLM Processing for TEI XML Generation

Automated TEI (Text Encoding Initiative) XML generation from plaintext sources using Large Language Models.

## Overview

This project processes plaintext files and generates structured output (TEI XML or JSON). It supports multiple LLM providers and allows for flexible prompt engineering and configuration.

**Key Features:**
- **Multi-model support**: OpenAI GPT, Anthropic Claude, Alibaba Qwen, OLMo
- **Flexible output formats**: Generate TEI-XML, RDF-XML, or JSON from text or JSON inputs
- **Flexible configuration**: Easily switch between models, prompts, parameters, and output formats
- **Configurable JSON processing**: Adapt to any JSON structure by configuring key mappings
- **Comprehensive metrics**: Token usage tracking, cost estimation, processing time
- **GPU monitoring**: Track GPU utilization for local models
- **Batch processing**: Process multiple text files with error handling and logging
- **Test mode**: Generate prompts without making API calls for development

## Using the Engine in Your Own Project

The engine is meant to be included in other projects, e.g. as a git subtree in a folder `llm_processing/`.
Keep everything project-specific outside that folder, so it stays identical to this repository and updates stay
clean:

```
your-project/
├── llm_processing/            # this engine, unchanged
├── configs/
│   └── my_run.py              # copy of llm_processing/config.py, adjusted
└── data/
    ├── prompts/my_prompt/     # prompt.txt, encoding_rules.txt, ...
    ├── input/
    └── output/
```

1. Copy `config.py` into your project (e.g. `configs/my_run.py`) and set `MODEL_NAME`, `PROMPT_DIR`, `INPUT_PATH`,
   `OUTPUT_DIR` and the workflow settings. Resolve paths relative to the config file, so the run works from any
   directory:
   ```python
   _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   PROMPT_DIR = os.path.join(_PROJECT_ROOT, "data", "prompts", PROMPT_VERSION)
   INPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "input", "my_input.json")
   OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "output")
   ```
   `keys.py` stays in `llm_processing/`; add that folder to `sys.path` in your config before `import keys`.
2. Pass the file with `--config` (or the `LLM_PROCESSING_CONFIG` environment variable). The engine's own
   `config.py` is then not read at all:
   ```bash
   python llm_processing/llm_processing.py --config configs/my_run.py --workflow json
   ```

The engine's own `config.py` is a template for exactly this layout: its paths point to `data/prompts/`,
`data/input/` and `data/output/` in the folder that contains `llm_processing/`. `ENABLE_API_CALLS` is `False` by
default, so a first run only builds the prompts and writes them to
`{OUTPUT_DIR}/<model>/processing_<timestamp>/log/prompts.txt`; no model is called and nothing is paid. Set it to
`True` and add your API key (see Installation) to send them to the model.

In the rest of this README, "`config.py`" means whichever configuration file you run with.

## Supported Workflows

### 1. Text Processing Workflow
Processes complete plaintext files into structured output (XML or JSON). This workflow:
- Takes plaintext files from a configured directory
- Generates structured output based on configuration
- Supports batch processing of multiple files
- Saves outputs with comprehensive logging
- Configurable output format via `OUTPUT_EXTENSION` and `XML_OUTPUT_TYPE` settings

**Input**: Plaintext files (`.txt` files in a directory)

**Output Options**:
- **XML files** (`.xml`): XML-based encodings with two types:
  - **TEI-XML** (`XML_OUTPUT_TYPE = "tei"`): Traditional text encoding with full TEI XML structure
  - **RDF-XML** (`XML_OUTPUT_TYPE = "rdf"`): RDF-XML semantic data output
- **JSON files** (`.json`): Plaintext to JSON workflow with two modes:
  - **Raw mode**: Individual JSON objects per input file
  - **JSON-array mode**: Combined JSON array file with all outputs

### 2. JSON Processing Workflow
Processes JSON files with two modes:

#### a) Key Extraction Mode
Extracts and analyzes specific keys from JSON objects. This mode supports two workflow types configured via `JSON_EXTRACTION_TYPE`:

**Workflow Type 1: TEI Encoding** (`JSON_EXTRACTION_TYPE = "tei_encoding"`)
- Encodes text segments (items) based on surrounding context
- Uses `JSON_CONTEXT_KEY` to specify the context field
- Uses `JSON_ITEMS_KEY` to specify the list/string of items to encode
- Example: "Encode bracketed sequences in letter text as TEI XML"

**Input**: JSON file with objects like:
```json
{
  "element_id": "p_123",
  "text": "He lived in [Paris] from 1920-1925.",
  "bracketed_sequences": ["Paris"]
}
```

**Workflow Type 2: Information Extraction** (`JSON_EXTRACTION_TYPE = "information_extraction"`)
- Extracts specific metadata values from JSON objects
- Uses `JSON_DATA_KEYS` to specify which keys to extract and analyze
- Uses `JSON_METADATA_KEYS_INFO` to specify which keys to include in output (but not analyze)
- Example: "Extract dateline and signature information from letter metadata"

**Input**: JSON file with objects like:
```json
{
  "letter_id": "L_456",
  "dateline": "Paris, June 15, 1925",
  "signature": "Yours truly, M.B."
}
```

**Output Options** (configurable via `KEY_EXTRACTION_OUTPUT_FORMAT` and `OUTPUT_EXTENSION`):
- **XML update mappings** (`KEY_EXTRACTION_OUTPUT_FORMAT = "xml_mapping"`): TEI encodings for updating existing XML files (TEI encoding workflow only)
- **JSON output** (`KEY_EXTRACTION_OUTPUT_FORMAT = "json"`): Analysis results as JSON files with two modes:
  - **Raw mode**: Individual JSON files per element
  - **JSON-array mode**: Combined JSON array with all analyses
- **XML files** (when using other processing modes): Can output as:
  - **TEI-XML** (`XML_OUTPUT_TYPE = "tei"`): Traditional TEI encoding
  - **RDF-XML** (`XML_OUTPUT_TYPE = "rdf"`): RDF semantic data output

#### b) Object Processing Mode
Processes complete JSON objects as units, generating output files based on `OUTPUT_EXTENSION` setting.

**Output Options**:
- **JSON output** (`OUTPUT_EXTENSION = ".json"`): Two modes available:
  - **Raw mode** (`JSON_OUTPUT_MODE = "raw"`): Extracts content after `<think>` tags, combined into one file (may need post-processing)
  - **JSON-array mode** (`JSON_OUTPUT_MODE = "json-array"`): Properly structured JSON array with individual objects (always valid JSON)
- **XML output** (`OUTPUT_EXTENSION = ".xml"`): Two types available:
  - **TEI-XML** (`XML_OUTPUT_TYPE = "tei"`): Traditional TEI encoding format
  - **RDF-XML** (`XML_OUTPUT_TYPE = "rdf"`): RDF semantic data format

**Input**: JSON file with any JSON structure

## Requirements

### Python Version
- **Required**: Python 3.8 or higher
- **Tested with**: Python 3.10, 3.11
- **Recommended**: Python 3.10 or 3.11 for best compatibility

### Core Dependencies (Required)
```bash
anthropic>=0.66.0       # Claude API
openai>=1.99.0          # GPT API
pandas>=2.3.0           # Data analysis
lxml>=6.0.0             # XML processing
```

**Note**: For reproducibility, record exact versions used for published results:
```bash
pip freeze > requirements_versions.txt
```

### Optional Dependencies (For Local Models)
Only required if using local models (Qwen, OLMo):
```bash
llama-cpp-python>=0.3.16    # Local model inference
torch>=2.8.0                # GPU support and diagnostics
gputil                      # GPU monitoring
psutil                      # System resource monitoring
```

**Important**: For GPU support with local models, install `llama-cpp-python` with CUDA wheels (see Installation section below).

## What to Prepare

Before you begin, ensure you have the following ready:

### Minimal Requirements
- **Input data**: A folder containing plaintext files (`.txt`) OR a JSON file, depending on your workflow
- **Prompt**: A prompt directory (anywhere, set as `PROMPT_DIR`) containing at least a `prompt.txt` file (see Prompt Configuration section for details)

### Model-Specific Requirements

**For Cloud Models (GPT, Claude):**
- **API Key**: Either:
  - Create a `keys.py` file next to `llm_processing.py` with your API keys, OR
  - Set environment variables with your API keys (see Installation section)

**For Local Models (Qwen, OLMo):**
- **Model file**: A downloaded GGUF model file
  - For GPU: Download a model appropriate for your VRAM (see Model Versions section)
  - For CPU: Download a quantized model from HuggingFace (e.g., [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF) - smallest version ~4.7 GB)
  - **Important**: Check your available RAM in Task Manager before downloading and ensure the model size is smaller than your available RAM

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Important for Local Models (Qwen, OLMo) used with GPU:**
   Please also install llama-cpp-python with CUDA support:
   ```bash
   pip uninstall llama-cpp-python
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   This ensures proper GPU acceleration. Without this step, local models may run very slowly or fail to utilize your GPU.

2. **Set up API keys** (for cloud models only)

   You can provide API keys in one of two ways:

   **Option A: Using a `keys.py` file** (default)

   Create a `keys.py` file next to `llm_processing.py` (it is git-ignored) and add the keys you need:
   ```python
   OPENAI = "your-openai-api-key-here"
   ANTHROPIC = "your-anthropic-api-key-here"
   DHINFRA = "your-dh-infra-api-key-here"  # only for models on the DH Infra cluster
   ```

   **Option B: Using environment variables** (recommended for production)

   Set environment variables instead:
   ```bash
   # On Linux/macOS:
   export OPENAI_API_KEY="your-openai-api-key-here"
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

   # On Windows (PowerShell):
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   $env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

   Then modify `config.py` to read from environment variables:
   ```python
   import os

   # In the API KEYS section, replace the keys.py import with:
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
   ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
   ```

3. **Configure the model and settings**

   Copy `config.py` into your project, adjust it and pass it with `--config` (see "Using the Engine in Your Own
   Project" above and the Configuration section below).

## Usage

### Basic Usage

**Interactive Mode** (default):
```bash
python llm_processing.py
```

The script will:
1. Prompt you to select a workflow (Text Processing or JSON Processing)
2. Load input files from the configured directory
3. Process each file using the selected LLM
4. Save TEI XML outputs and processing logs

**Command-Line Mode** (with argparse):
```bash
# Run text processing workflow directly
python llm_processing.py --workflow text
# or
python llm_processing.py -w text

# Run JSON processing workflow directly
python llm_processing.py --workflow json
# or
python llm_processing.py -w json

# Show version
python llm_processing.py --version

# Show help
python llm_processing.py --help
```

### Test Mode (No API Calls)

To generate prompts without making API calls:

```python
# In config.py
ENABLE_API_CALLS = False
```

This is useful for:
- Testing prompt configurations
- Validating input files
- Development and debugging
- Estimating costs before processing

### Processing Plaintext to XML

1. Place your plaintext files in a folder of your project (outside `llm_processing/`)
2. Update `config.py`:
   ```python
   INPUT_TYPE = "txt"
   INPUT_PATH = "/path/to/your/input"  # Your input directory
   OUTPUT_EXTENSION = ".xml"  # Generate TEI XML files
   ```
3. Run `python llm_processing.py --config <your config>`

### Processing Plaintext to JSON

**Option 1: Individual JSON Objects (Raw Mode)**
1. Place your plaintext files in a folder of your project (outside `llm_processing/`)
2. Update `config.py`:
   ```python
   INPUT_TYPE = "txt"
   INPUT_PATH = "/path/to/your/input"  # Your input directory
   OUTPUT_EXTENSION = ".json"  # Generate JSON files
   JSON_OUTPUT_MODE = "raw"  # One JSON file per input
   ```
3. Run `python llm_processing.py --config <your config>`

**Option 2: Combined JSON Array (JSON-Array Mode)**
1. Place your plaintext files in a folder of your project (outside `llm_processing/`)
2. Update `config.py`:
   ```python
   INPUT_TYPE = "txt"
   INPUT_PATH = "/path/to/your/input"  # Your input directory
   OUTPUT_EXTENSION = ".json"  # Generate JSON output
   JSON_OUTPUT_MODE = "json-array"  # Combine all outputs in one file
   ```
3. Run `python llm_processing.py --config <your config>`

**Result**: All outputs are combined into a single `output.json` file with this structure:
```json
[
  {
    "filename": "letter1.txt",
    "output": { ... }
  },
  {
    "filename": "letter2.txt",
    "output": { ... }
  }
]
```

### Processing JSON Files

#### Key Extraction Mode

**Option 1: XML Update Mappings**
1. Prepare your JSON file with the appropriate structure
2. Update `config.py`:
   ```python
   INPUT_TYPE = "json"
   INPUT_PATH = "/path/to/your_file.json"
   JSON_PROCESSING_MODE = "key_extraction"
   KEY_EXTRACTION_OUTPUT_FORMAT = "xml_mapping"  # Creates XML update mappings

   # Configure the keys to extract:
   JSON_CONTEXT_KEY = "your_context_field"
   JSON_ITEMS_KEY = "your_items_field"
   JSON_METADATA_KEYS = ["id", "filename", "xpath"]
   ```
3. Run `python llm_processing.py --config <your config>`

**Option 2: JSON Output**
1. Prepare your JSON file with the appropriate structure
2. Update `config.py`:
   ```python
   INPUT_TYPE = "json"
   INPUT_PATH = "/path/to/your_file.json"
   JSON_PROCESSING_MODE = "key_extraction"
   KEY_EXTRACTION_OUTPUT_FORMAT = "json"  # Output as JSON
   JSON_OUTPUT_MODE = "json-array"  # or "raw" for individual files

   # Configure the keys to extract:
   JSON_CONTEXT_KEY = "your_context_field"
   JSON_ITEMS_KEY = "your_items_field"
   JSON_METADATA_KEYS = ["id", "filename", "xpath"]
   ```
3. Run `python llm_processing.py --config <your config>`

#### Object Processing Mode

1. Prepare your JSON file with the appropriate structure
2. Update `config.py`:
   ```python
   INPUT_TYPE = "json"
   INPUT_PATH = "/path/to/your_file.json"
   JSON_PROCESSING_MODE = "object_processing"
   ```
3. Run `python llm_processing.py --config <your config>`

## Configuration

Edit `config.py` to customize the processing:

### Model Selection

**OpenAI GPT:**
```python
MODEL_NAME = "gpt-5-mini-2025-08-07"
TEMPERATURE = 0.3 # for cloud models 0.3 is recommended
MAX_TOKENS = 8000
```

**Anthropic Claude:**
```python
MODEL_NAME = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.3
MAX_TOKENS = 8000
```

**Local Models (Qwen):**
```python
MODEL_NAME = "qwen3-14B-Q6"
MODEL_PATH_QWEN3 = '../../../models/Qwen_Qwen3-14B-Q6_K.gguf'
QWEN_USE_THINKING = True  # Enable/disable thinking mode
TEMPERATURE = 0.6 # for local models this is the recommended temperature setting
MAX_TOKENS = 8000
```

**Note for CPU Usage:**
If you plan to use the CPU instead of GPU, you can download GGUF model files from HuggingFace. For CPU usage, we recommend using quantized models that are optimized for smaller memory footprints. A good option is the [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF) model, specifically the smallest quantized version (approximately 4.7 GB).

**Important:** Before downloading a model, check your available RAM in Task Manager (Windows) or Activity Monitor (macOS/Linux). Choose a model that is definitely smaller than your available RAM to ensure stable operation. The model file size should be significantly less than your total available RAM to leave room for the operating system and other processes.

**Local Models (OLMo):**
```python
MODEL_NAME = "olmo2-32B-instruct-Q4"
MODEL_PATH_OLMO2 = '../../../models/OLMo-2-0325-32B-Instruct-Q4_K_S.gguf'
TEMPERATURE = 0.6
MAX_TOKENS = 8000
```

**Note on Model Compatibility:**
The model processors have been tested with:
- **GPT**: GPT-4, GPT-5
- **Claude**: Claude 4 / 4.5 (Sonnet)
- **Qwen**: Qwen3
- **OLMo**: OLMo2

Newer model versions may have different API parameters or response formats. If you encounter issues with newer models, the processor code in `processors/` may need to be adapted to accommodate API changes.

### Prompt Configuration

Point the engine to a prompt directory:
```python
PROMPT_VERSION = "editorial_interventions"  # Name of the prompt design, recorded in processing_metadata.json
PROMPT_DIR = os.path.join(_PROJECT_ROOT, "data", "prompts", PROMPT_VERSION)  # Directory with the prompt files
```

The prompt directory can be anywhere (keep it outside the engine) and contains:
- `prompt.txt` (required) - Main prompt instructions for system message
- `encoding_rules.txt` (optional) - TEI encoding guidelines (added to system message)
- `few_shot_examples.txt` (optional) - Input-output examples (added to system message)
- `user_message.txt` (optional) - User message template (if not provided, raw content is sent)

```python
USER_MESSAGE = "user_message.txt"  # Optional template file in prompt directory
```


### Input/Output Paths

```python
# Input type: "txt" for plaintext files, or "json" for JSON processing
INPUT_TYPE = "txt"  # Options: "txt" or "json"
# Note: When INPUT_TYPE = "txt", the system always processes .txt files

# Input path (directory for txt, file path for json)
INPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "input", "my_input.json")

# JSON Processing Mode (only used when INPUT_TYPE = "json")
JSON_PROCESSING_MODE = "key_extraction"  # Options: "key_extraction" or "object_processing"

# Output directory for generated files
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "output")

# Output file extension (for text processing workflows)
# Options: ".xml" for XML files, ".json" for JSON output
OUTPUT_EXTENSION = ".xml"

# XML output type (only used when OUTPUT_EXTENSION = ".xml")
# Options: "tei" for TEI XML encoding, "rdf" for RDF-XML output
XML_OUTPUT_TYPE = "tei"  # Options: "tei" or "rdf"

# JSON output mode (only used when OUTPUT_EXTENSION = ".json" for text processing)
#   - "raw": Individual JSON files per input file
#   - "json-array": Combined JSON array with all outputs in one file
JSON_OUTPUT_MODE = "json-array"  # Options: "raw" or "json-array"

# Key extraction output format (only used when JSON_PROCESSING_MODE = "key_extraction")
#   - "xml_mapping": Create XML update mapping files (original behavior)
#   - "json": Output as JSON (uses JSON_OUTPUT_MODE for raw/json-array)
KEY_EXTRACTION_OUTPUT_FORMAT = "xml_mapping"  # Options: "xml_mapping" or "json"
```

### JSON Key Mapping Configuration

When using `JSON_PROCESSING_MODE = "key_extraction"`, you can configure which keys from your JSON structure to use. The system supports two workflow types configured via `JSON_EXTRACTION_TYPE`:

#### Workflow Type Selection

```python
# =============================================================================
# JSON KEY EXTRACTION TYPE
# =============================================================================

# Choose workflow type: "tei_encoding" or "information_extraction"
JSON_EXTRACTION_TYPE = "information_extraction"  # or "tei_encoding"
```

#### Workflow 1: TEI Encoding

**Purpose**: Encode text segments (items) based on surrounding context, generating TEI XML encodings.

**Configuration Options:**

```python
# =============================================================================
# TEI ENCODING WORKFLOW
# =============================================================================

# Key containing the context text
JSON_CONTEXT_KEY = "full_element_text"

# Key containing items to encode (can be a list OR a single string)
JSON_ITEMS_KEY = "bracketed_sequences"

# Keys to preserve as metadata in the output
JSON_METADATA_KEYS = ["element_id", "filename", "xpath", "index"]

# Labels used in user prompts (customize for your use case)
JSON_CONTEXT_LABEL = "Context"
JSON_ITEMS_LABEL = "Items to Encode"
```

**Example Use Case 1: Context + List of Items**

**JSON Structure:**
```json
{
  "element_id": "p_123",
  "filename": "letter10.xml",
  "full_element_text": "to you, so that I I [sic] might be able to announce their departure to him[?]",
  "bracketed_sequences": ["so that I I [sic]", "him[?]"]
}
```

**Configuration:**
```python
JSON_EXTRACTION_TYPE = "tei_encoding"
JSON_CONTEXT_KEY = "full_element_text"
JSON_ITEMS_KEY = "bracketed_sequences"
JSON_METADATA_KEYS = ["element_id", "filename"]
JSON_CONTEXT_LABEL = "Context"
JSON_ITEMS_LABEL = "Bracketed Sequences"
```

**Resulting Prompt:**
```
Context: "to you, so that I I [sic] might be able to announce their departure to him[?]"

Bracketed Sequences:
- so that I I [sic]
- him[?]
```


#### Workflow 2: Information Extraction

**Purpose**: Extract specific values containing metadata from JSON objects for LLM analysis.

**Configuration Options:**

```python
# =============================================================================
# INFORMATION EXTRACTION WORKFLOW
# =============================================================================

# Keys to extract and analyze (these go to the LLM)
JSON_DATA_KEYS = ["dateline", "signature"]

# Labels for data keys in user prompts (optional, must match order and length)
# If not provided, will use key names directly
JSON_DATA_LABELS = ["Dateline", "Signature"]

# Keys to preserve as metadata in output (not analyzed by LLM)
JSON_METADATA_KEYS_INFO = ["letter_id"]
```

**Example: Extract Date, Place and Writer from Dateline and Signature**

**JSON Input Structure:**
```json
{
  "letter_id": "L_456",
  "dateline": "Paris, June 15, 1925",
  "signature": "Yours truly, M.B."
}
```

**Configuration:**
```python
JSON_EXTRACTION_TYPE = "information_extraction"
JSON_DATA_KEYS = ["dateline", "signature"]
JSON_DATA_LABELS = ["Date and Place", "Writer"]  # Optional: customize prompt labels
JSON_METADATA_KEYS_INFO = ["letter_id"]
```

**Resulting Prompt:**
```
Date and Place: Paris, June 15, 1925
Writer: Yours truly, M.B.
```

**Note:** If `JSON_DATA_LABELS` is not provided or doesn't match the length of `JSON_DATA_KEYS`, the system will use the key names directly (e.g., "dateline:" instead of "Date and Place:").

**Key Points:**
- **TEI encoding**: Uses `JSON_CONTEXT_KEY` + `JSON_ITEMS_KEY` for context-based encoding
- `JSON_ITEMS_KEY` in TEI encoding can be **either a list OR a single string**
- Use `JSON_CONTEXT_LABEL` and `JSON_ITEMS_LABEL` to customize TEI encoding prompts

- **Information extraction**: Uses `JSON_DATA_KEYS` to specify which fields to analyze
- Use `JSON_DATA_LABELS` to customize information extraction prompt labels (must match order and length of `JSON_DATA_KEYS`)
- This allows the same code to work with different JSON formats without modification

### API Control

```python
# Set to False to generate prompts without making API calls
ENABLE_API_CALLS = True
```

## Output Structure

Processing results are organized in timestamped directories:

### XML Output (OUTPUT_EXTENSION = ".xml"):
```
{OUTPUT_DIR}/{model_name}/processing_{timestamp}/
├── letter1.xml                    # LLM-generated TEI XML files
├── letter2.xml
├── ...
└── log/
    ├── processing_metadata.json   # Processing metrics and costs
    ├── responses.txt              # Raw LLM responses
    └── prompts.txt                # User prompts sent to LLM
```

### JSON Output - Raw Mode (OUTPUT_EXTENSION = ".json", JSON_OUTPUT_MODE = "raw"):
```
{OUTPUT_DIR}/{model_name}/processing_{timestamp}/
├── letter1.json                   # Individual JSON files
├── letter2.json
├── ...
└── log/
    ├── processing_metadata.json   # Processing metrics and costs
    ├── responses.txt              # Raw LLM responses
    └── prompts.txt                # User prompts sent to LLM
```

### JSON Output - Array Mode (OUTPUT_EXTENSION = ".json", JSON_OUTPUT_MODE = "json-array"):
```
{OUTPUT_DIR}/{model_name}/processing_{timestamp}/
├── output.json                    # Combined JSON array with all outputs
└── log/
    ├── processing_metadata.json   # Processing metrics and costs
    ├── responses.txt              # Raw LLM responses
    └── prompts.txt                # User prompts sent to LLM
```

### Processing Metadata

The `processing_metadata.json` file contains:
- Model information (name, type, version)
- Processing time and timestamp
- Token usage (input/output/total)
- Cost estimation (for API models)
- GPU utilization (for local models)
- Temperature and prompt version

### Example Metadata:
```json
{
  "timestamp": "20250106_143022",
  "model_instance": "claude-sonnet-4-5-20250929",
  "model_type": "claude",
  "prompt_version": "prompt_v5",
  "temperature": 0.3,
  "processing_time": "00:05:23",
  "token_count": 145623,
  "input_tokens": 98234,
  "output_tokens": 47389,
  "estimated_cost_usd": 4.26
}
```

## Project Structure

```
llm_processing/
├── llm_processing.py              # Main coordinator script
├── config.py                      # Configuration template (copy it into your project)
├── config_loader.py               # Loads config.py, or the file given with --config
├── requirements.txt               # Python dependencies
├── processors/                    # Model-specific implementations
│   ├── __init__.py
│   ├── gpt.py                    # OpenAI GPT
│   ├── claude.py                 # Anthropic Claude
│   ├── qwen.py                   # Alibaba Qwen
│   └── olmo.py                   # OLMo
└── utils/                         # Shared utility functions
    ├── __init__.py
    └── utils.py                  # Utility functions for processors
```

Project-specific prompts, input and output do not belong in this folder (see "Using the Engine in Your Own
Project").

## Cost Estimation

The tool automatically calculates costs for API-based models:

**OpenAI GPT-5 Pricing (per million tokens):**
- GPT-5-nano: $0.05 input / $0.40 output
- GPT-5-mini: $0.25 input / $2.00 output
- GPT-5: $1.25 input / $10.00 output

**Anthropic Claude Pricing (per million tokens):**
- Claude Sonnet 4: $3.00 input / $15.00 output

For any other models please add pricing.

## Troubleshooting

### "MODEL_NAME not defined" error
Ensure `config.py` has exactly one uncommented `MODEL_NAME` line.

### "API key not found" error
1. Verify `keys.py` exists and contains your API keys
2. Or set `ENABLE_API_CALLS = False` for test mode

### Local model fails to load
1. Verify the model path in `config.py` points to a valid `.gguf` file
2. Ensure `llama-cpp-python` is installed with GPU support if needed
3. Check that you have sufficient RAM/VRAM for the model

### Local models running very slowly or not using GPU
This is a common issue with `llama-cpp-python`. The default pip installation doesn't include GPU support.

**Solution:**
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

This installs the CUDA 12.1 version with proper GPU acceleration. Adjust the `cu121` suffix if you're using a different CUDA version (e.g., `cu118` for CUDA 11.8).

### No output files generated
1. Check console output for errors
2. Verify input directory contains `.txt` files
3. Ensure output directory has write permissions
4. Review `log/responses.txt` for LLM errors

## Development

### Adding a New Model

1. Create a new processor in `processors/` with three required functions:
   ```python
   # processors/new_model_processor.py
   import config
   from utils.utils import (
       create_segment_error_response,
       create_test_response,
       create_test_tei_response,
       create_text_encoding_error,
       create_text_encoding_result,
       extract_tei_xml_from_response,
       extract_json_from_response,
       parse_json_response,
       validate_text_data,
       validate_text_segment
   )

   def encode_text_newmodel(text_data, coordinator=None):
       """
       Process plaintext file and generate TEI XML or JSON output.
       Used for text processing workflow.
       """
       # Implementation for plaintext to XML/JSON
       pass

   def encode_text_segment_newmodel(text_segment, coordinator=None):
       """
       Analyze specific keys from JSON and provide TEI encodings.
       Used for JSON key extraction mode.
       """
       # Implementation for JSON key extraction
       pass

   def process_json_object_newmodel(json_object, coordinator=None):
       """
       Process complete JSON object and generate output.
       Used for JSON object processing mode.
       """
       # Implementation for JSON object processing
       pass
   ```

2. Register in `llm_processing.py` (in the `__init__` method):
   ```python
   self.processors["newmodel"] = {
       "name": "New Model Name",
       "module": "processors.new_model_processor",
       "text_function": "encode_text_newmodel",
       "json_extraction_function": "encode_text_segment_newmodel",
       "json_object_function": "process_json_object_newmodel"
   }
   ```

3. Add model detection in `detect_model_from_config()` method:
   ```python
   elif "newmodel" in model_name_lower:
       return "newmodel"
   ```

4. Add configuration in `config.py`:
   ```python
   # MODEL_NAME = "newmodel-v1"  # Your model identifier
   # Add any model-specific settings like API keys or model paths
   ```

### Creating a New Prompt Version

1. Create a directory in your project, outside the engine, e.g. `data/prompts/prompt_v6/`
2. Add required files:
   - `prompt.txt` - Main instructions (required)
   - `encoding_rules.txt` - Guidelines (optional)
   - `few_shot_examples.txt` - Examples (optional)
   - `user_message.txt` - User message template (optional - if omitted, raw content is sent)
3. Update `config.py`:
   ```python
   PROMPT_VERSION = "prompt_v6"
   PROMPT_DIR = os.path.join(_PROJECT_ROOT, "data", "prompts", PROMPT_VERSION)
   USER_MESSAGE = "user_message.txt"  # Optional, defaults to this filename
   ```

### Using Different JSON Structures

The JSON key extraction workflow is designed to work with any JSON structure. To adapt it to your JSON format:

1. Identify your JSON structure:
   - Which field contains the context text? → `JSON_CONTEXT_KEY`
   - Which field contains the items to analyze? → `JSON_ITEMS_KEY`
   - Which fields should be preserved as metadata? → `JSON_METADATA_KEYS`

2. Update `config.py` with your field names:
   ```python
   JSON_CONTEXT_KEY = "your_context_field"
   JSON_ITEMS_KEY = "your_items_field"
   JSON_METADATA_KEYS = ["field1", "field2", "field3"]
   ```

3. Ensure your JSON file matches the expected structure:
   - `JSON_CONTEXT_KEY` should contain a string
   - `JSON_ITEMS_KEY` should contain a list
   - `JSON_METADATA_KEYS` can be any fields you want preserved

## Reproducibility

For reproducing published results, see the comprehensive reproducibility guide below.

### Code Version

**Current Version**: 1.0.0

For published results, record the git commit hash:
```bash
git rev-parse HEAD
```

### Environment Requirements

#### Python Version
- **Required**: Python 3.8 or higher
- **Tested with**: Python 3.10, 3.11

#### System Dependencies

**For Local Models (Qwen, OLMo):**
- **CUDA Toolkit** (for GPU processing): Version 11.8, 12.1, 12.4, or 13.0
- **NVIDIA GPU** with sufficient VRAM (see model-specific requirements below)
- **CPU alternative**: Models can run on CPU (slower, requires 8-9 GB RAM minimum)
  - For CPU usage, download GGUF model files from HuggingFace
  - Recommended model for CPU: [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF) (smallest quantized version, ~4.7 GB)
  - **Before downloading**: Check your available RAM in Task Manager (Windows) or Activity Monitor (macOS/Linux) and choose a model that is definitely smaller than your available RAM

**For Cloud Models (GPT, Claude):**
- Internet connection for API access
- Valid API keys (create `keys.py` file with your API keys)

### Model Versions

#### OpenAI GPT Models
- **Model Name**: As specified in `config.MODEL_NAME`
- **API Version**: Current as of 2025
- **Example**: `gpt-5-mini-2025-08-07`

#### Anthropic Claude Models
- **Model Name**: As specified in `config.MODEL_NAME`
- **API Version**: Current as of 2025
- **Example**: `claude-sonnet-4-5-20250929`

#### Local Models

**Qwen3:**
- **Model File**: GGUF format
- **Recommended Quantization**: Q6_K for GPU (14B model), IQ2_XS for CPU
- **File Size**: ~8-10 GB (Q6_K), ~4-5 GB (IQ2_XS)
- **VRAM Requirements**:
  - Q6_K: ~20 GB VRAM
  - IQ2_XS: ~8-9 GB RAM (CPU mode)
- **Model Path**: Configured in `config.MODEL_PATH_QWEN3`
- **CPU Usage**: For CPU processing, you can download GGUF files from HuggingFace. A recommended option is [bartowski/Qwen_Qwen3-14B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF), using the smallest quantized version (approximately 4.7 GB). **Important**: Check your available RAM in Task Manager before downloading and ensure the model size is significantly smaller than your available RAM.

**OLMo2:**
- **Model File**: GGUF format
- **Recommended Quantization**: Q4_K_S
- **File Size**: ~20 GB
- **VRAM Requirements**: ~24 GB VRAM
- **Model Path**: Configured in `config.MODEL_PATH_OLMO2`

### Critical Settings for Reproducibility

All settings are in `config.py`. For published results, record:

1. **Model Selection**:
   ```python
   MODEL_NAME = "qwen3-14B-Q6"  # or other model
   ```

2. **Temperature** (affects randomness):
   ```python
   TEMPERATURE = 0.1  # Lower = more deterministic
   ```

3. **Max Tokens**:
   ```python
   MAX_TOKENS = 10000
   ```

4. **Prompt Version** (and the prompt files in `PROMPT_DIR`):
   ```python
   PROMPT_VERSION = "editorial_interventions"
   ```

5. **Hardware Settings**:
   ```python
   USE_GPU = True  # or False for CPU
   ```

6. **Qwen Thinking Mode** (if using Qwen):
   ```python
   QWEN_USE_THINKING = False  # or True
   ```

### Non-Deterministic Operations

**Temperature Settings**:
- **Cloud Models (GPT, Claude)**: Temperature affects output randomness
  - Lower temperature (0.1-0.3) = more deterministic, consistent results
  - Higher temperature (0.6-1.0) = more creative, variable results
- **Local Models (Qwen, OLMo)**: Temperature also affects randomness
  - Recommended: 0.1-0.6 depending on use case

**Note**: Even with temperature=0.1, LLM outputs may vary slightly between runs due to:
- Model internal randomness
- API response variations (for cloud models)
- Floating-point precision differences

For maximum reproducibility:
- Use the lowest temperature that produces acceptable results
- Run multiple times and compare outputs
- Document any observed variations

### Reproducing Published Results

#### Step 1: Record Configuration

Before running, record the exact configuration used:
```bash
# Save git commit hash
git rev-parse HEAD > reproducibility_info.txt

# Keep the config file you pass with --config under version control in your project

# Record Python version
python --version >> reproducibility_info.txt

# Record package versions
pip freeze > requirements_versions.txt
```

#### Step 2: Set Configuration

1. Use the published run's config file (pass it with `--config`)
2. Ensure model files are in correct locations (for local models)
3. Verify API keys are set (for cloud models)

#### Step 3: Run Processing

```bash
# Interactive mode
python llm_processing.py --config <published config>

# Or with workflow selection
python llm_processing.py --config <published config> --workflow text
python llm_processing.py --config <published config> --workflow json
```

#### Step 4: Verify Outputs

Outputs are saved to:
- `{OUTPUT_DIR}/{MODEL_NAME}/processing_{timestamp}/` - Generated files
- `{OUTPUT_DIR}/{MODEL_NAME}/processing_{timestamp}/log/` - Logs and metadata

Compare outputs with published results:
- File structure should match
- File naming conventions should match
- Content should be similar (allowing for LLM variability)

### Environment-Specific Considerations

**GPU vs CPU:**
- **GPU**: Faster processing, requires CUDA setup
- **CPU**: Slower but more portable, no special setup needed
- Results should be identical, only speed differs

**Operating System:**
- **Windows**: Tested and working
- **Linux**: Should work (not extensively tested)
- **macOS**: May work with CPU mode (not tested)

**CUDA Versions:**
- Different CUDA versions may require different llama-cpp-python wheels
- Results should be identical regardless of CUDA version
- If GPU issues occur, fall back to CPU mode

### Known Variations

**Expected Variations:**
1. **LLM Output Variability**: Even with low temperature, outputs may vary slightly
2. **Token Counting**: Approximate counts may differ slightly between runs
3. **Timing Information**: Processing times will vary based on system load

**Unexpected Variations:**
If results differ significantly from published results, check:
1. Model version/quantization matches
2. Temperature setting matches
3. Prompt version matches
4. Input data matches exactly
5. Python package versions match (especially llama-cpp-python)

