"""
LLM Processing Coordinator for TEI XML Generation.

Coordinates LLM-based processing workflows for generating TEI XML from plaintext files.
Supports multiple LLM providers: OpenAI GPT, Anthropic Claude, Qwen, and OLMo.

WORKFLOWS SUPPORTED:
===================

1. TEXT PROCESSING WORKFLOW (Active - Currently Used)
   - Processes complete plaintext files into structured output
   - Input: Plaintext files from INPUT_PATH (when INPUT_TYPE='txt')
   - Output: Generated files based on OUTPUT_EXTENSION
     a) TEI XML files (.xml) - Traditional text encoding workflow
     b) JSON files (.json) - New plaintext to JSON workflow
        * Raw mode: Individual JSON files per input
        * JSON-array mode: Combined JSON array file
   - Functions: process_text_files_framework(), save_text_processing_outputs()

2. JSON PROCESSING WORKFLOW
   - Two modes based on JSON_PROCESSING_MODE:
     a) Key Extraction: Extracts and analyzes specific keys from JSON objects
        - Input: JSON file with objects containing keys to extract (config.INPUT_PATH when INPUT_TYPE='json')
        - Output: XML update mappings with TEI encodings
        - Functions: process_text_segments_framework(), encode_text_segment_*()
     b) Object Processing: Processes complete JSON objects as units
        - Input: JSON file with any JSON structure (config.INPUT_PATH when INPUT_TYPE='json')
        - Output: Direct JSON output files with two modes:
          * Raw mode: Extracts content after <think> tags (anything between { and })
          * Valid mode: Creates properly structured JSON array with validated objects
        - Functions: process_json_objects_framework(), process_json_object_*()
"""

# Version information
__version__ = "1.0.0"

# Standard library imports
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd

# Local imports
from config_loader import load_config
config = load_config()  # config.py, or the file given with --config
from utils.utils import extract_xml_from_response, extract_json_from_response


def configured_api_prices() -> Optional[Tuple[float, float]]:
    """
    Return the API prices set in the configuration, if any.

    INPUT_PRICE_USD_PER_MILLION and OUTPUT_PRICE_USD_PER_MILLION override the built-in
    price tables of calculate_openai_pricing() and calculate_anthropic_pricing().

    Returns:
        (input_rate, output_rate) in USD per million tokens, or None if not both are set.
    """
    input_rate = getattr(config, 'INPUT_PRICE_USD_PER_MILLION', None)
    output_rate = getattr(config, 'OUTPUT_PRICE_USD_PER_MILLION', None)
    if input_rate is None or output_rate is None:
        return None
    return float(input_rate), float(output_rate)


class LLMProcessingCoordinator:
    """
    Coordinates LLM processing workflows for TEI XML generation.

    Supports two workflows:
    1. Text Processing: Convert plaintext files to TEI XML (active)
    2. Editorial Intervention Analysis: Analyze bracketed sequences (future feature)
    """

    def __init__(self):
        self.current_model = None
        self.processing_start_time = None
        self.processing_end_time = None
        self.total_tokens_processed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.gpu_usage_data = []

        # Registry of available processors (functions loaded lazily)
        self.processors = {
            "gpt": {
                "name": "OpenAI GPT",
                "module": "processors.gpt",
                "text_function": "encode_text_gpt",  # For plaintext encoding to TEI XML
                "json_extraction_function": "encode_text_segment_gpt",  # For JSON key extraction mode
                "json_object_function": "process_json_object_gpt"  # For JSON object processing mode
            },
            "qwen": {
                "name": "Alibaba Cloud Qwen3",
                "module": "processors.qwen",
                "text_function": "encode_text_qwen",  # For plaintext encoding to TEI XML
                "json_extraction_function": "encode_text_segment_qwen",  # For JSON key extraction mode
                "json_object_function": "process_json_object_qwen"  # For JSON object processing mode
            },
            "claude": {
                "name": "Anthropic Claude",
                "module": "processors.claude",
                "text_function": "encode_text_claude",  # For plaintext encoding to TEI XML
                "json_extraction_function": "encode_text_segment_claude",  # For JSON key extraction mode
                "json_object_function": "process_json_object_claude"  # For JSON object processing mode
            },
            "olmo": {
                "name": "OLMo",
                "module": "processors.olmo",
                "text_function": "encode_text_olmo",  # For plaintext encoding to TEI XML
                "json_extraction_function": "encode_text_segment_olmo",  # For JSON key extraction mode
                "json_object_function": "process_json_object_olmo"  # For JSON object processing mode
            }
        }

    def get_timestamped_filename(self, base_name: str, model: str) -> str:
        """
        Generate a filename for logs (without timestamp since folder has it).
        """
        return f"{base_name}.txt"

    def ensure_log_directories(
        self,
        model: str,
        output_base_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Ensure the log directories exist within the processing output directory.

        Args:
            model: Model type (e.g., 'qwen', 'claude')
            output_base_dir: Base output directory (defaults to config.OUTPUT_DIR)

        Returns:
            Tuple of (log_dir_path, log_dir_path) for compatibility
        """
        output_base_dir = output_base_dir or config.OUTPUT_DIR

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name = self.get_run_folder_name()

        # Create timestamped processing directory
        timestamp = self.get_processing_timestamp()
        processing_dir = os.path.join(output_base_dir, model_name, f"processing_{timestamp}")
        log_dir = os.path.join(processing_dir, "log")

        os.makedirs(log_dir, exist_ok=True)

        # Return same path twice for compatibility with existing code
        return log_dir, log_dir

    def get_processing_timestamp(self) -> str:
        """
        Get the current processing timestamp in YYYYMMDD_HHMMSS format.
        This should match the timestamp used in XML mapping files.
        """
        # Use the stored timestamp to ensure consistency across all files
        if not hasattr(self, '_processing_timestamp'):
            self._processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._processing_timestamp

    def get_run_folder_name(self) -> str:
        """
        Get the name of the output folder for this run.

        Returns:
            config.RUN_NAME if it is set, otherwise config.MODEL_NAME. RUN_NAME
            separates several runs of the same model (e.g. different settings).
        """
        return getattr(config, 'RUN_NAME', None) or config.MODEL_NAME

    def format_processing_time(self, seconds: float) -> str:
        """
        Format processing time in HH:MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def calculate_cost_or_usage_metric(self, model_name: str) -> Tuple[str, float]:
        """
        Calculate appropriate cost or usage metric based on model type.
        Returns (metric_name, metric_value)
        """
        model_lower = model_name.lower()

        if "claude" in model_lower:
            _, _, total_cost = self.calculate_anthropic_pricing(
                self.total_input_tokens, self.total_output_tokens, model_name
            )
            return "Estimated Cost (USD)", total_cost
        elif "gpt" in model_lower:
            _, _, total_cost = self.calculate_openai_pricing(
                self.total_input_tokens, self.total_output_tokens, model_name
            )
            return "Estimated Cost (USD)", total_cost
        elif any(local_model in model_lower for local_model in ["qwen", "olmo"]):
            # For local models, return average GPU utilization
            if self.gpu_usage_data:
                avg_gpu_util = sum(gpu.get('after', {}).get('gpu_utilization', 0) for gpu in self.gpu_usage_data) / len(self.gpu_usage_data)
                return "Avg GPU Utilization (%)", avg_gpu_util
            else:
                return "Avg GPU Utilization (%)", 0.0
        else:
            # For other models (like Apertus), return 0 cost
            return "Estimated Cost (USD)", 0.0

    def save_processing_metrics_to_json(
        self,
        timestamp: str,
        model: str,
        model_name: str,
        output_base_dir: Optional[str] = None
    ) -> str:
        """
        Save processing metrics to a JSON file in the processing run's log directory.
        Returns the path to the JSON file.

        Args:
            timestamp: Processing timestamp
            model: Model type (e.g., 'qwen', 'claude')
            model_name: Model instance name
            output_base_dir: Base output directory (defaults to config.OUTPUT_DIR)
        """
        output_base_dir = output_base_dir or config.OUTPUT_DIR

        # Calculate processing time
        processing_time_seconds = 0.0
        if self.processing_start_time and self.processing_end_time:
            processing_time_seconds = self.processing_end_time - self.processing_start_time

        processing_time_formatted = self.format_processing_time(processing_time_seconds)

        # Create processing metrics entry
        metrics_entry = {
            'timestamp': timestamp,
            'model_instance': model_name,
            'model_type': model,
            'prompt_version': config.PROMPT_VERSION,
            'temperature': config.TEMPERATURE,
            'processing_time': processing_time_formatted,
            'processing_time_seconds': processing_time_seconds,
            'token_count': self.total_tokens_processed,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'thinking_mode': config.QWEN_USE_THINKING if (model == 'qwen' or 'qwen3' in config.MODEL_NAME.lower()) else None,
            'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Settings needed to reproduce the run
        metrics_entry['run_name'] = self.get_run_folder_name()
        metrics_entry['max_tokens'] = config.MAX_TOKENS
        metrics_entry['config_file'] = os.path.basename(getattr(config, '__file__', '') or 'config.py')
        if model == 'olmo':
            from processors.olmo import get_prompt_format
            metrics_entry['prompt_format'] = get_prompt_format()

        # Aggregate measured GPU energy for local models (NVML power sampling).
        # Each inference records a 'power' dict (energy_wh, elapsed_s) in gpu_usage;
        # sum them for the whole run so cost can be derived downstream.
        power_samples = [
            g.get('power') for g in self.gpu_usage_data
            if isinstance(g, dict) and isinstance(g.get('power'), dict)
        ]
        total_energy_wh = sum(p.get('energy_wh', 0.0) for p in power_samples)
        if total_energy_wh > 0:
            total_sampled_s = sum(p.get('elapsed_s', 0.0) for p in power_samples)
            metrics_entry['total_energy_wh'] = round(total_energy_wh, 4)
            if total_sampled_s > 0:
                # Energy-weighted mean board power across all sampled inferences.
                metrics_entry['avg_power_w'] = round(total_energy_wh * 3600.0 / total_sampled_s, 2)

        # If processing disambiguation, check for source extraction metadata
        if 'disambiguation' in config.PROMPT_VERSION.lower():
            input_path = Path(config.INPUT_PATH)
            input_metadata_path = input_path.parent / "processing_metadata.json"
            if input_metadata_path.exists():
                try:
                    with open(input_metadata_path, 'r', encoding='utf-8') as f:
                        input_metadata = json.load(f)
                    # Copy source extraction tracking from input metadata
                    if 'source_extraction_timestamp' in input_metadata:
                        metrics_entry['source_extraction_timestamp'] = input_metadata['source_extraction_timestamp']
                    if 'source_extraction_run' in input_metadata:
                        metrics_entry['source_extraction_run'] = input_metadata['source_extraction_run']
                except Exception as e:
                    print(f"Note: Could not read source extraction metadata from {input_metadata_path}: {e}")

        # Add estimated cost for Claude and GPT models
        model_name_lower = model_name.lower()
        if "claude" in model_name_lower:
            _, _, total_cost = self.calculate_anthropic_pricing(
                self.total_input_tokens, self.total_output_tokens, model_name
            )
            metrics_entry['estimated_cost_usd'] = round(total_cost, 6)
        elif "gpt" in model_name_lower:
            _, _, total_cost = self.calculate_openai_pricing(
                self.total_input_tokens, self.total_output_tokens, model_name
            )
            metrics_entry['estimated_cost_usd'] = round(total_cost, 6)
        if configured_api_prices():
            input_rate, output_rate = configured_api_prices()
            metrics_entry['input_price_usd_per_million'] = input_rate
            metrics_entry['output_price_usd_per_million'] = output_rate

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name_with_suffix = self.get_run_folder_name()

        # Create path to log directory in processing output
        processing_dir = os.path.join(output_base_dir, model_name_with_suffix, f"processing_{timestamp}")
        log_dir = os.path.join(processing_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        # Path to the processing-specific metadata JSON file (saved in processing folder, not log folder)
        json_filepath = os.path.join(processing_dir, "processing_metadata.json")

        # Save the single entry
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_entry, f, indent=2, ensure_ascii=False)

            print(f"Processing metrics saved to: {json_filepath}")
            return json_filepath

        except Exception as e:
            print(f"Error saving processing metrics to JSON: {e}")
            return ""

    def calculate_anthropic_pricing(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate estimated pricing for Anthropic Claude models.
        Returns: (input_cost, output_cost, total_cost) in USD
        """
        # Current Anthropic pricing (as of 2025) - per million tokens
        pricing_rates = {
            "claude-opus-4-6": {"input": 5.0, "output": 25.0},
            "claude-sonnet-4": {"input": 3.0, "output": 15.0}
        }

        # Use model from config if not provided
        if model_name is None:
            model_name = config.MODEL_NAME.lower()

        # Find matching pricing (handle partial matches)
        input_rate = 3.0  # Default to Sonnet rates
        output_rate = 15.0

        for model_key, rates in pricing_rates.items():
            if model_key in model_name or model_name in model_key:
                input_rate = rates["input"]
                output_rate = rates["output"]
                break

        # Prices set in the configuration take precedence
        if configured_api_prices():
            input_rate, output_rate = configured_api_prices()

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def calculate_openai_pricing(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate estimated pricing for OpenAI GPT models.
        Returns: (input_cost, output_cost, total_cost) in USD
        """
        # OpenAI list prices in USD per million tokens
        pricing_rates = {
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-5.2": {"input": 1.75, "output": 14.00},
            "gpt-5": {"input": 1.25, "output": 10.00}
        }

        # Use model from config if not provided
        if model_name is None:
            model_name = config.MODEL_NAME
        model_name = model_name.lower()

        # Find matching pricing: the longest key contained in the model name wins,
        # so "gpt-5.2-2025-12-11" gets the gpt-5.2 rates, not the gpt-5 ones
        input_rate = 0.05  # Default to GPT-5-nano rates
        output_rate = 0.40

        for model_key in sorted(pricing_rates, key=len, reverse=True):
            if model_key in model_name:
                input_rate = pricing_rates[model_key]["input"]
                output_rate = pricing_rates[model_key]["output"]
                break

        # Prices set in the configuration take precedence
        if configured_api_prices():
            input_rate, output_rate = configured_api_prices()

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost


    def get_analyzer_function(
        self,
        model_key: str,
        workflow_type: str = "json_extraction"
    ) -> Callable:
        """
        Dynamically import and return the analyzer function for the specified model.

        Args:
            model_key: The model key (e.g., 'claude', 'gpt')
            workflow_type: One of "json_extraction" (JSON key extraction), "text" (plaintext batch),
                          or "json" (JSON object processing)

        Returns:
            The analyzer function
        """
        import importlib

        processor_info = self.processors.get(model_key)
        if not processor_info:
            raise ValueError(f"Unknown model: {model_key}")

        # Choose the appropriate function name based on workflow type
        if workflow_type == "text":
            function_name = processor_info.get("text_function")
        elif workflow_type == "json":
            function_name = processor_info.get("json_object_function")
        else:  # Default to json_extraction
            function_name = processor_info.get("json_extraction_function")

        if not function_name:
            raise ValueError(f"No function found for workflow_type '{workflow_type}' in model '{model_key}'")

        try:
            # Import the module
            module = importlib.import_module(processor_info["module"])
            # Get the function from the module
            analyzer_function = getattr(module, function_name)
            return analyzer_function
        except ImportError as e:
            raise ImportError(f"Failed to import {processor_info['module']}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Function {function_name} not found in {processor_info['module']}: {e}")

    def detect_model_from_config(self) -> str:
        """
        Detect which model is configured in config.py based on MODEL_NAME.
        Returns the model key for processor lookup.
        """
        model_name_lower = config.MODEL_NAME.lower()

        # Check for each model type
        if "gpt" in model_name_lower or "openai" in model_name_lower:
            return "gpt"
        elif "claude" in model_name_lower or "anthropic" in model_name_lower:
            return "claude"
        elif "qwen3.5" in model_name_lower:
            return "gpt"  # OpenAI-compatible API via DH Infra cluster
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "olmo" in model_name_lower:
            return "olmo"
        else:
            # Default to first available processor if we can't detect
            print(f"Warning: Could not auto-detect model type from '{config.MODEL_NAME}'")
            print("Defaulting to GPT processor. Please check your MODEL_NAME in config.py")
            return "gpt"

    def validate_model_configuration(self) -> bool:
        """
        Validate that model-specific configuration is correct.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check if MODEL_NAME is defined
        if not hasattr(config, 'MODEL_NAME'):
            print("\n Configuration Error: MODEL_NAME is not defined!")

            return False

        # Check if MODEL_NAME is empty or None
        if not config.MODEL_NAME or config.MODEL_NAME.strip() == "":
            print("\n Configuration Error: MODEL_NAME is empty!")
            print("\nPlease set MODEL_NAME to a valid model name in config.py.")
            return False

        # Check for multiple MODEL_NAME definitions in the loaded configuration file
        # Read it and count uncommented MODEL_NAME lines
        config_path = getattr(config, '__file__', None) or os.path.join(os.path.dirname(__file__), 'config.py')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_lines = f.readlines()

            uncommented_model_names = []
            for i, line in enumerate(config_lines, 1):
                stripped = line.strip()
                # Check if line starts with MODEL_NAME = (not commented)
                if stripped.startswith('MODEL_NAME') and '=' in stripped and not stripped.startswith('#'):
                    uncommented_model_names.append((i, stripped))

            if len(uncommented_model_names) > 1:
                print("\n Configuration Error: Multiple MODEL_NAME definitions found!")
                print("\nYou have multiple MODEL_NAME lines in config.py:")
                for line_num, line_content in uncommented_model_names:
                    print(f"  Line {line_num}: {line_content}")
                print("\nPlease comment out all MODEL_NAME lines except the one you want to use.")
                return False
        except Exception as e:
            print(f"\nWarning: Could not validate MODEL_NAME uniqueness: {e}")
            # Continue anyway - this is a nice-to-have validation

        model_key = self.detect_model_from_config()

        # Check Qwen-specific requirements
        if model_key == "qwen":
            if not hasattr(config, 'QWEN_USE_THINKING'):
                print("\n Configuration Error: QWEN_USE_THINKING is not defined!")
                return False

            # Check if model path is defined
            if not hasattr(config, 'MODEL_PATH_QWEN3'):
                print("\n Configuration Error: MODEL_PATH_QWEN3 is not defined!")
                return False

        # Check OLMo-specific requirements
        if model_key == "olmo":
            if not hasattr(config, 'MODEL_PATH_OLMO2'):
                print("\n Configuration Error: MODEL_PATH_OLMO2 is not defined!")
                return False

        return True

    def display_configuration_and_confirm(
        self,
        file_count: int = 0,
        model_key: Optional[str] = None,
        segment_count: Optional[int] = None,
        total_sequences: Optional[int] = None,
        workflow_type: str = "text_processing"
    ) -> bool:
        """
        Display current configuration and ask user for confirmation.

        Args:
            file_count: Number of plaintext files found (for text processing workflow)
            model_key: Model key to use (if None, detects from config)
            segment_count: Number of segments (for editorial intervention workflow)
            total_sequences: Total bracketed sequences (for editorial intervention workflow)
            workflow_type: Type of workflow - "text_processing", "key_extraction", or "object_processing"

        Returns:
            True if user confirms, False otherwise
        """
        # Use provided model_key or detect from config
        if model_key is None:
            model_key = self.detect_model_from_config()
        processor_info = self.processors.get(model_key, {})

        # Supported models note (only for text processing workflow)
        if workflow_type == "text_processing":
            print("NOTE: This script supports the following models:")
            for key, info in self.processors.items():
                print(f"  - {info['name']}")
            print()
            print("Model implementations are based on APIs/formats as of November 2025.")
            print("Newer model versions may require script updates if APIs change.")
            print()

        print("\n"+ "═" * 70)
        if workflow_type == "key_extraction":
            print(" " * 15 + "JSON KEY EXTRACTION CONFIGURATION")
        elif workflow_type == "object_processing":
            print(" " * 15 + "JSON OBJECT PROCESSING CONFIGURATION")
        else:
            print(" " * 15 + "PLAINTEXT BATCH PROCESSING CONFIGURATION")
        print("═" * 70)
        print()

        # DATA STRUCTURE Section
        print("DATA STRUCTURE")
        if workflow_type == "key_extraction":
            input_files = f"{config.INPUT_PATH} ({segment_count} objects, {total_sequences} items to extract)"
        elif workflow_type == "object_processing":
            input_files = f"{config.INPUT_PATH} ({segment_count} JSON objects)"
        else:
            input_files = f"{config.INPUT_PATH}/ ({file_count} plaintext files found)"
        output_dir = f"{config.OUTPUT_DIR}/"

        print(f"  Input Type: {config.INPUT_TYPE}")
        if config.INPUT_TYPE == "json":
            print(f"  Processing Mode: {config.JSON_PROCESSING_MODE}")
        print(f"  Input: {input_files}")
        print(f"  Output: {output_dir}")

        # Show output format configuration for text processing workflow
        if workflow_type == "text_processing":
            print(f"  Output Format: {config.OUTPUT_EXTENSION}")
            if config.OUTPUT_EXTENSION == ".json":
                print(f"  JSON Output Mode: {config.JSON_OUTPUT_MODE}")

        # Show output format configuration for key extraction workflow
        if workflow_type == "key_extraction":
            output_format = getattr(config, 'KEY_EXTRACTION_OUTPUT_FORMAT', 'xml_mapping')
            print(f"  Output Format: {output_format}")
            if output_format == "json":
                output_mode = getattr(config, 'JSON_OUTPUT_MODE', 'json-array')
                print(f"  JSON Output Mode: {output_mode}")

        print()        # MODEL CONFIGURATION Section
        print("MODEL CONFIGURATION")
        print(f"  Model: {config.MODEL_NAME}")

        # Model-specific settings
        if model_key == "qwen":
            thinking_mode = "Enabled" if config.QWEN_USE_THINKING else "Disabled"
            print(f"  Thinking Mode: {thinking_mode}")
            if hasattr(config, 'MODEL_PATH_QWEN3'):
                print(f"  Model Path: {config.MODEL_PATH_QWEN3}")
        elif model_key == "olmo":
            if hasattr(config, 'MODEL_PATH_OLMO2'):
                print(f"  Model Path: {config.MODEL_PATH_OLMO2}")

        print(f"  Temperature: {config.TEMPERATURE}")
        print(f"  Max Tokens: {config.MAX_TOKENS:,}")

        # API Mode or Test Mode
        if config.ENABLE_API_CALLS:
            print(f"  Mode: Live API Calls (ENABLED)")

            # Check API key status for cloud models
            if model_key == "gpt":
                if config.OPENAI_API_KEY:
                    print(f"  API Key: Configured")
                else:
                    print(f"  API Key: NOT FOUND - Processing will fail!")
            elif model_key == "claude":
                if config.ANTHROPIC_API_KEY:
                    print(f"  API Key: Configured")
                else:
                    print(f"  API Key: NOT FOUND - Processing will fail!")
        else:
            print(f"  Mode: Test Mode (API calls disabled)")

        print()

        # PROMPT CONFIGURATION Section
        print("PROMPT CONFIGURATION")
        print(f"  Version: {config.PROMPT_VERSION}")

        # Check which components exist
        print(f"  Components:")
        prompt_dir = config.PROMPT_DIR
        components = [
            ("prompt.txt", True),  # required
            ("encoding_rules.txt", False),  # optional
            ("few_shot_examples.txt", False)  # optional
        ]

        for filename, required in components:
            filepath = os.path.join(prompt_dir, filename)
            if os.path.exists(filepath):
                print(f"    {filename}")
            elif required:
                print(f"    {filename} (REQUIRED - MISSING!)")
            else:
                print(f"    - {filename} (optional, not included)")

        print()
        print("═" * 68)
        print()

        # Confirmation prompt
        while True:
            try:
                response = input("Proceed with this configuration? [Y/n]: ").strip().lower()
                if response in ['', 'y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    print("\nProcessing cancelled by user.")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\n\nProcessing cancelled by user.")
                return False

    def get_workflow_choice(self) -> str:
        """
        Ask user to choose which workflow to run.
        Returns 'text_processing' or 'json_processing'.
        """
        print("\nAvailable Workflows:")
        print("1. Plaintext Batch Processing - Convert plaintext files to TEI XML")
        print("2. JSON Processing - Process JSON files (key extraction or object processing)")
        print()

        while True:
            try:
                choice = input("Please select a workflow (1-2): ").strip()
                choice_num = int(choice)

                if choice_num == 1:
                    print("Selected: Plaintext Batch Processing Workflow")
                    return 'text_processing'
                elif choice_num == 2:
                    print("Selected: JSON Processing Workflow")
                    return 'json_processing'
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

    def get_json_output_mode(self) -> str:
        """
        Ask user to choose output mode for JSON batch processing.
        Returns 'raw' or 'valid'.
        """
        print("\nOutput Mode Options:")
        print("1. Raw Extraction - Extract content after <think> tags (anything between { and })")
        print("   Combined into one file - may need post-processing for valid JSON")
        print("2. Valid JSON - Create properly structured JSON array with individual objects")
        print("   Always produces valid JSON code")
        print()

        while True:
            try:
                choice = input("Please select output mode (1-2): ").strip()
                choice_num = int(choice)

                if choice_num == 1:
                    print("Selected: Raw extraction mode (combined file)")
                    return 'raw'
                elif choice_num == 2:
                    print("Selected: Valid JSON mode (structured array)")
                    return 'valid'
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

    def get_model_choice(self) -> str:
        """
        Ask user to choose which LLM model to use.
        Returns the model key for processor lookup.
        """
        print("\nAvailable LLM Models:")

        # Build list of available models (all models are potentially available)
        available_models = []
        for key, info in self.processors.items():
            available_models.append((key, info["name"]))

        # Display options
        for i, (key, name) in enumerate(available_models, 1):
            print(f"{i}. {name}")
        print()

        while True:
            try:
                choice = input(f"Please select a model (1-{len(available_models)}): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(available_models):
                    model_key, model_name = available_models[choice_num - 1]
                    self.current_model = model_key
                    print(f"Selected: {model_name}")
                    return model_key
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(available_models)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

    def run_llm_processing(self, model_key: str) -> bool:
        """
        Run LLM processing using direct function calls.
        Returns True if successful, False otherwise.
        """
        try:
            # Get the processor info for the selected model
            processor_info = self.processors.get(model_key)
            if not processor_info:
                print(f"Error: Unknown model '{model_key}'")
                return False

            print(f"Running LLM processing with {processor_info['name']}...")

            # Dynamically import the analyzer function
            try:
                analyzer = self.get_analyzer_function(model_key, workflow_type="json_extraction")
            except (ImportError, AttributeError) as e:
                print(f"Error: Failed to load {processor_info['name']} processor: {e}")
                return False

            # Process the text segments using the shared framework
            try:
                results = self.process_text_segments_framework(analyzer, model_name=model_key)
            except RuntimeError as e:
                # Fatal error during processing (e.g., model loading failure)
                print(f"\nError: {e}")
                print(f"Cannot continue without a working model. Please fix the configuration and try again.")
                return False

            if not results:
                print("No results generated.")
                return False

            # Check output format configuration
            output_format = getattr(config, 'KEY_EXTRACTION_OUTPUT_FORMAT', 'xml_mapping')

            if output_format == 'json':
                # Save as JSON output (raw or json-array mode)
                output_mode = getattr(config, 'JSON_OUTPUT_MODE', 'json-array')
                print(f"\nSaving key extraction results as JSON ({output_mode} mode)...")
                stats = self.save_key_extraction_json_output(results, config.OUTPUT_DIR, output_mode)

                print(f"\n{processor_info['name']} processing completed successfully!")
                if stats['saved'] > 0:
                    print(f"JSON output saved to: {stats.get('output_dir', config.OUTPUT_DIR)}")

            else:
                # Save as XML update mapping (original behavior)
                print(f"\nCreating XML update mapping...")
                xml_update_map = self.create_xml_update_mapping_from_results(results)
                xml_mapping_file = self.save_xml_update_mapping(
                    xml_update_map,
                    model=model_key,
                    model_name=config.MODEL_NAME,
                    output_base_dir=config.OUTPUT_DIR
                )

                # Print statistics
                total_files = len(xml_update_map)
                total_elements = sum(len(elements) for elements in xml_update_map.values())
                total_replacements = sum(sum(elem['num_replacements'] for elem in elements) for elements in xml_update_map.values())

                print(f"\n{processor_info['name']} processing completed successfully!")
                print(f"XML update mapping created:")
                print(f"  Files to update: {total_files}")
                print(f"  Elements to update: {total_elements}")
                print(f"  Total replacements: {total_replacements}")
                print(f"Check {xml_mapping_file} for XML update operations.")

            return True

        except Exception as e:
            print(f"Error during LLM processing: {e}")
            return False

    def run_text_processing_workflow(self) -> None:
        """
        Run the text processing workflow.
        Processes plaintext files into TEI XML.
        """
        print("Starting Text Processing Workflow")
        print("=" * 50)

        # Validate input type
        if config.INPUT_TYPE != "txt":
            print(f"\nError: This workflow requires INPUT_TYPE='txt' in config.py")
            print(f"Current INPUT_TYPE is '{config.INPUT_TYPE}'")
            return

        # Check for plaintext input files
        input_dir = config.INPUT_PATH
        if not os.path.exists(input_dir):
            print(f"\nError: Input directory '{input_dir}' not found.")
            print(f"Please create the directory and add .txt plaintext files.")
            return

        # Count plaintext files
        text_files = [f for f in os.listdir(input_dir)
                       if f.endswith(".txt")
                       and os.path.isfile(os.path.join(input_dir, f))]

        if not text_files:
            print(f"\nError: No .txt files found in '{input_dir}'")
            print(f"Please add plaintext files to process.")
            return

        file_count = len(text_files)
        print()

        # Validate all required prompt files before processing
        if not self.validate_prompt_files():
            print("Error: Required prompt files are missing.")
            print("Please check your config.py settings and ensure all prompt files exist.")
            return

        # Validate model configuration (paths, thinking mode, etc.)
        if not self.validate_model_configuration():
            print("\nPlease update your config.py file with the required settings.")
            return

        # Display configuration and get user confirmation
        if not self.display_configuration_and_confirm(file_count):
            return

        # Detect model from config
        model_key = self.detect_model_from_config()

        # Run text processing
        try:
            # Get the processor info for the selected model
            processor_info = self.processors.get(model_key)
            if not processor_info:
                print(f"Error: Unknown model '{model_key}'")
                return

            print(f"Running text processing with {processor_info['name']}...")

            # Dynamically import the text analyzer function
            try:
                analyzer = self.get_analyzer_function(model_key, workflow_type="text")
            except (ImportError, AttributeError) as e:
                print(f"Error: Failed to load {processor_info['name']} text processor: {e}")
                return

            # Process the plaintext files using the shared framework
            try:
                results = self.process_text_files_framework(analyzer, model_name=model_key)
            except RuntimeError as e:
                # Fatal error during processing (e.g., model loading failure)
                print(f"\nError: {e}")
                print(f"Cannot continue without a working model. Please fix the configuration and try again.")
                return

            if not results:
                print("No results generated.")
                return

            # Save output files based on configured extension
            output_ext = config.OUTPUT_EXTENSION
            if output_ext == ".json":
                print(f"\nSaving JSON output files...")
            else:
                print(f"\nSaving TEI XML files...")
            stats = self.save_text_processing_outputs(results)

            # Print final summary
            print(f"\n{processor_info['name']} text processing completed successfully!")
            if stats['saved'] > 0:
                output_type = "JSON" if output_ext == ".json" else "TEI XML"
                print(f"Generated {output_type} files saved to: {stats.get('output_dir', config.OUTPUT_DIR)}")

        except Exception as e:
            print(f"Error during text processing: {e}")
            import traceback
            traceback.print_exc()
            return

        # Construct the correct output directory path
        output_model_name = self.get_run_folder_name()

        # Get the processing timestamp for display
        timestamp = self.get_processing_timestamp()

        print("\n" + "=" * 50)
        print("Text processing workflow finished successfully!")
        print("Check the following locations for results:")
        print(f"- {config.OUTPUT_DIR}/{output_model_name}/processing_{timestamp}/ - Generated TEI XML files")
        print(f"- {config.OUTPUT_DIR}/{output_model_name}/processing_{timestamp}/log/ - Prompts, responses, and metrics")

    def run_json_processing_workflow(self) -> None:
        """
        Run the unified JSON processing workflow.
        Routes to either key extraction or object processing based on JSON_PROCESSING_MODE.
        """
        # Validate input type
        if config.INPUT_TYPE != "json":
            print(f"\nError: This workflow requires INPUT_TYPE='json' in config.py")
            print(f"Current INPUT_TYPE is '{config.INPUT_TYPE}'")
            return

        # Validate processing mode
        if config.JSON_PROCESSING_MODE not in ["key_extraction", "object_processing"]:
            print(f"\nError: Invalid JSON_PROCESSING_MODE '{config.JSON_PROCESSING_MODE}'")
            print(f"Valid options: 'key_extraction' or 'object_processing'")
            return

        # Check for input file
        input_file = config.INPUT_PATH
        if not os.path.exists(input_file):
            print(f"\n [ERROR] Input file '{input_file}' not found.")
            return

        # Route based on processing mode
        if config.JSON_PROCESSING_MODE == "key_extraction":
            self._run_key_extraction_mode(input_file)
        else:  # object_processing
            self._run_object_processing_mode(input_file)

    def _run_key_extraction_mode(self, input_file: str) -> None:
        """
        Run JSON processing in key extraction mode.
        Extracts and analyzes specific keys from JSON objects.
        """
        print("Starting JSON Processing Workflow - Key Extraction Mode")
        print("=" * 50)

        # Try to load and count segments
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            segment_count = len(segments)

            # Count items based on extraction type
            extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')
            total_items = 0  # Initialize for both workflows

            if extraction_type == 'information_extraction':
                # Information extraction: count data keys (not individual items)
                data_keys = getattr(config, 'JSON_DATA_KEYS', [])
                total_items = len(data_keys) * segment_count  # Total extractions to perform
                print(f"\nFound {segment_count} JSON objects.")
                print(f"Configured to extract these data keys: {', '.join(data_keys)}")
            else:
                # TEI encoding: count items
                items_key = config.JSON_ITEMS_KEY
                for seg in segments:
                    # Use configured key name for items to extract
                    if items_key in seg:
                        total_items += len(seg.get(items_key, []))
                    else:
                        # If configured key not found, count the segment itself
                        total_items += 1
                print(f"\nFound {segment_count} JSON objects with {total_items} items to extract and analyze.")
        except Exception as e:
            print(f"\nError: Failed to read input file: {e}")
            return

        # Validate information extraction configuration if needed
        if extraction_type == 'information_extraction':
            data_keys = getattr(config, 'JSON_DATA_KEYS', [])
            data_labels = getattr(config, 'JSON_DATA_LABELS', [])

            if data_labels and len(data_labels) != len(data_keys):
                print(f"\nWarning: JSON_DATA_LABELS length ({len(data_labels)}) doesn't match JSON_DATA_KEYS length ({len(data_keys)})")
                print(f"JSON_DATA_KEYS: {data_keys}")
                print(f"JSON_DATA_LABELS: {data_labels}")
                print("Labels must be in the same order and have the same length as keys.")
                print("Will use key names instead of labels.\n")

        # Validate all required prompt files before processing
        if not self.validate_prompt_files():
            print("Error: Required prompt files are missing.")
            print("Please check your config.py settings and ensure all prompt files exist.")
            return

        # Validate model configuration (paths, thinking mode, etc.)
        if not self.validate_model_configuration():
            print("\nPlease update your config.py file with the required settings.")
            return

        # Detect model from config
        model_key = self.detect_model_from_config()

        # Display configuration and get user confirmation
        if not self.display_configuration_and_confirm(
            model_key=model_key,
            segment_count=segment_count,
            total_sequences=total_items,
            workflow_type="key_extraction"
        ):
            return

        # Run key extraction processing
        success = self.run_llm_processing(model_key)
        if not success:
            print("Key extraction processing failed. Exiting.")
            return

        print("\n" + "=" * 50)
        print("Key extraction workflow finished successfully!")

    def _run_object_processing_mode(self, input_file: str) -> None:
        """
        Run JSON processing in object processing mode.
        Processes complete JSON objects as units.
        """
        print("Starting JSON Processing Workflow - Object Processing Mode")
        print("=" * 50)

        # Try to load and count entries
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                json_objects = json.load(f)
                object_count = len(json_objects)
        except Exception as e:
            print(f"\nError: Failed to read input file: {e}")
            return

        print(f"\nFound {object_count} JSON objects to process.")
        print()

        # Validate all required prompt files before processing
        if not self.validate_prompt_files():
            print("Error: Required prompt files are missing.")
            print("Please check your config.py settings and ensure all prompt files exist.")
            return

        # Validate model configuration (paths, thinking mode, etc.)
        if not self.validate_model_configuration():
            print("\nPlease update your config.py file with the required settings.")
            return

        # Detect model from config
        model_key = self.detect_model_from_config()

        # Display configuration and get user confirmation
        if not self.display_configuration_and_confirm(
            model_key=model_key,
            segment_count  =object_count,
            total_sequences=object_count,
            workflow_type="object_processing"
        ):
            return

        # Get output mode from config instead of prompting
        # Map config JSON_OUTPUT_MODE to expected output_mode values
        # config "raw" → "raw", config "json-array" → "valid"
        output_mode = 'valid' if config.JSON_OUTPUT_MODE == 'json-array' else 'raw'

        # Run object processing
        success = self.run_json_batch_processing(model_key, output_mode=output_mode)
        if not success:
            print("Object processing failed. Exiting.")
            return

        # Get the processing timestamp for display
        timestamp = self.get_processing_timestamp()
        output_model_name = self.get_run_folder_name()

        print("\n" + "=" * 50)
        print("Object processing workflow finished successfully!")
        print("Check the following locations for results:")
        print(f"- {config.OUTPUT_DIR}/{output_model_name}/processing_{timestamp}/ - Generated output files")
        print(f"- {config.OUTPUT_DIR}/{output_model_name}/processing_{timestamp}/log/ - Prompts, responses, and metrics")

    def run_json_batch_processing(self, model_key: str, output_mode: str = 'separate') -> bool:
        """
        Run JSON object batch processing using direct function calls.
        Returns True if successful, False otherwise.

        Args:
            model_key: Model key to use
            output_mode: 'combined' or 'separate' - how to save outputs
        """
        try:
            # Get the processor info for the selected model
            processor_info = self.processors.get(model_key)
            if not processor_info:
                print(f"Error: Unknown model '{model_key}'")
                return False

            print(f"Running JSON batch processing with {processor_info['name']}...")

            # Dynamically import the JSON object analyzer function
            try:
                analyzer = self.get_analyzer_function(model_key, workflow_type="json")
            except (ImportError, AttributeError) as e:
                print(f"Error: Failed to load {processor_info['name']} JSON processor: {e}")
                return False

            # Process the JSON objects using the framework
            try:
                results = self.process_json_objects_framework(analyzer, model_name=model_key)
            except RuntimeError as e:
                # Fatal error during model loading
                print(f"\nError: {e}")
                print(f"Cannot continue without a working model. Please fix the configuration and try again.")
                return False

            if not results:
                print("No results generated.")
                return False

            # Save outputs
            print(f"\nSaving output files ({output_mode} mode)...")
            stats = self.save_json_batch_outputs(results, output_mode=output_mode)

            # Print final summary
            print(f"\n{processor_info['name']} JSON batch processing completed successfully!")
            if stats['saved'] > 0:
                print(f"Generated output files saved to: {stats.get('output_dir', config.OUTPUT_DIR)}")

            return True

        except Exception as e:
            print(f"Error during JSON batch processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ===== SHARED FRAMEWORK METHODS =====

    def validate_prompt_files(self) -> bool:
        """
        Validate that prompt version directory exists with required files.
        Only prompt.txt is mandatory; other components (encoding_rules.txt,
        few_shot_examples.txt) are optional.

        Returns True if validation passes, False otherwise.
        """
        prompt_dir = config.PROMPT_DIR

        # Check if directory exists
        if not os.path.exists(prompt_dir):
            print(f"   Error: Prompt directory '{prompt_dir}' not found!")
            print(f"   Please ensure PROMPT_VERSION='{config.PROMPT_VERSION}' points to a valid directory.")
            return False

        # Check for mandatory file
        prompt_file = os.path.join(prompt_dir, "prompt.txt")
        if not os.path.exists(prompt_file):
            print(f"  Error: Required file 'prompt.txt' not found in '{prompt_dir}'")
            return False

        # Check which optional files exist
        optional_files = ["encoding_rules.txt", "few_shot_examples.txt"]
        found_optional = []
        for filename in optional_files:
            filepath = os.path.join(prompt_dir, filename)
            if os.path.exists(filepath):
                found_optional.append(filename)

        if not found_optional:
            print(f"  - No optional components found")

        return True

    def load_prompt_component(self, filename: str, optional: bool = False) -> str:
        """
        Load a prompt component file from the configured prompt version directory.

        Args:
            filename: Name of the file to load (e.g., "prompt.txt", "encoding_rules.txt")
            optional: If True, returns empty string if file doesn't exist instead of raising error

        Returns:
            File contents as string, or empty string if optional and not found

        Raises:
            FileNotFoundError: If required file (optional=False) is not found
            Exception: If file exists but cannot be read
        """
        filepath = os.path.join(config.PROMPT_DIR, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            if optional:
                # File doesn't exist but that's okay - it's optional
                return ""
            else:
                # Required file is missing - this is an error
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in prompt version '{config.PROMPT_VERSION}'. "
                    f"Expected path: {filepath}"
                )
        except Exception as e:
            raise Exception(f"Error loading {filename} from '{config.PROMPT_VERSION}': {e}")

    def create_system_message(self, model: Optional[str] = None) -> str:
        """
        Create system message from prompt components.

        Only prompt.txt is required; other components (encoding_rules.txt,
        few_shot_examples.txt) are optional. Components are assembled with
        blank lines between them if they exist.

        Args:
            model: Optional model name (unused, kept for API compatibility)

        Returns:
            Complete system message string
        """
        # Load required component
        prompt_template = self.load_prompt_component("prompt.txt", optional=False)

        # Load optional components
        encoding_rules = self.load_prompt_component("encoding_rules.txt", optional=True)
        examples = self.load_prompt_component("few_shot_examples.txt", optional=True)

        # Build message from non-empty components
        components = [prompt_template]  # Always include the main prompt

        if encoding_rules.strip():  # Only add if not empty
            components.append(encoding_rules)

        if examples.strip():  # Only add if not empty
            components.append(examples)

        # Join with double newlines for separation
        complete_system_message = "\n\n".join(components)

        return complete_system_message

    def create_user_message(self, text_content: str, items_to_analyze: List[str]) -> str:
        """
        Create a user message prompt for analyzing items from JSON.

        Loads prefix text from config.USER_MESSAGE if available.
        The prefix text is prepended before the actual data.
        If template doesn't exist, returns only the formatted items list.

        Uses configurable labels from config.JSON_CONTEXT_LABEL and config.JSON_ITEMS_LABEL
        to support different use cases (e.g., "Source Text" / "Target Text" instead of
        "Context" / "Bracketed Sequences").

        Args:
            text_content: The context text (from JSON_CONTEXT_KEY)
            items_to_analyze: List of items to analyze (from JSON_ITEMS_KEY)
                Can be a list with one element if JSON_ITEMS_KEY was a string.

        Returns:
            Formatted user message string
        """
        # Get configurable labels
        context_label = getattr(config, 'JSON_CONTEXT_LABEL', 'Context')
        items_label = getattr(config, 'JSON_ITEMS_LABEL', 'Items')

        # Format the items - handle both single item and multiple items
        if len(items_to_analyze) == 1:
            # Single item (likely from two text segments use case)
            items_formatted = items_to_analyze[0]
        else:
            # Multiple items - format as bulleted list
            items_formatted = "\n".join([f"- {item}" for item in items_to_analyze])

        # Try to load prefix text from file
        prefix_text = self.load_prompt_component(config.USER_MESSAGE, optional=True)

        if prefix_text and prefix_text.strip():
            # Prepend prefix text before the data
            return f"{prefix_text.strip()}\n\n{context_label}: \"{text_content}\"\n\n{items_label}:\n{items_formatted}"

        # If no prefix text, return formatted data only
        return f"{context_label}: \"{text_content}\"\n\n{items_label}:\n{items_formatted}"

    def create_user_message_for_information_extraction(self, data_dict: Dict[str, Any]) -> str:
        """
        Create a user message prompt for information extraction workflow.

        Loads prefix text from config.USER_MESSAGE if available.
        Formats the data keys and their values for the LLM to process.

        Args:
            data_dict: Dictionary mapping data keys to their values (from JSON_DATA_KEYS)

        Returns:
            Formatted user message string
        """
        # Get configured labels (if any)
        data_keys = getattr(config, 'JSON_DATA_KEYS', [])
        data_labels = getattr(config, 'JSON_DATA_LABELS', [])

        # Create a mapping from keys to labels
        key_to_label = {}
        if data_labels and len(data_labels) == len(data_keys):
            # Use provided labels
            key_to_label = dict(zip(data_keys, data_labels))
        else:
            # Fall back to using key names as labels
            key_to_label = {key: key for key in data_keys}

        # Format the data as key-value pairs with labels
        data_lines = []
        for key, value in data_dict.items():
            label = key_to_label.get(key, key)  # Fall back to key if not found
            if value is not None:
                data_lines.append(f"{label}: {value}")
            else:
                data_lines.append(f"{label}: [not provided]")

        # Join the data lines into a formatted string
        data_formatted = "\n".join(data_lines)

        # Try to load prefix text from file
        prefix_text = self.load_prompt_component(config.USER_MESSAGE, optional=True)

        if prefix_text and prefix_text.strip():
            # Prepend prefix text before the data
            return f"{prefix_text.strip()}\n\n{data_formatted}"

        # If no prefix text, return formatted data only
        return data_formatted

    def create_user_message_for_text(self, text_content: str) -> str:
        """
        Create a user message prompt for encoding a complete plaintext file into TEI XML.

        Loads prefix text from config.USER_MESSAGE if available.
        The prefix text is prepended before the actual text content.
        If template doesn't exist, returns only the raw text content.

        Args:
            text_content: The complete plaintext content of the file

        Returns:
            Formatted user message string
        """
        # Try to load prefix text from file
        prefix_text = self.load_prompt_component(config.USER_MESSAGE, optional=True)

        if prefix_text and prefix_text.strip():
            # Prepend prefix text before the data
            return f"{prefix_text.strip()}\n\n{text_content}"

        # If no prefix text, return just the text content
        return text_content

    def create_user_message_for_json_object(self, json_object: Dict[str, Any]) -> str:
        """
        Create a user message prompt for processing a JSON object.

        Loads prefix text from config.USER_MESSAGE if available.
        The prefix text is prepended before the actual JSON object.
        If template doesn't exist, returns only the JSON-serialized object.

        Args:
            json_object: A single JSON object from the input array

        Returns:
            Formatted user message string with the JSON object
        """
        # Serialize JSON object
        json_str = json.dumps(json_object, indent=2, ensure_ascii=False)

        # Try to load prefix text from file
        prefix_text = self.load_prompt_component(config.USER_MESSAGE, optional=True)

        if prefix_text and prefix_text.strip():
            # Prepend prefix text before the data
            return f"{prefix_text.strip()}\n\n{json_str}"

        # If no prefix text, return just the JSON object
        return json_str

    def load_text_segments(self, segments_file: str) -> List[Dict[str, Any]]:
        """
        Load text segments from the JSON file created by the extractor.

        Part of Editorial Intervention Analysis Workflow.
        This workflow is implemented but not currently active in production.
        """
        try:
            with open(segments_file, 'r', encoding='utf-8') as f:
                text_segments = json.load(f)
            return text_segments
        except Exception as e:
            print(f"Error loading text segments file: {e}")
            return []

    def load_plaintext_files(self, input_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load plaintext files from the input directory.

        Args:
            input_dir: Directory containing plaintext files.
                      If None, uses config.INPUT_PATH (when INPUT_TYPE='txt')

        Returns:
            List of dictionaries with 'filename' and 'content' keys
        """
        # Use new config structure
        if config.INPUT_TYPE == "txt":
            input_dir = input_dir or config.INPUT_PATH
            extension = ".txt"
        else:
            raise ValueError(f"load_plaintext_files() requires INPUT_TYPE='txt', but got '{config.INPUT_TYPE}'")

        try:
            # Check if directory exists
            if not os.path.exists(input_dir):
                print(f"Error: Input directory '{input_dir}' does not exist.")
                return []

            # Get all files with the specified extension
            text_files = [f for f in os.listdir(input_dir)
                          if f.endswith(extension) and os.path.isfile(os.path.join(input_dir, f))]

            if not text_files:
                print(f"Warning: No {extension} files found in '{input_dir}'")
                return []

            # Sort files for consistent processing order
            text_files.sort()

            # Load each plaintext file
            files_data = []
            for filename in text_files:
                filepath = os.path.join(input_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    files_data.append({
                        'filename': filename,
                        'content': content
                    })

                except Exception as e:
                    print(f"Warning: Could not read file '{filename}': {e}")
                    continue

            return files_data

        except Exception as e:
            print(f"Error loading plaintext files: {e}")
            return []

    def process_text_segments_framework(
        self,
        model_analyzer: Callable,
        model_name: Optional[str] = None,
        segments_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Shared framework for processing text segments with any LLM model.
        This handles all the common workflow logic.

        Part of Editorial Intervention Analysis Workflow.
        This workflow analyzes bracketed sequences in extracted text segments.
        Currently implemented but not active in production.
        """
        # Use new config structure
        if config.INPUT_TYPE == "json":
            segments_file = segments_file or config.INPUT_PATH
        else:
            raise ValueError(f"process_text_segments_framework() requires INPUT_TYPE='json', but got '{config.INPUT_TYPE}'")

        # Set the current model for logging
        if model_name:
            self.current_model = model_name

        # Initialize timestamp for consistent file naming
        self._processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Start timing
        self.processing_start_time = time.time()
        self.total_tokens_processed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.gpu_usage_data = []

        print(f"Loading text segments from {segments_file}...")
        text_segments = self.load_text_segments(segments_file)

        if not text_segments:
            print("No text segments found or error loading file.")
            return []

        print(f"Found {len(text_segments)} text segments to process.")

        all_results = []
        all_responses = []
        all_user_prompts = []  # Store only user prompts
        system_message = None  # Store system message once

        # Track parsing statistics
        parsing_stats = {
            'total_responses': 0,
            'json_parsed': 0,
            'fallback_parsed': 0,
            'parse_errors': 0,
            'failed_elements': []
        }

        for index, text_segment in enumerate(text_segments):
            # Check extraction type to determine metadata handling
            extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')

            if extraction_type == 'information_extraction':
                # Information extraction: use JSON_METADATA_KEYS_INFO for metadata
                metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                element_id = text_segment.get(metadata_keys[0] if metadata_keys else 'id', f'element_{index}')

                print(f"\nProcessing element {index + 1}/{len(text_segments)}: {element_id}")

                # Don't print items since info extraction works differently
                data_keys = getattr(config, 'JSON_DATA_KEYS', [])
                print(f"Extracting data keys: {data_keys}")
            else:
                # TEI encoding: use element_id, filename, and JSON_ITEMS_KEY
                element_id = text_segment.get('element_id', f'element_{index}')
                filename = text_segment.get('filename', 'unknown')
                items_key = config.JSON_ITEMS_KEY
                items_to_analyze = text_segment.get(items_key, [])

                print(f"\nProcessing element {index + 1}/{len(text_segments)}: {element_id}")
                print(f"Found {len(items_to_analyze)} items to analyze: {items_to_analyze}")

            # Use the model-specific analyzer (pass coordinator instance)
            analyzer_result = model_analyzer(text_segment, coordinator=self)

            # Track parsing metadata
            parsing_metadata = None

            # Handle different return formats (now expecting 8/7/3 values with parsing_metadata)
            if len(analyzer_result) == 8:
                # Local models (Qwen, OLMo) with GPU usage tracking + parsing metadata
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens, gpu_usage, parsing_metadata = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.gpu_usage_data.append(gpu_usage)
            elif len(analyzer_result) == 7:
                # Check if 7th element is dict (parsing_metadata) or dict (gpu_usage for old format)
                if isinstance(analyzer_result[6], dict) and 'parse_success' in analyzer_result[6]:
                    # API models (Claude, GPT) with token tracking + parsing metadata
                    results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens, parsing_metadata = analyzer_result
                    self.total_tokens_processed += segment_tokens
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                else:
                    # Old format: Local models with GPU usage (no parsing metadata)
                    results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens, gpu_usage = analyzer_result
                    self.total_tokens_processed += segment_tokens
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.gpu_usage_data.append(gpu_usage)
            elif len(analyzer_result) == 6:
                # API models (Claude, GPT) with token tracking
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            else:
                # Other processors (Apertus, etc.)
                results, prompt_tuple, raw_response = analyzer_result

            # Update parsing statistics
            if parsing_metadata:
                parsing_stats['total_responses'] += 1
                if parsing_metadata['parse_method'] == 'json':
                    parsing_stats['json_parsed'] += 1
                elif parsing_metadata['parse_method'] == 'fallback':
                    parsing_stats['fallback_parsed'] += 1
                    parsing_stats['failed_elements'].append(element_id)
                elif parsing_metadata['parse_method'] == 'error':
                    parsing_stats['parse_errors'] += 1
                    parsing_stats['failed_elements'].append(element_id)

            # Extract system and user messages from tuple
            if isinstance(prompt_tuple, tuple) and len(prompt_tuple) == 2:
                sys_msg, user_msg = prompt_tuple
                if system_message is None:  # Store system message only once
                    system_message = sys_msg
                all_user_prompts.append(user_msg)
            else:
                # Fallback for old format (combined prompt)
                all_user_prompts.append(str(prompt_tuple))

            # Add metadata to the results - structure depends on extraction type
            extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')

            if extraction_type == 'information_extraction':
                # Information extraction: simple structure with configured metadata
                metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                element_result = {}

                # Add configured metadata fields
                for key in metadata_keys:
                    element_result[key] = text_segment.get(key, None)

                # Add the analysis results
                element_result['results'] = results

            else:
                # TEI encoding: detailed structure with context and items
                context_key = config.JSON_CONTEXT_KEY
                items_key = config.JSON_ITEMS_KEY
                element_result = {
                    "element_id": element_id,
                    "filename": filename,
                    "xpath": text_segment.get('xpath', 'unknown'),
                    context_key: text_segment.get(context_key, ''),
                    items_key: items_to_analyze,
                    "tei_encodings": results
                }

            all_results.append(element_result)

            # Save LLM response with metadata - format depends on extraction type
            if extraction_type == 'information_extraction':
                # Information extraction: show data keys and metadata
                metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                data_keys = getattr(config, 'JSON_DATA_KEYS', [])
                metadata_info = "\n".join(f"{key}: {text_segment.get(key, 'N/A')}" for key in metadata_keys)
                data_info = "\n".join(f"{key}: {text_segment.get(key, 'N/A')}" for key in data_keys)

                response_with_metadata = f"""
{'='*80}
ELEMENT {index + 1} - LLM RESPONSE
{'='*80}
Metadata:
{metadata_info}

Data Keys Analyzed:
{data_info}

RAW LLM RESPONSE:
{raw_response}

{'='*80}
"""
            else:
                # TEI encoding: show context and items
                response_with_metadata = f"""
{'='*80}
ELEMENT {index + 1} - LLM RESPONSE
{'='*80}
ID: {element_id}
Filename: {filename}
Items ({items_key}): {items_to_analyze}
Context ({context_key}): {text_segment.get(context_key, '')}

RAW LLM RESPONSE:
{raw_response}

{'='*80}
"""
            all_responses.append(response_with_metadata)

        # End timing
        self.processing_end_time = time.time()


        # Ensure log directories exist and get paths
        responses_file = ""
        prompts_file = ""
        if self.current_model:
            prompt_log_dir, response_log_dir = self.ensure_log_directories(self.current_model)

            # Generate timestamped filenames
            responses_filename = self.get_timestamped_filename("responses", self.current_model)
            prompts_filename = self.get_timestamped_filename("prompts", self.current_model)

            responses_file = os.path.join(response_log_dir, responses_filename)
            prompts_file = os.path.join(prompt_log_dir, prompts_filename)

        # Save LLM responses to file - EXACT RESPONSES FROM LLM
        if responses_file:
            with open(responses_file, 'w', encoding='utf-8') as f:
                f.write("EXACT RESPONSES FROM LLM\n")
                f.write("=" * 50 + "\n\n")

                # Technical details section
                f.write("TECHNICAL DETAILS OF LLM PROCESSING\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model instance name: {config.MODEL_NAME}\n")
                f.write(f"Date of processing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Calculate processing time
                if self.processing_start_time and self.processing_end_time:
                    processing_time = self.processing_end_time - self.processing_start_time
                    f.write(f"Time of LLM processing: {processing_time:.2f} seconds\n")
                else:
                    f.write("Time of LLM processing: Not available\n")

                f.write(f"Temperature setting: {config.TEMPERATURE}\n")
                f.write(f"Token count: {self.total_tokens_processed:,} tokens\n")

                # Add thinking mode information for Qwen
                if self.current_model == 'qwen':
                    f.write(f"Thinking mode: {'Enabled' if config.QWEN_USE_THINKING else 'Disabled'}\n")

                # Add detailed token breakdown and pricing for supported models
                if "claude" in config.MODEL_NAME.lower():
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                    # Calculate and display pricing
                    input_cost, output_cost, total_cost = self.calculate_anthropic_pricing(
                        self.total_input_tokens, self.total_output_tokens, config.MODEL_NAME
                    )
                    f.write(f"Estimated cost breakdown ({config.MODEL_NAME}):\n")
                    f.write(f"  Input cost: ${input_cost:.6f}\n")
                    f.write(f"  Output cost: ${output_cost:.6f}\n")
                    f.write(f"  Total estimated cost: ${total_cost:.6f}\n")
                elif "gpt" in config.MODEL_NAME.lower():
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                    # Calculate and display pricing
                    input_cost, output_cost, total_cost = self.calculate_openai_pricing(
                        self.total_input_tokens, self.total_output_tokens, config.MODEL_NAME
                    )
                    f.write(f"Estimated cost breakdown ({config.MODEL_NAME}):\n")
                    f.write(f"  Input cost: ${input_cost:.6f}\n")
                    f.write(f"  Output cost: ${output_cost:.6f}\n")
                    f.write(f"  Total estimated cost: ${total_cost:.6f}\n")
                elif any(model in config.MODEL_NAME.lower() for model in ["qwen", "olmo"]):
                    # Local models with GPU usage tracking
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                f.write("\n")

                f.write("This file contains the EXACT responses received from the LLM.\n")
                f.write("Each response includes the raw output and the parsed results.\n\n")
                f.write("RESPONSES:\n")
                f.write("=" * 50 + "\n\n")
                for response in all_responses:
                    f.write(response)

        # Save prompts to file - EXACT PROMPTS SENT TO LLM
        if prompts_file:
            with open(prompts_file, 'w', encoding='utf-8') as f:
                f.write("EXACT PROMPTS SENT TO LLM\n")
                f.write("=" * 50 + "\n\n")
                f.write("This file contains the EXACT prompts that were sent to the LLM.\n")
                f.write("The system message is shown once at the top, followed by user messages for each element.\n\n")

                # Write system message once at the top
                if system_message:
                    f.write("=" * 80 + "\n")
                    f.write("SYSTEM MESSAGE (used for all elements)\n")
                    f.write("=" * 80 + "\n")
                    f.write(system_message)
                    f.write("\n\n" + "=" * 80 + "\n\n")

                f.write("USER MESSAGES:\n")
                f.write("=" * 50 + "\n\n")

                # Write only user messages for each element
                extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')

                for index, (element_result, user_msg) in enumerate(zip(all_results, all_user_prompts)):
                    if extraction_type == 'information_extraction':
                        # Information extraction: show metadata fields
                        metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                        metadata_info = "\n".join(f"{key}: {element_result.get(key, 'N/A')}" for key in metadata_keys)

                        prompt_with_metadata = f"""{'='*80}
ELEMENT {index + 1} - USER PROMPT
{'='*80}
Metadata:
{metadata_info}

USER MESSAGE:
{user_msg}

{'='*80}
"""
                    else:
                        # TEI encoding: show element_id, filename, items
                        items_key = config.JSON_ITEMS_KEY
                        element_id = element_result.get('element_id', f'element_{index}')
                        filename = element_result.get('filename', 'unknown')
                        items_to_analyze = element_result.get(items_key, [])

                        prompt_with_metadata = f"""{'='*80}
ELEMENT {index + 1} - USER PROMPT
{'='*80}
ID: {element_id}
Filename: {filename}
Items ({items_key}): {items_to_analyze}

USER MESSAGE:
{user_msg}

{'='*80}
"""
                    f.write(prompt_with_metadata)

        # Save processing metrics to JSON
        json_file = ""
        if self.current_model:
            processing_timestamp = self.get_processing_timestamp()
            json_file = self.save_processing_metrics_to_json(
                processing_timestamp,
                self.current_model,
                config.MODEL_NAME,
                config.OUTPUT_DIR
            )

        print(f"\nLLM processing complete!")
        print(f"LLM responses saved to {responses_file}")
        print(f"LLM prompts saved to {prompts_file}")
        if json_file:
            print(f"Processing metrics saved to {json_file}")
        print(f"Total elements analyzed: {len(all_results)}")

        # Print summary statistics - depends on extraction type
        extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')

        if extraction_type == 'information_extraction':
            # Information extraction: report on data keys processed
            data_keys = getattr(config, 'JSON_DATA_KEYS', [])
            print(f"Data keys analyzed: {', '.join(data_keys)}")
        else:
            # TEI encoding: report on items processed
            items_key = config.JSON_ITEMS_KEY
            total_items_processed = sum(len(result.get(items_key, [])) for result in all_results)
            print(f"Total items processed ({items_key}): {total_items_processed}")

        # Print parsing statistics
        if parsing_stats["total_responses"] > 0:
            print(f"\n{'='*60}")
            print("JSON PARSING REPORT")
            print(f"{'='*60}")
            print(f"Total LLM responses: {parsing_stats['total_responses']}")
            print(f"Successfully parsed as JSON: {parsing_stats['json_parsed']} ({parsing_stats['json_parsed']/parsing_stats['total_responses']*100:.1f}%)")
            if parsing_stats["fallback_parsed"] > 0:
                print(f"Fallback parsing used: {parsing_stats['fallback_parsed']} ({parsing_stats['fallback_parsed']/parsing_stats['total_responses']*100:.1f}%)")
            if parsing_stats["parse_errors"] > 0:
                print(f"Parse errors: {parsing_stats['parse_errors']} ({parsing_stats['parse_errors']/parsing_stats['total_responses']*100:.1f}%)")
            if parsing_stats["fallback_parsed"] > 0 or parsing_stats["parse_errors"] > 0:
                print(f"\nWarning: Some LLM responses could not be parsed as valid JSON.")
                print(f"Check the responses file for details: {responses_file}")
            print(f"{'='*60}")

        # Count intervention types - only for TEI encoding workflow
        extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')

        if extraction_type == 'tei_encoding':
            intervention_types = []
            type_field = config.JSON_OUTPUT_TYPE_FIELD
            for result in all_results:
                tei_encodings = result.get('tei_encodings', {})
                for seq, encoding in tei_encodings.items():
                    # Safety check: ensure encoding is a dictionary before calling .get()
                    if isinstance(encoding, dict):
                        intervention_types.append(encoding.get(type_field, 'unknown'))
                    else:
                        # Handle case where encoding is not a dictionary (e.g., list, string, etc.)
                        print(f"Warning: Unexpected encoding format for sequence '{seq}': {type(encoding).__name__}")
                        intervention_types.append('unknown')

            if intervention_types:
                type_counts = pd.Series(intervention_types).value_counts()
                print("\nIntervention type distribution:")
                for intervention_type, count in type_counts.items():
                    print(f"  {intervention_type}: {count}")

        return all_results

    def process_text_files_framework(
        self,
        model_analyzer: Callable,
        model_name: Optional[str] = None,
        input_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Shared framework for processing complete plaintext files with any LLM model.
        This handles the text processing workflow logic.

        Args:
            model_encoder: The model-specific encoder function (e.g., encode_text_claude)
            model_name: Name of the model for logging purposes
            input_dir: Directory with plaintext files (defaults to config.INPUT_PATH when INPUT_TYPE='txt')

        Returns:
            List of result dictionaries with filename, tei_xml, and success status
        """
        # Use new config structure
        if config.INPUT_TYPE == "txt":
            input_dir = input_dir or config.INPUT_PATH
        else:
            raise ValueError(f"process_text_files_framework() requires INPUT_TYPE='txt', but got '{config.INPUT_TYPE}'")

        # Set the current model for logging
        if model_name:
            self.current_model = model_name

        # Initialize timestamp for consistent file naming
        self._processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Start timing
        self.processing_start_time = time.time()
        self.total_tokens_processed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.gpu_usage_data = []

        print(f"Loading plaintext files from {input_dir}...")
        files_data = self.load_plaintext_files(input_dir)

        if not files_data:
            print("No plaintext files found or error loading files.")
            return []

        print(f"Found {len(files_data)} plaintext files to process.")

        all_results = []
        all_responses = []
        all_user_prompts = []  # Store only user prompts
        system_message = None  # Store system message once

        for index, file_data in enumerate(files_data):
            filename = file_data['filename']

            print(f"\nProcessing file {index + 1}/{len(files_data)}: {filename}")

            # Use the model-specific analyzer (pass coordinator instance)
            analyzer_result = model_analyzer(file_data, coordinator=self)

            # Handle different return formats (local models return 7 values, API models return 6)
            if len(analyzer_result) == 7:
                # Local models (Qwen, OLMo) with GPU usage tracking
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens, gpu_usage = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.gpu_usage_data.append(gpu_usage)
            elif len(analyzer_result) == 6:
                # API models (Claude, GPT) with token tracking
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            else:
                # Other processors (Apertus, etc.)
                results, prompt_tuple, raw_response = analyzer_result

            # Extract system and user messages from tuple
            if isinstance(prompt_tuple, tuple) and len(prompt_tuple) == 2:
                sys_msg, user_msg = prompt_tuple
                if system_message is None:  # Store system message only once
                    system_message = sys_msg
                all_user_prompts.append(user_msg)
            else:
                # Fallback for old format (combined prompt)
                all_user_prompts.append(str(prompt_tuple))

            # Determine output format based on config
            output_ext = config.OUTPUT_EXTENSION

            if output_ext == ".json":
                # Extract JSON from response
                output_content = extract_json_from_response(raw_response)
                success = bool(output_content and output_content.strip())

                # Add metadata to the results
                text_result = {
                    "filename": filename,
                    "input_text": file_data['content'],
                    "output_content": output_content,
                    "success": success
                }
            else:
                # Default to XML extraction (backward compatible)
                # Add metadata to the results
                text_result = {
                    "filename": filename,
                    "input_text": file_data['content'],
                    "tei_xml": results.get('tei_xml', ''),
                    "success": results.get('success', False)
                }

            all_results.append(text_result)

            # Save LLM response with metadata
            response_with_metadata = f"""
{'='*80}
FILE {index + 1} - LLM RESPONSE
{'='*80}
Filename: {filename}

RAW LLM RESPONSE:
{raw_response}

{'='*80}
"""
            all_responses.append(response_with_metadata)

        # End timing
        self.processing_end_time = time.time()

        # Ensure log directories exist and get paths
        responses_file = ""
        prompts_file = ""
        if self.current_model:
            prompt_log_dir, response_log_dir = self.ensure_log_directories(self.current_model)

            # Generate timestamped filenames
            responses_filename = self.get_timestamped_filename("responses", self.current_model)
            prompts_filename = self.get_timestamped_filename("prompts", self.current_model)

            responses_file = os.path.join(response_log_dir, responses_filename)
            prompts_file = os.path.join(prompt_log_dir, prompts_filename)

        # Save LLM responses to file - EXACT RESPONSES FROM LLM
        if responses_file:
            with open(responses_file, 'w', encoding='utf-8') as f:
                f.write("EXACT RESPONSES FROM LLM - TEXT PROCESSING\n")
                f.write("=" * 50 + "\n\n")

                # Technical details section
                f.write("TECHNICAL DETAILS OF LLM PROCESSING\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model instance name: {config.MODEL_NAME}\n")
                f.write(f"Date of processing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Calculate processing time
                if self.processing_start_time and self.processing_end_time:
                    processing_time = self.processing_end_time - self.processing_start_time
                    f.write(f"Time of LLM processing: {processing_time:.2f} seconds\n")
                else:
                    f.write("Time of LLM processing: Not available\n")

                f.write(f"Temperature setting: {config.TEMPERATURE}\n")
                f.write(f"Token count: {self.total_tokens_processed:,} tokens\n")

                # Add thinking mode information for Qwen
                if self.current_model == 'qwen':
                    f.write(f"Thinking mode: {'Enabled' if config.QWEN_USE_THINKING else 'Disabled'}\n")

                # Add detailed token breakdown and pricing for supported models
                if "claude" in config.MODEL_NAME.lower():
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                    # Calculate and display pricing
                    input_cost, output_cost, total_cost = self.calculate_anthropic_pricing(
                        self.total_input_tokens, self.total_output_tokens, config.MODEL_NAME
                    )
                    f.write(f"Estimated cost breakdown ({config.MODEL_NAME}):\n")
                    f.write(f"  Input cost: ${input_cost:.6f}\n")
                    f.write(f"  Output cost: ${output_cost:.6f}\n")
                    f.write(f"  Total estimated cost: ${total_cost:.6f}\n")
                elif "gpt" in config.MODEL_NAME.lower():
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                    # Calculate and display pricing
                    input_cost, output_cost, total_cost = self.calculate_openai_pricing(
                        self.total_input_tokens, self.total_output_tokens, config.MODEL_NAME
                    )
                    f.write(f"Estimated cost breakdown ({config.MODEL_NAME}):\n")
                    f.write(f"  Input cost: ${input_cost:.6f}\n")
                    f.write(f"  Output cost: ${output_cost:.6f}\n")
                    f.write(f"  Total estimated cost: ${total_cost:.6f}\n")
                elif any(model in config.MODEL_NAME.lower() for model in ["qwen", "olmo"]):
                    # Local models with GPU usage tracking
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                f.write("\n")

                f.write("This file contains the EXACT responses received from the LLM.\n\n")
                f.write("=" * 50 + "\n")
                f.write("RESPONSES:\n")
                f.write("=" * 50 + "\n\n")
                for response in all_responses:
                    f.write(response)

        # Save prompts to file - EXACT PROMPTS SENT TO LLM
        if prompts_file:
            with open(prompts_file, 'w', encoding='utf-8') as f:
                f.write("EXACT PROMPTS SENT TO LLM - TEXT PROCESSING\n")
                f.write("=" * 50 + "\n\n")
                f.write("This file contains the EXACT prompts that were sent to the LLM.\n")
                f.write("The system message is shown once at the top, followed by user messages for each file.\n\n")

                # Write system message once at the top
                if system_message:
                    f.write("=" * 80 + "\n")
                    f.write("SYSTEM MESSAGE (used for all files)\n")
                    f.write("=" * 80 + "\n")
                    f.write(system_message)
                    f.write("\n\n" + "=" * 80 + "\n\n")

                f.write("USER MESSAGES:\n")
                f.write("=" * 50 + "\n\n")

                # Write only user messages for each file
                for index, (text_result, user_msg) in enumerate(zip(all_results, all_user_prompts)):
                    filename = text_result['filename']

                    prompt_with_metadata = f"""{'='*80}
FILE {index + 1} - USER PROMPT
{'='*80}
Filename: {filename}

USER MESSAGE:
{user_msg}

{'='*80}
"""
                    f.write(prompt_with_metadata)

        # Save processing metrics to JSON
        json_file = ""
        if self.current_model:
            processing_timestamp = self.get_processing_timestamp()
            json_file = self.save_processing_metrics_to_json(
                processing_timestamp,
                self.current_model,
                config.MODEL_NAME,
                config.OUTPUT_DIR
            )

        print(f"\nLLM processing complete!")
        print(f"LLM responses saved to {responses_file}")
        print(f"LLM prompts saved to {prompts_file}")
        if json_file:
            print(f"Processing metrics saved to {json_file}")
        print(f"Total files processed: {len(all_results)}")

        # Count successful encodings
        successful = sum(1 for result in all_results if result['success'])
        failed = len(all_results) - successful
        print(f"Successful encodings: {successful}")
        if failed > 0:
            print(f"Failed encodings: {failed}")
            print("Check the response log for details on failures.")

        return all_results

    def process_json_objects_framework(
        self,
        model_analyzer: Callable,
        model_name: Optional[str] = None,
        input_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Shared framework for processing JSON objects with any LLM model.
        This handles the JSON object batch processing workflow logic.

        Args:
            model_processor: The model-specific processor function (e.g., process_json_object_qwen)
            model_name: Name of the model for logging purposes
            input_file: JSON file with array of objects (defaults to config.INPUT_PATH when INPUT_TYPE='json')

        Returns:
            List of result dictionaries with object_id, output_content, and success status
        """
        # Use new config structure
        if config.INPUT_TYPE == "json":
            input_file = input_file or config.INPUT_PATH
        else:
            raise ValueError(f"process_json_objects_framework() requires INPUT_TYPE='json', but got '{config.INPUT_TYPE}'")

        # Set the current model for logging
        if model_name:
            self.current_model = model_name

        # Initialize timestamp for consistent file naming
        self._processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Start timing
        self.processing_start_time = time.time()
        self.total_tokens_processed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.gpu_usage_data = []

        print(f"Loading JSON objects from {input_file}...")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                json_objects = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []

        if not json_objects:
            print("No JSON objects found or error loading file.")
            return []

        print(f"Found {len(json_objects)} JSON objects to process.")

        # Extract base filename for output naming
        input_base_name = os.path.splitext(os.path.basename(input_file))[0]

        all_results = []
        all_responses = []
        all_user_prompts = []  # Store only user prompts
        system_message = None  # Store system message once

        for index, json_object in enumerate(json_objects):
            # Try to extract an ID from JSON_METADATA_KEYS_INFO, then fall back to common id field names
            metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
            object_id = None

            # First try to get ID from JSON_METADATA_KEYS_INFO
            if metadata_keys:
                for key in metadata_keys:
                    if key in json_object:
                        object_id = json_object[key]
                        break

            # Fall back to common id field names if not found
            if object_id is None:
                object_id = json_object.get('id') or json_object.get('entry_id') or json_object.get('element_id') or f"object_{index + 1}"

            print(f"\nProcessing object {index + 1}/{len(json_objects)}: {object_id}")

            # Use the model-specific analyzer (pass coordinator instance)
            analyzer_result = model_analyzer(json_object, coordinator=self)

            # Handle different return formats (local models return 7 values, API models return 6)
            if len(analyzer_result) == 7:
                # Local models (Qwen, OLMo) with GPU usage tracking
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens, gpu_usage = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.gpu_usage_data.append(gpu_usage)
            elif len(analyzer_result) == 6:
                # API models (Claude, GPT) with token tracking
                results, prompt_tuple, raw_response, segment_tokens, input_tokens, output_tokens = analyzer_result
                self.total_tokens_processed += segment_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            else:
                # Other processors
                results, prompt_tuple, raw_response = analyzer_result

            # Extract system and user messages from tuple
            if isinstance(prompt_tuple, tuple) and len(prompt_tuple) == 2:
                sys_msg, user_msg = prompt_tuple
                if system_message is None:  # Store system message only once
                    system_message = sys_msg
                all_user_prompts.append(user_msg)
            else:
                # Fallback for old format (combined prompt)
                all_user_prompts.append(str(prompt_tuple))

            # Add metadata to the results
            # Extract metadata keys from input JSON
            metadata_dict = {}
            if metadata_keys:
                for key in metadata_keys:
                    if key in json_object:
                        metadata_dict[key] = json_object[key]

            object_result = {
                "object_id": object_id,
                "object_index": index + 1,
                "input_filename": input_base_name,
                "total_objects": len(json_objects),
                "metadata": metadata_dict,  # Include metadata from JSON_METADATA_KEYS_INFO
                "input_json": json_object,
                "output_content": results.get('output_content', ''),
                "success": results.get('success', False)
            }

            all_results.append(object_result)

            # Save LLM response with metadata
            response_with_metadata = f"""
{'='*80}
OBJECT {index + 1} - LLM RESPONSE
{'='*80}
Object ID: {object_id}

RAW LLM RESPONSE:
{raw_response}

{'='*80}
"""
            all_responses.append(response_with_metadata)

        # End timing
        self.processing_end_time = time.time()

        # Ensure log directories exist and get paths
        responses_file = ""
        prompts_file = ""
        if self.current_model:
            prompt_log_dir, response_log_dir = self.ensure_log_directories(self.current_model)

            # Generate timestamped filenames
            responses_filename = self.get_timestamped_filename("responses", self.current_model)
            prompts_filename = self.get_timestamped_filename("prompts", self.current_model)

            responses_file = os.path.join(response_log_dir, responses_filename)
            prompts_file = os.path.join(prompt_log_dir, prompts_filename)

        # Save LLM responses to file
        if responses_file:
            with open(responses_file, 'w', encoding='utf-8') as f:
                f.write("EXACT RESPONSES FROM LLM - JSON BATCH PROCESSING\n")
                f.write("=" * 50 + "\n\n")

                # Technical details section
                f.write("TECHNICAL DETAILS OF LLM PROCESSING\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model instance name: {config.MODEL_NAME}\n")
                f.write(f"Date of processing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Calculate processing time
                if self.processing_start_time and self.processing_end_time:
                    processing_time = self.processing_end_time - self.processing_start_time
                    f.write(f"Time of LLM processing: {processing_time:.2f} seconds\n")
                else:
                    f.write("Time of LLM processing: Not available\n")

                f.write(f"Temperature setting: {config.TEMPERATURE}\n")
                f.write(f"Token count: {self.total_tokens_processed:,} tokens\n")

                # Add thinking mode information for Qwen
                if self.current_model == 'qwen':
                    f.write(f"Thinking mode: {'Enabled' if config.QWEN_USE_THINKING else 'Disabled'}\n")

                # Add detailed token breakdown for supported models
                if any(model in config.MODEL_NAME.lower() for model in ["claude", "gpt", "qwen", "olmo"]):
                    f.write(f"Input tokens: {self.total_input_tokens:,} tokens\n")
                    f.write(f"Output tokens: {self.total_output_tokens:,} tokens\n")

                f.write("\n")
                f.write("This file contains the EXACT responses received from the LLM.\n\n")
                f.write("=" * 50 + "\n")
                f.write("RESPONSES:\n")
                f.write("=" * 50 + "\n\n")
                for response in all_responses:
                    f.write(response)

        # Save prompts to file
        if prompts_file:
            with open(prompts_file, 'w', encoding='utf-8') as f:
                f.write("EXACT PROMPTS SENT TO LLM - JSON BATCH PROCESSING\n")
                f.write("=" * 50 + "\n\n")
                f.write("This file contains the EXACT prompts that were sent to the LLM.\n")
                f.write("The system message is shown once at the top, followed by user messages for each object.\n\n")

                # Write system message once at the top
                if system_message:
                    f.write("=" * 80 + "\n")
                    f.write("SYSTEM MESSAGE (used for all objects)\n")
                    f.write("=" * 80 + "\n")
                    f.write(system_message)
                    f.write("\n\n" + "=" * 80 + "\n\n")

                f.write("USER MESSAGES:\n")
                f.write("=" * 50 + "\n\n")

                # Write only user messages for each object
                for index, (object_result, user_msg) in enumerate(zip(all_results, all_user_prompts)):
                    object_id = object_result['object_id']

                    prompt_with_metadata = f"""{'='*80}
OBJECT {index + 1} - USER PROMPT
{'='*80}
Object ID: {object_id}

USER MESSAGE:
{user_msg}

{'='*80}
"""
                    f.write(prompt_with_metadata)

        # Save processing metrics to JSON
        json_file = ""
        if self.current_model:
            processing_timestamp = self.get_processing_timestamp()
            json_file = self.save_processing_metrics_to_json(
                processing_timestamp,
                self.current_model,
                config.MODEL_NAME,
                config.OUTPUT_DIR
            )

        print(f"\nLLM processing complete!")
        print(f"LLM responses saved to {responses_file}")
        print(f"LLM prompts saved to {prompts_file}")
        if json_file:
            print(f"Processing metrics saved to {json_file}")
        print(f"Total objects processed: {len(all_results)}")

        # Count successful outputs
        successful = sum(1 for result in all_results if result['success'])
        failed = len(all_results) - successful
        print(f"Successful outputs: {successful}")
        if failed > 0:
            print(f"Failed outputs: {failed}")
            print("Check the response log for details on failures.")

        return all_results

    def save_json_batch_outputs(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        output_mode: str = 'valid'
    ) -> Dict[str, Any]:
        """
        Save JSON batch processing outputs to files.
        Supports both JSON and XML output based on config.OUTPUT_EXTENSION.

        Args:
            results: List of result dictionaries from process_json_objects_framework
            output_dir: Base directory to save output files (defaults to config.OUTPUT_DIR)
            output_mode: 'raw' to extract content after <think> tags, 'valid' for structured JSON array

        Returns:
            Dictionary with statistics about saved files
        """
        output_base_dir = output_dir or config.OUTPUT_DIR

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name = self.get_run_folder_name()

        # Create timestamped processing directory
        timestamp = self.get_processing_timestamp()
        output_dir = os.path.join(output_base_dir, model_name, f"processing_{timestamp}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        stats = {
            'total': len(results),
            'saved': 0,
            'failed': 0,
            'skipped': 0,
            'files': [],
            'output_dir': output_dir
        }

        # Check if we're outputting XML or JSON
        is_xml_output = config.OUTPUT_EXTENSION == ".xml"

        def sanitize_filename(filename: str) -> str:
            """Sanitize a string to be a valid filename by replacing invalid characters."""
            # Replace characters invalid in Windows filenames
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '#']
            sanitized = filename
            for char in invalid_chars:
                sanitized = sanitized.replace(char, '_')
            # Also replace multiple underscores with single underscore
            while '__' in sanitized:
                sanitized = sanitized.replace('__', '_')
            return sanitized

        if output_mode == 'raw':
            # Raw extraction mode
            if is_xml_output:
                # Extract XML content and save as separate files
                for result in results:
                    object_id = result['object_id']
                    input_filename = result.get('input_filename', 'output')
                    object_index = result.get('object_index', 1)
                    total_objects = result.get('total_objects', 1)
                    output_content = result.get('output_content', '')
                    success = result.get('success', False)

                    if not success or not output_content or not output_content.strip():
                        print(f"  Skipping {object_id}: No valid output generated")
                        stats['skipped'] += 1
                        continue

                    # Extract XML using the appropriate function
                    extracted_xml = extract_xml_from_response(output_content)

                    if extracted_xml:
                        # Generate filename based on input file and object count
                        if total_objects == 1:
                            xml_filename = f"{input_filename}.xml"
                        else:
                            xml_filename = f"{input_filename}_{object_index}.xml"
                        xml_path = os.path.join(output_dir, xml_filename)

                        try:
                            with open(xml_path, 'w', encoding='utf-8') as f:
                                f.write(extracted_xml)

                            print(f"  Saved: {model_name}/processing_{timestamp}/{xml_filename}")
                            stats['files'].append(xml_filename)
                            stats['saved'] += 1
                        except Exception as e:
                            print(f"  Error saving {xml_filename}: {e}")
                            stats['failed'] += 1
                    else:
                        print(f"  Skipping {object_id}: No XML content found")
                        stats['skipped'] += 1
            else:
                # JSON output - Extract content after <think> tags (anything between { and })
                combined_content = []

                for result in results:
                    object_id = result['object_id']
                    output_content = result.get('output_content', '')
                    success = result.get('success', False)

                    if not success or not output_content or not output_content.strip():
                        print(f"  Skipping {object_id}: No valid output generated")
                        stats['skipped'] += 1
                        continue

                    # Extract content after <think> tags or </think> tags
                    extracted = self.extract_json_after_think(output_content)

                    if extracted:
                        combined_content.append(f"// Object: {object_id}")
                        combined_content.append(extracted)
                        combined_content.append("")  # Empty line between objects
                        stats['saved'] += 1
                    else:
                        print(f"  Skipping {object_id}: No JSON content found after <think> tags")
                        stats['skipped'] += 1

                # Save combined file
                combined_filename = "output.json"
                combined_path = os.path.join(output_dir, combined_filename)

                try:
                    with open(combined_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(combined_content))

                    print(f"  Saved: {model_name}/processing_{timestamp}/{combined_filename}")
                    stats['files'].append(combined_filename)

                except Exception as e:
                    print(f"  Error saving combined output: {e}")
                    stats['failed'] = 1

        else:
            # Valid mode
            if is_xml_output:
                # Extract XML content and save as separate files
                for result in results:
                    object_id = result['object_id']
                    input_filename = result.get('input_filename', 'output')
                    object_index = result.get('object_index', 1)
                    total_objects = result.get('total_objects', 1)
                    output_content = result.get('output_content', '')
                    success = result.get('success', False)

                    if not success or not output_content or not output_content.strip():
                        print(f"  Skipping {object_id}: No valid output generated")
                        stats['skipped'] += 1
                        continue

                    # Extract XML using the appropriate function
                    extracted_xml = extract_xml_from_response(output_content)

                    if extracted_xml:
                        # Generate filename based on input file and object count
                        if total_objects == 1:
                            xml_filename = f"{input_filename}.xml"
                        else:
                            xml_filename = f"{input_filename}_{object_index}.xml"
                        xml_path = os.path.join(output_dir, xml_filename)

                        try:
                            with open(xml_path, 'w', encoding='utf-8') as f:
                                f.write(extracted_xml)

                            print(f"  Saved: {model_name}/processing_{timestamp}/{xml_filename}")
                            stats['files'].append(xml_filename)
                            stats['saved'] += 1
                        except Exception as e:
                            print(f"  Error saving {xml_filename}: {e}")
                            stats['failed'] += 1
                    else:
                        print(f"  Skipping {object_id}: No XML content found")
                        stats['skipped'] += 1
            else:
                # Valid JSON mode: Create properly structured JSON array
                combined_data = []

                for result in results:
                    object_id = result['object_id']
                    output_content = result.get('output_content', '')
                    success = result.get('success', False)
                    metadata = result.get('metadata', {})

                    if not success or not output_content or not output_content.strip():
                        print(f"  Skipping {object_id}: No valid output generated")
                        stats['skipped'] += 1
                        continue

                    # Try to parse as JSON, otherwise store as string
                    try:
                        parsed_output = json.loads(output_content)
                    except (json.JSONDecodeError, ValueError):
                        # Try to extract JSON from response
                        extracted = self.extract_json_after_think(output_content)
                        if extracted:
                            try:
                                parsed_output = json.loads(extracted)
                            except (json.JSONDecodeError, ValueError):
                                parsed_output = output_content.strip()
                        else:
                            parsed_output = output_content.strip()

                    # Handle array responses: if LLM returns an array of objects with the same ID,
                    # merge them into a single object (common in disambiguation with both writer and place)
                    if isinstance(parsed_output, list) and len(parsed_output) > 0:
                        # Check if all items have the same ID
                        ids = [item.get('id') for item in parsed_output if isinstance(item, dict) and 'id' in item]
                        if ids and len(set(ids)) == 1:
                            # All items have the same ID - merge them
                            # Special handling for 'conf' field: rename to field-specific confidence
                            merged = {}
                            for item in parsed_output:
                                if isinstance(item, dict):
                                    # If this item has a 'conf' field, rename it based on what field it's for
                                    item_copy = item.copy()
                                    if 'conf' in item_copy:
                                        conf_value = item_copy.pop('conf')
                                        # Determine which field this confidence is for
                                        if 'writer' in item_copy or 'gnd' in item_copy:
                                            item_copy['writer_conf'] = conf_value
                                        elif 'place' in item_copy or 'geonamesId' in item_copy:
                                            item_copy['place_conf'] = conf_value
                                        else:
                                            # Generic conf - keep as is
                                            item_copy['conf'] = conf_value
                                    merged.update(item_copy)
                            parsed_output = merged
                        else:
                            # Different IDs or no IDs - keep as array (will be wrapped in output key)
                            pass

                    # Build output entry with metadata if available
                    output_entry = {}

                    # Include metadata keys at the top level (similar to how plaintext uses 'filename')
                    output_entry.update(metadata)

                    # Add the output
                    output_entry['output'] = parsed_output

                    combined_data.append(output_entry)
                    stats['saved'] += 1

                # Save combined JSON file
                combined_filename = "output.json"
                combined_path = os.path.join(output_dir, combined_filename)

                try:
                    with open(combined_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, indent=2, ensure_ascii=False)

                    print(f"  Saved: {model_name}/processing_{timestamp}/{combined_filename}")
                    stats['files'].append(combined_filename)

                except Exception as e:
                    print(f"  Error saving combined output: {e}")
                    stats['failed'] = 1

        # Print summary
        output_type = "XML" if is_xml_output else "JSON"
        print(f"\n{'='*50}")
        print(f"{output_type} Batch Output Summary ({output_mode} mode):")
        print(f"{'='*50}")
        print(f"Total objects processed: {stats['total']}")
        print(f"Successfully saved: {stats['saved']}")
        if stats['skipped'] > 0:
            print(f"Skipped (no valid output): {stats['skipped']}")
        if stats['failed'] > 0:
            print(f"Failed to save: {stats['failed']}")
        print(f"Output directory: {output_dir}")

        return stats

    @staticmethod
    def _iter_balanced_json_spans(content: str):
        """Yield every top-level {...} / [...] substring in reading order.

        String-aware: braces and brackets inside JSON string literals (and any
        prose the model writes between blocks) do not affect nesting, so a run
        of reasoning that happens to contain a stray '{' cannot merge two real
        blocks into one malformed span. Each yielded span is a self-contained,
        balanced value — an object or an array — but is not guaranteed to parse
        (the model may still emit invalid JSON inside it); the caller validates.
        """
        openers = {'{': '}', '[': ']'}
        i, n = 0, len(content)
        while i < n:
            ch = content[i]
            if ch in openers:
                close = openers[ch]
                depth = 0
                in_str = False
                esc = False
                j = i
                while j < n:
                    cj = content[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif cj == '\\':
                            esc = True
                        elif cj == '"':
                            in_str = False
                    else:
                        if cj == '"':
                            in_str = True
                        elif cj == ch:
                            depth += 1
                        elif cj == close:
                            depth -= 1
                            if depth == 0:
                                yield content[i:j + 1]
                                i = j
                                break
                    j += 1
                else:
                    # No matching close before end of string; nothing balanced
                    # starts here — advance one char and keep scanning.
                    pass
            i += 1

    def extract_json_after_think(self, content: str) -> str:
        """
        Extract the JSON answer that follows the model's reasoning.

        Reasoning-parser endpoints (e.g. the DH Infra qwen3.5 cluster) wrap the
        chain-of-thought in <think>...</think> and place the answer after it, but
        the model also frequently writes its reasoning as plain prose (no tags)
        and then emits the answer — sometimes emitting the JSON block more than
        once. The answer may be a JSON object ({...}) OR a JSON array ([...]) —
        the disambiguation task returns an array — so both are handled.

        Strategy, robust to prose reasoning and repeated blocks:
          1. Drop everything up to the last </think>.
          2. Prefer the contents of ```...``` code fences, last fence first
             (the final fenced block is the model's committed answer).
          3. Otherwise scan for balanced top-level {...}/[...] spans, last first.
          4. In every case, return the last candidate that actually parses as
             JSON; only if none parse fall back to the first-open/last-close
             heuristic so behaviour is never worse than a naive slice.

        Args:
            content: Raw output from LLM

        Returns:
            Extracted JSON string (parseable when possible), or empty string.
        """
        if not content or not content.strip():
            return ""

        # 1) Drop everything up to and including the reasoning block so braces or
        # brackets inside the thinking can't be picked up as the answer. Use the
        # last </think> in case the reasoning echoes the tag.
        close_tag = '</think>'
        tag_idx = content.lower().rfind(close_tag)
        if tag_idx != -1:
            content = content[tag_idx + len(close_tag):]
        content = content.strip()

        # 2) Fenced code blocks first. Take the JSON value inside each fence and
        # try them from last to first — when the model restates its answer, the
        # final fence is the committed one.
        fence_bodies = re.findall(r'```[a-zA-Z0-9_]*\s*\n?(.*?)```', content, re.DOTALL)
        for body in reversed(fence_bodies):
            for span in reversed(list(self._iter_balanced_json_spans(body))):
                try:
                    json.loads(span)
                    return span.strip()
                except (json.JSONDecodeError, ValueError):
                    continue

        # 3) No usable fenced JSON — scan the whole (fence-free or fence-mixed)
        # content for balanced spans and return the last one that parses.
        spans = list(self._iter_balanced_json_spans(content))
        for span in reversed(spans):
            try:
                json.loads(span)
                return span.strip()
            except (json.JSONDecodeError, ValueError):
                continue

        # 4) Nothing parsed. Fall back to the legacy first-open/last-close slice
        # (better than nothing for a single truncated block); prefer the last
        # balanced span's opener if we found any.
        if spans:
            return spans[-1].strip()
        candidates = []
        first_brace = content.find('{')
        if first_brace != -1:
            candidates.append((first_brace, '}'))
        first_bracket = content.find('[')
        if first_bracket != -1:
            candidates.append((first_bracket, ']'))
        if not candidates:
            return ""
        start, close_char = min(candidates, key=lambda c: c[0])
        end = content.rfind(close_char)
        if end != -1 and end > start:
            return content[start:end + 1].strip()
        return ""

    def clean_rdf_output(self, content: str) -> str:
        """
        Clean RDF/XML output by removing XML declarations and outer root tags
        so it can be combined into a single file.

        Args:
            content: Raw RDF/XML output from LLM

        Returns:
            Cleaned content ready for combination
        """
        if not content or not content.strip():
            return ""

        # Remove XML declaration if present
        content = re.sub(r'<\?xml[^>]*\?>', '', content, flags=re.IGNORECASE)
        content = content.strip()

        # Remove outer <rdf:RDF> tags if present (we'll add our own)
        if content.startswith('<rdf:RDF') or content.startswith('<RDF'):
            # Find the closing tag
            if '</rdf:RDF>' in content:
                start = content.find('>') + 1
                end = content.rfind('</rdf:RDF>')
                content = content[start:end].strip()
            elif '</RDF>' in content:
                start = content.find('>') + 1
                end = content.rfind('</RDF>')
                content = content[start:end].strip()

        return content

    def save_text_processing_outputs(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save text processing outputs based on configured OUTPUT_EXTENSION.
        Routes to appropriate saver (XML or JSON) based on configuration.

        Args:
            results: List of result dictionaries from process_text_files_framework
            output_dir: Base directory to save output files (defaults to config.OUTPUT_DIR)

        Returns:
            Dictionary with statistics about saved files
        """
        output_ext = config.OUTPUT_EXTENSION

        if output_ext == ".json":
            # Use JSON output mode
            output_mode = config.JSON_OUTPUT_MODE
            return self.save_json_text_outputs(results, output_dir, output_mode)
        else:
            # Default to XML (backward compatible)
            return self.save_tei_xml_outputs(results, output_dir)

    def save_json_text_outputs(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        output_mode: str = 'json-array'
    ) -> Dict[str, Any]:
        """
        Save JSON outputs from text processing workflow.

        Args:
            results: List of result dictionaries from process_text_files_framework
            output_dir: Base directory to save output files (defaults to config.OUTPUT_DIR)
            output_mode: 'raw' to extract content, 'json-array' for structured JSON array

        Returns:
            Dictionary with statistics about saved files
        """
        output_base_dir = output_dir or config.OUTPUT_DIR

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name = self.get_run_folder_name()

        # Create timestamped processing directory
        timestamp = self.get_processing_timestamp()
        output_dir = os.path.join(output_base_dir, model_name, f"processing_{timestamp}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        stats = {
            'total': len(results),
            'saved': 0,
            'failed': 0,
            'skipped': 0,
            'files': [],
            'output_dir': output_dir
        }

        if output_mode == 'raw':
            # Raw extraction mode: Save each file's JSON separately
            for result in results:
                filename = result['filename']
                output_content = result.get('output_content', '')
                success = result.get('success', False)

                # Generate output filename (replace .txt with .json)
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.json"
                output_path = os.path.join(output_dir, output_filename)

                if not success or not output_content or not output_content.strip():
                    print(f"  Skipping {filename}: No valid JSON generated")
                    stats['skipped'] += 1
                    continue

                try:
                    # Save the raw JSON content
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_content)

                    print(f"  Saved: {model_name}/processing_{timestamp}/{output_filename}")
                    stats['saved'] += 1
                    stats['files'].append(output_filename)

                except Exception as e:
                    print(f"  Error saving {output_filename}: {e}")
                    stats['failed'] += 1

        else:
            # json-array mode: Create properly structured JSON array
            combined_data = []

            for result in results:
                filename = result['filename']
                output_content = result.get('output_content', '')
                success = result.get('success', False)

                if not success or not output_content or not output_content.strip():
                    print(f"  Skipping {filename}: No valid JSON generated")
                    stats['skipped'] += 1
                    continue

                # Try to parse as JSON, otherwise store as string
                try:
                    parsed_output = json.loads(output_content)
                except (json.JSONDecodeError, ValueError):
                    parsed_output = output_content.strip()

                combined_data.append({
                    'filename': filename,
                    'output': parsed_output
                })
                stats['saved'] += 1

            # Save combined JSON file
            combined_filename = "output.json"
            combined_path = os.path.join(output_dir, combined_filename)

            try:
                with open(combined_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)

                print(f"  Saved: {model_name}/processing_{timestamp}/{combined_filename}")
                stats['files'].append(combined_filename)

            except Exception as e:
                print(f"  Error saving combined output: {e}")
                stats['failed'] = 1

        # Print summary
        print(f"\n{'='*50}")
        print(f"JSON Output Summary ({output_mode} mode):")
        print(f"{'='*50}")
        print(f"Total files processed: {stats['total']}")
        print(f"Successfully saved: {stats['saved']}")
        if stats['skipped'] > 0:
            print(f"Skipped (no valid JSON): {stats['skipped']}")
        if stats['failed'] > 0:
            print(f"Failed to save: {stats['failed']}")
        print(f"Output directory: {output_dir}")

        return stats

    def save_tei_xml_outputs(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save TEI XML outputs to individual files in a timestamped processing subdirectory.

        Args:
            results: List of result dictionaries from process_text_files_framework
            output_dir: Base directory to save XML files (defaults to config.OUTPUT_DIR)

        Returns:
            Dictionary with statistics about saved files
        """
        output_base_dir = output_dir or config.OUTPUT_DIR
        output_ext = config.OUTPUT_EXTENSION

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name = self.get_run_folder_name()

        # Create timestamped processing directory
        timestamp = self.get_processing_timestamp()
        output_dir = os.path.join(output_base_dir, model_name, f"processing_{timestamp}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        stats = {
            'total': len(results),
            'saved': 0,
            'failed': 0,
            'skipped': 0,
            'files': [],
            'output_dir': output_dir  # Store the actual output directory
        }

        for result in results:
            filename = result['filename']
            tei_xml = result.get('tei_xml', '')
            success = result.get('success', False)

            # Generate output filename (replace .txt with .xml)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}{output_ext}"
            output_path = os.path.join(output_dir, output_filename)

            # Only save if encoding was successful and we have XML content
            if not success or not tei_xml or not tei_xml.strip():
                print(f"  Skipping {filename}: No valid TEI XML generated")
                stats['skipped'] += 1
                continue

            try:
                # Save the TEI XML to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(tei_xml)

                print(f"  Saved: {model_name}/processing_{timestamp}/{output_filename}")
                stats['saved'] += 1
                stats['files'].append(output_filename)

            except Exception as e:
                print(f"  Error saving {output_filename}: {e}")
                stats['failed'] += 1

        # Print summary
        print(f"\n{'='*50}")
        print(f"TEI XML Output Summary:")
        print(f"{'='*50}")
        print(f"Total files processed: {stats['total']}")
        print(f"Successfully saved: {stats['saved']}")
        if stats['skipped'] > 0:
            print(f"Skipped (no valid XML): {stats['skipped']}")
        if stats['failed'] > 0:
            print(f"Failed to save: {stats['failed']}")
        print(f"Output directory: {output_dir}")

        return stats

    def create_xml_update_mapping_from_results(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create a detailed XML update mapping directly from analysis results.
        Returns a dictionary organized by filename with detailed replacement information.

        Part of Editorial Intervention Analysis Workflow.
        Creates mappings for updating existing XML files with TEI encodings.

        Uses configurable keys from config:
        - XML_MAPPING_FILENAME_KEY, XML_MAPPING_ELEMENT_ID_KEY, XML_MAPPING_XPATH_KEY
        - JSON_CONTEXT_KEY for context text
        - JSON_OUTPUT_* fields for LLM output structure
        """
        try:
            xml_update_map = {}
            context_key = config.JSON_CONTEXT_KEY
            filename_key = config.XML_MAPPING_FILENAME_KEY
            element_id_key = config.XML_MAPPING_ELEMENT_ID_KEY
            xpath_key = config.XML_MAPPING_XPATH_KEY

            # Get configurable JSON output field names
            tei_field = config.JSON_OUTPUT_TEI_FIELD
            type_field = config.JSON_OUTPUT_TYPE_FIELD
            explanation_field = config.JSON_OUTPUT_EXPLANATION_FIELD

            for element_result in results:
                filename = element_result.get(filename_key)
                element_id = element_result.get(element_id_key)
                xpath = element_result.get(xpath_key)
                context_text = element_result.get(context_key, '')

                if not filename:
                    print(f"Warning: Missing '{filename_key}' in result, skipping element")
                    continue

                if filename not in xml_update_map:
                    xml_update_map[filename] = []

                # Create replacements for this element
                element_replacements = []
                for item_seq, encoding_info in element_result['tei_encodings'].items():
                    # Strip duplicate index suffix (e.g., "__2", "__3") from key
                    # These are added by parse_json_response when LLM returns duplicate keys
                    original_item = re.sub(r'__\d+$', '', item_seq)
                    replacement = {
                        'original_item': original_item,
                        'tei_encoding': encoding_info.get(tei_field, ''),
                        'intervention_type': encoding_info.get(type_field, 'unknown')
                    }
                    # Only add explanation if the field is configured
                    if explanation_field:
                        replacement['explanation'] = encoding_info.get(explanation_field, '')
                    element_replacements.append(replacement)

                # Build element update using configurable keys
                element_update = {
                    element_id_key: element_id,
                    xpath_key: xpath,
                    context_key: context_text,
                    'replacements': element_replacements,
                    'num_replacements': len(element_replacements)
                }

                xml_update_map[filename].append(element_update)

            return xml_update_map

        except Exception as e:
            print(f"Error creating XML update mapping: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def save_xml_update_mapping(
        self,
        xml_update_map: Dict[str, List[Dict]],
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        output_base_dir: Optional[str] = None
    ) -> str:
        """
        Save the XML update mapping to a file in the processing run's log directory.
        Returns the path to the saved file.

        Part of Editorial Intervention Analysis Workflow.
        Saves mappings for updating existing XML files with TEI encodings.

        Args:
            xml_update_map: Dictionary mapping filenames to element updates
            model: Model type (e.g., 'qwen', 'claude')
            model_name: Model instance name
            output_base_dir: Base output directory (defaults to config.OUTPUT_DIR)
        """
        output_base_dir = output_base_dir or config.OUTPUT_DIR

        # Use current model if not provided
        if model is None:
            model = self.current_model or "unknown"

        # Use config model name if not provided
        if model_name is None:
            model_name = config.MODEL_NAME

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name_with_suffix = self.get_run_folder_name()

        # Create path to log directory in processing output
        timestamp = self.get_processing_timestamp()
        processing_dir = os.path.join(output_base_dir, model_name_with_suffix, f"processing_{timestamp}")
        log_dir = os.path.join(processing_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        # Path to the XML mapping file (saved in processing folder, not log folder)
        xml_mapping_file = os.path.join(processing_dir, "xml_update_mapping.json")

        # Save the mapping
        with open(xml_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(xml_update_map, f, indent=2, ensure_ascii=False)

        print(f"XML update mapping saved to: {xml_mapping_file}")
        return xml_mapping_file

    def save_key_extraction_json_output(
        self,
        results: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        output_mode: str = 'json-array'
    ) -> Dict[str, Any]:
        """
        Save key extraction results as JSON output.
        Alternative to XML update mapping for key extraction workflow.

        Args:
            results: List of result dictionaries from process_text_segments_framework
            output_dir: Base directory to save output files (defaults to config.OUTPUT_DIR)
            output_mode: 'raw' for individual files, 'json-array' for combined array

        Returns:
            Dictionary with statistics about saved files
        """
        output_base_dir = output_dir or config.OUTPUT_DIR

        # Create run-specific subdirectory (RUN_NAME, or MODEL_NAME by default)
        model_name = self.get_run_folder_name()

        # Create timestamped processing directory
        timestamp = self.get_processing_timestamp()
        output_dir = os.path.join(output_base_dir, model_name, f"processing_{timestamp}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        stats = {
            'total': len(results),
            'saved': 0,
            'failed': 0,
            'skipped': 0,
            'files': [],
            'output_dir': output_dir
        }

        # Determine extraction type to check for appropriate results field
        extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')
        results_field = 'results' if extraction_type == 'information_extraction' else 'tei_encodings'

        if output_mode == 'raw':
            # Raw mode: Save each element's analysis separately
            for result in results:
                # Try to get an identifier for the element
                if extraction_type == 'information_extraction':
                    # For info extraction, use first metadata key as identifier
                    metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                    element_id = result.get(metadata_keys[0] if metadata_keys else 'id', f"element_{stats['total']}")
                else:
                    # For TEI encoding, use element_id
                    element_id = result.get(config.XML_MAPPING_ELEMENT_ID_KEY) or result.get('id') or f"element_{stats['total']}"

                output_filename = f"{element_id}.json"
                output_path = os.path.join(output_dir, output_filename)

                # Check if we have results
                analysis_data = result.get(results_field, {})
                if not analysis_data:
                    print(f"  Skipping {element_id}: No {results_field} generated")
                    stats['skipped'] += 1
                    continue

                try:
                    # Save the result as JSON
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    print(f"  Saved: {model_name}/processing_{timestamp}/{output_filename}")
                    stats['saved'] += 1
                    stats['files'].append(output_filename)

                except Exception as e:
                    print(f"  Error saving {output_filename}: {e}")
                    stats['failed'] += 1

        else:
            # json-array mode: Combine all results into one file
            combined_data = []

            for result in results:
                # Try to get identifier based on extraction type
                if extraction_type == 'information_extraction':
                    metadata_keys = getattr(config, 'JSON_METADATA_KEYS_INFO', [])
                    element_id = result.get(metadata_keys[0] if metadata_keys else 'id', f"element_{len(combined_data)}")
                else:
                    element_id = result.get(config.XML_MAPPING_ELEMENT_ID_KEY) or result.get('id') or f"element_{len(combined_data)}"

                # Check if we have results
                analysis_data = result.get(results_field, {})
                if not analysis_data:
                    print(f"  Skipping {element_id}: No {results_field} generated")
                    stats['skipped'] += 1
                    continue

                # Add to combined data - for info extraction, use result directly; for TEI encoding, wrap it
                if extraction_type == 'information_extraction':
                    combined_data.append(result)
                else:
                    combined_data.append({
                        'element_id': element_id,
                        'data': result
                    })
                stats['saved'] += 1

            # Save combined JSON file
            combined_filename = "output.json"
            combined_path = os.path.join(output_dir, combined_filename)

            try:
                with open(combined_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)

                print(f"  Saved: {model_name}/processing_{timestamp}/{combined_filename}")
                stats['files'].append(combined_filename)

            except Exception as e:
                print(f"  Error saving combined output: {e}")
                stats['failed'] = 1

        # Print summary
        extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')
        workflow_name = "Information Extraction" if extraction_type == 'information_extraction' else "TEI Encoding"

        print(f"\n{'='*50}")
        print(f"{workflow_name} JSON Output Summary ({output_mode} mode):")
        print(f"{'='*50}")
        print(f"Total elements processed: {stats['total']}")
        print(f"Successfully saved: {stats['saved']}")
        if stats['skipped'] > 0:
            results_field = 'results' if extraction_type == 'information_extraction' else 'tei_encodings'
            print(f"Skipped (no {results_field}): {stats['skipped']}")
        if stats['failed'] > 0:
            print(f"Failed to save: {stats['failed']}")
        print(f"Output directory: {output_dir}")

        return stats


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LLM Processing Coordinator for TEI XML Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for workflow selection)
  python llm_processing.py

  # Run text processing workflow directly
  python llm_processing.py --workflow text

  # Run JSON processing workflow directly
  python llm_processing.py --workflow json

  # Use a project-specific configuration file instead of config.py
  python llm_processing.py --config ../configs/my_run.py --workflow json

  # Show version
  python llm_processing.py --version
        """
    )

    parser.add_argument(
        '--workflow',
        '-w',
        choices=['text', 'json', 'text_processing', 'json_processing'],
        help='Workflow to run: "text" or "json". If not specified, interactive mode is used.'
    )

    parser.add_argument(
        '--config',
        '-c',
        help='Path to a configuration file to use instead of config.py '
             '(also settable via the LLM_PROCESSING_CONFIG environment variable).'
    )

    parser.add_argument(
        '--version',
        '-v',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the LLM processing workflow.

    Supports both command-line arguments and interactive mode:
    - Use --workflow to specify workflow directly
    - Run without arguments for interactive workflow selection
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize the coordinator
    coordinator = LLMProcessingCoordinator()

    # Determine workflow choice
    if args.workflow:
        # Map short forms to full names
        workflow_map = {
            'text': 'text_processing',
            'json': 'json_processing',
            'text_processing': 'text_processing',
            'json_processing': 'json_processing'
        }
        workflow = workflow_map[args.workflow]
    else:
        # Interactive mode: get workflow choice from user
        workflow = coordinator.get_workflow_choice()

    # Run the selected workflow
    if workflow == 'text_processing':
        coordinator.run_text_processing_workflow()
    elif workflow == 'json_processing':
        coordinator.run_json_processing_workflow()
    else:
        print(f"Error: Unknown workflow '{workflow}'")
        sys.exit(1)

if __name__ == "__main__":
    main()