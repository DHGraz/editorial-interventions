"""
AI2 OLMo processor for TEI XML generation.

This module provides OLMo-specific implementations for encoding text segments
and complete plaintext files into TEI XML format using llama.cpp for local inference.
"""

# Standard library imports
import json
import multiprocessing
import sys
from typing import Any, Dict, Optional, Tuple

# Third-party imports
# (llama_cpp imported conditionally below)

# Local imports
import config
from utils.utils import (
    check_response_completeness,
    create_segment_error_response,
    create_test_response,
    create_test_tei_response,
    create_text_encoding_error,
    create_text_encoding_result,
    extract_xml_from_response,
    get_gpu_usage,
    get_system_usage,
    parse_json_response,
    prepare_prompts_for_segment,
    validate_text_data
)

# Initialize availability flags
LLAMA_AVAILABLE = False

# Only import llama_cpp when needed
if config.ENABLE_API_CALLS:
    try:
        from llama_cpp import Llama
        LLAMA_AVAILABLE = True
    except ImportError:
        print("Warning: llama_cpp not installed. Only test mode will be available.")
        LLAMA_AVAILABLE = False

# Global model instance to avoid reloading
_olmo_model = None


# Model parameters
MODEL_PARAMS = {
    'n_gpu_layers': -1 if config.USE_GPU else 0,  # -1 for GPU, 0 for CPU
    'n_ctx': 8192,      # Context window
    'n_batch': 1024,     # Batch size
    'max_tokens': config.MAX_TOKENS,  # Max tokens per response
    'n_threads': multiprocessing.cpu_count() - 1,
    'temperature': config.TEMPERATURE
}
if getattr(config, 'SEED', None) is not None:
    MODEL_PARAMS['seed'] = config.SEED  # fixed sampling seed for repeatable runs

PROMPT_FORMATS = ("chat_template", "legacy")


def get_prompt_format() -> str:
    """
    Get the configured prompt format for OLMo (config.OLMO_PROMPT_FORMAT).

    - "chat_template" (default): llama.cpp renders the chat template stored in
      the GGUF file, with separate system, user and assistant turns.
    - "legacy": the hand-built prompt used until September 2026, which puts the
      system prompt in the user turn and the user message in the assistant turn.
      Only for reproducing runs made with that version.

    Returns:
        The prompt format name.

    Raises:
        ValueError: If OLMO_PROMPT_FORMAT has an unknown value.
    """
    prompt_format = getattr(config, 'OLMO_PROMPT_FORMAT', 'chat_template')
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"OLMO_PROMPT_FORMAT must be one of {PROMPT_FORMATS}, got '{prompt_format}'")
    return prompt_format


def _format_legacy_prompt(system_message: str, user_message: str) -> str:
    """Build the legacy prompt exactly as the published OLMo run received it."""
    return f"""<|user|>
{system_message}


<|assistant|>
{user_message}
"""


def _run_olmo_inference(llm: Any, system_message: str, user_message: str) -> Tuple[str, str, int, int]:
    """
    Generate a response with the configured prompt format.

    Checks the context window before generating and shortens max_tokens if needed.

    Args:
        llm: Loaded llama_cpp.Llama model.
        system_message: System prompt message.
        user_message: User prompt message.

    Returns:
        Tuple of (raw_response, finish_reason, input_tokens, output_tokens).

    Raises:
        Exception: If the prompt leaves too little room for a response.
    """
    prompt_format = get_prompt_format()
    if prompt_format == "legacy":
        prompt_text = _format_legacy_prompt(system_message, user_message)
    else:
        # Only used to estimate the prompt length; llama.cpp renders the real prompt.
        prompt_text = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>\n"

    # Check if prompt exceeds context window BEFORE making API call
    prompt_tokens = count_tokens(prompt_text)
    context_window = MODEL_PARAMS['n_ctx']
    max_output_tokens = MODEL_PARAMS['max_tokens']

    # Calculate available space for response
    available_tokens = context_window - prompt_tokens

    # Fail immediately if prompt is too large (leaving no room for response)
    # 200 tokens is an estimated minimum needed for a useful response
    if available_tokens < 200:
        raise Exception(
            f"Prompt is too large and leaves insufficient space for response.\n"
            f"  Prompt tokens: {prompt_tokens}\n"
            f"  Context window: {context_window}\n"
            f"  Available for output: {available_tokens}\n"
            f"  Solution: Reduce prompt size (use fewer examples, shorter text, etc.)"
        )

    # Warn if output will be constrained, and adjust max_tokens if needed
    actual_max_tokens = max_output_tokens
    if prompt_tokens + max_output_tokens > context_window:
        actual_max_tokens = available_tokens
        print(f"Warning: Reducing max_tokens to {actual_max_tokens} to fit context window ({context_window} tokens)")

    if prompt_format == "legacy":
        response = llm(
            prompt_text,
            max_tokens=actual_max_tokens,
            temperature=MODEL_PARAMS['temperature'],
            stop=["<|user|>", "<|assistant|>"]
        )
        raw_response = response['choices'][0]['text'].strip()
        finish_reason = response['choices'][0].get('finish_reason', '')
        input_tokens = count_tokens(prompt_text)
        output_tokens = count_tokens(raw_response)
    else:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=actual_max_tokens,
            temperature=MODEL_PARAMS['temperature']
        )
        raw_response = (response['choices'][0]['message'].get('content') or '').strip()
        finish_reason = response['choices'][0].get('finish_reason', '')
        usage = response.get('usage') or {}
        input_tokens = usage.get('prompt_tokens') or count_tokens(prompt_text)
        output_tokens = usage.get('completion_tokens') or count_tokens(raw_response)

    return raw_response, finish_reason, input_tokens, output_tokens


def count_tokens(text: str) -> int:
    """
    Count tokens in text using llama_cpp tokenizer for OLMo.

    Args:
        text: Text string to count tokens for.

    Returns:
        Token count, or 0 if counting fails. Falls back to approximate count
        (words * 1.3) if tokenizer is unavailable.
    """
    if not text:
        return 0

    try:
        model = get_olmo_model()
        if model:
            return len(model.tokenize(text.encode()))
    except Exception:
        # Token counting failed - fallback to approximate count
        return int(len(text.split()) * 1.3)

    return 0


def get_olmo_model() -> Optional[Any]:
    """
    Get or initialize the OLMo model using llama.cpp (singleton pattern).

    Returns:
        Llama model instance or None if not available
    """
    global _olmo_model

    if _olmo_model is None and config.ENABLE_API_CALLS:
        # Check if llama_cpp is available
        if not LLAMA_AVAILABLE:
            error_msg = "llama-cpp-python is not installed. Please install it to use local models.\n"
            error_msg += "Install with: pip install llama-cpp-python\n"
            error_msg += "For CPU-only: pip install llama-cpp-python\n"
            error_msg += "For GPU (CUDA): pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            print(error_msg, file=sys.stderr)
            raise ImportError("llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")

        try:
            _olmo_model = Llama(
                model_path=config.MODEL_PATH_OLMO2,
                **MODEL_PARAMS,
                verbose=False
            )
            # Without an embedded chat template llama.cpp would fall back to a generic
            # format, which is wrong for OLMo 2.
            metadata = getattr(_olmo_model, 'metadata', None) or {}
            if get_prompt_format() == "chat_template" and not metadata.get('tokenizer.chat_template'):
                raise ValueError(
                    "The GGUF file contains no chat template (tokenizer.chat_template). "
                    "Use an OLMo 2 Instruct GGUF that includes it."
                )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading OLMo model: {e}")
            _olmo_model = None

    return _olmo_model


def encode_text_segment_olmo(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any], Dict[str, Any]]:
    """
    OLMo-specific implementation for encoding text segments extracted from JSON objects.

    Encodes text segments (values from specific keys in JSON) into TEI encodings.
    Includes context window validation and response completeness checking.

    Args:
        text_segment: Dictionary containing text segment data with keys to extract.
            Must contain keys specified in config.JSON_CONTEXT_KEY and config.JSON_ITEMS_KEY.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary mapping extracted key values to their TEI encodings.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from OLMo model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system).
            - parsing_metadata: Dictionary with parse_success and parse_method fields.

    Raises:
        Exception: If prompt is too large for context window or model returns empty response.
    """
    try:
        # Prepare prompts using centralized routing logic
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            results = create_test_response(items_to_analyze)
            test_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"

            # For test mode, estimate token count and get system usage
            test_prompt = f"<|system|>\n{system_message}\n\n<|user|>\n{user_message}"
            input_tokens = count_tokens(test_prompt)
            output_tokens = count_tokens(test_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage()}

            return results, (system_message, user_message), test_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # Get the model instance
        if coordinator:
            llm = get_olmo_model()
            if not llm:
                raise Exception("Failed to initialize OLMo model")
        else:
            raise Exception("Coordinator required for OLMo in API mode")

        # Get GPU usage before inference
        gpu_usage_before = get_gpu_usage()

        # Generate response (prompt format from config.OLMO_PROMPT_FORMAT)
        raw_response, finish_reason, input_tokens, output_tokens = _run_olmo_inference(
            llm, system_message, user_message
        )

        # Check if response was truncated
        if finish_reason == 'length':
            print("Warning: Response truncated. Consider increasing MAX_TOKENS in config.py")

        # Check for incomplete response indicators
        if not raw_response:
            raise Exception("Model returned empty response")

        # Check response completeness for JSON format
        completeness = check_response_completeness(raw_response, expected_format="json")
        if not completeness['is_complete']:
            for warning in completeness['warnings']:
                print(f"Warning: {warning}")

        # Get GPU usage after inference
        gpu_usage_after = get_gpu_usage()

        total_tokens = input_tokens + output_tokens

        # Combine GPU usage info
        gpu_usage = {
            'before': gpu_usage_before,
            'after': gpu_usage_after,
            'system': get_system_usage()
        }

        # Parse the response
        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)
        
        parsing_metadata = {
            'parse_success': parse_success,
            'parse_method': parse_method
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage, parsing_metadata

    except Exception as e:
        # Create error response for all items
        items_key = config.JSON_ITEMS_KEY
        
        # Check if this is a model loading error - these should be fatal
        error_str = str(e).lower()
        if "failed to initialize" in error_str or "model file not found" in error_str or "llama-cpp-python" in error_str:
            # Fatal error - re-raise to stop execution
            raise RuntimeError(f"Model loading failed: {e}") from e
        
        # For other errors, return error response
        results = create_segment_error_response(
            text_segment.get(items_key, []), "OLMo", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        parsing_metadata = {"parse_success": False, "parse_method": "error"}
        return results, ("", ""), error_response, 0, 0, 0, {}, parsing_metadata


def encode_text_olmo(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    OLMo-specific implementation for encoding a complete plaintext file into TEI XML.

    Includes context window validation and response completeness checking.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from OLMo model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system).

    Raises:
        ValueError: If 'filename' or 'content' fields are missing from text_data.
        Exception: If prompt is too large for context window or model returns empty response.
    """
    try:
        # Validate and extract required fields
        filename, text_content = validate_text_data(text_data)

        # Create the prompt using the coordinator's methods
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_text(text_content)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            user_message = f"Encode this text: {text_content}"

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Test mode - return realistic fake TEI XML response
            raw_response = create_test_tei_response("OLMo", filename)

            # For test mode, estimate token count and get system usage
            test_prompt = f"<|system|>\n{system_message}\n\n<|user|>\n{user_message}"
            input_tokens = count_tokens(test_prompt)
            output_tokens = count_tokens(raw_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage()}

            # Parse the response to extract TEI XML
            tei_xml = extract_xml_from_response(raw_response)

            results = create_text_encoding_result(tei_xml)

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # API calls enabled - use actual model
        # Get the model instance
        llm = get_olmo_model()
        if not llm:
            raise Exception("Failed to initialize OLMo model")

        # Get GPU usage before inference
        gpu_usage_before = get_gpu_usage()

        # Generate response (prompt format from config.OLMO_PROMPT_FORMAT)
        raw_response, finish_reason, input_tokens, output_tokens = _run_olmo_inference(
            llm, system_message, user_message
        )

        # Check if response was truncated
        if finish_reason == 'length':
            print("Warning: Response truncated. Consider increasing MAX_TOKENS in config.py")

        # Check for incomplete response indicators
        if not raw_response:
            raise Exception("Model returned empty response")

        # Check response completeness for XML format
        completeness = check_response_completeness(raw_response, expected_format="xml")
        if not completeness['is_complete']:
            for warning in completeness['warnings']:
                print(f"Warning: {warning}")

        # Get GPU usage after inference
        gpu_usage_after = get_gpu_usage()

        total_tokens = input_tokens + output_tokens

        # Combine GPU usage info
        gpu_usage = {
            'before': gpu_usage_before,
            'after': gpu_usage_after,
            'system': get_system_usage()
        }

        # Parse the response to extract TEI XML
        tei_xml = extract_xml_from_response(raw_response)

        # Create results dictionary
        results = create_text_encoding_result(tei_xml)

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

    except Exception as e:
        results = create_text_encoding_error(str(e))
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0, {}


if __name__ == "__main__":
    print("Please run: python llm_processing.py")