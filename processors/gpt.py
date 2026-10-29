"""
OpenAI GPT processor for TEI XML generation.

This module provides GPT-specific implementations for encoding text segments
and complete plaintext files into TEI XML format using OpenAI's API.
"""

# Standard library imports
import json
from typing import Any, Dict, Optional, Tuple

# Third-party imports
from openai import BadRequestError, OpenAI

# Local imports
import config
from utils.utils import (
    create_segment_error_response,
    create_test_response,
    create_test_tei_response,
    create_text_encoding_error,
    create_text_encoding_result,
    extract_xml_from_response,
    parse_json_response,
    prepare_prompts_for_segment,
    validate_text_data
)


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client. Routes to DH Infra cluster endpoint for qwen3.5 models,
    otherwise uses the standard OpenAI API.
    """
    if "qwen3.5" in config.MODEL_NAME.lower():
        return OpenAI(
            api_key=config.DH_INFRA_API_KEY,
            base_url=config.DH_INFRA_BASE_URL
        )
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _build_api_params(system_message: str, user_message: str) -> Dict[str, Any]:
    """Build the API parameters dict based on the configured model."""
    api_params = {
        "model": config.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": config.TEMPERATURE
    }
    is_reasoning_model = any(m in config.MODEL_NAME.lower() for m in ['o1-', 'o3-'])
    is_gpt5 = 'gpt-5' in config.MODEL_NAME.lower()

    if is_reasoning_model:
        api_params['max_completion_tokens'] = config.MAX_TOKENS
    elif not is_gpt5:
        api_params['max_tokens'] = config.MAX_TOKENS
    else:
        # GPT-5: no token limit is sent (API default).
        reasoning_effort = getattr(config, 'GPT5_REASONING_EFFORT', None)
        if reasoning_effort:
            api_params['reasoning_effort'] = reasoning_effort

    if "qwen3" in config.MODEL_NAME.lower():
        # The DH Infra cluster (vLLM/SGLang-style OpenAI endpoint) reads the
        # thinking toggle from chat_template_kwargs; a top-level "enable_thinking"
        # key is silently ignored. Verified against qwen3.5-397b: nested True keeps
        # the reasoning field, nested False removes it.
        api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": config.QWEN_USE_THINKING}}

    return api_params


def _stream_completion(
    client: OpenAI,
    api_params: Dict[str, Any],
    system_message: str,
    user_message: str
) -> Tuple[str, int, int, int]:
    """
    Call the API with streaming enabled, printing tokens live to stdout.

    Returns:
        Tuple of (raw_response, total_tokens, input_tokens, output_tokens).
        Token counts come from the API if available, otherwise estimated.
    """
    stream_params = {**api_params, 'stream': True, 'stream_options': {'include_usage': True}}

    try:
        stream = client.chat.completions.create(**stream_params)
    except Exception:
        # Endpoint does not support stream_options — retry without it
        stream_params.pop('stream_options', None)
        stream = client.chat.completions.create(**stream_params)

    chunks = []
    reasoning_chunks = []
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Reasoning-parser endpoints (e.g. the DH Infra qwen3.5 cluster) stream
            # thinking on delta.reasoning, separate from the answer on delta.content.
            reasoning = getattr(delta, 'reasoning', None)
            if reasoning:
                print(reasoning, end='', flush=True)
                reasoning_chunks.append(reasoning)
            if delta.content:
                print(delta.content, end='', flush=True)
                chunks.append(delta.content)
        if hasattr(chunk, 'usage') and chunk.usage:
            input_tokens = chunk.usage.prompt_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0
            total_tokens = chunk.usage.total_tokens or 0

    print(flush=True)
    content = ''.join(chunks).strip()
    reasoning_text = ''.join(reasoning_chunks).strip()
    # Preserve the reasoning trace in the raw response by wrapping it in <think>
    # tags. Every downstream parser strips through </think>, so JSON/XML extraction
    # is unaffected, and the response log now shows whether thinking actually ran
    # (mirroring the inline <think> blocks the local qwen3 model emits).
    if reasoning_text:
        raw_response = f"<think>\n{reasoning_text}\n</think>\n{content}"
    else:
        raw_response = content

    if total_tokens == 0:
        input_tokens = int(len((system_message + user_message).split()) * 1.3)
        output_tokens = int(len(raw_response.split()) * 1.3)
        total_tokens = input_tokens + output_tokens

    return raw_response, total_tokens, input_tokens, output_tokens


def encode_text_segment_gpt(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    GPT-specific implementation for encoding text segments extracted from JSON objects.

    Encodes text segments (values from specific keys in JSON) into TEI encodings.

    Args:
        text_segment: Dictionary containing text segment data with keys to extract.
            Must contain keys specified in config.JSON_CONTEXT_KEY and config.JSON_ITEMS_KEY.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary mapping extracted key values to their TEI encodings.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from GPT API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - parsing_metadata: Dictionary with parse_success and parse_method fields.

    Raises:
        ValueError: If required fields are missing from text_segment.
    """
    try:
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        if config.ENABLE_API_CALLS:
            api_params = _build_api_params(system_message, user_message)
            client = get_openai_client()
            raw_response, total_tokens, input_tokens, output_tokens = _stream_completion(
                client, api_params, system_message, user_message
            )
        else:
            results = create_test_response(items_to_analyze)
            raw_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"
            input_tokens = int(len((system_message + user_message).split()) * 1.3)
            output_tokens = int(len(raw_response.split()) * 1.3)
            total_tokens = input_tokens + output_tokens
            parsing_metadata = {'parse_success': True, 'parse_method': 'test_mode'}
            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, parsing_metadata

        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)
        parsing_metadata = {'parse_success': parse_success, 'parse_method': parse_method}

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, parsing_metadata

    except Exception as e:
        if isinstance(e, BadRequestError) and 'unsupported' in str(e).lower():
            # An unsupported parameter (e.g. a temperature the model does not accept)
            # would fail for every segment, so stop the run instead.
            raise RuntimeError(
                f"The API rejected the request parameters: {e}\n"
                "If the model does not accept a temperature, set GPT5_REASONING_EFFORT = \"none\" "
                "in the configuration."
            ) from e
        items_key = config.JSON_ITEMS_KEY
        context_key = config.JSON_CONTEXT_KEY
        results = create_segment_error_response(
            text_segment.get(items_key, []), "GPT", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        system_message = "Error occurred before message creation"
        user_message = f"Error prompt for: {text_segment.get(context_key, 'unknown text')}"
        parsing_metadata = {'parse_success': False, 'parse_method': 'error'}
        return results, (system_message, user_message), error_response, 0, 0, 0, parsing_metadata


def encode_text_gpt(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int]:
    """
    GPT-specific implementation for encoding a complete plaintext file into TEI XML.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from GPT API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.

    Raises:
        ValueError: If 'filename' or 'content' fields are missing from text_data.
    """
    try:
        filename, text_content = validate_text_data(text_data)

        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_text(text_content)
        else:
            system_message = "System message placeholder"
            user_message = f"Encode this text: {text_content}"

        if config.ENABLE_API_CALLS:
            api_params = _build_api_params(system_message, user_message)
            client = get_openai_client()
            raw_response, total_tokens, input_tokens, output_tokens = _stream_completion(
                client, api_params, system_message, user_message
            )
        else:
            raw_response = create_test_tei_response("GPT", filename)
            input_tokens = int(len((system_message + user_message).split()) * 1.3)
            output_tokens = int(len(raw_response.split()) * 1.3)
            total_tokens = input_tokens + output_tokens

        tei_xml = extract_xml_from_response(raw_response)
        results = create_text_encoding_result(tei_xml)

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

    except Exception as e:
        results = create_text_encoding_error(str(e))
        error_response = f"ERROR: {str(e)}"
        system_message = "Error occurred before message creation"
        user_message = f"Error prompt for file: {text_data.get('filename', 'unknown')}"
        return results, (system_message, user_message), error_response, 0, 0, 0


def process_json_object_gpt(
    json_object: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    GPT-specific implementation for processing a complete JSON object.

    Args:
        json_object: A single JSON object from the input array (any structure).
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.

    Returns:
        Tuple containing:
            - results: Dictionary with 'output_content' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from GPT API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Empty dict (not applicable for API models).
    """
    try:
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_json_object(json_object)
        else:
            system_message = "System message placeholder"
            user_message = f"Process this JSON object: {json.dumps(json_object, indent=2)}"

        if config.ENABLE_API_CALLS:
            api_params = _build_api_params(system_message, user_message)
            client = get_openai_client()
            raw_response, total_tokens, input_tokens, output_tokens = _stream_completion(
                client, api_params, system_message, user_message
            )
        else:
            object_id = json_object.get('id') or json_object.get('object_id') or 'unknown'
            raw_response = f"TEST MODE RESPONSE for object: {object_id}"
            input_tokens = int(len((system_message + user_message).split()) * 1.3)
            output_tokens = int(len(raw_response.split()) * 1.3)
            total_tokens = input_tokens + output_tokens

        results = {
            'output_content': raw_response,
            'success': bool(raw_response and raw_response.strip())
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, {}

    except Exception as e:
        results = {'output_content': '', 'success': False, 'error': str(e)}
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0, {}


if __name__ == "__main__":
    print("Please run: python llm_processing.py")
