"""
Anthropic Claude processor for TEI XML generation.

This module provides Claude-specific implementations for encoding text segments
and complete plaintext files into TEI XML format using Anthropic's API.
"""

# Standard library imports
import json
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import anthropic

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

def get_anthropic_client() -> anthropic.Anthropic:
    """
    Get Anthropic client with lazy-loaded API key.

    Returns:
        Initialized Anthropic client instance.

    Raises:
        ValueError: If API key is not configured when API calls are enabled.
    """
    return anthropic.Anthropic(api_key=config.get_anthropic_api_key())


def count_tokens_claude(text: str, model: Optional[str] = None, system_message: Optional[str] = None) -> int:
    """
    Count tokens in text using Anthropic's count_tokens API.

    Args:
        text: Text to count tokens for.
        model: Model name to use (defaults to config.MODEL_NAME if None).
        system_message: Optional system message to include in token count.

    Returns:
        Token count, or 0 if counting fails.
    """
    if not text:
        return 0

    try:
        # Use the model from config if not provided
        if model is None:
            model = config.MODEL_NAME

        # Prepare the API call parameters
        api_params = {
            "model": model,
            "messages": [{"role": "user", "content": text}]
        }

        # Add system message if provided
        if system_message:
            api_params["system"] = system_message

        # Count tokens using Anthropic's API
        client = get_anthropic_client()
        response = client.messages.count_tokens(**api_params)

        # Handle different response formats
        if hasattr(response, 'input_tokens'):
            return response.input_tokens
        elif hasattr(response, 'json'):
            # If response has json method, parse it
            response_data = response.json()
            return response_data.get('input_tokens', 0)
        else:
            # Fallback: try to access as dict
            return getattr(response, 'input_tokens', 0)

    except Exception:
        # Token counting failed - return 0 (will use approximate count)
        return 0


def encode_text_segment_claude(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    Claude-specific implementation for encoding text segments extracted from JSON objects.

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
            - raw_response: Raw response text from Claude API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - parsing_metadata: Dictionary with parse_success and parse_method fields.

    Raises:
        ValueError: If required fields are missing from text_segment.
    """
    try:
        # Prepare prompts using centralized routing logic
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        # Call the Anthropic API (if enabled in config)
        if config.ENABLE_API_CALLS:
            # Count input tokens (including system message)
            input_tokens = count_tokens_claude(user_message, system_message=system_message)

            # Set up the API parameters for Claude (proper format with separate system message)
            api_params = {
                "model": config.MODEL_NAME,
                "system": system_message,
                "messages": [{"role": "user", "content": user_message}],
                "max_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE
            }

            client = get_anthropic_client()
            response = client.messages.create(**api_params)

            # Get the raw response content from Claude's response
            raw_response = response.content[0].text.strip()

            # Get exact token usage from API response if available
            if hasattr(response, 'usage') and response.usage:
                # Use exact token counts from API response
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
            else:
                # Fallback: count output tokens (approximate)
                output_tokens = count_tokens_claude(raw_response)
                total_tokens = input_tokens + output_tokens
        else:
            # Test mode - use shared test response generator
            results = create_test_response(items_to_analyze)
            raw_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"

            # For test mode, estimate token count
            input_tokens = count_tokens_claude(user_message, system_message=system_message)
            output_tokens = count_tokens_claude(raw_response)
            total_tokens = input_tokens + output_tokens

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

        # Parse the response using shared JSON parser
        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)
        
        parsing_metadata = {
            'parse_success': parse_success,
            'parse_method': parse_method
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, parsing_metadata

    except Exception as e:
        items_key = config.JSON_ITEMS_KEY
        results = create_segment_error_response(
            text_segment.get(items_key, []), "Claude", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        parsing_metadata = {'parse_success': False, 'parse_method': 'error'}
        return results, ("", ""), error_response, 0, 0, 0, parsing_metadata


def encode_text_claude(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int]:
    """
    Claude-specific implementation for encoding a complete plaintext file into TEI XML.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Claude API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.

    Raises:
        ValueError: If 'filename' or 'content' fields are missing from text_data.
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

        # Call the Anthropic API (if enabled in config)
        if config.ENABLE_API_CALLS:
            # Count input tokens (including system message)
            input_tokens = count_tokens_claude(user_message, system_message=system_message)

            # Set up the API parameters for Claude
            api_params = {
                "model": config.MODEL_NAME,
                "system": system_message,
                "messages": [{"role": "user", "content": user_message}],
                "max_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE
            }

            client = get_anthropic_client()
            response = client.messages.create(**api_params)

            # Get the raw response content from Claude's response
            raw_response = response.content[0].text.strip()

            # Get exact token usage from API response if available
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
            else:
                # Fallback: count output tokens (approximate)
                output_tokens = count_tokens_claude(raw_response)
                total_tokens = input_tokens + output_tokens
        else:
            # Test mode - return realistic fake TEI XML response
            raw_response = create_test_tei_response("Claude", filename)

            # For test mode, estimate token count
            input_tokens = count_tokens_claude(user_message, system_message=system_message)
            output_tokens = count_tokens_claude(raw_response)
            total_tokens = input_tokens + output_tokens

        # Parse the response to extract TEI XML
        tei_xml = extract_xml_from_response(raw_response)

        # Create results dictionary
        results = create_text_encoding_result(tei_xml)

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

    except Exception as e:
        results = create_text_encoding_error(str(e))
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0


def process_json_object_claude(
    json_object: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int]:
    """
    Claude-specific implementation for processing a complete JSON object.

    Processes the entire JSON object as a unit (object processing mode).

    Args:
        json_object: A single JSON object from the input array (any structure).
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'output_content' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Claude API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
    """
    try:
        # Create the prompt using the coordinator's methods
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_json_object(json_object)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            user_message = f"Process this JSON object: {json.dumps(json_object, indent=2)}"

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            object_id = json_object.get('id') or json_object.get('object_id') or 'unknown'
            raw_response = f"""<!-- CLAUDE TEST MODE OUTPUT for object: {object_id} -->
<rdf:RDF>
  <rdf:Description>
    <content>{json_object.get('content', 'N/A')}</content>
  </rdf:Description>
</rdf:RDF>"""

            # For test mode, estimate token count
            input_tokens = count_tokens_claude(user_message, system_message=system_message)
            output_tokens = count_tokens_claude(raw_response)
            total_tokens = input_tokens + output_tokens

            results = {
                'output_content': raw_response,
                'success': True
            }

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

        # API calls enabled - call the Anthropic API
        input_tokens = count_tokens_claude(user_message, system_message=system_message)

        api_params = {
            "model": config.MODEL_NAME,
            "system": system_message,
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE
        }

        client = get_anthropic_client()
        response = client.messages.create(**api_params)

        raw_response = response.content[0].text.strip()

        # Get exact token usage from API response if available
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
        else:
            output_tokens = count_tokens_claude(raw_response)
            total_tokens = input_tokens + output_tokens

        # Strip markdown code blocks if present
        output_content = raw_response
        if output_content.startswith('```'):
            lines = output_content.split('\n')
            lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            output_content = '\n'.join(lines).strip()

        results = {
            'output_content': output_content,
            'success': bool(output_content and output_content.strip())
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

    except Exception as e:
        results = {
            'output_content': '',
            'success': False,
            'error': str(e)
        }
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0


if __name__ == "__main__":
    print("Please run: python llm_processing.py")
