"""
Shared utility functions for LLM processors.

This module contains common functions used across multiple processor implementations
to reduce code duplication and maintain consistency.
"""

# Standard library imports
import json
import re
from typing import Any, Dict, List, Tuple


# =============================================================================
# TEI XML EXTRACTION
# =============================================================================

def extract_tei_xml_from_response(response: str) -> str:
    """
    Extract TEI XML content from LLM response.
    Handles cases where XML is wrapped in markdown code blocks or other text.
    Works with responses from Claude, GPT, Qwen, and OLMo.

    Args:
        response: Raw response from any LLM

    Returns:
        Extracted TEI XML string, or empty string if not found
    """
    # Remove thinking process tags if present (Qwen with thinking mode)
    if '<think>' in response and '</think>' in response:
        think_end = response.find('</think>')
        response = response[think_end + len('</think>'):].strip()

    # Remove markdown code blocks if present
    response = response.strip()

    # Remove ```xml and ``` markers if present
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```xml or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines).strip()

    # Try to extract content between <body> and </body> tags
    if '<body>' in response and '</body>' in response:
        start = response.find('<body>')
        end = response.find('</body>') + len('</body>')
        return response[start:end]

    # If no body tags, try to extract content between <text> and </text> tags
    if '<text>' in response and '</text>' in response:
        start = response.find('<text>')
        end = response.find('</text>') + len('</text>')
        return response[start:end]

    # If no XML tags found, return the entire response (might be malformed)
    return response


def extract_rdf_xml_from_response(response: str) -> str:
    """
    Extract RDF-XML content from LLM response.
    Handles cases where XML is wrapped in markdown code blocks or other text.
    Works with responses from Claude, GPT, Qwen, and OLMo.

    Args:
        response: Raw response from any LLM

    Returns:
        Extracted RDF-XML string, or empty string if not found
    """
    # Remove thinking process tags if present (Qwen with thinking mode)
    if '<think>' in response and '</think>' in response:
        think_end = response.find('</think>')
        response = response[think_end + len('</think>'):].strip()
    elif '</think>' in response:
        # Handle case where only closing tag is present
        think_end = response.find('</think>')
        response = response[think_end + len('</think>'):].strip()

    # Remove markdown code blocks if present
    response = response.strip()

    # Remove ```xml and ``` markers if present
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```xml or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines).strip()

    # Try to extract content between <rdf:RDF> and </rdf:RDF> tags
    if '<rdf:RDF' in response and '</rdf:RDF>' in response:
        start = response.find('<rdf:RDF')
        end = response.find('</rdf:RDF>') + len('</rdf:RDF>')
        return response[start:end]

    # Alternative: Try <RDF> tags (without namespace prefix)
    if '<RDF' in response and '</RDF>' in response:
        start = response.find('<RDF')
        end = response.find('</RDF>') + len('</RDF>')
        return response[start:end]

    # If no RDF wrapper tags, try to extract XML content directly
    # Find the first XML tag (starts with <) and extract until the end
    if '<' in response and '>' in response:
        # Find the first opening tag
        start = response.find('<')
        # Find the last closing tag
        last_close = response.rfind('>')
        if start != -1 and last_close != -1 and last_close > start:
            return response[start:last_close + 1].strip()

    # If no XML found, return empty string
    return ""


def extract_xml_from_response(response: str) -> str:
    """
    Extract XML content from LLM response based on configured XML output type.
    Routes to appropriate extraction function based on config.XML_OUTPUT_TYPE.

    Args:
        response: Raw response from any LLM

    Returns:
        Extracted XML string (TEI or RDF based on config), or empty string if not found
    """
    import config
    
    xml_type = getattr(config, 'XML_OUTPUT_TYPE', 'tei')
    
    if xml_type == 'rdf':
        return extract_rdf_xml_from_response(response)
    else:
        return extract_tei_xml_from_response(response)


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON content from LLM response.
    Handles cases where JSON is wrapped in markdown code blocks or thinking tags.
    Works with responses from Claude, GPT, Qwen, and OLMo.

    Args:
        response: Raw response from any LLM

    Returns:
        Extracted JSON string, or empty string if not found
    """
    # Remove thinking process tags if present (Qwen with thinking mode).
    # Cut at the last </think> in case the reasoning itself echoes the tag.
    if '<think>' in response and '</think>' in response:
        think_end = response.rfind('</think>')
        response = response[think_end + len('</think>'):].strip()

    # Remove markdown code blocks if present
    response = response.strip()

    # Remove ```json and ``` markers if present
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines).strip()

    # Extract the outermost JSON value — object ({...}) or array ([...]),
    # whichever opens first. The disambiguation task returns an array, so
    # matching only braces would drop it and store the raw string instead.
    candidates = []
    first_brace = response.find('{')
    if first_brace != -1:
        candidates.append((first_brace, '}'))
    first_bracket = response.find('[')
    if first_bracket != -1:
        candidates.append((first_bracket, ']'))
    if candidates:
        start, close_char = min(candidates, key=lambda c: c[0])
        end = response.rfind(close_char)
        if end != -1 and end > start:
            return response[start:end + 1].strip()

    # If no JSON delimiters found, return the entire response (might be malformed)
    return response


# =============================================================================
# GPU AND SYSTEM MONITORING
# =============================================================================

def get_gpu_usage() -> Dict[str, Any]:
    """
    Get current GPU usage information.
    Returns dict with GPU stats or empty dict if GPUtil is unavailable.

    Returns:
        Dictionary with keys: gpu_utilization, gpu_memory_used, gpu_memory_total,
        gpu_memory_percent, gpu_temperature (if available)
    """
    gpu_info = {}

    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            gpu_info = {
                'gpu_utilization': gpu.load * 100,  # Percentage
                'gpu_memory_used': gpu.memoryUsed,  # MB
                'gpu_memory_total': gpu.memoryTotal,  # MB
                'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'gpu_temperature': gpu.temperature if hasattr(gpu, 'temperature') else None
            }
    except ImportError:
        # GPUtil not installed - return empty dict
        pass
    except Exception:
        # Silently fail - returns empty dict
        pass

    return gpu_info


def get_system_usage() -> Dict[str, Any]:
    """
    Get current system resource usage.
    Returns dict with system stats or empty dict if psutil is unavailable.

    Returns:
        Dictionary with keys: cpu_percent, memory_percent, memory_used_gb, memory_total_gb
    """
    system_info = {}

    try:
        import psutil
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    except ImportError:
        # psutil not installed - return empty dict
        pass
    except Exception:
        # Silently fail - returns empty dict
        pass

    return system_info


# =============================================================================
# TEST MODE RESPONSE GENERATION
# =============================================================================

def create_test_response(items_to_analyze: List[str]) -> Dict[str, Any]:
    """
    Create a test mode response for when API calls are disabled.
    Generates realistic fake TEI encodings based on item patterns.

    Args:
        items_to_analyze: List of items to analyze (from configured JSON_ITEMS_KEY)

    Returns:
        Dictionary mapping each item to intervention analysis
    """
    results = {}
    for seq in items_to_analyze:
        if '[' in seq and ']' in seq:
            if seq.startswith('[') and seq.endswith(']'):
                # Pattern like [qu'à] - likely addition
                intervention_type = "addition"
                content = seq[1:-1]
                explanation = f"TEST MODE: Added text '{content}'"
                tei_encoding = f"<supplied>{content}</supplied>"
            else:
                # Pattern like Chev[alier] - likely abbreviation
                intervention_type = "abbreviation"
                parts = seq.split('[')
                abbr = parts[0]
                expan = parts[1].rstrip(']')
                explanation = f"TEST MODE: Expanded abbreviation '{abbr}' to '{abbr}{expan}'"
                tei_encoding = f"<choice><abbr>{abbr}</abbr><expan>{abbr}{expan}</expan></choice>"
        else:
            # Fallback for unexpected patterns
            intervention_type = "unknown"
            explanation = f"TEST MODE: Unknown intervention type for '{seq}'"
            tei_encoding = f"<!-- Unknown intervention: {seq} -->"

        results[seq] = {
            "intervention_type": intervention_type,
            "explanation": explanation,
            "tei_encoding": tei_encoding
        }

    return results


# =============================================================================
# JSON RESPONSE PARSING
# =============================================================================

def parse_json_response(raw_response: str, items_to_analyze: List[str]) -> Tuple[Dict[str, Any], bool, str]:
    """
    Parse JSON response from LLM, handling both thinking mode and regular responses.
    
    Attempts to extract JSON object and falls back to structured text parsing.
    Handles duplicate keys by indexing them (e.g., "key", "key__2", "key__3").

    Args:
        raw_response: Raw text response from LLM.
        items_to_analyze: List of items that should be in response (from configured JSON_ITEMS_KEY).

    Returns:
        Tuple containing:
            - results_dict: Dictionary mapping items to their analyses.
            - parse_success: True if JSON was successfully parsed, False if fallback used.
            - parse_method: "json" if JSON parsing succeeded, "fallback" if structured text 
                parsing used, "error" if parsing failed.
    """
    def handle_duplicate_keys(pairs):
        """
        Custom object_pairs_hook to handle duplicate keys in JSON.
        Appends __2, __3, etc. suffix to duplicate keys.
        """
        result = {}
        key_counts = {}
        for key, value in pairs:
            if key in result:
                # Duplicate key found - append index suffix
                key_counts[key] = key_counts.get(key, 1) + 1
                new_key = f"{key}__{key_counts[key]}"
                result[new_key] = value
            else:
                result[key] = value
        return result

    try:
        # Extract thinking process and final output (for models with thinking mode)
        content = raw_response
        if '<think>' in raw_response and '</think>' in raw_response:
            think_end = raw_response.find('</think>')
            content = raw_response[think_end + len('</think>'):].strip()

        # Try to extract JSON object from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                # Use custom hook to handle duplicate keys
                parsed_json = json.loads(json_match.group(), object_pairs_hook=handle_duplicate_keys)
                # Validate that the parsed JSON is a dictionary, not a list
                if isinstance(parsed_json, dict):
                    return parsed_json, True, 'json'
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, try to parse as structured text
        results = {}

        # Split content by items and try to extract information
        for seq in items_to_analyze:
            # Look for lines containing this sequence
            lines = content.split('\n')
            relevant_lines = [line for line in lines if seq in line]

            if relevant_lines:
                # Try to extract intervention type and encoding from the response
                line_text = ' '.join(relevant_lines)

                # Default values
                intervention_type = "unknown"
                explanation = line_text[:200] + "..." if len(line_text) > 200 else line_text
                tei_encoding = f"<!-- Could not determine encoding for: {seq} -->"

                # Try to identify common patterns
                if 'addition' in line_text.lower() or 'supplied' in line_text.lower():
                    intervention_type = "addition"
                    if seq.startswith('[') and seq.endswith(']'):
                        content_inner = seq[1:-1]
                        tei_encoding = f"<supplied>{content_inner}</supplied>"
                elif 'abbreviation' in line_text.lower() or 'expan' in line_text.lower():
                    intervention_type = "abbreviation"
                    if '[' in seq and ']' in seq and not seq.startswith('['):
                        parts = seq.split('[')
                        if len(parts) == 2:
                            abbr = parts[0]
                            expan = parts[1].rstrip(']')
                            tei_encoding = f"<choice><abbr>{abbr}</abbr><expan>{abbr}{expan}</expan></choice>"

                results[seq] = {
                    "intervention_type": intervention_type,
                    "explanation": explanation,
                    "tei_encoding": tei_encoding
                }
            else:
                # No relevant information found for this sequence
                results[seq] = {
                    "intervention_type": "unknown",
                    "explanation": f"No information found in response for: {seq}",
                    "tei_encoding": f"<!-- No encoding determined for: {seq} -->"
                }

        return results, False, "fallback"

    except Exception as e:
        # Create error response for all items
        results = {}
        for seq in items_to_analyze:
            results[seq] = {
                "intervention_type": "error",
                "explanation": f"Error parsing response: {str(e)}",
                "tei_encoding": f"<!-- Parsing error for: {seq} -->"
            }
        return results, False, "error"


# =============================================================================
# RESPONSE COMPLETENESS VALIDATION
# =============================================================================

def check_response_completeness(raw_response: str, expected_format: str = "json") -> Dict[str, Any]:
    """
    Check if the model response appears complete or truncated.
    Useful for detecting when max_tokens limit was reached.

    Args:
        raw_response: The raw text response from the model
        expected_format: Expected format - "json" or "xml"

    Returns:
        Dictionary with 'is_complete' (bool) and 'warnings' (list of strings)
    """
    warnings = []
    is_complete = True

    if not raw_response or not raw_response.strip():
        warnings.append("Response is empty")
        is_complete = False
        return {'is_complete': is_complete, 'warnings': warnings}

    # Check for JSON completeness
    if expected_format == "json":
        # Count braces
        open_braces = raw_response.count('{')
        close_braces = raw_response.count('}')
        if open_braces != close_braces:
            warnings.append(f"Mismatched braces: {open_braces} open, {close_braces} close - response may be truncated")
            is_complete = False

    # Check for XML completeness
    elif expected_format == "xml":
        # Check for common XML tags
        if '<body>' in raw_response and '</body>' not in raw_response:
            warnings.append("Missing closing </body> tag - response may be truncated")
            is_complete = False
        if '<text>' in raw_response and '</text>' not in raw_response:
            warnings.append("Missing closing </text> tag - response may be truncated")
            is_complete = False

    # Check for common truncation indicators
    if raw_response.endswith('...'):
        warnings.append("Response ends with '...' - may indicate truncation")
        is_complete = False

    # Check if response seems to end mid-sentence
    last_chars = raw_response.strip()[-50:] if len(raw_response.strip()) > 50 else raw_response.strip()
    if last_chars and not any(last_chars.endswith(end) for end in ['.', '>', '}', ']', '"']):
        warnings.append("Response may end abruptly without proper termination")
        is_complete = False

    return {'is_complete': is_complete, 'warnings': warnings}


# =============================================================================
# TEST MODE TEI XML GENERATION
# =============================================================================

def create_test_tei_response(model_name: str, filename: str) -> str:
    """
    Create a test mode TEI XML response for when API calls are disabled.
    Used by all processors for consistent test output.

    Args:
        model_name: Name of the model (e.g., 'Claude', 'GPT', 'Qwen', 'OLMo')
        filename: Name of the file being processed

    Returns:
        Fake TEI XML response string
    """
    return f"""<body>
  <div type="text">
    <opener>
      <dateline>[DATELINE]</dateline>
      <salute>[SALUTATION]</salute>
    </opener>
    <p>[TEXT CONTENT - {model_name.upper()} TEST MODE - File: {filename}]</p>
    <closer>
      <signed>[SIGNATURE]</signed>
    </closer>
  </div>
</body>"""


# =============================================================================
# ERROR RESPONSE GENERATION
# =============================================================================

def create_segment_error_response(items_to_analyze: List[str], model_name: str, error_msg: str) -> Dict[str, Any]:
    """
    Create an error response dictionary for text segment analysis failures.
    Used when an exception occurs during segment processing.

    Args:
        items_to_analyze: List of items that were being analyzed
        model_name: Name of the model (e.g., 'Claude', 'GPT', 'Qwen', 'OLMo')
        error_msg: The error message to include

    Returns:
        Dictionary mapping each item to an error response
    """
    results = {}
    for seq in items_to_analyze:
        results[seq] = {
            "intervention_type": "error",
            "explanation": f"Error during {model_name} analysis: {error_msg}",
            "tei_encoding": f"<!-- {model_name} analysis failed -->"
        }
    return results


def create_text_encoding_result(tei_xml: str) -> Dict[str, Any]:
    """
    Create a success result dictionary for text encoding.

    Args:
        tei_xml: The generated TEI XML content

    Returns:
        Dictionary with 'tei_xml' and 'success' keys
    """
    return {
        'tei_xml': tei_xml,
        'success': bool(tei_xml and tei_xml.strip())
    }


def create_text_encoding_error(error_msg: str) -> Dict[str, Any]:
    """
    Create an error result dictionary for text encoding failures.

    Args:
        error_msg: The error message to include

    Returns:
        Dictionary with 'tei_xml', 'success', and 'error' keys
    """
    return {
        'tei_xml': '',
        'success': False,
        'error': error_msg
    }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_text_segment(text_segment: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Validate and extract required fields from a text segment dictionary.

    Validates that the text segment contains the required keys as configured
    in config.JSON_CONTEXT_KEY and config.JSON_ITEMS_KEY.

    Args:
        text_segment: Dictionary containing text segment data with keys to extract.
            Must contain keys specified in config.JSON_CONTEXT_KEY and config.JSON_ITEMS_KEY.

    Returns:
        Tuple containing:
            - context_value: The value from the context key (string)
            - items_to_analyze: List of items from the items key
              (converted to list if single string provided)

    Raises:
        ValueError: If required fields are missing from text_segment.
    """
    import config

    context_key = config.JSON_CONTEXT_KEY
    items_key = config.JSON_ITEMS_KEY

    if context_key not in text_segment:
        raise ValueError(f"text_segment missing '{context_key}' field (configured as JSON_CONTEXT_KEY)")
    if items_key not in text_segment:
        raise ValueError(f"text_segment missing '{items_key}' field (configured as JSON_ITEMS_KEY)")

    context_value = text_segment[context_key]
    items_value = text_segment[items_key]
    
    # Flexibly handle both list and string for items_key
    # This allows for two text segments OR a list of items
    if isinstance(items_value, list):
        items_to_analyze = items_value
    elif isinstance(items_value, str):
        # Single string - wrap in list for uniform processing
        items_to_analyze = [items_value]
    else:
        # Try to convert to list if it's another iterable type
        try:
            items_to_analyze = list(items_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"'{items_key}' must be a string or list, got {type(items_value).__name__}"
            )

    return context_value, items_to_analyze


def validate_information_extraction(json_object: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and extract data keys for information extraction workflow.

    Extracts values for keys specified in config.JSON_DATA_KEYS.

    Args:
        json_object: Dictionary containing JSON object data.
            Should contain keys specified in config.JSON_DATA_KEYS.

    Returns:
        Dictionary mapping data keys to their values from the JSON object.
        Missing keys will have None as their value.

    Raises:
        ValueError: If JSON_DATA_KEYS is not configured or empty.
    """
    import config

    # Get the data keys to extract
    data_keys = getattr(config, 'JSON_DATA_KEYS', None)
    
    if not data_keys or not isinstance(data_keys, list):
        raise ValueError("JSON_DATA_KEYS must be configured as a list of key names")
    
    # Extract values for each data key
    extracted_data = {}
    for key in data_keys:
        extracted_data[key] = json_object.get(key)
    
    return extracted_data


def prepare_prompts_for_segment(text_segment: Dict[str, Any], coordinator=None) -> Tuple[str, str, list]:
    """
    Prepare system and user prompts for a text segment based on extraction type.

    Routes to appropriate validation and message creation based on JSON_EXTRACTION_TYPE.
    Handles both TEI encoding and information extraction workflows.

    Args:
        text_segment: Dictionary containing text segment/JSON object data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - system_message: System prompt string
            - user_message: User prompt string
            - items_to_analyze: List of items (for test mode compatibility)

    Raises:
        ValueError: If validation fails or required fields are missing.
    """
    import config

    # Check extraction type to route validation and message creation
    extraction_type = getattr(config, 'JSON_EXTRACTION_TYPE', 'tei_encoding')
    
    if extraction_type == 'information_extraction':
        # Information extraction workflow: extract specific key-value pairs
        data_dict = validate_information_extraction(text_segment)
        if data_dict is None:
            raise ValueError("Failed to extract required data keys from JSON object")
        
        # Create the prompt for information extraction
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_information_extraction(data_dict)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            data_str = "\n".join(f"{k}: {v}" for k, v in data_dict.items())
            user_message = f"Analyze:\n{data_str}"
        
        # The rest of processing is same for both workflows
        items_to_analyze = list(data_dict.keys())  # For test mode compatibility
        
    else:
        # TEI encoding workflow: validate text segment structure
        context_value, items_to_analyze = validate_text_segment(text_segment)
        
        # Create the prompt using the text segment data via centralized functions
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message(context_value, items_to_analyze)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            user_message = f"Analyze: {context_value}"
    
    return system_message, user_message, items_to_analyze


def validate_text_data(text_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Validate and extract required fields from a text data dictionary.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.

    Returns:
        Tuple containing:
            - filename: The filename from text_data
            - content: The text content from text_data

    Raises:
        ValueError: If 'filename' or 'content' fields are missing from text_data.
    """
    if 'content' not in text_data:
        raise ValueError("text_data missing 'content' field")
    if 'filename' not in text_data:
        raise ValueError("text_data missing 'filename' field")

    return text_data['filename'], text_data['content']