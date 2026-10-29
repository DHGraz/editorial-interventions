"""
Utilities package for LLM processing.

This package provides shared utility functions used across multiple processor
implementations to reduce code duplication and maintain consistency.
"""

from .utils import (
    check_response_completeness,
    create_segment_error_response,
    create_test_response,
    create_test_tei_response,
    create_text_encoding_error,
    create_text_encoding_result,
    extract_tei_xml_from_response,
    get_gpu_usage,
    get_system_usage,
    parse_json_response,
    validate_text_data,
    validate_text_segment
)

__all__ = [
    'check_response_completeness',
    'create_segment_error_response',
    'create_test_response',
    'create_test_tei_response',
    'create_text_encoding_error',
    'create_text_encoding_result',
    'extract_tei_xml_from_response',
    'get_gpu_usage',
    'get_system_usage',
    'parse_json_response',
    'validate_text_data',
    'validate_text_segment'
]
