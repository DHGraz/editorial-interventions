"""
Alibaba Qwen processor for TEI XML generation.

This module provides Qwen-specific implementations for encoding text segments
and complete plaintext files into TEI XML format using llama.cpp for local inference.
Supports both GPU and CPU execution with extensive diagnostics.
"""

# Standard library imports
import json
import multiprocessing
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

# Third-party imports
# (llama_cpp imported conditionally below)

# Local imports
import config
from utils.utils import (
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
TORCH_AVAILABLE = False

# Only import llama_cpp when needed
if config.ENABLE_API_CALLS:
    try:
        from llama_cpp import Llama
        LLAMA_AVAILABLE = True

        # Add GPU diagnostic imports
        try:
            import torch
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False

    except ImportError:
        print("Warning: llama_cpp not installed. Only test mode will be available.")
        LLAMA_AVAILABLE = False

# NVML (pynvml) for GPU power sampling - optional
NVML_AVAILABLE = False
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Track whether NVML has been initialized (init is process-global and idempotent-ish)
_nvml_initialized = False

# Global model instance to avoid reloading
_qwen_model = None


def _get_nvml_handle(gpu_index: int = 0) -> Optional[Any]:
    """
    Get an NVML device handle for the given GPU, initializing NVML on first use.

    Args:
        gpu_index: Index of the GPU to query (0 for the first/only GPU).

    Returns:
        NVML device handle, or None if pynvml is unavailable or init fails.
        Degrades gracefully so inference is never blocked by power tracking.
    """
    global _nvml_initialized

    if not NVML_AVAILABLE:
        return None

    try:
        if not _nvml_initialized:
            pynvml.nvmlInit()
            _nvml_initialized = True
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception:
        # Driver missing, index out of range, etc. - fall back to no power data.
        return None


class GPUPowerSampler:
    """
    Context manager that samples GPU board power in a background thread and
    reports average power and total energy for the wrapped code block.

    Uses NVIDIA NVML (pynvml). Power fluctuates during generation (prefill spike,
    decode plateau), so a single before/after reading is not enough for energy;
    this samples continuously at a fixed interval and averages. If NVML is
    unavailable or the GPU does not report power, it degrades to zeros so
    inference is never blocked.

    Attributes (populated on __exit__):
        avg_power_w: Mean sampled board power in watts (0.0 if unmeasured).
        energy_wh: Energy for the block in watt-hours (avg_power_w * elapsed / 3600).
        elapsed_s: Wall-clock duration of the block in seconds.
        sample_count: Number of successful power samples taken.

    Usage:
        with GPUPowerSampler(gpu_index=0, interval=0.5) as power:
            response = llm(...)
        power.as_dict()  # -> {'avg_power_w': ..., 'energy_wh': ..., ...}
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.5):
        self.gpu_index = gpu_index
        self.interval = interval
        self._handle = _get_nvml_handle(gpu_index)
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self.avg_power_w = 0.0
        self.energy_wh = 0.0
        self.elapsed_s = 0.0
        self.sample_count = 0

    def _read_power_w(self) -> Optional[float]:
        """Read instantaneous board power in watts, or None if unavailable."""
        if self._handle is None:
            return None
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
        except Exception:
            return None

    def _loop(self):
        """Background sampling loop; runs until stopped."""
        while not self._stop.is_set():
            power = self._read_power_w()
            if power is not None:
                self._samples.append(power)
            # Event.wait doubles as an interruptible sleep.
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.time()
        # Take one immediate sample so very short calls still get a reading.
        power = self._read_power_w()
        if power is not None:
            self._samples.append(power)
        # Only spin up a thread if we can actually read power.
        if self._handle is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.elapsed_s = time.time() - self._t0 if self._t0 else 0.0
        self.sample_count = len(self._samples)
        if self._samples:
            self.avg_power_w = sum(self._samples) / len(self._samples)
            self.energy_wh = self.avg_power_w * self.elapsed_s / 3600.0
        return False  # never suppress exceptions from the wrapped block

    def as_dict(self) -> Dict[str, Any]:
        """Return sampled power/energy metrics as a JSON-serializable dict."""
        return {
            'avg_power_w': round(self.avg_power_w, 2),
            'energy_wh': round(self.energy_wh, 6),
            'elapsed_s': round(self.elapsed_s, 3),
            'sample_count': self.sample_count,
        }


# Model parameters
MODEL_PARAMS = {
    'n_gpu_layers': -1 if config.USE_GPU else 0,  # -1 for GPU, 0 for CPU
    'n_ctx': 32768,      # Context window
    'n_batch': 1024,     # Batch size
    'max_tokens': config.MAX_TOKENS,  # Max tokens per response
    'n_threads': multiprocessing.cpu_count() - 1,
    'temperature': config.TEMPERATURE
}
if getattr(config, 'SEED', None) is not None:
    MODEL_PARAMS['seed'] = config.SEED  # fixed sampling seed for repeatable runs


def count_tokens_qwen(text: str) -> int:
    """
    Count tokens in text using llama_cpp tokenizer for Qwen.

    Args:
        text: Text string to count tokens for.

    Returns:
        Token count, or 0 if counting fails. Falls back to approximate count
        (words * 1.3) if tokenizer is unavailable.
    """
    if not text:
        return 0

    try:
        model = get_qwen_model()
        if model:
            return len(model.tokenize(text.encode()))
    except Exception:
        # Token counting failed - fallback to approximate count
        return int(len(text.split()) * 1.3)

    return 0


def _format_qwen_prompt(system_message: str, user_message: str) -> str:
    """
    Format system and user messages into Qwen chat template.

    Args:
        system_message: System prompt message.
        user_message: User prompt message.

    Returns:
        Formatted prompt string for Qwen model. Includes /no_think tag
        if thinking mode is disabled.
    """
    if config.QWEN_USE_THINKING:
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
    else:
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""


def _run_qwen_inference(
    system_message: str,
    user_message: str,
    coordinator: Optional[Any] = None
) -> Tuple[str, int, int, int, Dict[str, Any]]:
    """
    Run Qwen model inference with GPU tracking and token counting.

    Args:
        system_message: System prompt message.
        user_message: User prompt message.
        coordinator: LLMProcessingCoordinator instance (required for API mode).

    Returns:
        Tuple containing:
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system).

    Raises:
        Exception: If coordinator is None or model initialization fails.
    """
    # Get the model instance
    if coordinator:
        llm = get_qwen_model()
        if not llm:
            raise Exception("Failed to initialize Qwen model")
    else:
        raise Exception("Coordinator required for Qwen in API mode")

    # Format prompt for Qwen
    combined_prompt = _format_qwen_prompt(system_message, user_message)

    # Get GPU usage before inference
    gpu_usage_before = get_gpu_usage()

    # Generate response, sampling GPU power draw for the duration of inference
    with GPUPowerSampler(gpu_index=0, interval=0.5) as power:
        response = llm(
            combined_prompt,
            max_tokens=MODEL_PARAMS['max_tokens'],
            temperature=MODEL_PARAMS['temperature'],
            stop=["<|im_end|>"]
        )

    raw_response = response['choices'][0]['text'].strip()

    # Get GPU usage after inference
    gpu_usage_after = get_gpu_usage()

    # Calculate token counts
    input_tokens = count_tokens_qwen(combined_prompt)
    output_tokens = count_tokens_qwen(raw_response)
    total_tokens = input_tokens + output_tokens

    # Combine GPU usage info and thinking mode info
    gpu_usage = {
        'before': gpu_usage_before,
        'after': gpu_usage_after,
        'system': get_system_usage(),
        'thinking_mode': config.QWEN_USE_THINKING,
        'power': power.as_dict()
    }

    return raw_response, total_tokens, input_tokens, output_tokens, gpu_usage


def get_qwen_model() -> Optional[Any]:
    """
    Get or initialize the Qwen model using llama.cpp (singleton pattern).

    Returns:
        Llama model instance or None if not available
    """
    global _qwen_model

    if _qwen_model is None and config.ENABLE_API_CALLS:
        # Check if llama_cpp is available
        if not LLAMA_AVAILABLE:
            error_msg = "llama-cpp-python is not installed. Please install it to use local models.\n"
            error_msg += "Install with: pip install llama-cpp-python\n"
            error_msg += "For CPU-only: pip install llama-cpp-python\n"
            error_msg += "For GPU (CUDA): pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            try:
                print(error_msg, file=sys.stderr)
            except:
                pass
            raise ImportError("llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")

        try:
            # Check if model file exists
            model_path = config.MODEL_PATH_QWEN3
            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}\nPlease check the MODEL_PATH_QWEN3 setting in config.py"
                try:
                    print(error_msg, file=sys.stderr)
                except:
                    pass  # If even stderr fails, just raise the exception
                raise FileNotFoundError(error_msg)

            # Check model file size
            try:
                model_size_bytes = os.path.getsize(model_path)
                model_size_gb = model_size_bytes / (1024**3)
                print(f"Model file size: {model_size_gb:.2f} GB")
            except Exception as e:
                print(f"Warning: Could not determine model file size: {e}")
                model_size_gb = None

            # GPU diagnostics before loading (only check VRAM for size validation)
            gpu_vram_gb = None
            if TORCH_AVAILABLE:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        break  # Use first GPU

                        # Check if model is too large for GPU
                        if config.USE_GPU and model_size_gb is not None:
                            # Model needs ~1.1-1.2x its file size in VRAM (due to context, activations, etc.)
                            required_vram = model_size_gb * 1.2
                            if required_vram > gpu_vram_gb:
                                print(f"\n[ERROR] Model is too large for GPU VRAM!")
                                print(f"  Model size: {model_size_gb:.2f} GB")
                                print(f"  Required VRAM (estimated): {required_vram:.2f} GB")
                                print(f"  Available GPU VRAM: {gpu_vram_gb:.2f} GB")
                                print(f"  Shortage: {required_vram - gpu_vram_gb:.2f} GB")
                                print(f"\n  Solutions:")
                                print(f"  1. Use a smaller model (e.g., 7B-Q4 or 7B-Q5 instead of 14B)")
                                print(f"  2. Set USE_GPU = False in config.py to use CPU instead")
                                print(f"  3. Use a GPU with more VRAM (need at least {required_vram:.1f} GB)")
                                raise ValueError(
                                    f"Model ({model_size_gb:.2f} GB) is too large for GPU VRAM ({gpu_vram_gb:.2f} GB). "
                                    f"Estimated requirement: {required_vram:.2f} GB. "
                                    f"Set USE_GPU = False to use CPU, or use a smaller model."
                                )
                            elif required_vram > gpu_vram_gb * 0.9:
                                print(f"\n[WARNING] Model is close to VRAM limit!")
                                print(f"  Model: {model_size_gb:.2f} GB, Required: {required_vram:.2f} GB, Available: {gpu_vram_gb:.2f} GB")
                                print(f"  Consider using a smaller model or CPU mode (USE_GPU = False)")

            # Also check with GPUtil if PyTorch not available
            if gpu_vram_gb is None:
                try:
                    gpu_info = get_gpu_usage()
                    if gpu_info and 'gpu_memory_total' in gpu_info:
                        gpu_vram_gb = gpu_info['gpu_memory_total'] / 1024  # Convert MB to GB
                        if config.USE_GPU and model_size_gb is not None:
                            required_vram = model_size_gb * 1.2
                            if required_vram > gpu_vram_gb:
                                print(f"\n[ERROR] Model too large for GPU VRAM!")
                                print(f"  Model: {model_size_gb:.2f} GB, Required: {required_vram:.2f} GB, Available: {gpu_vram_gb:.2f} GB")
                                raise ValueError(
                                    f"Model ({model_size_gb:.2f} GB) is too large for GPU VRAM ({gpu_vram_gb:.2f} GB). "
                                    f"Set USE_GPU = False to use CPU, or use a smaller model."
                                )
                except:
                    pass  # GPUtil not available or failed

            # Check CUDA support only if GPU is requested and we need to warn
            if config.USE_GPU:
                try:
                    import llama_cpp
                    # Only check and warn if CUDA is clearly not available
                    if hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUDA'):
                        cuda_available = llama_cpp.llama_cpp.GGML_USE_CUDA
                        if not cuda_available:
                            print("\n[ERROR] llama-cpp-python was compiled WITHOUT CUDA support!")
                            print("  Install a CUDA-enabled wheel or set USE_GPU = False in config.py")
                except:
                    pass  # llama_cpp not available or check failed

            # Use MODEL_PARAMS which already respects config.USE_GPU
            model_params = MODEL_PARAMS.copy()

            # Additional GPU-specific parameters (only if using GPU)
            if config.USE_GPU:
                model_params.update({
                    'main_gpu': 0,       # Use GPU 0 as main GPU
                    'split_mode': 1,     # LLAMA_SPLIT_LAYER (default for multi-GPU)
                    'tensor_split': None, # Let it auto-distribute
                    'rope_scaling_type': -1,  # Default rope scaling
                    'rope_freq_base': 0.0,    # Use model default
                    'rope_freq_scale': 0.0,   # Use model default
                    'low_vram': False,   # Don't use low VRAM mode initially
                })

            # Common parameters for both GPU and CPU
            model_params.update({
                'use_mmap': True,    # Use memory mapping
                'use_mlock': False,  # Don't lock memory initially
            })

            _qwen_model = Llama(
                model_path=model_path,
                **model_params,
                verbose=False  # Enable verbose to see loading details
            )

            # Post-loading diagnostics - only check for problems
            print("Model loaded successfully!")

            # Check if GPU is actually being used (only if GPU was requested)
            if config.USE_GPU:
                backend = None
                actual_gpu_layers = None

                try:
                    # Check backend
                    if hasattr(_qwen_model, 'backend'):
                        backend = _qwen_model.backend
                        if backend in ('metal', 'cpu'):
                            print(f"\n[ERROR] Model is using {backend.upper()} backend despite USE_GPU=True!")
                            print("  Set USE_GPU = False in config.py to use CPU mode, or fix GPU setup.")
                            raise RuntimeError(f"GPU requested but model backend is {backend}. Set USE_GPU=False or fix GPU setup.")

                    # Check GPU layers
                    if hasattr(_qwen_model, '_model'):
                        try:
                            actual_gpu_layers = _qwen_model._model.n_gpu_layers
                        except:
                            pass
                    elif hasattr(_qwen_model, 'ctx'):
                        try:
                            actual_gpu_layers = _qwen_model.ctx.n_gpu_layers
                        except:
                            pass

                    # Error if no GPU layers loaded when GPU was requested
                    if actual_gpu_layers == 0:
                        error_msg = "\n[ERROR] No GPU layers loaded despite USE_GPU=True!\n"
                        error_msg += "  Model is running on CPU. Set USE_GPU = False in config.py or fix GPU setup."
                        if model_size_gb is not None and gpu_vram_gb is not None:
                            required_vram = model_size_gb * 1.2
                            if required_vram > gpu_vram_gb:
                                error_msg += f"\n  Model ({model_size_gb:.2f} GB) may be too large for VRAM ({gpu_vram_gb:.2f} GB)."
                        print(error_msg)
                        raise RuntimeError("GPU requested but model is running on CPU. Set USE_GPU=False or fix GPU setup.")

                except RuntimeError:
                    raise  # Re-raise RuntimeErrors
                except Exception:
                    pass  # Ignore other errors in diagnostics

        except ValueError as e:
            # Model too large error - provide specific guidance
            error_msg = f"\n{'='*70}\n"
            error_msg += f"MODEL TOO LARGE FOR GPU VRAM\n"
            error_msg += f"{'='*70}\n"
            error_msg += f"{str(e)}\n\n"
            error_msg += f"RECOMMENDED SOLUTIONS:\n"
            error_msg += f"  1. Use CPU mode (slower but works):\n"
            error_msg += f"     Set USE_GPU = False in config.py\n\n"
            error_msg += f"  2. Use a smaller model:\n"
            error_msg += f"     - Qwen3-7B-Q5_K_M (~5-6 GB) - fits in 6GB VRAM\n"
            error_msg += f"     - Qwen3-7B-Q4_K_M (~4-5 GB) - safer option\n"
            error_msg += f"     - Qwen3-8B-Q4_K_M (~5-6 GB) - might fit\n\n"
            error_msg += f"  3. Upgrade GPU to one with more VRAM (12GB+ recommended)\n"
            error_msg += f"{'='*70}\n"
            try:
                print(error_msg, file=sys.stderr)
            except:
                pass
            _qwen_model = None
            raise
        except Exception as e:
            # Use stderr for error messages to avoid invalid handle issues
            error_msg = f"Error loading Qwen model: {e}\n"
            error_msg += "\n=== Troubleshooting ===\n"
            error_msg += "1. Check if model file exists at the path in config.py\n"

            # Add model size check if we have that info
            if 'model_path' in locals():
                try:
                    if os.path.exists(model_path):
                        model_size = os.path.getsize(model_path) / (1024**3)
                        error_msg += f"   Model file size: {model_size:.2f} GB\n"
                except:
                    pass

            error_msg += "\n2. If using GPU and model is too large:\n"
            error_msg += "   - Set USE_GPU = False to use CPU\n"
            error_msg += "   - Or use a smaller model (7B instead of 14B)\n"
            error_msg += "\n3. If CUDA issues:\n"
            error_msg += "   - Install llama-cpp-python with CUDA:\n"
            error_msg += "     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n"
            error_msg += "\n4. Check if your model file is compatible"
            try:
                print(error_msg, file=sys.stderr)
            except:
                # If stderr also fails, try to at least log the basic error
                try:
                    import traceback
                    traceback.print_exc()
                except:
                    pass
            _qwen_model = None

    return _qwen_model


def encode_text_segment_qwen(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any], Dict[str, Any]]:
    """
    Qwen-specific implementation for encoding text segments extracted from JSON objects.

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
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).
            - parsing_metadata: Dictionary with parse_success and parse_method fields.
    """
    try:
        # Prepare prompts using centralized routing logic
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Create test response for each item
            results = create_test_response(items_to_analyze)
            test_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"

            # Use general prompt formatting helper
            test_prompt = _format_qwen_prompt(system_message, user_message)

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(test_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            return results, (system_message, user_message), test_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # Use general inference helper
        raw_response, total_tokens, input_tokens, output_tokens, gpu_usage = _run_qwen_inference(
            system_message, user_message, coordinator
        )

        # Parse response expecting the configured items
        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)
        
        # Add parsing metadata to results
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
            text_segment.get(items_key, []), "Qwen", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        parsing_metadata = {'parse_success': False, 'parse_method': 'error'}
        return results, ("", ""), error_response, 0, 0, 0, {}, parsing_metadata


def encode_text_qwen(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    Qwen-specific implementation for encoding a complete plaintext file into TEI XML.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).

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

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Test mode - return realistic fake TEI XML response
            raw_response = create_test_tei_response("Qwen", filename)

            # Create the combined prompt in the same format as API calls
            test_prompt = _format_qwen_prompt(system_message, user_message)

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(raw_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            # Parse the response to extract TEI XML
            tei_xml = extract_xml_from_response(raw_response)

            results = create_text_encoding_result(tei_xml)

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # API calls enabled - use actual model
        # Get the model instance
        llm = get_qwen_model()
        if not llm:
            raise Exception("Failed to initialize Qwen model")

        # Format prompt for Qwen
        if config.QWEN_USE_THINKING:
            combined_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
        else:
            combined_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""

        # Get GPU usage before inference
        gpu_usage_before = get_gpu_usage()

        # Generate response, sampling GPU power draw for the duration of inference
        with GPUPowerSampler(gpu_index=0, interval=0.5) as power:
            response = llm(
                combined_prompt,
                max_tokens=MODEL_PARAMS['max_tokens'],
                temperature=MODEL_PARAMS['temperature'],
                stop=["<|im_end|>"]
            )

        raw_response = response['choices'][0]['text'].strip()

        # Get GPU usage after inference
        gpu_usage_after = get_gpu_usage()

        # Calculate token counts
        input_tokens = count_tokens_qwen(combined_prompt)
        output_tokens = count_tokens_qwen(raw_response)
        total_tokens = input_tokens + output_tokens

        # Combine GPU usage info and thinking mode info
        gpu_usage = {
            'before': gpu_usage_before,
            'after': gpu_usage_after,
            'system': get_system_usage(),
            'thinking_mode': config.QWEN_USE_THINKING,
            'power': power.as_dict()
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


def process_json_object_qwen(
    json_object: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    Qwen-specific implementation for processing a complete JSON object.

    Processes the entire JSON object as a unit (object processing mode).

    Args:
        json_object: A single JSON object from the input array (any structure).
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'output_content' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).
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
            # Test mode - return placeholder response
            object_id = json_object.get('id') or json_object.get('object_id') or 'unknown'
            raw_response = f"""<!-- QWEN TEST MODE OUTPUT for object: {object_id} -->
<rdf:RDF>
  <rdf:Description>
    <content>{json_object.get('content', 'N/A')}</content>
  </rdf:Description>
</rdf:RDF>"""

            # Create the combined prompt in the same format as API calls
            if config.QWEN_USE_THINKING:
                test_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
            else:
                test_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(raw_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            results = {
                'output_content': raw_response,
                'success': True
            }

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # API calls enabled - use actual model
        llm = get_qwen_model()
        if not llm:
            raise Exception("Failed to initialize Qwen model")

        # Format prompt for Qwen
        if config.QWEN_USE_THINKING:
            combined_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
        else:
            combined_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""

        # Get GPU usage before inference
        gpu_usage_before = get_gpu_usage()

        # Generate response, sampling GPU power draw for the duration of inference
        with GPUPowerSampler(gpu_index=0, interval=0.5) as power:
            response = llm(
                combined_prompt,
                max_tokens=MODEL_PARAMS['max_tokens'],
                temperature=MODEL_PARAMS['temperature'],
                stop=["<|im_end|>"]
            )

        raw_response = response['choices'][0]['text'].strip()

        # Get GPU usage after inference
        gpu_usage_after = get_gpu_usage()

        # Calculate token counts
        input_tokens = count_tokens_qwen(combined_prompt)
        output_tokens = count_tokens_qwen(raw_response)
        total_tokens = input_tokens + output_tokens

        # Combine GPU usage info
        gpu_usage = {
            'before': gpu_usage_before,
            'after': gpu_usage_after,
            'system': get_system_usage(),
            'thinking_mode': config.QWEN_USE_THINKING,
            'power': power.as_dict()
        }

        # Extract content from response (remove thinking tags if present)
        output_content = extract_output_from_response(raw_response)

        results = {
            'output_content': output_content,
            'success': bool(output_content and output_content.strip())
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

    except Exception as e:
        results = {
            'output_content': '',
            'success': False,
            'error': str(e)
        }
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0, {}


def extract_output_from_response(response: str) -> str:
    """
    Extract output content from Qwen's response.

    Handles thinking tags and markdown code blocks. This is Qwen-specific
    as it handles the /no_think tag and thinking mode responses.

    Args:
        response: Raw response string from Qwen model.

    Returns:
        Cleaned output content with thinking tags and markdown code blocks removed.
    """
    # Remove thinking process if present
    if '<think>' in response and '</think>' in response:
        think_end = response.find('</think>')
        response = response[think_end + len('</think>'):].strip()

    # Remove markdown code blocks if present
    response = response.strip()

    # Remove ```xml or ```rdf and ``` markers if present
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```xml or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines).strip()

    return response


if __name__ == "__main__":
    print("Please run: python llm_processing.py")