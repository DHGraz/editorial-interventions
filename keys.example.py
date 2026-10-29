"""
Template for the API keys of the cloud models (GPT, Claude).

Copy this file to ``llm_processing/keys.py`` (the engine folder, see README) and fill in your keys.
``keys.py`` is ignored by git; never commit it. The configs in ``configs/`` import it from there
and set the keys to None if it is missing. Only needed for new LLM runs, not for reproducing the
published results.
"""

OPENAI = "your-openai-api-key"
ANTHROPIC = "your-anthropic-api-key"
