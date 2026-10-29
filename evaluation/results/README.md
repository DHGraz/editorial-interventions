# Evaluation results of the published runs

One folder per model instance, as produced by `evaluation/eval.py`. File names end with the timestamp of the
processing run in `data/output/<model instance>/processing_<timestamp>/`.

| Folder | Run (timestamp) | Model and mode |
|---|---|---|
| `gpt-5.2-2025-12-11` | `20260301_015947` | GPT-5.2 |
| `claude-sonnet-4-6` | `20260301_020434` | Claude Sonnet 4.6 |
| `qwen3-14B-Q6` | `20260301_021912` | Qwen3-14B, standard mode (`thinking_mode: false`) |
| `qwen3-14B-Q6` | `20260301_022635` | Qwen3-14B, thinking mode (`thinking_mode: true`) |
| `olmo2-32B-instruct-Q4` | `20260927_114615` | OLMo2-32B-Instruct |

In file names, `-` and `.` of the model instance are replaced by `_` (e.g.
`evaluation_olmo2_32B_instruct_Q4_20260927_114615.xlsx`); the `model_instance` field keeps the folder name.

## Estimated cost

`estimated_cost_usd` in the metadata copies:

- GPT and Claude: calculated by the pipeline from the token counts and the list prices set in the configs in
  `configs/`.
- Local runs (Qwen, OLMo): added by `eval.py` when it copies the metadata. It is the estimated cost of local
  inference (electricity plus amortised GPU cost) from the runtime and the measured GPU power of each run, with
  the parameters in `evaluation/config.py` (Qwen3-14B standard mode 0.08, thinking mode 0.85, OLMo2-32B-Instruct 0.15).

## Hard-coded overrides in `eval.py`

`evaluation/eval.py` contains hard-coded per-letter corrections (search for `HACK`). They were used to produce
these results.

- **Normalisation overrides** (start of `normalize_bracketed_content_by_gold_types`): they set the expected text for
  the source-fidelity check in six edge cases (letter174, letter31, letter732, letter2864, letter7514, letter7726),
  where automatic normalisation produces the wrong expected text. Effect: at most 0.5 points.

The gold-standard elements of each bracketed sequence are found without overrides: when a paragraph contains several
sequences, `find_sampled_occurrences` aligns the source paragraph with the gold-standard paragraph to find the
occurrence the annotators encoded, since the same bracketed text can also occur unencoded elsewhere in the paragraph
(e.g. `H[er]r` before the sampled `[er]` in letter1177).
