# Editorial interventions: LLM-assisted TEI encoding

A pipeline that asks LLMs to turn bracket-marked editorial interventions in historical letters into TEI XML
markup. A separate, optional component evaluates how well a model did this, against a hand-made gold
standard.

This repository contains the code, the sample, the stored LLM output and the evaluation results of the
article

> Sabrina Strutz, Martina Scholger, Georg Vogeler: *Beyond Rule-Based Processing: LLM-Assisted TEI Encoding
> of Editorial Interventions in Historical Correspondence.* Accepted for publication in the Journal of the
> Text Encoding Initiative (jTEI).

All results in this repository can be re-computed from the stored LLM output; see
[Reproducing the published results](#reproducing-the-published-results).

## What it does

Historical letter editions often mark editorial interventions with square brackets, e.g. `Fr[au]`
(abbreviation), `[ha]ve` (supplied text), `know[sic]` (error), `selection[?]` (unclear reading), `[...]`
(omission). The pipeline itself has three steps:

| Step | What happens | Code |
|---|---|---|
| 1. Extraction | Find text nodes with brackets in TEI letters and cut out the bracketed sequences | `data/sampling/analysis/extract_editorial_interventions.py`, `data/sampling/sampling.py` |
| 2. LLM processing | Send each text element and its bracketed sequences to a model; get back a TEI encoding per sequence (`<choice><abbr>`/`<expan>`, `<supplied>`, `<sic>`/`<corr>`, `<unclear>`, `<gap/>`, `<surplus>`) | `llm_processing/`, prompts in `data/prompts/` |
| 3. Write-back | Insert the encodings back into the TEI letter files | `evaluation/update_xml_files.py` |

`llm_processing/` is a separate engine also used by other projects; everything specific to this project
(prompts, run configurations) lives outside it, in `configs/` and `data/prompts/`. The engine is included in
`llm_processing/` as a git subtree of `https://github.com/strubrina/llm-processing` (see
[Installation](#installation)).

`evaluation/` scores an LLM's output against a hand-annotated gold standard, for benchmarking model/prompt
choices; see [Evaluating output quality](#evaluating-output-quality).

## Project structure

```
configs/                    run configuration of each published LLM run (read by the engine)
data/
  input/                    the published sample: 90 TEI letters (bracket_sample/), the 169 bracketed
                            sequences (brackets_sample.json/.xlsx), letter metadata
  prompts/                  prompt templates of the LLM runs
  output/<run>/processing_<timestamp>/
                            stored raw output of each LLM run (see "Output files")
  references/gold/          gold standard: the 90 letters with hand-made TEI encodings
  sampling/                 how the sample was drawn (needs the unpublished full corpus);
                            brackets_statistics.xlsx = gold-standard type of every sequence
evaluation/                 evaluation scripts and their results (results/, comparisons/,
                            aggregated_results.*, language_scores.xlsx)
  utils/                    helpers used by eval.py
llm_processing/             the LLM engine (git subtree of strubrina/llm-processing)
utils/                      shared XML helpers (Jing validation, normalisation, gold-standard types)
resources/                  TEI schema (tei_all.rng) and the Jing validator
keys.example.py             template for the API keys of new LLM runs
```

## Installation

- Python 3.9.11. The published results were produced and checked with this version on Windows 11; other
  versions are untested.
- `pip install -r requirements.txt`: exact versions of every package needed to reproduce the results
  (evaluation, tables, figures, sampling)
- Java 8 or newer (tested with 1.8.0_471), for TEI schema validation with the bundled Jing validator
  (`resources/tools/jing-RELEASE220.jar`); `java` must be on the `PATH`

Only for **new** LLM runs:

- `pip install -r requirements-llm.txt`: exact versions of the API clients and `llama-cpp-python`, and also
  installs `requirements.txt`; use it instead of `llm_processing/requirements.txt`, which only gives minimum
  versions (for GPU support install a CUDA build of `llama-cpp-python`, see the engine's README)
- The engine is included in `llm_processing/` (version `31348e5` of
  `https://github.com/strubrina/llm-processing`, added as a git subtree).
- API keys for cloud models: copy `keys.example.py` to `llm_processing/keys.py` and fill in your keys
  (`keys.py` is ignored by git)

## Reproducing the published results

Everything after the LLM calls is deterministic and is re-computed from the stored LLM output in
`data/output/`. The commands below reproduce all published results. Run them from
`evaluation/`, in a POSIX shell (on Windows: Git Bash); they take about 15 minutes, mostly for TEI validation
with Jing:

```bash
cd evaluation
export MPLBACKEND=Agg PYTHONIOENCODING=utf-8
printf '\n'  | python eval.py gpt-5.2-2025-12-11
printf '\n'  | python eval.py claude-sonnet-4-6
printf '\n'  | python eval.py olmo2-32B-instruct-Q4
# standard mode (processing_20260301_021912)
printf '1\n' | python eval.py qwen3-14B-Q6
# thinking mode (processing_20260301_022635)
printf '2\n' | python eval.py qwen3-14B-Q6
python aggregate_results.py
python language_scores.py
python compare_models.py
python comp_vis.py
```

`eval.py` asks which processing run to evaluate; the piped answers select the published runs (Enter = the
only run; for `qwen3-14B-Q6`, `1` and `2` are the two runs listed above; check the line "Using mapping file").
Evaluate the two Qwen runs in this order: `aggregate_results.py` sorts runs of the same model by the time
`eval.py` was run, and both runs write the same heatmap files (the published heatmaps are those of the
thinking-mode run).

Things to know when comparing with the committed files:

- `unified_evaluation_summary_*.json` and `aggregated_results.*` contain the time of the `eval.py` run
  (`summary.timestamp`, `Timestamp`); everything else is identical.
- Excel files differ byte-wise on every save; compare their cell values.
- `comp_vis.py` writes a new set of figures into `evaluation/comparisons/`, with the current time in the file
  names: `overall_metrics` and, per dimension (`d4a`, `d4b`, `d1`), `*_macro_averaged` and
  `*_per_category`.

What cannot be re-run identically:

- **The LLM calls** (`data/output/`). API models are not deterministic, and local models depend on the
  hardware and the llama.cpp build even with a fixed seed. A new run
  gives new output; its evaluation then gives different scores.
- **The sampling** (`data/sampling/sampling.py`, `data/sampling/analysis/count_brackets_per_letter.py`): needs
  the full letter corpus, which is not part of this repository. Its outputs are stored (`data/input/`,
  `data/sampling/brackets_statistics.xlsx`).

The five runs, each with its own configuration in `configs/`:

| Run folder | Timestamp | Temperature | Note |
|---|---|---|---|
| `gpt-5.2-2025-12-11` | `20260301_015947` | 0.3 | commercial API |
| `claude-sonnet-4-6` | `20260301_020434` | 0.3 | commercial API |
| `qwen3-14B-Q6` | `20260301_021912` | 0.6 | local GPU, standard mode |
| `qwen3-14B-Q6` | `20260301_022635` | 0.6 | local GPU, thinking mode |
| `olmo2-32B-instruct-Q4` | `20260927_114615` | 0.6 | local GPU |

Hard-coded corrections in the evaluation:

- **Hard-coded normalisation overrides** in `evaluation/eval.py` (search for `HACK`): in six letters they set
  or adjust the expected text of the source-fidelity check (D1); documented with their effect on scores in
  `evaluation/results/README.md`.

## Entry points

Every script prints its usage with `--help`. Scripts in `evaluation/` are run from `evaluation/`.

| Script | Usage | What it does |
|---|---|---|
| `evaluation/eval.py` | `python eval.py [<run>]` | Evaluate one LLM run against the gold standard (menu for the processing run; without `<run>`, a menu lists the runs) |
| `evaluation/update_xml_files.py` | `python update_xml_files.py [<run>]` | Write a run's encodings back into the TEI letters, into `xml_edited/<run>/`; not needed for the evaluation |
| `evaluation/aggregate_results.py` | `python aggregate_results.py` | One summary table of all evaluated runs |
| `evaluation/compare_models.py` | `python compare_models.py` | Sequences that the five published runs fail, per dimension |
| `evaluation/language_scores.py` | `python language_scores.py` | Mean overall score per letter language and run |
| `evaluation/comp_vis.py` | `python comp_vis.py` | Comparison figures of the five published runs |
| `data/sampling/sampling.py` | `python data/sampling/sampling.py` | Draw the sample from the full corpus (needs the unpublished corpus in `data/sampling/analysis/`) |
| `data/sampling/analysis/extract_editorial_interventions.py` | `python extract_editorial_interventions.py [--letters-dir DIR] [--metadata FILE] [--output FILE]` | Extract all bracketed sequences of a folder of TEI letters into an Excel file (see [Using it with your own TEI data](#using-it-with-your-own-tei-data)) |
| `llm_processing/llm_processing.py` (engine) | `python llm_processing/llm_processing.py --config configs/<run>.py --workflow json` | New LLM run (from the repository root) |


## Output files

**LLM runs**, `data/output/<run>/processing_<timestamp>/` (written by the engine):

- `xml_update_mapping.json`: for every text element, its bracketed sequences and the model's TEI encoding of
  each; this is what the evaluation reads
- `processing_metadata.json`: model, settings, runtime, token counts, estimated cost
- `log/prompts.txt`, `log/responses.txt`: every prompt sent and every raw model response

**Evaluation of one run**, `evaluation/results/<run>/` (written by `eval.py`; file names end with the
processing timestamp):

- `evaluation_<model>_<timestamp>.xlsx`: sheets `Summary`; `Details` (one row per sequence: input, gold
  standard, model output, result of every check); and F1 per category for `Source Fidelity (D1)`,
  `Classification (D4a)`, `Element Application (D4b)`, `Element Categories (Relaxed)`
- `unified_evaluation_summary_<model>_<timestamp>.json`: the same as JSON: `summary` (counts and percentages
  for `dimension_0_…` to `dimension_4b_…`, `overall_tiered_scoring`), the per-category TP/FP/FN/F1 lists, and
  `details` (one entry per sequence)
- `processing_metadata_<timestamp>.json`: copy of the run's metadata; for local runs with the estimated cost added
  (see `evaluation/results/README.md`)
- `unified_evaluation_heatmap.png`, `unified_evaluation_sequence_heatmap.png`

**Across runs**:

- `evaluation/aggregated_results.xlsx` / `.json`: one summary row per run, F1 per category, sequences that
  every run failed
- `evaluation/results/model_comparison_failures.xlsx`: sequences that all five runs failed (D1, D4a, project
  rules) and failure counts
- `evaluation/language_scores.xlsx`: mean overall score per language and run
- `evaluation/comparisons/*.png`: comparison figures

**Sample**: `data/input/brackets_sample.json` (input of the LLM runs: elements, context windows, sequences),
`data/input/brackets_sample.xlsx`, `data/sampling/brackets_statistics.xlsx` (gold-standard intervention type
and language of every sequence; sheet `Summary` = the sample description).

## Using it with your own TEI data

1. **Find bracketed sequences** in your own TEI corpus. `data/sampling/analysis/extract_editorial_interventions.py`
   and `data/sampling/sampling.py` are this project's extraction scripts, not a generic tool — the XPath, the
   bracket-parsing rules, and the expected metadata file are specific to this corpus's TEI structure and
   bracket convention. Use them as a reference and rewrite the extraction logic for your own corpus and
   bracket convention, producing the same JSON shape as `data/input/brackets_sample.json`. To see the
   extraction script actually run (the full letter corpus it was written against is not part of this
   repository), try it against the published 90-letter sample:

   ```
   cd data/sampling/analysis
   python extract_editorial_interventions.py \
       --letters-dir ../../input/bracket_sample \
       --metadata ../../input/corresp-metadata_sample.json \
       --output editorial_interventions_brackets_demo.xlsx
   ```

   `data/sampling/sampling.py` and `data/sampling/analysis/count_brackets_per_letter.py` need the full corpus
   and cannot be run with the published data; they document how the sample was drawn.
2. **Configure a run**: copy one of the files in `configs/` and adjust `MODEL_NAME` and the input path. Useful
   engine options (all optional):

   | Option | What it does |
   |---|---|
   | `--config <file>` | Use a config file from anywhere instead of the engine's default `config.py` |
   | `RUN_NAME` | Output folder name; lets two runs of the same model sit side by side |
   | `INPUT_PRICE_USD_PER_MILLION`, `OUTPUT_PRICE_USD_PER_MILLION` | API prices used for the estimated cost of GPT and Claude runs (recorded in `processing_metadata.json`); the GPT and Claude configs in `configs/` set them |

   **Check the prices before relying on the cost estimates.** The prices in this repository are those used for
   the published runs: the providers' API list prices for these models, and electricity and hardware costs
   from Austria in 2025 for the local models. For other models, providers or hardware, check the current
   prices of your model provider and your own electricity and hardware costs, and set them in your config
   (`INPUT_PRICE_USD_PER_MILLION`, `OUTPUT_PRICE_USD_PER_MILLION`) and in `evaluation/config.py` (cost
   parameters and `MEASURED_POWER_W` for local runs).

3. **Run the pipeline**:

   ```
   python llm_processing/llm_processing.py --config configs/<run>.py --workflow json
   ```

4. **Write the encodings back into your TEI files**:

   ```
   cd evaluation
   python update_xml_files.py <run folder>
   ```

This produces the encoded TEI files. If you also have a gold standard to check quality against, see
[Evaluating output quality](#evaluating-output-quality).

## Evaluating output quality

`evaluation/eval.py` scores an LLM's TEI output against a hand-made gold standard (reference XML in
`data/references/gold/`):

```
cd evaluation
python eval.py <run folder>
```

Each sequence is scored on:

- **D0** well-formed XML
- **D1** source fidelity (the letter's text is unchanged apart from the markup)
- **D2** TEI schema validity and project rules (right elements, no attributes)
- **D4a** intervention type correct
- **D4b** TEI elements correct
- **Overall score** = mean of D1, D4a and D4b (each pass = 100, fail = 0; D1 counts as 0 when D4b fails),
  × 0.75 if the project rules are broken; XML that is not well-formed scores 0

The dimensions follow the evaluation framework [tei-evaluation](https://github.com/strubrina/tei-evaluation)
(D0 well-formedness, D1 source fidelity, D2 schema validity, D3 structural fidelity, D4 semantic content
accuracy), but `evaluation/` is a separate, project-specific implementation and does not use its code. It
scores each bracketed sequence rather than whole files, leaves out D3 (its scope is document structure; each
encoding here is a single element), splits D4 into intervention type (D4a) and elements (D4b), and checks the
project rules in D2 against the intervention type the model assigned, which a schema cannot express.

Scores per intervention type are F1 scores; summary tables report their unweighted average (macro F1). Other
per-run notes (cost estimates, which run is which) are in `evaluation/results/README.md`.

To summarise scores across several evaluated runs:

```
python aggregate_results.py
```

`comp_vis.py` draws the figures for the runs listed in its `RUNS` constant (the five published runs); add
your own runs there to include them.

## Data and license

**Data:** The bracketed sequences are drawn from a historical letter edition (90 letters, 100 sampled text elements, 169
bracketed sequences); the full letter corpus is not part of this repository. Source of the letters:
[*Hammer-Purgstall Briefedition*](https://gams.uni-graz.at/context:hpe), version 2024-12-01 (Zentrum für
Wissenschaftsgeschichte, University of Graz; project lead Martina Scholger), based on Walter Höflechner,
*Joseph von Hammer-Purgstall. Briefe*; see the TEI header of each letter. In `data/input/bracket_sample/` and `data/references/gold/` the TEI header is shortened to title,
publisher, authority, license and source; the text of the letters is unchanged (`sampling.py` copies the letters with
the edition's full header).

**Code:** MIT. Data (letters, sequences, gold standard, LLM output, evaluation results): CC BY-NC 4.0. Details,
attribution and the third-party files in `resources/` are given in `LICENSE`.

## Citation and version

See `CITATION.cff` for how to cite this repository. The published results were produced by the code of the
release this README belongs to.
