# Stealth Model Fingerprint Lab

A small, reproducible toolkit for asking a disciplined question about an
unidentified model: **which supplied reference model has the most similar
observable writing behavior under the same prompts?**

It began as an investigation of OpenRouter's `stealth/ox-alpha`. That original
corpus remains intact at the repository root as a retrospective case study:
Ox Alpha was closest to the endpoint then exposed as `z-ai/glm-5.3` on every
one of 11 matched prompts. The later public appearance of GLM-5.3-Flash makes
the investigation a useful real-world check of the workflow—not a claim that
stylometry alone can prove model weights or lineage.

The toolkit is provider-neutral at the analysis layer. Collect outputs by hand
from any interface, or use the included collector with OpenRouter (the default)
or another OpenAI-compatible Chat Completions endpoint.

> A result is a ranking within the candidate set, not an identity verdict. It
> cannot distinguish a checkpoint from fine-tuning, a system prompt, a shared
> serving stack, or post-processing.

![Ox Alpha similarity ranking](results/target_similarity.png)

## Start a new investigation

Python 3, NumPy, and Matplotlib are the only required dependencies.

```bash
python3 -m pip install -r requirements.txt
python3 new_study.py studies/my-stealth-model
```

Edit the new study's `study.json`, `manifest.csv`, `models.json`, and
`prompts.md`. The template has three references and one target. Keep exactly
one row with role `target`; use at least two references, ideally including
nearby versions as well as plausible alternatives.

To collect through OpenRouter, set the key only in your terminal, then run the
same prompt battery against every source:

```bash
export OPENROUTER_API_KEY='your-key-here'
python3 collect_chat_completions.py --study studies/my-stealth-model \
  --all-prompts --all-sources --workers 4
python3 analyze.py --study studies/my-stealth-model validate
python3 analyze.py --study studies/my-stealth-model run
```

For another OpenAI-compatible service, point the collector at its Chat
Completions endpoint and name the environment variable that holds its key:

```bash
export PROVIDER_API_KEY='your-key-here'
python3 collect_chat_completions.py --study studies/my-stealth-model \
  --all-prompts --all-sources \
  --base-url 'https://provider.example/v1/chat/completions' \
  --api-key-env PROVIDER_API_KEY
```

The collector is resumable: existing `data/raw/<source>/<prompt>.txt` files are
skipped unless `--overwrite` is passed. It saves visible final answers and
sanitized response metadata, never reasoning traces or API keys. If a platform
is not Chat-Completions-compatible, collect manually in the same directory
layout and run the analysis normally.

See [the protocol](docs/PROTOCOL.md) before interpreting a result. The bundled
template is deliberately a small screening battery; the root `prompts.md`
contains the larger 30-prompt battery used for the Ox Alpha study.

## What a study contains

```text
studies/my-stealth-model/
├── study.json                 # title and contextual notes
├── manifest.csv               # source ID, role, display name
├── models.json                # source ID → endpoint model/settings
├── prompts.md                 # ### p01 — label, followed by prompt text
├── experiment_log.md          # collection dates and deviations
├── data/
│   ├── raw/<source>/<prompt>.txt
│   └── metadata/<source>/<prompt>.json
└── results/                   # generated reports, tables, and figures
```

Prompt IDs may use lowercase letters, numbers, `_`, and `-`. The analysis uses
only the prompt IDs present for every reference and the target. Metadata is
recommended for auditability but not required for analysis.

`models.json` supports a short model string or a request object. This lets a
study record the same generation controls for every source where the provider
supports them:

```json
{
  "candidate_a": {
    "model": "provider/candidate-a",
    "max_tokens": 6000,
    "temperature": 0.7,
    "reasoning": {"effort": "high", "exclude": true}
  }
}
```

`provider` is also passed through when needed by a gateway such as OpenRouter.
Do not invent a setting just because one provider exposes it: record
unavoidable differences in a study log and treat them as a limitation.

## What the analysis does

Each final answer becomes a deterministic 460-feature representation: hashed
character n-grams, function-word rates, structural measures, discourse-marker
rates, punctuation rates, and morphology measures. Features are standardized
using reference outputs only; feature families are block-weighted; and each
prompt's reference-model mean is removed to reduce the prompt's main effect.

For each held-out prompt, the tool compares an output with source centroids
built from the other prompts. That produces two useful checks:

- Reference leave-one-prompt-out accuracy asks whether the candidate set is
  separable at all.
- Target distances, prompt votes, and 4,000 prompt-bootstrap resamples show
  which supplied reference is closest and how stable that ordering is across
  this prompt set.

The report calls fewer than 8 matched prompts or poor reference separability
`NOT INTERPRETABLE`. It calls 8–19 matched prompts with reasonable controls
`SCREENING ONLY`; use an untouched confirmation battery before making a
stronger statement. Twenty matched prompts, at least 70% control accuracy, and
a stable bootstrap leader are a minimum threshold for an `INTERPRETABLE`
ranking—not proof of model identity.

## Retrospective case study: Ox Alpha → GLM-5.3-Flash

The original data and results are preserved at the repository root. It used
fresh single-turn OpenRouter requests on August 25, 2026, saved final answers
only, and tested the balanced intersection of 11 prompts across seven known
references and Ox Alpha. The model was closest to the historical
`z-ai/glm-5.3` endpoint, now useful as the GLM-5.3-Flash comparison point.

| Rank | Reference | Mean distance | Bootstrap winner | Prompt votes |
|---:|---|---:|---:|---:|
| 1 | GLM-5.3-Flash | 1.8094 | 100.0% | 11/11 |
| 2 | GLM-5.2 | 1.9363 | 0.0% | 0/11 |
| 3 | Gemini 3.7 Flash | 1.9995 | 0.0% | 0/11 |
| 4 | GLM-5 | 2.0036 | 0.0% | 0/11 |
| 5 | MiMo V2.5 | 2.0116 | 0.0% | 0/11 |
| 6 | DeepSeek V4 Flash | 2.0299 | 0.0% | 0/11 |
| 7 | MiniMax M3 | 2.0423 | 0.0% | 0/11 |

Known-control leave-one-prompt-out accuracy was 72.7%. The 6.6% distance
advantage over GLM-5.2, unanimous prompt votes, and 100% bootstrap win rate
were strong screening evidence within this candidate set. But the study is
still correctly labeled `SCREENING ONLY`: it has 11 matched prompts, an
incomplete candidate universe, one generation per cell, and non-identical
reasoning controls where providers required them.

Read the full [Ox Alpha case study](docs/CASE_STUDY_OX_ALPHA.md), the generated
[screening report](results/report.md), [per-prompt distances](results/predictions.csv),
and [control confusion matrix](results/confusion.csv). The original collection
incidents and deviations are in [experiment_log.md](experiment_log.md).

Re-run the preserved case study with:

```bash
python3 analyze.py validate
python3 analyze.py run
```

## Contributing evidence

Useful contributions are new, fully documented study directories—not just a
ranking screenshot. Include the exact prompt battery, candidate manifest,
visible final answers, non-secret request metadata, collection dates, and all
known deviations. Add candidates before inspecting outcomes when possible;
reserve a second prompt battery before choosing a narrative about the first.

Please avoid publishing private prompts, API keys, hidden reasoning, or claims
that a writing-fingerprint comparison establishes weights, training data, or a
provider relationship. The goal is a reusable mental map of behavioral
similarity as models appear, not a false sense of certainty.
