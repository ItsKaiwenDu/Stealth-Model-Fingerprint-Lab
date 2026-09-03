# Case study: Ox Alpha and GLM-5.3-Flash

## Why this case remains in the repository

This repository was originally built to investigate OpenRouter's anonymous
`stealth/ox-alpha` endpoint. The subsequent public appearance of GLM-5.3-Flash
gives the experiment a rare retrospective outcome: the mystery model's behavior
was consistently closest to the endpoint collected as `z-ai/glm-5.3`.

That makes the corpus valuable as a worked reliability example. It does not
retroactively change what the analysis measured. The analysis found the nearest
writing fingerprint among a bounded candidate set; the later release is the
outside context that connects that behavior to GLM-5.3-Flash.

## Frozen corpus and protocol

The included corpus was collected on August 25, 2026 through the OpenRouter
Chat Completions API. Each cell used a fresh single-turn request. Only final
visible answers were saved; generation IDs, API keys, and reasoning traces are
absent. The full prompt text is in [`prompts.md`](../prompts.md), and request
metadata is in [`data/metadata`](../data/metadata).

The original candidate set was GLM-5.3, GLM-5.2, GLM-5, MiMo V2.5, DeepSeek V4
Flash, Gemini 3.7 Flash, and MiniMax M3. Ox Alpha was the sole target. The
target's p12 request repeatedly stalled or was rate-limited, so the analysis
uses p01–p11—the balanced intersection—rather than selectively imputing a
response. Unmatched reference p12 outputs stay in the corpus for auditability.

Several models used native/default reasoning because forcing an incompatible
gateway setting produced empty visible outputs. This is a real limitation, not
a data-cleaning decision. Details, including retries and a length-truncated
GLM-5.2 p10 response retained unchanged, are in the
[`experiment log`](../experiment_log.md).

## Result

The deterministic analysis used 460 surface-style features, reference-only
normalization, prompt-effect removal, leave-one-prompt-out reference controls,
and 4,000 prompt-bootstrap resamples. GLM-5.3 was the nearest reference on all
11 target prompts.

| Measure | Result |
|---|---:|
| Matched prompts | 11 |
| Analyzed documents | 88 |
| Reference leave-one-prompt-out accuracy | 72.7% |
| GLM-5.3 mean distance | 1.8094 |
| GLM-5.2 mean distance | 1.9363 |
| GLM-5.3 advantage over GLM-5.2 | 6.6% |
| GLM-5.3 prompt votes | 11 / 11 |
| GLM-5.3 bootstrap winner rate | 100.0% |

The reproducible artifacts are the [report](../results/report.md),
[per-prompt predictions](../results/predictions.csv),
[control confusion matrix](../results/confusion.csv), and
[summary JSON](../results/summary.json).

## What it supports—and what it does not

At collection time, the responsible conclusion was: *Ox Alpha exhibited
GLM-5.3-like writing behavior among the tested candidates and might be
GLM-5.3-derived or closely related.* The later GLM-5.3-Flash release makes that
working hypothesis a meaningful retrospective check.

It remains inappropriate to infer exact model weights, whether Ox Alpha used a
base checkpoint versus a tuned variant, the identity of a serving provider, or
the cause of the similarity from this experiment alone. The original study also
has 11 rather than 20 matched prompts, incomplete candidate coverage, one
generation per cell, and inconsistent reasoning controls. Those constraints are
why its generated report remains `SCREENING ONLY` even with a unanimous leader.

The right lesson is methodological: with matched prompts, credible controls,
reference-only preprocessing, and conservative reporting, a small stylometry
screen can provide a useful directional map when a new model appears. Follow
the [general protocol](PROTOCOL.md) before treating this case as a template for
any new attribution claim.
