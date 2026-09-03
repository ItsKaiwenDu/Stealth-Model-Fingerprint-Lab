# Ox Alpha → GLM-5.3-Flash retrospective — writing-fingerprint report

**Status: SCREENING ONLY**

Matched prompts: 11

Reference leave-one-prompt-out accuracy: 72.7%

Reference macro recall: 72.7%

## Target ranking

| Rank | Candidate | Mean distance | 95% bootstrap interval | Bootstrap winner | Prompt votes |
|---:|---|---:|---:|---:|---:|
| 1 | GLM-5.3-Flash | 1.8094 | 1.6536–2.0103 | 100.0% | 11 |
| 2 | GLM-5.2 | 1.9363 | 1.7710–2.1442 | 0.0% | 0 |
| 3 | Gemini 3.7 Flash | 1.9995 | 1.8520–2.1898 | 0.0% | 0 |
| 4 | GLM-5 | 2.0036 | 1.8400–2.2277 | 0.0% | 0 |
| 5 | MiMo V2.5 | 2.0116 | 1.8585–2.2257 | 0.0% | 0 |
| 6 | DeepSeek V4 Flash | 2.0299 | 1.8707–2.2445 | 0.0% | 0 |
| 7 | MiniMax M3 | 2.0423 | 1.8600–2.2548 | 0.0% | 0 |

## Interpretation

The closest reference is **GLM-5.3-Flash**, followed by **GLM-5.2**. The leading candidate wins 100.0% of prompt-bootstrap resamples.

This is not a final attribution. Collect more matched prompts and/or improve reference separability before treating the ranking as evidence.

## Warnings

- sources have unmatched prompt IDs; only the intersection will be analyzed

## Guardrails

- Prompt IDs, not individual documents, are held out during validation.
- Prompt-level mean style is removed using references before classification.
- The target is excluded from all feature scaling, prompt means, and training centroids.
- Do not combine reasoning traces with final answers.
- A close result cannot distinguish a base model from a fine-tune, system prompt, or shared serving stack.
