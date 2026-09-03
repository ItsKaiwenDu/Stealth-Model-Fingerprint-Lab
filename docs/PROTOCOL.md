# Protocol for stealth-model fingerprint studies

This repository is a behavioral-screening tool. Its output answers only:
“Within the references tested, which source is closest under this prompt and
collection protocol?” It does not identify weights, training data, ownership,
or the hidden provider behind a model.

## 1. Form the question before collecting

Write down the target, the candidate set, the intended endpoint/interface, the
date, and the conclusion that would be warranted at each possible outcome.
Treat a mystery model as a target, not a reference. Choose candidates that can
actually distinguish hypotheses: include nearby versions, likely relatives, and
credible alternatives rather than only distant brands.

Do not add or remove candidates because early output “looks like” a favorite
answer. If the first screen points somewhere unexpected, call it exploratory
and test the enlarged candidate set on an untouched confirmation battery.

## 2. Keep cells comparable

Every source/prompt pair is one experimental cell. For a given battery, use:

- a fresh, single-turn conversation for each cell;
- the exact same prompt text and prompt ID;
- final visible answer only, with no reasoning trace;
- the same system message, tools, browsing, temperature, token cap, and other
  controls wherever the interface permits; and
- the same collection path where feasible.

If a setting cannot be made equivalent, do not hide it. Preserve the returned
model/provider, finish reason, usage, request settings, and any retry in
metadata or an experiment log. Such a difference can cause apparent style
similarity or separation.

One sample per cell is enough for a low-cost screen, but it measures a single
stochastic draw. Stronger work should use multiple independent generations per
prompt or a repeated-battery design and pre-specify how they will be aggregated.

## 3. Use a varied, matched prompt battery

Prompt text can overwhelm a writing fingerprint, so variation matters. Include
several genres or tasks: fiction, explanation, structured operational writing,
argument, dialogue, and constraint-following. Avoid prompts that demand a
signature phrase, named entity, citation style, or literal answer that will
dominate the response.

Eight fully matched prompts is the minimum supported screen in this repository.
Twenty or more prompts and a held-out second battery are much more informative.
The bundled 30 prompts are split into a first screening phase and a second
confirmation phase for that reason. Do not inspect phase two until phase one,
the candidate list, and the analysis plan are frozen.

## 4. Validate before interpreting

Run `analyze.py --study <study> validate` before `run`. It reports per-source
file counts, mean word counts, the common prompt intersection, short outputs,
and unmatched IDs. Repair obvious collection failures rather than silently
mixing prompt subsets.

The analysis intentionally uses only prompts held by every source. If a target
or candidate is missing a prompt, the effective study shrinks. Do not replace a
weak but valid answer after seeing an analysis outcome; retry only documented
transport failures or empty final answers under the same settings.

## 5. Read the result in the right order

First inspect reference leave-one-prompt-out accuracy and macro recall. If
known candidate models are not separable under this feature set and prompt
battery, the target ranking is not meaningful. Then inspect the closest
candidate, its distance margin, prompt votes, bootstrap win rate, and the
control confusion matrix together.

The bootstrap resamples prompts, so it measures stability across the collected
prompt set. It does not correct for a missing candidate, a shared system prompt,
provider routing, model updates, or collection artifacts. A 100% bootstrap win
rate is not 100% confidence in model identity.

Status labels are deliberately conservative:

| Status | Meaning |
|---|---|
| `NOT INTERPRETABLE` | Fewer than 8 matched prompts or reference controls are too weak. Collect or redesign before drawing a comparison. |
| `SCREENING ONLY` | A directional candidate-set ranking. Use it to choose what to test next, not as a public attribution. |
| `INTERPRETABLE` | At least 20 matched prompts, reasonable control accuracy, and a stable leader. Still behavior-only evidence, not proof of lineage. |

## 6. Confirm rather than narrate

For a promising result, collect a distinct battery before declaring success.
Include the leader, close competitors, new plausible candidates, and the target;
hold settings fixed; and test whether the same ordering persists. Then report
both batteries, all exclusions, and every known limitation.

The most responsible conclusion is normally phrased as “the target exhibited
writing behavior closest to X among the tested candidates under this protocol.”
Reserve “is X” for independent evidence outside stylometry.
