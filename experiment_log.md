# Experiment log

## Collection window

- Local date: August 25, 2026 (America/Los_Angeles)
- UTC metadata range: 2026-08-26 00:44–03:31
- Interface: OpenRouter Chat Completions API
- Conversation structure: fresh, single-turn request for every model/prompt pair
- Tools, browsing, memory, and custom instructions: none
- Saved content: final answer only; reasoning traces were not saved
- Sampling: provider-default temperature and top-p

## Reasoning settings

- High reasoning requested and required where supported: GLM 5.3, GLM 5,
  DeepSeek V4 Flash, Gemini 3.7 Flash, and Ox Alpha.
- Native/default reasoning: GLM 5.2, MiMo V2.5, and MiniMax M3. Enforcing the
  generic High setting on GLM 5.2 and MiniMax caused completion budgets to be
  consumed without a visible final answer during the pilot. MiMo was moved to
  the same native/default path because its routed providers did not consistently
  expose the generic control.
- The initial `p01` pilot for most sources predates protocol version 2. Its
  metadata therefore lacks `protocol_version` and `require_parameters`. GLM 5.2
  and MiniMax `p01` were recollected under version 2 after empty pilot outputs.

## Routing

OpenRouter selected providers dynamically except where only one provider was
available. Provider names are preserved in each metadata JSON. This means some
reference sources were served by multiple inference providers across prompts;
GLM 5.3, Gemini 3.7 Flash, MiniMax M3, and Ox Alpha remained on one provider each.

## Deviations and incidents

- MiniMax M3 `p01` initially exhausted 16,000 completion tokens in reasoning and
  returned no visible answer. The invalid placeholder file was replaced by a
  successful native/default-reasoning collection.
- GLM 5.2 produced intermittent empty final answers when High reasoning was
  enforced. It was recollected using native/default reasoning.
- GLM 5.2 `p10` ended with `finish_reason=length` at 6,001 completion tokens. It
  was retained unchanged as required by the protocol.
- Ox Alpha experienced repeated upstream shared-pool HTTP 429 responses and one
  request that remained open for more than nine minutes.
- Ox Alpha `p12` could not be collected. Phase-1 analysis therefore uses the
  intersection `p01`–`p11`: 11 matched prompts across all eight sources.
- The collector was updated during the pilot with certificate handling,
  heartbeats, resumable collection, bounded parallelism, empty-response retries,
  and patient 429 backoff. These operational changes do not edit saved answers.

No response was regenerated because its writing seemed weak or surprising.
Only transport failures or responses containing no final text were retried.
