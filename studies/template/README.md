# New stealth-model study

1. Give the study a factual title in `study.json`.
2. Put exactly one `target` and at least two `reference` rows in `manifest.csv`.
3. Map each source ID to its endpoint model ID and shared generation settings in
   `models.json`; record dates and deviations in `experiment_log.md`.
4. Use at least eight matched prompts for a screening run; collect a separate,
   untouched prompt set before making a stronger claim.
5. Collect fresh single-turn answers, preserve the final text and metadata, then
   run the repository's validator and analysis commands.

See the repository's `docs/PROTOCOL.md` for design and interpretation rules.
