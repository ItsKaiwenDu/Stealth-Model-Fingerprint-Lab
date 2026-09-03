#!/usr/bin/env python3
"""Resumable Chat Completions collector for a writing-fingerprint study.

The default endpoint is OpenRouter, but any OpenAI-compatible Chat Completions
endpoint can be supplied with --base-url.  It saves only final visible answers
and sanitized metadata; never put an API key in a study directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_HEADING = re.compile(r"^###\s+([a-z][a-z0-9_-]*)\b", re.IGNORECASE)
PROMPT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
PROTOCOL_VERSION = 3
REQUEST_TIMEOUT_SECONDS = 600


class EmptyFinalAnswerError(RuntimeError):
    """A provider completed the request without returning visible text."""


def tls_context() -> ssl.SSLContext:
    """Use certifi when python.org macOS Python lacks a configured CA link."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


TLS_CONTEXT = tls_context()


def resolve_study(path: str) -> Path:
    study = Path(path).expanduser().resolve()
    if not study.is_dir():
        raise ValueError(f"study directory does not exist: {study}")
    return study


def load_models(study: Path) -> dict[str, dict]:
    """Load source IDs to request specifications.

    A source may be a plain model string for a compact setup, or an object with
    model, max_tokens, reasoning, provider, temperature, top_p, and/or seed.
    """
    primary = study / "models.json"
    legacy = study / "openrouter_models.json"
    path = primary if primary.exists() else legacy
    if not path.exists():
        raise ValueError(f"missing models.json in {study}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must be a non-empty object")
    if "sources" in raw:
        raw = raw["sources"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} sources must be a non-empty object")
    models: dict[str, dict] = {}
    for source, spec in raw.items():
        if not isinstance(source, str) or not PROMPT_ID_RE.match(source):
            raise ValueError(f"invalid source ID in {path}: {source!r}")
        if isinstance(spec, str):
            spec = {"model": spec}
        if not isinstance(spec, dict) or not isinstance(spec.get("model"), str) or not spec["model"].strip():
            raise ValueError(f"model spec for {source} must contain a non-empty 'model' string")
        if "max_tokens" in spec and (not isinstance(spec["max_tokens"], int) or spec["max_tokens"] < 1):
            raise ValueError(f"model spec for {source} has an invalid max_tokens value")
        models[source] = spec
    return models


def load_manifest_source_ids(study: Path) -> set[str]:
    path = study / "manifest.csv"
    if not path.exists():
        raise ValueError(f"missing study manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source_id" not in rows[0]:
        raise ValueError(f"{path} must contain at least one source_id row")
    sources = {row["source_id"].strip() for row in rows}
    if any(not PROMPT_ID_RE.match(source) for source in sources) or len(sources) != len(rows):
        raise ValueError(f"{path} contains invalid or duplicate source_id values")
    return sources


def load_prompts(study: Path) -> dict[str, str]:
    path = study / "prompts.md"
    if not path.exists():
        raise ValueError(f"missing prompt battery: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in lines:
        match = PROMPT_HEADING.match(line)
        if match:
            if current:
                prompts[current] = "\n".join(buffer).strip()
            current = match.group(1).lower()
            if current in prompts:
                raise ValueError(f"duplicate prompt ID: {current}")
            buffer = []
        elif current:
            if line.startswith("## "):
                prompts[current] = "\n".join(buffer).strip()
                current = None
                buffer = []
            else:
                buffer.append(line)
    if current:
        prompts[current] = "\n".join(buffer).strip()
    if not prompts or any(not text for text in prompts.values()):
        raise ValueError(f"Expected one or more non-empty prompts, found {len(prompts)}")
    return prompts


def prompt_selection(args, available: dict[str, str]) -> list[str]:
    if args.prompts:
        selected = [item.strip().lower() for item in args.prompts.split(",") if item.strip()]
    elif args.all_prompts:
        selected = sorted(available)
    elif args.phase == 1:
        selected = [f"p{i:02d}" for i in range(1, 13) if f"p{i:02d}" in available]
    elif args.phase == 2:
        selected = [f"p{i:02d}" for i in range(13, 31) if f"p{i:02d}" in available]
    else:
        raise ValueError("Choose --all-prompts, --phase 1, --phase 2, or --prompts p01,p02")
    if not selected:
        raise ValueError("prompt selection did not match any prompts in prompts.md")
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown prompt IDs: {', '.join(unknown)}")
    return selected


def source_selection(args, models: dict[str, dict]) -> list[str]:
    selected = list(models) if args.all_sources else (args.source or [])
    if not selected:
        raise ValueError("Choose one or more --source values, or --all-sources")
    unknown = [item for item in selected if item not in models]
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(unknown)}")
    return selected


def final_text(message: dict) -> str:
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def request_payload(spec: dict, prompt: str) -> dict:
    payload = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": spec.get("max_tokens", 6000),
        "stream": False,
    }
    for field in ("temperature", "top_p", "seed", "reasoning", "provider"):
        if field in spec and spec[field] is not None:
            payload[field] = spec[field]
    return payload


def request_completion(api_url: str, headers: dict[str, str], spec: dict, prompt: str,
                       label: str, max_attempts: int = 8):
    payload = request_payload(spec, prompt)
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
        failure = "unknown temporary failure"
        delay = None
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS, context=TLS_CONTEXT
            ) as response:
                return json.loads(response.read().decode("utf-8")), payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise RuntimeError(f"Chat Completions HTTP {exc.code}: {detail}") from exc
            failure = f"Chat Completions HTTP {exc.code}: {detail}"
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(5, min(300, int(float(retry_after))))
                except (TypeError, ValueError):
                    delay = min(120, 5 * (2 ** (attempt - 1)))
        except (urllib.error.URLError, TimeoutError) as exc:
            is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            if attempt == max_attempts or (is_timeout and attempt >= 2):
                raise RuntimeError(f"Chat Completions network error: {exc}") from exc
            failure = f"network error: {exc}"
        if delay is None:
            delay = min(60, 2 ** attempt)
        print(f"    {label}: {failure}; retrying in {delay}s", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def request_with_heartbeat(api_url: str, headers: dict[str, str], spec: dict, prompt: str,
                           label: str):
    """Run a blocking request while showing that the process is still alive."""
    outcome: dict[str, object] = {}

    def worker() -> None:
        try:
            outcome["result"] = request_completion(
                api_url, headers, spec, prompt, label
            )
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    started = time.monotonic()
    thread.start()
    while thread.is_alive():
        thread.join(timeout=15)
        if thread.is_alive():
            elapsed = int(time.monotonic() - started)
            print(f"    {label}: request still running ({elapsed}s elapsed)...", flush=True)
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["result"]


def save_response(study: Path, source: str, prompt_id: str, spec: dict, response: dict,
                  payload: dict) -> None:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {json.dumps(response)[:1000]}")
    message = choices[0].get("message") or {}
    text = final_text(message)
    if not text:
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        raise EmptyFinalAnswerError(
            "The Chat Completions endpoint returned an empty final answer "
            f"(provider={response.get('provider')!r}, "
            f"finish_reason={choices[0].get('finish_reason')!r}, "
            f"completion_tokens={usage.get('completion_tokens')!r}, "
            f"reasoning_tokens={completion_details.get('reasoning_tokens')!r})"
        )
    raw_dir = study / "data" / "raw" / source
    metadata_dir = study / "data" / "metadata" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{prompt_id}.txt").write_text(text + "\n", encoding="utf-8")
    metadata = {
        "source_id": source,
        "prompt_id": prompt_id,
        "requested_model": spec["model"],
        "returned_model": response.get("model"),
        "provider": response.get("provider"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
        "settings": {
            "protocol_version": PROTOCOL_VERSION,
            "max_tokens": payload["max_tokens"],
            "reasoning": payload.get("reasoning"),
            "provider": payload.get("provider"),
            "temperature": payload.get("temperature", "provider default"),
            "top_p": payload.get("top_p", "provider default"),
            "seed": payload.get("seed", "provider default"),
            "tools": "none",
        },
        "prompt_sha256": hashlib.sha256(
            (payload["messages"][0]["content"] + "\n").encode("utf-8")
        ).hexdigest(),
        "final_answer_characters": len(text),
    }
    (metadata_dir / f"{prompt_id}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", default=".",
        help="Study directory containing manifest.csv, models.json, and prompts.md (default: current directory)",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--phase", type=int, choices=[1, 2])
    prompt_group.add_argument("--prompts", help="Comma-separated IDs, e.g. p01,p02")
    prompt_group.add_argument("--all-prompts", action="store_true", help="Collect every prompt in prompts.md")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source", action="append", help="Source ID; repeat as needed")
    source_group.add_argument("--all-sources", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of models to request concurrently per prompt (1-8; default: 1)",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_API_URL,
        help="OpenAI-compatible /chat/completions endpoint (default: OpenRouter)",
    )
    parser.add_argument(
        "--api-key-env", default="OPENROUTER_API_KEY",
        help="Environment variable holding the API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument("--referer", help="Optional HTTP-Referer header, useful for OpenRouter")
    parser.add_argument("--title", help="Optional X-Title header, useful for OpenRouter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        print("Configuration error: --workers must be between 1 and 8", file=sys.stderr)
        return 2
    try:
        study = resolve_study(args.study)
        models = load_models(study)
        manifest_sources = load_manifest_source_ids(study)
        prompts = load_prompts(study)
        selected_prompts = prompt_selection(args, prompts)
        selected_sources = source_selection(args, models)
        if args.all_sources and set(models) != manifest_sources:
            raise ValueError("--all-sources requires models.json to cover exactly the manifest source IDs")
        unlisted = [source for source in selected_sources if source not in manifest_sources]
        if unlisted:
            raise ValueError(f"selected sources are absent from manifest.csv: {', '.join(unlisted)}")
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    jobs = []
    for prompt_id in selected_prompts:
        for source in selected_sources:
            output = study / "data" / "raw" / source / f"{prompt_id}.txt"
            if output.exists() and not args.overwrite:
                continue
            jobs.append((prompt_id, source))

    print(f"Selected {len(selected_prompts)} prompts × {len(selected_sources)} sources")
    print(f"Pending API requests: {len(jobs)}")
    for source in selected_sources:
        print(f"  {source:24s} {models[source]['model']}")
    if args.dry_run or not jobs:
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"{args.api_key_env} is not set in this terminal.", file=sys.stderr)
        return 2
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if args.referer:
        headers["HTTP-Referer"] = args.referer
    if args.title:
        headers["X-Title"] = args.title

    def collect_one(index: int, prompt_id: str, source: str) -> None:
        spec = models[source]
        label = f"{source} {prompt_id}"
        print(f"[{index}/{len(jobs)}] {label}", flush=True)
        for response_attempt in range(1, 3):
            response, payload = request_with_heartbeat(
                args.base_url, headers, spec, prompts[prompt_id], label
            )
            try:
                save_response(study, source, prompt_id, spec, response, payload)
                return
            except EmptyFinalAnswerError as exc:
                if response_attempt == 2:
                    raise
                print(
                    f"    {label}: {exc}; retrying once with the same settings",
                    flush=True,
                )

    indexed_jobs = [
        (index, prompt_id, source)
        for index, (prompt_id, source) in enumerate(jobs, 1)
    ]
    for prompt_id in selected_prompts:
        prompt_jobs = [job for job in indexed_jobs if job[1] == prompt_id]
        if not prompt_jobs:
            continue
        failures = []
        with ThreadPoolExecutor(max_workers=min(args.workers, len(prompt_jobs))) as executor:
            future_jobs = {
                executor.submit(collect_one, index, job_prompt, source): (job_prompt, source)
                for index, job_prompt, source in prompt_jobs
            }
            for future in as_completed(future_jobs):
                job_prompt, source = future_jobs[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((source, job_prompt, exc))
        if failures:
            for source, job_prompt, exc in failures:
                print(f"FAILED {source} {job_prompt}: {exc}", file=sys.stderr)
            print("Progress is saved. Re-run the same command to resume.", file=sys.stderr)
            return 1
    print(f"Collection complete. Next: python3 analyze.py --study {study} validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
