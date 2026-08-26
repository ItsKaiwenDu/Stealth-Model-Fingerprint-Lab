#!/usr/bin/env python3
"""Resumable OpenRouter collector for the Ox Alpha fingerprint experiment."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_HEADING = re.compile(r"^###\s+(p\d{2})\b", re.IGNORECASE)
NATIVE_REASONING_SOURCES = {"glm_5_2", "mimo_v2_5", "minimax_m3"}
PROTOCOL_VERSION = 2
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


def load_models() -> dict[str, str]:
    return json.loads((ROOT / "openrouter_models.json").read_text(encoding="utf-8"))


def load_prompts() -> dict[str, str]:
    lines = (ROOT / "prompts.md").read_text(encoding="utf-8").splitlines()
    prompts: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in lines:
        match = PROMPT_HEADING.match(line)
        if match:
            if current:
                prompts[current] = "\n".join(buffer).strip()
            current = match.group(1).lower()
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
    if len(prompts) != 30 or any(not text for text in prompts.values()):
        raise ValueError(f"Expected 30 non-empty prompts, found {len(prompts)}")
    return prompts


def prompt_selection(args, available: dict[str, str]) -> list[str]:
    if args.prompts:
        selected = [item.strip().lower() for item in args.prompts.split(",") if item.strip()]
    elif args.phase == 1:
        selected = [f"p{i:02d}" for i in range(1, 13)]
    elif args.phase == 2:
        selected = [f"p{i:02d}" for i in range(13, 31)]
    else:
        raise ValueError("Choose --phase 1, --phase 2, or --prompts p01,p02")
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown prompt IDs: {', '.join(unknown)}")
    return selected


def source_selection(args, models: dict[str, str]) -> list[str]:
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


def request_completion(api_key: str, model: str, prompt: str, use_reasoning_control: bool,
                       label: str, max_attempts: int = 8):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000 if use_reasoning_control else 6000,
        "provider": {"require_parameters": use_reasoning_control},
        "stream": False,
    }
    if use_reasoning_control:
        payload["reasoning"] = {"effort": "high", "exclude": True}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/ox-alpha-fingerprint",
        "X-Title": "Ox Alpha writing fingerprint experiment",
    }
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")
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
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            failure = f"OpenRouter HTTP {exc.code}: {detail}"
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(5, min(300, int(float(retry_after))))
                except (TypeError, ValueError):
                    delay = min(120, 5 * (2 ** (attempt - 1)))
        except (urllib.error.URLError, TimeoutError) as exc:
            is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            if attempt == max_attempts or (is_timeout and attempt >= 2):
                raise RuntimeError(f"OpenRouter network error: {exc}") from exc
            failure = f"network error: {exc}"
        if delay is None:
            delay = min(60, 2 ** attempt)
        print(f"    {label}: {failure}; retrying in {delay}s", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def request_with_heartbeat(api_key: str, model: str, prompt: str,
                           use_reasoning_control: bool, label: str):
    """Run a blocking request while showing that the process is still alive."""
    outcome: dict[str, object] = {}

    def worker() -> None:
        try:
            outcome["result"] = request_completion(
                api_key, model, prompt, use_reasoning_control, label
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


def save_response(source: str, prompt_id: str, model: str, response: dict, payload: dict) -> None:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {json.dumps(response)[:1000]}")
    message = choices[0].get("message") or {}
    text = final_text(message)
    if not text:
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        raise EmptyFinalAnswerError(
            "OpenRouter returned an empty final answer "
            f"(provider={response.get('provider')!r}, "
            f"finish_reason={choices[0].get('finish_reason')!r}, "
            f"completion_tokens={usage.get('completion_tokens')!r}, "
            f"reasoning_tokens={completion_details.get('reasoning_tokens')!r})"
        )
    raw_dir = ROOT / "data" / "raw" / source
    metadata_dir = ROOT / "data" / "metadata" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{prompt_id}.txt").write_text(text + "\n", encoding="utf-8")
    metadata = {
        "source_id": source,
        "prompt_id": prompt_id,
        "requested_model": model,
        "returned_model": response.get("model"),
        "provider": response.get("provider"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
        "settings": {
            "protocol_version": PROTOCOL_VERSION,
            "max_tokens": payload["max_tokens"],
            "reasoning_effort": (payload.get("reasoning") or {}).get("effort", "native default"),
            "reasoning_excluded": (payload.get("reasoning") or {}).get("exclude", False),
            "require_parameters": payload["provider"]["require_parameters"],
            "tools": "none",
            "temperature": "provider default",
            "top_p": "provider default",
        },
        "final_answer_characters": len(text),
    }
    (metadata_dir / f"{prompt_id}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument("--prompts", help="Comma-separated IDs, e.g. p01,p02")
    parser.add_argument("--source", action="append", help="Source ID; repeat as needed")
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of models to request concurrently per prompt (1-8; default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        print("Configuration error: --workers must be between 1 and 8", file=sys.stderr)
        return 2
    try:
        models = load_models()
        prompts = load_prompts()
        selected_prompts = prompt_selection(args, prompts)
        selected_sources = source_selection(args, models)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    jobs = []
    for prompt_id in selected_prompts:
        for source in selected_sources:
            output = ROOT / "data" / "raw" / source / f"{prompt_id}.txt"
            if output.exists() and not args.overwrite:
                continue
            jobs.append((prompt_id, source))

    print(f"Selected {len(selected_prompts)} prompts × {len(selected_sources)} sources")
    print(f"Pending API requests: {len(jobs)}")
    for source in selected_sources:
        print(f"  {source:24s} {models[source]}")
    if args.dry_run or not jobs:
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set in this terminal.", file=sys.stderr)
        return 2

    def collect_one(index: int, prompt_id: str, source: str) -> None:
        model = models[source]
        label = f"{source} {prompt_id}"
        print(f"[{index}/{len(jobs)}] {label}", flush=True)
        use_reasoning_control = source not in NATIVE_REASONING_SOURCES
        for response_attempt in range(1, 3):
            response, payload = request_with_heartbeat(
                api_key, model, prompts[prompt_id], use_reasoning_control, label
            )
            try:
                save_response(source, prompt_id, model, response, payload)
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
    print("Collection complete. Next: python3 analyze.py validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
