#!/usr/bin/env python3
"""Local, paired-prompt writing-fingerprint analysis for a model study.

This is deliberately deterministic and API-free. It measures surface style,
not model internals, and must be interpreted as similarity evidence only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROMPT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
WORD_RE = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?|\d+(?:[.,]\d+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-Z0-9\"“‘])")

FUNCTION_WORDS = """
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how however i if in into is it its itself just may me might
more most must my myself neither no nor not of off on once only or other ought
our ours ourselves out over own rather same she should since so some such than
that the their theirs them themselves then there therefore these they this
those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves yet
""".split()

MARKERS = [
    "for example", "for instance", "in other words", "on the other hand",
    "as a result", "at the same time", "in contrast", "more importantly",
    "ultimately", "crucially", "not only", "rather than", "of course",
    "it is important", "the key", "in practice", "to be clear",
    "in conclusion", "this means", "the result", "consider",
]

PUNCTUATION = [".", ",", ";", ":", "!", "?", "—", "–", "-", "(", ")", "\"", "…"]
SUFFIXES = ["ing", "ed", "ly", "ion", "tion", "ment", "ness", "ity", "ive", "ous", "able"]


def resolve_study(path: str) -> Path:
    study = Path(path).expanduser().resolve()
    if not study.is_dir():
        raise ValueError(f"study directory does not exist: {study}")
    return study


def load_study_metadata(study: Path) -> dict[str, str]:
    path = study / "study.json"
    if not path.exists():
        return {"title": "Writing-fingerprint study"}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain a JSON object")
    title = metadata.get("title", "Writing-fingerprint study")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{path} field 'title' must be a non-empty string")
    return {"title": title.strip(), **{k: str(v) for k, v in metadata.items() if k != "title"}}


def load_manifest(study: Path) -> list[dict[str, str]]:
    path = study / "manifest.csv"
    if not path.exists():
        raise ValueError(f"missing study manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "role", "display_name"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError("manifest.csv must contain source_id, role, and display_name columns")
    source_ids = [row["source_id"].strip() for row in rows]
    if any(not PROMPT_RE.match(source) for source in source_ids):
        raise ValueError("manifest source_id values must use lowercase letters, digits, underscores, or hyphens")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("manifest.csv contains duplicate source_id values")
    for row, source in zip(rows, source_ids):
        row["source_id"] = source
        row["role"] = row["role"].strip().lower()
        row["display_name"] = row["display_name"].strip()
    references = [row for row in rows if row["role"] == "reference"]
    targets = [row for row in rows if row["role"] == "target"]
    if len(references) < 2 or len(targets) != 1 or len(references) + len(targets) != len(rows):
        raise ValueError("manifest.csv must contain at least two references and exactly one target")
    if any(not row["display_name"] for row in rows):
        raise ValueError("manifest.csv display_name values cannot be blank")
    return rows


def read_corpus(rows: list[dict[str, str]], raw: Path) -> dict[str, dict[str, str]]:
    corpus: dict[str, dict[str, str]] = {}
    for row in rows:
        source = row["source_id"]
        samples: dict[str, str] = {}
        folder = raw / source
        if folder.exists():
            for path in sorted(folder.glob("*.txt")):
                prompt = path.stem.lower()
                if not PROMPT_RE.match(prompt):
                    continue
                samples[prompt] = path.read_text(encoding="utf-8").strip()
        corpus[source] = samples
    return corpus


def validate(rows: list[dict[str, str]], corpus: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    references = [r["source_id"] for r in rows if r["role"] == "reference"]
    target = next(r["source_id"] for r in rows if r["role"] == "target")
    sets = [set(corpus[s]) for s in references + [target]]
    common = sorted(set.intersection(*sets)) if sets else []
    print("Dataset inventory")
    for row in rows:
        source = row["source_id"]
        samples = corpus[source]
        words = [len(WORD_RE.findall(text)) for text in samples.values()]
        mean = round(float(np.mean(words))) if words else 0
        print(f"  {source:22s} {len(samples):3d} files, mean {mean:4d} words")
        if samples and any(n < 250 for n in words):
            warnings.append(f"{source} has at least one response below 250 words")
    print(f"  {'common prompt IDs':22s} {len(common):3d}")
    if len(common) < 8:
        warnings.append("fewer than 8 matched prompts; attribution is not meaningful yet")
    if any(set(corpus[s]) != set(common) for s in references + [target]):
        warnings.append("sources have unmatched prompt IDs; only the intersection will be analyzed")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return common, warnings


def stable_bucket(value: str, size: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(np.median(arr)), float(np.max(arr))


def syllable_guess(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def extract(text: str) -> tuple[np.ndarray, list[str], list[str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    words_original = WORD_RE.findall(normalized)
    words = [w.lower().replace("’", "'") for w in words_original]
    word_count = max(1, len(words))
    alpha_words = [w for w in words if w[0].isalpha()]
    counts = Counter(alpha_words)
    sentences = [s.strip() for s in SENTENCE_RE.split(normalized) if s.strip()]
    if not sentences:
        sentences = [normalized.strip()]
    sentence_lengths = [len(WORD_RE.findall(s)) for s in sentences]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    paragraph_lengths = [len(WORD_RE.findall(p)) for p in paragraphs]
    sent_mean, sent_std, sent_median, sent_max = safe_stats(sentence_lengths)
    para_mean, para_std, _, _ = safe_stats(paragraph_lengths)
    unique = len(counts)
    hapax = sum(v == 1 for v in counts.values())
    long_words = sum(len(w) >= 8 for w in alpha_words)
    syllables = sum(syllable_guess(w) for w in alpha_words)
    entropy = 0.0
    for value in counts.values():
        p = value / max(1, len(alpha_words))
        entropy -= p * math.log2(p)

    scalar_names = [
        "sentence_mean", "sentence_std", "sentence_cv", "sentence_median",
        "sentence_max_ratio", "paragraph_mean", "paragraph_std",
        "paragraphs_per_1k", "type_token", "hapax_ratio", "lexical_entropy",
        "mean_word_length", "long_word_ratio", "syllables_per_word",
        "flesch_proxy", "uppercase_word_ratio", "numeric_token_ratio",
        "contraction_ratio", "question_sentence_ratio", "exclaim_sentence_ratio",
        "dialogue_line_ratio", "heading_line_ratio", "list_line_ratio",
    ]
    mean_word_len = float(np.mean([len(w) for w in alpha_words])) if alpha_words else 0.0
    syllables_per_word = syllables / max(1, len(alpha_words))
    flesch = 206.835 - 1.015 * sent_mean - 84.6 * syllables_per_word
    lines = [line for line in normalized.splitlines() if line.strip()]
    dialogue_lines = sum(bool(re.match(r"\s*[\"“‘']", line)) for line in lines)
    heading_lines = sum(bool(re.match(r"\s*(#{1,6}\s|[A-Z][^.!?]{0,55}:?\s*$)", line)) for line in lines)
    list_lines = sum(bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)) for line in lines)
    scalar = [
        sent_mean, sent_std, sent_std / max(1.0, sent_mean), sent_median,
        sent_max / word_count, para_mean, para_std, len(paragraphs) * 1000 / word_count,
        unique / max(1, len(alpha_words)), hapax / max(1, unique), entropy,
        mean_word_len, long_words / max(1, len(alpha_words)), syllables_per_word,
        flesch, sum(w.isupper() and len(w) > 1 for w in words_original) / word_count,
        sum(w[0].isdigit() for w in words) / word_count,
        sum("'" in w for w in words) / word_count,
        sum("?" in s for s in sentences) / len(sentences),
        sum("!" in s for s in sentences) / len(sentences),
        dialogue_lines / max(1, len(lines)), heading_lines / max(1, len(lines)),
        list_lines / max(1, len(lines)),
    ]

    lower = normalized.lower().replace("’", "'")
    punct_names = [f"punct_{ord(char):x}" for char in PUNCTUATION]
    punct = [normalized.count(char) * 1000 / word_count for char in PUNCTUATION]
    function_names = [f"fw_{word}" for word in FUNCTION_WORDS]
    function = [counts[word] * 1000 / word_count for word in FUNCTION_WORDS]
    marker_names = [f"marker_{i:02d}" for i in range(len(MARKERS))]
    markers = [lower.count(marker) * 1000 / word_count for marker in MARKERS]
    suffix_names = [f"suffix_{suffix}" for suffix in SUFFIXES]
    suffixes = [sum(w.endswith(suffix) for w in alpha_words) * 1000 / word_count for suffix in SUFFIXES]

    char_size = 256
    char_counts = np.zeros(char_size, dtype=float)
    char_text = re.sub(r"\s+", " ", lower)
    for n in (3, 4):
        for i in range(max(0, len(char_text) - n + 1)):
            gram = char_text[i:i + n]
            char_counts[stable_bucket(f"{n}:{gram}", char_size)] += 1
    if char_counts.sum():
        char_counts *= 1000 / char_counts.sum()
    char_names = [f"charhash_{i:03d}" for i in range(char_size)]

    values = np.asarray(scalar + punct + function + markers + suffixes + char_counts.tolist(), dtype=float)
    names = scalar_names + punct_names + function_names + marker_names + suffix_names + char_names
    blocks = (["structure"] * len(scalar) + ["punctuation"] * len(punct) +
              ["function_words"] * len(function) + ["discourse"] * len(markers) +
              ["morphology"] * len(suffixes) + ["char_ngrams"] * char_size)
    return values, names, blocks


def prepare(rows: list[dict[str, str]], corpus: dict[str, dict[str, str]], prompts: list[str]):
    sources = [r["source_id"] for r in rows]
    raw_vectors: dict[tuple[str, str], np.ndarray] = {}
    names: list[str] = []
    blocks: list[str] = []
    for source in sources:
        for prompt in prompts:
            vector, current_names, current_blocks = extract(corpus[source][prompt])
            raw_vectors[(source, prompt)] = vector
            if not names:
                names, blocks = current_names, current_blocks

    refs = [r["source_id"] for r in rows if r["role"] == "reference"]
    ref_matrix = np.vstack([raw_vectors[(s, p)] for s in refs for p in prompts])
    mean = ref_matrix.mean(axis=0)
    std = ref_matrix.std(axis=0)
    std[std < 1e-8] = 1.0

    block_counts = Counter(blocks)
    block_weight = {
        "structure": 1.0, "punctuation": 0.8, "function_words": 1.2,
        "discourse": 0.8, "morphology": 0.8, "char_ngrams": 1.0,
    }
    weights = np.asarray([
        block_weight[b] / math.sqrt(block_counts[b]) for b in blocks
    ])
    z = {key: ((value - mean) / std) * weights for key, value in raw_vectors.items()}

    # Remove the main effect of each prompt using reference models only.
    residual: dict[tuple[str, str], np.ndarray] = {}
    for prompt in prompts:
        prompt_mean = np.vstack([z[(s, prompt)] for s in refs]).mean(axis=0)
        for source in sources:
            residual[(source, prompt)] = z[(source, prompt)] - prompt_mean
    return raw_vectors, residual, names, blocks


def centroid(residual, source: str, prompts: list[str], exclude: str | None = None) -> np.ndarray:
    chosen = [p for p in prompts if p != exclude]
    return np.vstack([residual[(source, p)] for p in chosen]).mean(axis=0)


def distances(vector: np.ndarray, centers: dict[str, np.ndarray]) -> dict[str, float]:
    return {source: float(np.linalg.norm(vector - center)) for source, center in centers.items()}


def classify_references(residual, refs: list[str], prompts: list[str]):
    predictions = []
    confusion = {actual: {pred: 0 for pred in refs} for actual in refs}
    for prompt in prompts:
        centers = {s: centroid(residual, s, prompts, exclude=prompt) for s in refs}
        for actual in refs:
            ds = distances(residual[(actual, prompt)], centers)
            predicted = min(ds, key=ds.get)
            predictions.append((prompt, actual, predicted, ds))
            confusion[actual][predicted] += 1
    accuracy = sum(a == p for _, a, p, _ in predictions) / max(1, len(predictions))
    recalls = [confusion[s][s] / max(1, sum(confusion[s].values())) for s in refs]
    return predictions, confusion, accuracy, float(np.mean(recalls))


def classify_target(residual, refs: list[str], target: str, prompts: list[str]):
    rows = []
    for prompt in prompts:
        centers = {s: centroid(residual, s, prompts, exclude=prompt) for s in refs}
        ds = distances(residual[(target, prompt)], centers)
        ordered = sorted(ds, key=ds.get)
        rows.append({"prompt": prompt, "prediction": ordered[0], "distances": ds,
                     "margin": ds[ordered[1]] - ds[ordered[0]]})
    return rows


def bootstrap_target(target_rows, refs: list[str], rounds: int = 4000, seed: int = 20260825):
    rng = np.random.default_rng(seed)
    matrix = np.asarray([[row["distances"][s] for s in refs] for row in target_rows])
    winner_counts = np.zeros(len(refs), dtype=int)
    mean_samples = np.zeros((rounds, len(refs)), dtype=float)
    for i in range(rounds):
        indices = rng.integers(0, len(matrix), size=len(matrix))
        means = matrix[indices].mean(axis=0)
        mean_samples[i] = means
        winner_counts[int(np.argmin(means))] += 1
    summary = {}
    for j, source in enumerate(refs):
        summary[source] = {
            "mean_distance": float(matrix[:, j].mean()),
            "ci95_low": float(np.percentile(mean_samples[:, j], 2.5)),
            "ci95_high": float(np.percentile(mean_samples[:, j], 97.5)),
            "bootstrap_win_rate": float(winner_counts[j] / rounds),
        }
    return summary


def rarity_scores(residual, sources: list[str], prompts: list[str], k: int = 5):
    keys = [(s, p) for s in sources for p in prompts]
    matrix = np.vstack([residual[key] for key in keys])
    raw = []
    for i in range(len(keys)):
        d = np.linalg.norm(matrix - matrix[i], axis=1)
        d[i] = np.inf
        raw.append(float(np.sort(d)[:min(k, len(d) - 1)].mean()))
    order = np.argsort(np.argsort(raw))
    percentiles = order / max(1, len(raw) - 1)
    return {key: float(score) for key, score in zip(keys, percentiles)}


def write_outputs(study, metadata, rows, prompts, warnings, refs, target, confusion, accuracy,
                  macro_recall, target_rows, bootstrap, rarity):
    results = study / "results"
    results.mkdir(exist_ok=True)
    display = {r["source_id"]: r["display_name"] for r in rows}

    with (results / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["prompt", "prediction", "margin"] + [f"distance_{s}" for s in refs]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in target_rows:
            output = {"prompt": row["prompt"], "prediction": row["prediction"], "margin": row["margin"]}
            output.update({f"distance_{s}": row["distances"][s] for s in refs})
            writer.writerow(output)

    with (results / "confusion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted"] + refs)
        for actual in refs:
            writer.writerow([actual] + [confusion[actual][pred] for pred in refs])

    vote_counts = Counter(row["prediction"] for row in target_rows)
    ranked = sorted(refs, key=lambda s: bootstrap[s]["mean_distance"])
    summary = {
        "study_title": metadata["title"],
        "matched_prompts": prompts,
        "reference_accuracy": accuracy,
        "reference_macro_recall": macro_recall,
        "target": target,
        "target_vote_counts": dict(vote_counts),
        "target_ranking": ranked,
        "bootstrap": bootstrap,
        "warnings": warnings,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    verdict_ok = len(prompts) >= 20 and accuracy >= 0.70
    screening_ok = len(prompts) >= 8 and accuracy >= 0.60
    top = ranked[0]
    runner = ranked[1]
    top_win = bootstrap[top]["bootstrap_win_rate"]
    status = "INTERPRETABLE" if verdict_ok and top_win >= 0.75 else "SCREENING ONLY"
    if not screening_ok:
        status = "NOT INTERPRETABLE"

    report = [
        f"# {metadata['title']} — writing-fingerprint report", "",
        f"**Status: {status}**", "",
        f"Matched prompts: {len(prompts)}", "",
        f"Reference leave-one-prompt-out accuracy: {accuracy:.1%}", "",
        f"Reference macro recall: {macro_recall:.1%}", "",
        "## Target ranking", "",
        "| Rank | Candidate | Mean distance | 95% bootstrap interval | Bootstrap winner | Prompt votes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, source in enumerate(ranked, 1):
        b = bootstrap[source]
        report.append(
            f"| {i} | {display[source]} | {b['mean_distance']:.4f} | "
            f"{b['ci95_low']:.4f}–{b['ci95_high']:.4f} | "
            f"{b['bootstrap_win_rate']:.1%} | {vote_counts[source]} |"
        )
    report += [
        "", "## Interpretation", "",
        f"The closest reference is **{display[top]}**, followed by **{display[runner]}**. "
        f"The leading candidate wins {top_win:.1%} of prompt-bootstrap resamples.", "",
    ]
    if status != "INTERPRETABLE":
        report.append(
            "This is not a final attribution. Collect more matched prompts and/or improve "
            "reference separability before treating the ranking as evidence."
        )
    else:
        report.append(
            "This ranking is usable as one line of evidence, but it identifies the closest "
            "writing behavior among the supplied candidates—not weights, provider, or exact checkpoint."
        )
    report += ["", "## Warnings", ""]
    report += [f"- {warning}" for warning in warnings] or ["- None generated by validation."]
    report += [
        "", "## Guardrails", "",
        "- Prompt IDs, not individual documents, are held out during validation.",
        "- Prompt-level mean style is removed using references before classification.",
        "- The target is excluded from all feature scaling, prompt means, and training centroids.",
        "- Do not combine reasoning traces with final answers.",
        "- A close result cannot distinguish a base model from a fine-tune, system prompt, or shared serving stack.",
    ]
    (results / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    try:
        mpl_cache = study / ".mplconfig"
        mpl_cache.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt

        labels = [display[s] for s in refs]
        means = [bootstrap[s]["mean_distance"] for s in refs]
        lows = [bootstrap[s]["ci95_low"] for s in refs]
        highs = [bootstrap[s]["ci95_high"] for s in refs]
        order = np.argsort(means)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        y = np.arange(len(refs))
        ordered_means = np.asarray(means)[order]
        ax.errorbar(ordered_means, y,
                    xerr=[ordered_means - np.asarray(lows)[order], np.asarray(highs)[order] - ordered_means],
                    fmt="o", color="#7b2cbf", ecolor="#b8a1d9", capsize=4)
        ax.set_yticks(y, [labels[i] for i in order])
        ax.invert_yaxis()
        ax.set_xlabel("Mean standardized distance (lower is closer)")
        ax.set_title(f"{display[target]} similarity to reference writing fingerprints")
        fig.tight_layout()
        fig.savefig(results / "target_similarity.png", dpi=180)
        plt.close(fig)

        all_sources = refs + [target]
        values = [[rarity[(s, p)] for p in prompts] for s in all_sources]
        fig, ax = plt.subplots(figsize=(10, 5.2))
        violin = ax.violinplot(values, showmeans=True, showmedians=True)
        for body in violin["bodies"]:
            body.set_facecolor("#4c78a8")
            body.set_alpha(0.65)
        ax.set_xticks(range(1, len(all_sources) + 1), [display[s] for s in all_sources], rotation=18)
        ax.set_ylabel("Small-sample rarity percentile")
        ax.set_title("Writing-feature rarity by source (exploratory; not a model identity test)")
        fig.tight_layout()
        fig.savefig(results / "rarity_violin.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


def run(study: Path) -> int:
    metadata = load_study_metadata(study)
    rows = load_manifest(study)
    corpus = read_corpus(rows, study / "data" / "raw")
    prompts, warnings = validate(rows, corpus)
    if len(prompts) < 4:
        print("Need at least four matched prompts to run a smoke analysis.")
        return 2
    refs = [r["source_id"] for r in rows if r["role"] == "reference"]
    target = next(r["source_id"] for r in rows if r["role"] == "target")
    _, residual, _, _ = prepare(rows, corpus, prompts)
    _, confusion, accuracy, macro_recall = classify_references(residual, refs, prompts)
    target_rows = classify_target(residual, refs, target, prompts)
    bootstrap = bootstrap_target(target_rows, refs)
    rarity = rarity_scores(residual, refs + [target], prompts)
    write_outputs(study, metadata, rows, prompts, warnings, refs, target, confusion, accuracy,
                  macro_recall, target_rows, bootstrap, rarity)
    print(f"Wrote analysis to {study / 'results'}")
    print(f"Reference accuracy: {accuracy:.1%}")
    print("Target ranking:")
    for source in sorted(refs, key=lambda s: bootstrap[s]["mean_distance"]):
        print(f"  {source:22s} distance={bootstrap[source]['mean_distance']:.4f} "
              f"bootstrap_win={bootstrap[source]['bootstrap_win_rate']:.1%}")
    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["validate", "run"], default="run")
    parser.add_argument(
        "--study", default=".",
        help="Study directory containing manifest.csv and data/ (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        study = resolve_study(args.study)
        rows = load_manifest(study)
        corpus = read_corpus(rows, study / "data" / "raw")
        if args.command == "validate":
            validate(rows, corpus)
            return 0
        return run(study)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
