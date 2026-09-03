#!/usr/bin/env python3
"""Create a new, isolated writing-fingerprint study from the bundled template."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "studies" / "template"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", help="New directory for the study, for example studies/my-model")
    args = parser.parse_args()
    destination = Path(args.study_dir).expanduser().resolve()
    if destination.exists():
        print(f"Refusing to overwrite existing path: {destination}", file=sys.stderr)
        return 2
    if not TEMPLATE.is_dir():
        print(f"Missing template directory: {TEMPLATE}", file=sys.stderr)
        return 2
    shutil.copytree(TEMPLATE, destination)
    print(f"Created study scaffold: {destination}")
    print("Next: edit study.json, manifest.csv, models.json, and prompts.md, then collect matched outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
