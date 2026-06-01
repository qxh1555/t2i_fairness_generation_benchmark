#!/usr/bin/env python3
"""
Lightweight validation for generated prompt sets.

Usage:
  python scripts/validate_prompt_set.py data/nurse_prompt_set.jsonl
  python scripts/validate_prompt_set.py data/generated/firefighter_prompt_set.jsonl
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def has_word(text: str, word: str) -> bool:
    text = re.sub(r"[_\-]+", " ", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    word = re.sub(r"[_\-]+", " ", word.lower())
    word = re.sub(r"[^a-z0-9]+", " ", word).strip()
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    path = Path(args.path)
    rows = read_jsonl(path)

    errors = []
    warnings = []
    ids = set()

    for row in rows:
        pid = row.get("prompt_id")
        prompt = row.get("prompt", "")
        module = row.get("module")
        occ = row.get("target_occupation")

        if not pid:
            errors.append("Missing prompt_id.")
        elif pid in ids:
            errors.append(f"Duplicate prompt_id: {pid}")
        ids.add(pid)

        if not prompt:
            errors.append(f"{pid}: missing prompt.")
        if module not in {"A", "B", "C", "D", "E", "F"}:
            errors.append(f"{pid}: invalid module {module}")

        # D must not leak occupation/confusable terms.
        if module == "D":
            if "clear face visible" not in prompt.lower():
                errors.append(f"{pid}: contextual trigger must contain `clear face visible`.")
            if row.get("requires_face_visible") is not True:
                errors.append(f"{pid}: contextual trigger must set requires_face_visible=true.")
            target_cues = [
                str(cue).strip()
                for cue in row.get("context_target_specific_cues", [])
                if str(cue).strip()
            ]
            if target_cues:
                if len(target_cues) < 3:
                    errors.append(f"{pid}: contextual trigger should declare at least 3 target-specific cues.")
                missing_cues = [cue for cue in target_cues if not has_word(prompt, cue)]
                if missing_cues:
                    errors.append(f"{pid}: prompt missing declared target-specific cues: {missing_cues}")
            forbidden = row.get("forbidden_terms", [])
            leaked = [t for t in forbidden if t and has_word(prompt, t)]
            if leaked:
                errors.append(f"{pid}: contextual trigger leaks forbidden terms: {leaked}")

        # E should contain explicit gender.
        if module == "E" and not row.get("contains_explicit_gender", False):
            errors.append(f"{pid}: role-binding prompt should contain explicit gender.")

        # F common should not expect people.
        if module == "F" and row.get("slice") == "irrelevant_side_effect_common":
            if row.get("human_expected") is not False:
                warnings.append(f"{pid}: side-effect common prompt should set human_expected=false.")

    counts = {}
    for row in rows:
        counts[row["module"]] = counts.get(row["module"], 0) + 1

    print("Counts:", json.dumps(counts, ensure_ascii=False))
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(" -", w)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    print(f"Validation passed: {len(rows)} prompts.")


if __name__ == "__main__":
    main()
