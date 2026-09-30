#!/usr/bin/env python3
"""Rebuild notebooks or sync their Jupytext sources without executing cells."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "paper_odds"


def git_paths(root: Path, *args: str, input: str | None = None) -> set[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        text=True,
        check=False,
    )
    # check-ignore returns 1 when no supplied path is ignored.
    if result.returncode and not (args[0] == "check-ignore" and result.returncode == 1):
        raise RuntimeError(result.stderr.strip())
    fields = result.stdout.rstrip("\0").split("\0") if result.stdout else []
    if args[0] == "check-ignore":
        # Verbose records are: ignore file, line, pattern, path. Git also
        # reports matching !exceptions; those paths are eligible for tracking.
        return {Path(fields[i + 3]) for i in range(0, len(fields), 4)
                if not fields[i + 2].startswith("!")}
    return {Path(name) for name in fields}


def discover(root: Path) -> list[Path]:
    """Find eligible paper_odds pairs, including new ignored notebooks."""
    candidates = git_paths(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", f"{SCOPE}/*.py", f"{SCOPE}/*.ipynb",
    )
    candidates |= git_paths(
        root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
        "--", f"{SCOPE}/*.ipynb",
    )
    sources = {path.with_suffix(".py") for path in candidates}
    if not sources:
        return []
    ignored = git_paths(
        root, "check-ignore", "--no-index", "-v", "-z", "--stdin",
        input="".join(f"{path}\0" for path in sorted(sources)),
    )
    pairs = []
    for source in sorted(sources - ignored):
        notebook = root / source.with_suffix(".ipynb")
        script = root / source
        if notebook.is_file():
            if script.is_file() and "#   jupytext:" not in script.read_text(encoding="utf-8")[:4096].splitlines():
                raise RuntimeError(f"Refusing to replace non-Jupytext source: {source}")
            pairs.append(source)
        elif script.is_file():
            with script.open(encoding="utf-8") as stream:
                if "#   jupytext:" in stream.read(4096).splitlines():
                    pairs.append(source)
    return pairs


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "sync"),
        help="build missing notebooks, or sync the newest inputs in each pair",
    )
    args = parser.parse_args(argv)
    if importlib.util.find_spec("jupytext") is None:
        parser.exit(1, 'Install notebook tooling first: python -m pip install -e ".[notebooks]"\n')

    try:
        pairs = discover(root)
        changed = 0
        for source in pairs:
            notebook = source.with_suffix(".ipynb")
            if args.command == "build":
                # Preserve existing local notebooks, including their outputs and edits.
                if (root / notebook).exists() or not (root / source).is_file():
                    continue
                options = ["--to", "notebook", str(source)]
            else:
                # Jupytext selects the newer inputs and keeps notebook outputs.
                path = source if (root / source).is_file() else notebook
                options = ["--sync", str(path)]
            subprocess.run([sys.executable, "-m", "jupytext", *options], cwd=root, check=True)
            changed += 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Notebook {args.command} failed: {exc}\n")
    print(f"{args.command}: processed {changed} of {len(pairs)} notebook pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
