#!/usr/bin/env python3
"""
Clean corrupted Jupyter notebooks by extracting source code and discarding outputs.
Handles truncated files, bad control characters, and other JSON corruption.
"""
import argparse
import json
import sys
from pathlib import Path


def extract_cells(filepath: str) -> list:
    """Extract cells by finding source sections line-by-line."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    cells = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if '"cell_type":' in line:
            cell_type = "code" if "code" in line else "markdown"
            cell_id = None
            source_lines = []

            j = i
            in_source = False
            bracket_count = 0

            while j < len(lines):
                l = lines[j]

                # Extract cell ID
                if '"id":' in l and cell_id is None:
                    import re
                    m = re.search(r'"id":\s*"([^"]+)"', l)
                    if m:
                        cell_id = m.group(1)

                # Find and extract source array
                if '"source":' in l:
                    in_source = True
                    bracket_count = l.count('[') - l.count(']')
                    source_start = l.find('[')
                    if source_start >= 0:
                        source_lines.append(l[source_start:])
                    j += 1
                    continue

                if in_source:
                    source_lines.append(l)
                    bracket_count += l.count('[') - l.count(']')
                    if bracket_count <= 0:
                        break

                j += 1

            # Parse and create cell
            if source_lines:
                source_text = ''.join(source_lines)
                try:
                    source = json.loads(source_text)
                    cell = {
                        "cell_type": cell_type,
                        "id": cell_id or f"cell_{len(cells)}",
                        "metadata": {},
                        "source": source
                    }
                    if cell_type == "code":
                        cell["execution_count"] = None
                        cell["outputs"] = []
                    cells.append(cell)
                except json.JSONDecodeError:
                    pass  # Skip cells with unparseable source

            i = j

        i += 1

    return cells


def main():
    parser = argparse.ArgumentParser(description='Clean corrupted Jupyter notebooks')
    parser.add_argument('input', help='Input notebook path')
    parser.add_argument('-o', '--output', help='Output path (default: overwrite input)')
    parser.add_argument('--backup', action='store_true', help='Create .backup file')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.backup:
        backup_path = input_path.with_suffix('.ipynb.backup')
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"Backup: {backup_path}")

    cells = extract_cells(str(input_path))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"Saved {len(cells)} cells to {output_path}")


if __name__ == '__main__':
    main()
