#!/usr/bin/env python3
"""
Main runner — run all Python scripts in the `run/` directory sequentially.

This script runs every `*.py` under `run/` (sorted by name) using the current
Python interpreter. It prints start/finish status for each script and a
compact summary at the end.

Usage:
  python main.py
"""

import subprocess
import sys
from pathlib import Path


def find_run_scripts(run_dir: Path):
    return sorted([p for p in run_dir.glob('*.py') if p.name != 'test_refactoring.sh'])


def run_scripts(scripts):
    results = []
    py = sys.executable or 'python3'
    for script in scripts:
        print(f"--- Running: {script.name}")
        proc = subprocess.run([py, str(script)])
        results.append((script.name, proc.returncode))
        print(f"--- Finished: {script.name} (code={proc.returncode})\n")
    return results


def main():
    root = Path.cwd()
    run_dir = root / 'run'
    if not run_dir.exists():
        print('run/ directory not found')
        return

    scripts = find_run_scripts(run_dir)
    if not scripts:
        print('No scripts found in run/')
        return

    results = run_scripts(scripts)

    print('Summary:')
    for name, code in results:
        print(f"  {name}: returncode={code}")


if __name__ == '__main__':
    main()
