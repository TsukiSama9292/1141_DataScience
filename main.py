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
import re
from pathlib import Path


def find_run_scripts(run_dir: Path):
    return sorted([p for p in run_dir.glob('*.py') if p.name != 'test_refactoring.sh'])


def _extract_experiment_name(script_path: Path):
    """Try to extract the Experiment `name` from a run script.

    Falls back to the script stem when no `name=` is found.
    """
    try:
        text = script_path.read_text(encoding='utf-8')
    except Exception:
        return script_path.stem

    # look for name="實驗..." or name='...'
    m = re.search(r"name\s*=\s*[\"']([^\"']+)[\"']", text)
    if m:
        return m.group(1)
    return script_path.stem


def run_scripts(scripts):
    results = []
    py = sys.executable or 'python3'
    log_dir = Path('log')
    log_dir.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        exp_name = _extract_experiment_name(script)
        json_path = log_dir / f"{exp_name}.json"

        # Skip if a JSON summary exists
        if json_path.exists():
            print(f"--- Skipping: {script.name} (already completed: {json_path.name})")
            results.append((script.name, 'skipped'))
            continue

        # Fallback: check text log for a completion marker
        log_file = log_dir / f"{exp_name}.log"
        if log_file.exists():
            try:
                text = log_file.read_text(encoding='utf-8')
                if '實驗完成' in text or '完成' in text:
                    print(f"--- Skipping: {script.name} (log indicates completed: {log_file.name})")
                    results.append((script.name, 'skipped'))
                    continue
            except Exception:
                # if reading fails, fall back to running the script
                pass

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
