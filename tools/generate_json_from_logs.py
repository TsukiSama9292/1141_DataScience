#!/usr/bin/env python3
"""
Generate missing JSON summaries by parsing existing `.log` files.

This script is safe to run locally and will create `log/{name}.json`
for any `log/{name}.log` that does not already have a JSON.
"""
import re
import json
from pathlib import Path


LOG_DIR = Path('log')


def parse_log(path: Path):
    text = path.read_text(encoding='utf-8')
    name = path.stem

    # n_samples
    m = re.search(r'評估結果 \(樣本數:\s*(\d+)\)', text)
    n_samples = int(m.group(1)) if m else None

    metrics = {}
    # ranking metrics
    for k in ['Precision@10', 'Recall@10', 'Hit Rate@10', 'MRR', 'NDCG@10']:
        m = re.search(rf'{re.escape(k)}\s*=\s*([0-9\.]+)', text)
        if m:
            key = k.replace(' ', '_').replace('@', '_at_').lower()
            metrics[key] = float(m.group(1))

    # rating metrics
    for k in ['RMSE', 'MAE']:
        m = re.search(rf'{k}\s*=\s*([0-9\.]+)', text)
        if m:
            metrics[k.lower()] = float(m.group(1))

    # peak memory
    peak = None
    m = re.search(r'記憶體峰值:\s*([0-9\.]+)\s*MB', text)
    if m:
        peak = float(m.group(1))

    # time records
    time_records = {}
    m_total = re.search(r'總執行時間:\s*([0-9\.]+)\s*秒', text)
    if m_total:
        time_records['total_time'] = float(m_total.group(1))

    # capture stage times anywhere in the line (log prefix may exist)
    for line in text.splitlines():
        m = re.search(r'([\u4e00-\u9fffA-Za-z0-9 _@#\-]+):\s*([0-9\.]+)\s*秒', line)
        if m:
            stage = m.group(1).strip()
            # strip common log prefix fragments like '2026-01-02 04:44:24,501 - INFO - '
            stage = re.sub(r"^[0-9\-:,\s]+-\s*INFO\s*-\s*", '', stage)
            stage = stage.strip()
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            # avoid duplicating total_time
            if '總執行時間' in stage and 'total_time' in time_records:
                continue
            time_records[stage] = val

    summary = {
        'name': name,
        'n_samples': n_samples,
        'metrics': metrics,
        'time_records': time_records,
        'peak_memory_mb': peak
    }
    return summary


def main():
    LOG_DIR.mkdir(exist_ok=True)
    for log_file in sorted(LOG_DIR.glob('*.log')):
        json_file = LOG_DIR / f"{log_file.stem}.json"
        if json_file.exists():
            print(f"JSON exists, skipping: {json_file.name}")
            continue
        print(f"Parsing log -> creating JSON: {log_file.name}")
        try:
            summary = parse_log(log_file)
            tmp = json_file.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(json_file)
            print(f"Wrote {json_file.name}")
        except Exception as e:
            print(f"Failed to parse {log_file.name}: {e}")


if __name__ == '__main__':
    main()
