"""Shared helpers for the group-D builder scripts: HF datasets-server fetch with
an on-disk cache, plus task-file writing. Python stdlib only."""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
TASKS = os.path.join(ROOT, "tasks")
SEED = 7

os.makedirs(CACHE, exist_ok=True)
os.makedirs(TASKS, exist_ok=True)


def _get(url):
    key = hashlib.sha1(url.encode()).hexdigest() + ".json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    last = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = json.load(r)
            time.sleep(1.5)  # datasets-server rate-limits aggressively
            break
        except Exception as e:
            last = e
            time.sleep(min(60, 5 * 2**attempt))
    else:
        raise RuntimeError(f"fetch failed {url}: {last}")
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def rows(dataset, config, split, offset, length=100):
    url = (
        "https://datasets-server.huggingface.co/rows?"
        f"dataset={urllib.parse.quote(dataset)}&config={config}&split={split}"
        f"&offset={offset}&length={length}"
    )
    d = _get(url)
    return [r["row"] for r in d["rows"]], d["num_rows_total"]


def rows_blocks(dataset, config, split, offsets, length=100):
    """Fetch several 100-row blocks and return them as one flat list."""
    out = []
    for off in offsets:
        rs, _ = rows(dataset, config, split, off, length)
        out.extend(rs)
    return out


def total_rows(dataset, config, split):
    _, n = rows(dataset, config, split, 0, 1)
    return n


def write_task(task):
    path = os.path.join(TASKS, task["id"] + ".json")
    with open(path, "w") as f:
        json.dump(task, f, ensure_ascii=False, indent=1)
    print(f"wrote {path} ({len(task['items'])} items, {os.path.getsize(path)} bytes)")
    return path
