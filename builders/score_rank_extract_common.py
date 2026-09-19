"""Shared helpers for the groupC builders: cached HF datasets-server paging + task writing."""

import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, ".cache")
TASKS = os.path.join(ROOT, "tasks")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(TASKS, exist_ok=True)

BASE = "https://datasets-server.huggingface.co/rows"


_last_call = [0.0]
MIN_GAP = 4.0  # datasets-server rate-limits anonymous callers hard (CloudFront 429)


def _get(url, tries=20):
    last = None
    for i in range(tries):
        gap = MIN_GAP - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jevbench-builder"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            last = e
            code = getattr(e, "code", None)
            # Retry storms extend the block, so back off on a flat, long cooldown.
            time.sleep(75 if code == 429 else 5 + 5 * i)
    raise last


def rows(dataset, config, split, offset=0, length=100):
    """One page (max 100) of rows, cached on disk."""
    key = "%s_%s_%s_%d_%d.json" % (
        dataset.replace("/", "__"),
        config,
        split,
        offset,
        length,
    )
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    url = "%s?dataset=%s&config=%s&split=%s&offset=%d&length=%d" % (
        BASE,
        urllib.parse.quote(dataset, safe=""),
        urllib.parse.quote(config, safe=""),
        urllib.parse.quote(split, safe=""),
        offset,
        length,
    )
    data = _get(url)
    out = [r["row"] for r in data["rows"]]
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def rows_many(dataset, config, split, total, start=0):
    out = []
    off = start
    while len(out) < total:
        page = rows(dataset, config, split, off, 100)
        if not page:
            break
        out.extend(page)
        off += 100
    return out[:total]


def write_task(task):
    path = os.path.join(TASKS, task["id"] + ".json")
    with open(path, "w") as f:
        json.dump(task, f, ensure_ascii=False, indent=1)
    print("wrote", path, len(task["items"]), "items")
    return path


def squash(s):
    return " ".join(str(s).split())
