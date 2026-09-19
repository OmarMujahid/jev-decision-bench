"""Shared helpers for the group B (multiple-choice) jevbench builders.

Python stdlib only. Rows come from the Hugging Face datasets-server.
"""

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
TASKS = os.path.join(ROOT, "tasks")
SEED = 7
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

os.makedirs(CACHE, exist_ok=True)
os.makedirs(TASKS, exist_ok=True)


_last_call = [0.0]
MIN_INTERVAL = 1.2  # datasets-server returns 429 well below one request per second


def _get(url, tries=8):
    key = hashlib.sha1(url.encode()).hexdigest()[:24]
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    last = None
    for i in range(tries):
        gap = MIN_INTERVAL - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jevbench-builder"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            _last_call[0] = time.time()
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        except Exception as e:  # transient 5xx / rate limits are common here
            last = e
            _last_call[0] = time.time()
            wait = 60 if isinstance(e, urllib.error.HTTPError) and e.code == 429 else 3 + 4 * i
            time.sleep(wait)
    raise RuntimeError(f"fetch failed {url}: {last}")


def rows(dataset, config, split, offset=0, length=100):
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    d = _get("https://datasets-server.huggingface.co/rows?" + q)
    return [r["row"] for r in d["rows"]], d["num_rows_total"]


def all_rows(dataset, config, split, cap=None):
    out, total = rows(dataset, config, split, 0, 100)
    if cap:
        total = min(total, cap)
        out = out[:total]
    off = len(out)
    while off < total:
        chunk, _ = rows(dataset, config, split, off, min(100, total - off))
        if not chunk:
            break
        out.extend(chunk)
        off += len(chunk)
    return out


def sample_rows(dataset, config, split, n, rng):
    """Sample n rows without loading the whole split: pick indices, fetch covering windows."""
    _, total = rows(dataset, config, split, 0, 1)
    idx = sorted(rng.sample(range(total), min(n, total)))
    got = {}
    i = 0
    while i < len(idx):
        start = idx[i]
        length = min(100, total - start)
        chunk, _ = rows(dataset, config, split, start, length)
        for j, r in enumerate(chunk):
            got[start + j] = r
        while i < len(idx) and idx[i] < start + length:
            i += 1
    return [got[k] for k in idx]


def window_sample(dataset, config, split, n, rng, windows=10):
    """Cluster sample for big splits: `windows` random 100-row windows, then sample n from them.

    The datasets-server rate-limits hard, so fetching a scattered index sample row by row is
    not practical for a 10k-row split.
    """
    first, total = rows(dataset, config, split, 0, 100)
    starts = sorted(rng.sample(range(0, max(1, total - 100)), windows))
    pool = []
    for s in starts:
        chunk, _ = rows(dataset, config, split, s, 100)
        pool.extend(chunk)
    if len(pool) < n:
        pool.extend(first)
    return rng.sample(pool, n)


def choice_item(item_id, state, options, gold_index, instructions, rng):
    """Build an item with a single per-item choice question `q`, options shuffled."""
    order = list(range(len(options)))
    rng.shuffle(order)
    criteria = {}
    gold_key = None
    for pos, src in enumerate(order):
        key = LETTERS[pos]
        criteria[key] = options[src]
        if src == gold_index:
            gold_key = key
    assert gold_key is not None
    return {
        "id": item_id,
        "state": state,
        "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        "gold": {"q": gold_key},
    }


def write_task(task):
    path = os.path.join(TASKS, task["id"] + ".json")
    with open(path, "w") as f:
        json.dump(task, f, ensure_ascii=False, indent=1)
    dist = {}
    for it in task["items"]:
        g = it["gold"]["q"] if "questions" in it else it["gold"]
        dist[g] = dist.get(g, 0) + 1
    print(path, len(task["items"]), "items, gold dist", dict(sorted(dist.items())))
    return path


def new_rng():
    return random.Random(SEED)
