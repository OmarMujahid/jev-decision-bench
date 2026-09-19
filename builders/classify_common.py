"""Shared helpers for the groupA task builders. Python stdlib only."""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
TASKS = os.path.join(ROOT, "tasks")
SERVER = "https://datasets-server.huggingface.co/rows"
SEED = 7


def _get(url, tries=8):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jevbench-builder"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # datasets-server rate-limits (429) and 500s intermittently
            last = e
            time.sleep(min(60, 5 * (i + 1) ** 2))
    raise RuntimeError(f"failed {url}: {last}")


def rows(dataset, config, split, offset, length=100):
    """One page of rows from the HF datasets-server, cached on disk."""
    url = (
        f"{SERVER}?dataset={urllib.parse.quote(dataset, safe='')}"
        f"&config={urllib.parse.quote(config, safe='')}"
        f"&split={urllib.parse.quote(split, safe='')}"
        f"&offset={offset}&length={length}"
    )
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = _get(url)
    time.sleep(1.0)  # datasets-server rate-limits anonymous callers
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def total(dataset, config, split):
    return rows(dataset, config, split, 0, 1)["num_rows_total"]


def fetch_all(dataset, config, split, limit=None):
    """Every row of a split, in order."""
    n = total(dataset, config, split)
    if limit:
        n = min(n, limit)
    out = []
    off = 0
    while off < n:
        page = rows(dataset, config, split, off, min(100, n - off))
        out.extend(r["row"] for r in page["rows"])
        off += 100
    return out


def fetch_pages(dataset, config, split, offsets):
    """Rows from specific page offsets (for big splits we only sample from)."""
    out = []
    for off in offsets:
        page = rows(dataset, config, split, off, 100)
        out.extend(r["row"] for r in page["rows"])
    return out


def sample_pages(dataset, config, split, rng, classify, classes, need, start_pages=24):
    """Pull random 100-row pages until every class has `need` usable rows.

    classify(row) -> class key, or None to drop the row.
    Page offsets are drawn with the seeded rng, so the set of pages is reproducible.
    """
    n = total(dataset, config, split)
    all_offsets = list(range(0, n, 100))
    rng.shuffle(all_offsets)
    by = {c: [] for c in classes}
    used = 0
    while used < len(all_offsets):
        for off in all_offsets[used : used + start_pages]:
            for r in rows(dataset, config, split, off, 100)["rows"]:
                c = classify(r["row"])
                if c in by:
                    by[c].append(r["row"])
        used += start_pages
        if all(len(v) >= need for v in by.values()):
            break
        start_pages = max(8, start_pages // 2)
    return by


def balanced(rng, by_class, per_class, order_classes):
    """Take per_class items from each class, then shuffle the result."""
    picked = []
    for c in order_classes:
        pool = list(by_class[c])
        rng.shuffle(pool)
        assert len(pool) >= per_class, f"class {c}: only {len(pool)} available"
        picked.extend((c, x) for x in pool[:per_class])
    rng.shuffle(picked)
    return picked


def write(task):
    os.makedirs(TASKS, exist_ok=True)
    path = os.path.join(TASKS, task["id"] + ".json")
    with open(path, "w") as f:
        json.dump(task, f, ensure_ascii=False, indent=1)
    print("wrote", path, len(task["items"]), "items")
    return path
