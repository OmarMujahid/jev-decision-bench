import random

from classify_common import SEED, balanced, fetch_all, write

rng = random.Random(SEED)
data = fetch_all("nyu-mll/glue", "sst2", "validation")
by = {1: [], 0: []}
for r in data:
    if r["sentence"].strip():
        by[r["label"]].append(r["sentence"].strip())

picked = balanced(rng, by, 100, [1, 0])

write(
    {
        "id": "sst2",
        "title": "Movie review sentiment (SST-2)",
        "category": "classification",
        "source": "nyu-mll/glue sst2 validation, 200 sampled seed 7, balanced 100 positive / 100 negative",
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`excerpt` is a fragment of a movie review written by a film critic. "
                "The opinion the writer expresses about the movie in `excerpt` is positive."
            ),
            "criteria": {
                "true": "The writer is favourable about the movie: they praise it, recommend it, or describe it in approving terms.",
                "false": "The writer is unfavourable about the movie: they criticise it, dismiss it, or describe it in disapproving terms.",
            },
        },
        "items": [
            {"id": f"sst2-{i}", "state": {"excerpt": text}, "gold": label == 1}
            for i, (label, text) in enumerate(picked)
        ],
    }
)
