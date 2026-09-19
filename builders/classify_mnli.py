import random

from classify_common import SEED, balanced, sample_pages, write

rng = random.Random(SEED)

LABELS = ["entailment", "neutral", "contradiction"]  # glue mnli label ids 0,1,2


def classify(row):
    if not row["premise"].strip() or not row["hypothesis"].strip():
        return None
    return LABELS[row["label"]] if row["label"] in (0, 1, 2) else None


by = sample_pages(
    "nyu-mll/glue", "mnli", "validation_matched", rng, classify, LABELS, 67
)
picked = balanced(rng, by, 67, LABELS)[:200]

write(
    {
        "id": "mnli",
        "title": "Natural language inference (MultiNLI matched)",
        "category": "understanding",
        "source": "nyu-mll/glue mnli validation_matched, 200 sampled seed 7, balanced 67 per class then trimmed to 200",
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "Assume `premise` is true. What is the relationship between `premise` and `hypothesis`? "
                "Judge only what `premise` says; do not use outside knowledge about the topic."
            ),
            "criteria": {
                "entailment": "`hypothesis` must be true if `premise` is true.",
                "neutral": "`hypothesis` might be true or might be false; `premise` neither establishes it nor rules it out.",
                "contradiction": "`hypothesis` must be false if `premise` is true.",
            },
        },
        "items": [
            {
                "id": f"mnli-{i}",
                "state": {
                    "premise": row["premise"].strip(),
                    "hypothesis": row["hypothesis"].strip(),
                },
                "gold": cls,
            }
            for i, (cls, row) in enumerate(picked)
        ],
    }
)
