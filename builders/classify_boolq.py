import random

from classify_common import SEED, balanced, sample_pages, write

rng = random.Random(SEED)


def classify(row):
    if not row["question"].strip() or not row["passage"].strip():
        return None
    if len(row["passage"]) > 6000:
        return None
    return "yes" if row["answer"] else "no"


by = sample_pages("google/boolq", "default", "validation", rng, classify, ["yes", "no"], 100)
picked = balanced(rng, by, 100, ["yes", "no"])

write(
    {
        "id": "boolq",
        "title": "Yes/no reading comprehension (BoolQ)",
        "category": "understanding",
        "source": "google/boolq validation, passages under 6000 chars, 200 sampled seed 7, balanced 100 yes / 100 no",
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`passage` is an extract from an encyclopedia article and `question` is a "
                "yes/no question about it. The answer to `question` is yes, according to `passage`. "
                "Decide using `passage` alone."
            ),
            "criteria": {
                "true": "`passage` states or directly implies that the answer to `question` is yes.",
                "false": "`passage` states or directly implies that the answer to `question` is no.",
            },
        },
        "items": [
            {
                "id": f"boolq-{i}",
                "state": {"passage": row["passage"].strip(), "question": row["question"].strip()},
                "gold": cls == "yes",
            }
            for i, (cls, row) in enumerate(picked)
        ],
    }
)
