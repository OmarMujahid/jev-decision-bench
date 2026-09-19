import random

from classify_common import SEED, fetch_all, write

rng = random.Random(SEED)

# PolyAI/banking77 is a loading script the HF dataset viewer refuses to run,
# so we use mteb/banking77, the parquet mirror of the same 77-intent data.
data = fetch_all("mteb/banking77", "default", "test")

by = {}
for r in data:
    if r["text"].strip():
        by.setdefault(r["label_text"], []).append(r["text"].strip())

labels = sorted(by)
assert len(labels) == 77, len(labels)


def describe(label):
    """Short option description derived mechanically from the intent name."""
    phrase = label.replace("_", " ").strip().rstrip("?")
    phrase = phrase[0].lower() + phrase[1:]
    return f"The customer is asking about {phrase}."


criteria = {label: describe(label) for label in labels}

# 200 items over 77 intents: every intent gets 2, and 46 intents drawn at random get a 3rd.
counts = {label: 2 for label in labels}
for label in rng.sample(labels, 200 - 2 * len(labels)):
    counts[label] = 3

picked = []
for label in labels:
    pool = list(by[label])
    rng.shuffle(pool)
    assert len(pool) >= counts[label]
    picked.extend((label, t) for t in pool[: counts[label]])
rng.shuffle(picked)
assert len(picked) == 200

write(
    {
        "id": "banking77",
        "title": "Banking intent among 77 (Banking77)",
        "category": "classification",
        "source": (
            "mteb/banking77 test (parquet mirror of PolyAI/banking77, whose loading script the HF "
            "dataset viewer cannot run), 200 sampled seed 7, 2 per intent plus a 3rd for 46 "
            "randomly chosen intents; all 77 intents are offered as options on every item"
        ),
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "`query` is a message a customer sent to the support team of an online bank. "
                "Which one of these 77 support intents does `query` belong to? "
                "Pick the intent that matches what the customer is actually asking for."
            ),
            "criteria": criteria,
        },
        "items": [
            {"id": f"banking77-{i}", "state": {"query": text}, "gold": label}
            for i, (label, text) in enumerate(picked)
        ],
    }
)
