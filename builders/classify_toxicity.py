import random

from classify_common import SEED, balanced, sample_pages, write

rng = random.Random(SEED)


def classify(row):
    text = (row.get("text") or "").strip()
    # Keep a clean margin: crowd-rated toxicity >= 0.5 is toxic, < 0.1 is not.
    # Comments in between are ambiguous to the human raters themselves and are dropped.
    if not (20 <= len(text) <= 1500):
        return None
    t = row["toxicity"]
    if t >= 0.5:
        return "toxic"
    if t < 0.1:
        return "clean"
    return None


by = sample_pages(
    "google/civil_comments", "default", "test", rng, classify, ["toxic", "clean"], 100
)
picked = balanced(rng, by, 100, ["toxic", "clean"])

write(
    {
        "id": "toxicity",
        "title": "Toxic comment detection (Civil Comments)",
        "category": "classification",
        "source": (
            "google/civil_comments test, crowd toxicity >= 0.5 as toxic vs < 0.1 as not toxic "
            "(0.1-0.5 dropped as ambiguous), 20-1500 chars, 200 sampled seed 7, balanced 100/100"
        ),
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`comment` was posted by a reader under a news article. "
                "`comment` is toxic: rude, disrespectful, or unreasonable enough that a "
                "reader would be likely to leave the discussion because of it."
            ),
            "criteria": {
                "true": "Toxic. The comment insults, demeans, attacks, mocks, or threatens a person or a group, or uses obscene or hateful language.",
                "false": "Not toxic. The comment is civil, even if it is critical, strongly worded, sarcastic, political, or about an unpleasant topic.",
            },
        },
        "items": [
            {"id": f"toxicity-{i}", "state": {"comment": row["text"].strip()}, "gold": cls == "toxic"}
            for i, (cls, row) in enumerate(picked)
        ],
    }
)
