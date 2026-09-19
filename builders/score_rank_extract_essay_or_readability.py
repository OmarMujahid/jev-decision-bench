"""essay_or_readability: HelpSteer2 human helpfulness rating (0-4) of an assistant response."""

import json
import random

from score_rank_extract_common import rows_many, write_task

N = 200
SEED = 7
MAX_ITEM_CHARS = 40000

# Levels follow the five points of the HelpSteer2 helpfulness scale
# (not helpful / borderline unhelpful / partially helpful / mostly helpful / perfectly helpful),
# written as concrete situations.
LEVELS = [
    "The response gives the user nothing they asked for: it answers a different question, "
    "refuses without cause, is off topic, or is wrong throughout.",
    "The response barely serves the request. Most of it misses what the user wanted, and only "
    "a small part of it is usable.",
    "The response does part of what was asked and misses another part, or mixes correct and "
    "useful content with content that is wrong, irrelevant or padding.",
    "The response does what the user asked and is correct, but has a clear shortcoming: a "
    "missing detail, a small error, an unrequested tangent, or a form the user did not ask for.",
    "The response does exactly what the user asked, is correct, and needs nothing added or "
    "removed for the user to act on it.",
]

INSTRUCTIONS = (
    "`prompt` is a request written by a user and `response` is an assistant's reply to it. "
    "Rate how helpful `response` is to the user who wrote `prompt`. Consider whether the reply "
    "does what was asked, whether what it says is correct, and whether it contains material the "
    "user did not ask for. Judge the reply as a whole, not any single sentence in it. A longer "
    "reply is not more helpful for being longer."
)


def main():
    data = rows_many("nvidia/HelpSteer2", "default", "validation", 1000)
    rng = random.Random(SEED)

    by_label = {i: [] for i in range(5)}
    for r in data:
        h = r.get("helpfulness")
        if h is None or r.get("response") is None or r.get("prompt") is None:
            continue
        size = len(json.dumps(r["prompt"])) + len(json.dumps(r["response"]))
        if size > MAX_ITEM_CHARS:
            continue
        by_label[int(h)].append(r)
    for v in by_label.values():
        rng.shuffle(v)

    # Even quota where possible; the rare levels are small, so redistribute the shortfall.
    need, picked = N, []
    order = sorted(by_label, key=lambda k: len(by_label[k]))
    left = len(order)
    for lab in order:
        quota = min(len(by_label[lab]), -(-need // left))
        picked.extend(by_label[lab][:quota])
        need -= quota
        left -= 1
    assert len(picked) == N, len(picked)
    rng.shuffle(picked)

    items = [
        {
            "id": "essay_or_readability-%d" % i,
            "state": {"prompt": r["prompt"], "response": r["response"]},
            "gold": int(r["helpfulness"]),
        }
        for i, r in enumerate(picked)
    ]

    write_task(
        {
            "id": "essay_or_readability",
            "title": "Response helpfulness rating (HelpSteer2)",
            "category": "scoring",
            "source": "nvidia/HelpSteer2 validation, 200 sampled seed 7 spread as evenly as the label counts allow over helpfulness 0-4; gold is the dataset's human helpfulness rating; prompt+response pairs longer than 40k characters skipped",
            "primitive": "score",
            "metric": "spearman",
            "question": {
                "type": "score",
                "instructions": INSTRUCTIONS,
                "criteria": LEVELS,
            },
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
