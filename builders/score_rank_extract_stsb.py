"""stsb: semantic textual similarity, 6 score levels from the official STS annotation guideline."""

import random

from score_rank_extract_common import rows_many, squash, write_task

N = 200
SEED = 7

# Wording of the six levels as published in the SemEval STS annotation guidelines
# (and reproduced on the STS-Benchmark page), lowest first.
LEVELS = [
    "The two sentences are completely dissimilar.",
    "The two sentences are not equivalent, but are on the same topic.",
    "The two sentences are not equivalent, but share some details.",
    "The two sentences are roughly equivalent, but some important information differs or is missing.",
    "The two sentences are mostly equivalent, but some unimportant details differ.",
    "The two sentences are completely equivalent, as they mean the same thing.",
]

INSTRUCTIONS = (
    "Rate the degree of semantic similarity between the two sentences in `sentence1` and "
    "`sentence2`. Judge only how much of the same meaning the two sentences express. "
    "Ignore differences in wording, grammar, sentence length and writing style that do not "
    "change the meaning. Do not judge whether either sentence is true, well written or useful."
)


def main():
    data = rows_many("nyu-mll/glue", "stsb", "validation", 1500)
    rng = random.Random(SEED)

    # Spread across the 0-5 range: 6 buckets by floor(label), even quota, shortfall
    # redistributed to the remaining buckets.
    buckets = {b: [] for b in range(6)}
    for r in data:
        buckets[min(5, int(r["label"]))].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    picked, need = [], N
    order = sorted(buckets, key=lambda b: len(buckets[b]))
    left = len(order)
    for b in order:
        quota = min(len(buckets[b]), -(-need // left))
        picked.extend(buckets[b][:quota])
        need -= quota
        left -= 1
    rng.shuffle(picked)

    items = [
        {
            "id": "stsb-%d" % i,
            "state": {
                "sentence1": squash(r["sentence1"]),
                "sentence2": squash(r["sentence2"]),
            },
            "gold": round(float(r["label"]), 3),
        }
        for i, r in enumerate(picked)
    ]

    write_task(
        {
            "id": "stsb",
            "title": "Semantic textual similarity (STS-Benchmark)",
            "category": "scoring",
            "source": "nyu-mll/glue stsb validation, 200 sampled seed 7, spread evenly over the six 0-5 similarity buckets; gold is the dataset's averaged human similarity score",
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
