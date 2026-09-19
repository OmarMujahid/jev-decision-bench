"""anli_r3: 200 balanced ANLI round-3 test items, task-level entailment choice."""

from reason_common import all_rows, new_rng, write_task

LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
COUNTS = {"entailment": 67, "neutral": 67, "contradiction": 66}

QUESTION = {
    "type": "choice",
    "instructions": (
        "Treat `premise` as true. Decide the relationship between `premise` and `hypothesis` and "
        "select the matching option. Judge only what `premise` states or clearly implies; do not "
        "use outside knowledge to fill gaps, and do not treat a merely plausible hypothesis as "
        "supported."
    ),
    "criteria": {
        "entailment": "`premise` makes `hypothesis` true: `hypothesis` follows from `premise`.",
        "neutral": (
            "`premise` neither makes `hypothesis` true nor makes it false: `hypothesis` could be "
            "true or false as far as `premise` says, including when it adds details `premise` "
            "does not mention."
        ),
        "contradiction": (
            "`premise` makes `hypothesis` false: `hypothesis` conflicts with `premise`."
        ),
    },
}


def main():
    rng = new_rng()
    pool = all_rows("facebook/anli", "plain_text", "test_r3")
    by_label = {v: [] for v in LABELS.values()}
    for r in pool:
        lab = LABELS.get(int(r["label"]))
        if lab and r.get("premise") and r.get("hypothesis"):
            by_label[lab].append(r)
    items = []
    for lab, n in COUNTS.items():
        for k, r in enumerate(rng.sample(by_label[lab], n)):
            items.append(
                {
                    "id": f"anli-{lab}-{k}",
                    # `reason` is a human rationale that gives the label away; it is left out.
                    "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                    "gold": lab,
                }
            )
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"anli-{i}"
    write_task(
        {
            "id": "anli_r3",
            "title": "Adversarial natural language inference (ANLI R3)",
            "category": "reasoning",
            "source": (
                "facebook/anli plain_text test_r3 split, 200 sampled seed 7, balanced "
                "(67 entailment / 67 neutral / 66 contradiction); the dataset's `reason` field "
                "is excluded from state because it explains the gold label"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "question": QUESTION,
            "items": items,
        }
    )


main()
