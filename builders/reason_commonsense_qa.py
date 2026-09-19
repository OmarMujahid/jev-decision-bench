"""commonsense_qa: 200 CommonsenseQA validation questions."""

from reason_common import all_rows, choice_item, new_rng, write_task

INSTRUCTIONS = (
    "`question` is an everyday-knowledge question with five candidate answers. Each option below "
    "is one of those candidates, copied verbatim. Several options may look possible; exactly one "
    "is the answer most people would give. Select that option."
)


def main():
    rng = new_rng()
    pool = all_rows("tau/commonsense_qa", "default", "validation")
    pool = [r for r in pool if r.get("answerKey") in r["choices"]["label"]]
    picked = rng.sample(pool, 200)
    items = []
    for k, r in enumerate(picked):
        labels = list(r["choices"]["label"])
        texts = [str(t) for t in r["choices"]["text"]]
        items.append(
            choice_item(
                f"csqa-{k}",
                {"question": r["question"]},
                texts,
                labels.index(r["answerKey"]),
                INSTRUCTIONS,
                rng,
            )
        )
    write_task(
        {
            "id": "commonsense_qa",
            "title": "Commonsense question answering (CommonsenseQA)",
            "category": "reasoning",
            "source": (
                "tau/commonsense_qa validation split, 200 sampled seed 7; "
                "answer options shuffled and re-keyed A..E with seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
