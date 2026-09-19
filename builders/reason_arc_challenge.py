"""arc_challenge: 200 ARC-Challenge test questions."""

from reason_common import all_rows, choice_item, new_rng, write_task

INSTRUCTIONS = (
    "`question` is a grade-school science exam question. Each option below is one of the answer "
    "choices, copied verbatim from the exam. Exactly one option is the correct answer. "
    "Select that option."
)


def main():
    rng = new_rng()
    pool = all_rows("allenai/ai2_arc", "ARC-Challenge", "test")
    pool = [r for r in pool if r["answerKey"] in r["choices"]["label"] and len(r["choices"]["text"]) >= 2]
    picked = rng.sample(pool, 200)
    items = []
    for k, r in enumerate(picked):
        labels = list(r["choices"]["label"])
        texts = [str(t) for t in r["choices"]["text"]]
        items.append(
            choice_item(
                f"arc-{k}",
                {"question": r["question"]},
                texts,
                labels.index(r["answerKey"]),
                INSTRUCTIONS,
                rng,
            )
        )
    write_task(
        {
            "id": "arc_challenge",
            "title": "Science exam questions (ARC-Challenge)",
            "category": "reasoning",
            "source": (
                "allenai/ai2_arc ARC-Challenge test split, 200 sampled seed 7; "
                "answer options shuffled and re-keyed A..E with seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
