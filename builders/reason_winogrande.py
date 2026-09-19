"""winogrande: 200 WinoGrande XL validation items - fill the blank with option 1 or 2."""

from reason_common import all_rows, choice_item, new_rng, write_task

INSTRUCTIONS = (
    "`sentence` contains exactly one blank, written as a single underscore character. "
    "Each option below is a candidate filler for that blank. Read the whole sentence and decide "
    "which filler makes the sentence consistent and sensible. Select that option."
)


def main():
    rng = new_rng()
    pool = all_rows("allenai/winogrande", "winogrande_xl", "validation")
    pool = [r for r in pool if r.get("answer") in ("1", "2", 1, 2) and "_" in r["sentence"]]
    picked = rng.sample(pool, 200)
    items = []
    for k, r in enumerate(picked):
        items.append(
            choice_item(
                f"winogrande-{k}",
                {"sentence": r["sentence"]},
                [r["option1"], r["option2"]],
                int(r["answer"]) - 1,
                INSTRUCTIONS,
                rng,
            )
        )
    write_task(
        {
            "id": "winogrande",
            "title": "Pronoun-style blank filling (WinoGrande)",
            "category": "reasoning",
            "source": (
                "allenai/winogrande winogrande_xl validation split, 200 sampled seed 7; "
                "the two fillers shuffled into keys A/B with seed 7 so the gold key is not the "
                "dataset's answer index"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
