"""logiqa: 200 LogiQA test items - logical reasoning over a short passage."""

from reason_common import all_rows, choice_item, new_rng, write_task

INSTRUCTIONS = (
    "`passage` states the facts of a logical-reasoning problem and `question` asks something about "
    "them. Each option below is one of the answer choices, copied verbatim. Exactly one option is "
    "correct. Reason only from what `passage` states; do not add outside assumptions. "
    "Select the correct option."
)


def main():
    rng = new_rng()
    pool = all_rows("lucasmccabe/logiqa", "default", "test")
    pool = [
        r
        for r in pool
        if r.get("query") and len(r.get("options", [])) >= 2 and 0 <= int(r["correct_option"]) < len(r["options"])
    ]
    picked = rng.sample(pool, 200)
    items = []
    for k, r in enumerate(picked):
        items.append(
            choice_item(
                f"logiqa-{k}",
                {"passage": r["context"], "question": r["query"]},
                [str(o) for o in r["options"]],
                int(r["correct_option"]),
                INSTRUCTIONS,
                rng,
            )
        )
    write_task(
        {
            "id": "logiqa",
            "title": "Logical reasoning over a passage (LogiQA)",
            "category": "reasoning",
            "source": (
                "lucasmccabe/logiqa test split (English LogiQA, from the Chinese civil-service "
                "exam), 200 sampled seed 7; answer options shuffled and re-keyed A..D with seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
