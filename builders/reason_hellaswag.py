"""hellaswag: 200 HellaSwag validation items - pick the most plausible continuation."""

from reason_common import choice_item, new_rng, window_sample, write_task

INSTRUCTIONS = (
    "`context` is the beginning of a description of a real situation, cut off mid-sentence. "
    "`topic` names the activity it describes. Each option below is a candidate continuation of "
    "`context`, written to be read directly after it. Exactly one option is the real continuation; "
    "the others were written to sound plausible but describe something that does not physically or "
    "logically follow from `context`. Select the option that is the most plausible continuation."
)


def main():
    rng = new_rng()
    picked = window_sample("Rowan/hellaswag", "default", "validation", 200, rng, windows=12)
    items = []
    for k, r in enumerate(picked):
        items.append(
            choice_item(
                f"hellaswag-{k}",
                {"topic": r["activity_label"], "context": r["ctx"]},
                [str(e) for e in r["endings"]],
                int(r["label"]),
                INSTRUCTIONS,
                rng,
            )
        )
    write_task(
        {
            "id": "hellaswag",
            "title": "Most plausible continuation (HellaSwag)",
            "category": "reasoning",
            "source": (
                "Rowan/hellaswag validation split, 200 sampled seed 7 from 1,200 rows drawn as "
                "12 random 100-row windows (the datasets-server rate limit makes a scattered "
                "row-by-row sample of the 10,042-row split impractical); "
                "continuations shuffled and re-keyed A..D with seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
