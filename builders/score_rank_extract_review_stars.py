"""review_stars: predict the star rating a Yelp reviewer gave, 5 ordered levels."""

import random

from score_rank_extract_common import rows_many, write_task

N = 200
SEED = 7
MAXCHARS = 1500

LEVELS = [
    "The reviewer is angry or feels wronged. The visit went badly, and they warn other people "
    "off or say they will never come back.",
    "The reviewer was let down. Specific things went wrong and the visit fell short of what "
    "they wanted, even if they name one or two redeeming details.",
    "The reviewer is in the middle. They name real positives and real negatives, or describe an "
    "ordinary experience they neither recommend nor warn against.",
    "The reviewer liked the place and would go back, but names a flaw, a reservation or "
    "something that could be better.",
    "The reviewer is enthusiastic and has no real complaint. They praise the place, call it a "
    "favourite, or tell other people to go.",
]

INSTRUCTIONS = (
    "The text in `review` is a customer review of a business, written by someone who also gave "
    "that business a star rating from 1 to 5. Rate how satisfied the writer of `review` is with "
    "the business, so that the level you pick matches the star rating they gave: the lowest "
    "level corresponds to 1 star and the highest level to 5 stars. Judge only the opinion the "
    "writer expresses about the business. Ignore spelling, grammar, length and how well the "
    "review is written."
)


def main():
    data = rows_many("Yelp/yelp_review_full", "yelp_review_full", "test", 2000)
    rng = random.Random(SEED)

    by_label = {i: [] for i in range(5)}
    for r in data:
        by_label[int(r["label"])].append(r)
    for v in by_label.values():
        rng.shuffle(v)

    per = N // 5
    picked = []
    for lab in range(5):
        assert len(by_label[lab]) >= per, (lab, len(by_label[lab]))
        picked.extend(by_label[lab][:per])
    rng.shuffle(picked)

    items = []
    for i, r in enumerate(picked):
        text = r["text"].replace("\\n", "\n").replace('\\"', '"').strip()
        if len(text) > MAXCHARS:
            text = text[:MAXCHARS].rsplit(" ", 1)[0] + " ..."
        items.append(
            {
                "id": "review_stars-%d" % i,
                "state": {"review": text},
                "gold": int(r["label"]),
            }
        )

    write_task(
        {
            "id": "review_stars",
            "title": "Star rating from review text (Yelp Review Full)",
            "category": "scoring",
            "source": "Yelp/yelp_review_full test, 200 sampled seed 7 from the first 2000 rows, 40 per star; gold is the dataset label (stars - 1); reviews truncated to 1500 characters",
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
