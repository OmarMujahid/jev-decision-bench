import random

from classify_common import SEED, fetch_all, write

rng = random.Random(SEED)

# CogComp/trec has no dataset-viewer support (loading script), so we use the
# SetFit mirror of the same TREC question-classification data, coarse labels.
CLASSES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

test = fetch_all("SetFit/TREC-QC", "default", "test")
train = fetch_all("SetFit/TREC-QC", "default", "train", limit=5500)

by_test = {c: [] for c in CLASSES}
for r in test:
    by_test[r["label_coarse_original"]].append(r["text"].strip())

picked = []
NEED = 34  # 6 * 34 = 204, trimmed to 200 below
topped = {}
for c in CLASSES:
    pool = list(by_test[c])
    rng.shuffle(pool)
    take = pool[:NEED]
    if len(take) < NEED:
        # The TREC test split holds only 9 ABBR questions, so the shortfall for a
        # class is topped up from the train split of the same dataset.
        extra = [r["text"].strip() for r in train if r["label_coarse_original"] == c]
        rng.shuffle(extra)
        topped[c] = NEED - len(take)
        take += extra[: NEED - len(take)]
    picked.extend((c, t) for t in take)

rng.shuffle(picked)
picked = picked[:200]

write(
    {
        "id": "trec",
        "title": "Question type (TREC coarse)",
        "category": "classification",
        "source": (
            "SetFit/TREC-QC test (mirror of CogComp/trec, which the HF dataset viewer cannot serve), "
            "coarse labels, 200 sampled seed 7, balanced ~33 per class; classes short in the test "
            f"split topped up from the train split of the same dataset: {topped or 'none'}"
        ),
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "`question` is a factual question typed into a question-answering system. "
                "What kind of thing would a correct answer to `question` be? "
                "Judge the answer that is being asked for, not the wording of the question."
            ),
            "criteria": {
                "ABBR": "An abbreviation: the answer is an acronym or initialism, or the expansion of one.",
                "DESC": "A description: the answer is a definition, an explanation, a reason, or a description of how something works.",
                "ENTY": "An entity: the answer is a thing such as an animal, plant, colour, food, product, language, event, technique, substance, work, or currency.",
                "HUM": "A human: the answer is a person, a group of people, or an organisation, or a description of a person.",
                "LOC": "A location: the answer is a place such as a city, country, state, mountain, or other geographic location.",
                "NUM": "A number: the answer is a numeric value such as a count, a date, an age, a distance, a price, a speed, a temperature, or a percentage.",
            },
        },
        "items": [
            {"id": f"trec-{i}", "state": {"question": text}, "gold": cls}
            for i, (cls, text) in enumerate(picked)
        ],
    }
)
