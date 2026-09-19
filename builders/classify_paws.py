import random

from classify_common import SEED, balanced, sample_pages, write

rng = random.Random(SEED)


def classify(row):
    if not row["sentence1"].strip() or not row["sentence2"].strip():
        return None
    return "para" if row["label"] == 1 else "not_para"


by = sample_pages(
    "google-research-datasets/paws",
    "labeled_final",
    "test",
    rng,
    classify,
    ["para", "not_para"],
    100,
)
picked = balanced(rng, by, 100, ["para", "not_para"])

write(
    {
        "id": "paws",
        "title": "Adversarial paraphrase detection (PAWS)",
        "category": "understanding",
        "source": "google-research-datasets/paws labeled_final test, 200 sampled seed 7, balanced 100 paraphrase / 100 not",
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`sentence_a` and `sentence_b` use almost the same words. "
                "They state the same thing: every fact `sentence_a` asserts, `sentence_b` also asserts, and the other way round. "
                "Word order and which name fills which role both matter."
            ),
            "criteria": {
                "true": "The two sentences mean the same thing. They describe the same facts, with the same entities in the same roles, even if the wording or order differs.",
                "false": "The two sentences mean different things. For example, two names or places are swapped between roles, a relation runs the opposite way, or a number, date, or detail differs.",
            },
        },
        "items": [
            {
                "id": f"paws-{i}",
                "state": {
                    "sentence_a": row["sentence1"].strip(),
                    "sentence_b": row["sentence2"].strip(),
                },
                "gold": cls == "para",
            }
            for i, (cls, row) in enumerate(picked)
        ],
    }
)
