"""mmlu: 200 MMLU test questions, 8 each from 25 diverse subjects."""

from reason_common import all_rows, choice_item, new_rng, write_task

SUBJECTS = [
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_computer_science",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_geography",
    "high_school_psychology",
    "international_law",
    "machine_learning",
    "marketing",
    "medical_genetics",
    "moral_disputes",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "world_religions",
]
PER = 8

INSTRUCTIONS = (
    "`question` is a multiple-choice exam question from the academic subject named in `subject`. "
    "Each option below is one of the answer choices, copied verbatim from the exam. "
    "Exactly one option is the correct answer. Select that option."
)


def main():
    rng = new_rng()
    items = []
    for subject in SUBJECTS:
        pool = all_rows("cais/mmlu", subject, "test")
        pool = [r for r in pool if r["question"] and len(r["choices"]) >= 2]
        picked = rng.sample(pool, PER)
        for k, r in enumerate(picked):
            items.append(
                choice_item(
                    f"mmlu-{subject}-{k}",
                    {"subject": subject.replace("_", " "), "question": r["question"]},
                    [str(c) for c in r["choices"]],
                    int(r["answer"]),
                    INSTRUCTIONS,
                    rng,
                )
            )
    rng.shuffle(items)
    write_task(
        {
            "id": "mmlu",
            "title": "Academic knowledge (MMLU)",
            "category": "knowledge",
            "source": (
                "cais/mmlu test split, 25 per-subject configs "
                "(" + ", ".join(SUBJECTS) + "), 8 questions sampled per subject with seed 7, "
                "200 total; answer options shuffled with seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
