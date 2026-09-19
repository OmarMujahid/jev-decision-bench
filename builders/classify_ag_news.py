import random

from classify_common import SEED, balanced, sample_pages, write

rng = random.Random(SEED)

LABELS = ["world", "sports", "business", "sci_tech"]  # ag_news label ids 0..3, in order


def classify(row):
    return LABELS[row["label"]] if row["text"].strip() else None


by = sample_pages("fancyzhx/ag_news", "default", "test", rng, classify, LABELS, 50)
picked = balanced(rng, by, 50, LABELS)

write(
    {
        "id": "ag_news",
        "title": "News topic (AG News)",
        "category": "classification",
        "source": "fancyzhx/ag_news test, 200 sampled seed 7, balanced 50 per topic",
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "`article` is the headline and opening sentences of a news wire story. "
                "Which of the four news sections did this story run in?"
            ),
            "criteria": {
                "world": "World news: international affairs, war and conflict, politics and government outside of business, crime, disasters.",
                "sports": "Sports: matches, athletes, teams, leagues, tournaments, transfers, results.",
                "business": "Business: companies, markets, earnings, deals, oil and commodity prices, jobs and the economy.",
                "sci_tech": "Science and technology: research and discovery, space, computing, software, the internet, telecoms hardware and gadgets.",
            },
        },
        "items": [
            {"id": f"ag_news-{i}", "state": {"article": row["text"].strip()}, "gold": cls}
            for i, (cls, row) in enumerate(picked)
        ],
    }
)
