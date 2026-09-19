"""msmarco_rerank: rerank the ~10 retrieved passages of an MS MARCO query, one score per passage."""

import random

from score_rank_extract_common import rows_many, squash, write_task

N = 100
SEED = 7
MIN_P, MAX_P = 8, 12

LEVELS = [
    "The passage does not answer the question. It is about something else, or it touches the "
    "same subject but contains none of the information the question asks for.",
    "The passage is about what the question asks about and gives related background or a "
    "partial detail, but a reader could not get the answer to the question from it.",
    "The passage contains the answer to the question. A reader could read this passage on its "
    "own and know the answer.",
]


def q_for(i):
    return {
        "type": "score",
        "instructions": (
            "`query` is a question typed into a search engine and `passages` is a list of web "
            "passages retrieved for it. Rate how well the passage at position %d of `passages`, "
            "that is `passages[%d]`, answers `query`. Judge only `passages[%d]`; ignore every "
            "other passage in the list, and ignore whether the passage is well written."
        )
        % (i, i, i),
        "criteria": LEVELS,
    }


def main():
    data = rows_many("microsoft/ms_marco", "v1.1", "validation", 800)
    rng = random.Random(SEED)

    usable = []
    for r in data:
        p = r["passages"]
        sel, texts = p["is_selected"], p["passage_text"]
        if len(sel) != len(texts):
            continue
        if not (MIN_P <= len(texts) <= MAX_P):
            continue
        if sum(sel) != 1:
            continue
        usable.append(r)

    rng.shuffle(usable)
    picked = usable[:N]
    assert len(picked) == N, len(picked)

    items = []
    for i, r in enumerate(picked):
        texts = [squash(t) for t in r["passages"]["passage_text"]]
        sel = r["passages"]["is_selected"]
        qs = {"p%d" % j: q_for(j) for j in range(len(texts))}
        gold = {"p%d" % j: int(sel[j]) for j in range(len(texts))}
        items.append(
            {
                "id": "msmarco_rerank-%d" % i,
                "state": {"query": squash(r["query"]), "passages": texts},
                "questions": qs,
                "gold": gold,
            }
        )

    write_task(
        {
            "id": "msmarco_rerank",
            "title": "Passage reranking (MS MARCO v1.1)",
            "category": "ranking",
            "source": "microsoft/ms_marco v1.1 validation, 100 queries sampled seed 7 from the first 800 rows, each with 8-12 retrieved passages and exactly one is_selected=1; gold relevance is the dataset's is_selected flag",
            "primitive": "score",
            "metric": "mrr",
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
