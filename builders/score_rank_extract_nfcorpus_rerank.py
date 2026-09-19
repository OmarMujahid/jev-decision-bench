"""nfcorpus_rerank: medical/nutrition rerank (BEIR NFCorpus) with lexical hard negatives."""

import random
import re

from score_rank_extract_common import rows_many, squash, write_task

N_QUERIES = 60
SEED = 7
MAX_REL = 5
MAX_CAND = 15
DOC_CHARS = 1200
POOL = 2000

LEVELS = [
    "The document is not about what the query asks for. It may share a general medical or "
    "nutrition topic, but it does not report on the specific question, food, substance or "
    "condition named in the query.",
    "The document is about the subject of the query and reports related findings, but it does "
    "not cover the specific link, effect or claim the query asks about.",
    "The document reports directly on what the query asks about: it studies or reviews the "
    "specific food, substance, condition or effect named in the query.",
]

STOP = set(
    "a an and are as at be by for from has have how in is it its of on or that the to was were "
    "what which who why with does do can may might effect effects".split()
)


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def q_for(i):
    return {
        "type": "score",
        "instructions": (
            "`query` is a health question a person is searching for, and `documents` is a list "
            "of biomedical article titles with their abstracts. Rate how relevant the document "
            "at position %d of `documents`, that is `documents[%d]`, is to `query`. Judge only "
            "`documents[%d]`; ignore every other document in the list. A document is relevant "
            "when reading it would help answer `query`, whatever the answer turns out to be."
        )
        % (i, i, i),
        "criteria": LEVELS,
    }


def main():
    rng = random.Random(SEED)

    corpus = rows_many("BeIR/nfcorpus", "corpus", "corpus", 3700)
    queries = rows_many("BeIR/nfcorpus", "queries", "queries", 3300)
    qrels = rows_many("BeIR/nfcorpus-qrels", "default", "test", 3000)

    docs = {d["_id"]: d for d in corpus}
    qtext = {q["_id"]: q["text"] for q in queries}

    rel = {}
    for r in qrels:
        rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    # qrels rows are grouped by query-id, so the query at the edge of the fetched window may
    # have had its judgments cut off. Drop it.
    rel.pop(qrels[-1]["query-id"], None)

    cand_qids = sorted(
        qid
        for qid, m in rel.items()
        if qid in qtext and any(v > 0 and cid in docs for cid, v in m.items())
    )
    rng.shuffle(cand_qids)

    pool_ids = sorted(docs)
    rng.shuffle(pool_ids)
    pool_ids = pool_ids[:POOL]
    pool_toks = {cid: set(toks(docs[cid]["title"] + " " + docs[cid]["text"])) for cid in pool_ids}

    def doctext(cid):
        d = docs[cid]
        t = squash(d["title"])
        body = squash(d["text"])
        s = (t + ". " + body).strip() if t else body
        return s[:DOC_CHARS]

    items = []
    for qid in cand_qids:
        if len(items) == N_QUERIES:
            break
        query = squash(qtext[qid])
        judged = rel[qid]
        rels = [cid for cid, v in sorted(judged.items(), key=lambda kv: -kv[1]) if v > 0 and cid in docs]
        rels = rels[:MAX_REL]
        if not rels:
            continue

        qt = set(toks(query))
        scored = [
            (len(qt & pool_toks[cid]), cid)
            for cid in pool_ids
            if cid not in judged
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        negs = [cid for _, cid in scored[: MAX_CAND - len(rels)]]

        cands = rels + negs
        rng.shuffle(cands)

        qs = {"d%d" % i: q_for(i) for i in range(len(cands))}
        gold = {"d%d" % i: int(judged.get(cid, 0)) for i, cid in enumerate(cands)}
        items.append(
            {
                "id": "nfcorpus_rerank-%d" % len(items),
                "state": {"query": query, "documents": [doctext(c) for c in cands]},
                "questions": qs,
                "gold": gold,
            }
        )

    assert len(items) == N_QUERIES, len(items)

    write_task(
        {
            "id": "nfcorpus_rerank",
            "title": "Biomedical document reranking (BEIR NFCorpus)",
            "category": "ranking",
            "source": (
                "BeIR/nfcorpus corpus+queries with BeIR/nfcorpus-qrels test, 60 test queries "
                "sampled seed 7 from those with at least one relevant document; candidates = up "
                "to 5 relevant docs (highest graded first) + lexical hard negatives (most query "
                "token overlap among a random 2000-doc pool, excluding all judged docs) up to 15 "
                "candidates, shuffled; docs = title + abstract truncated to 1200 chars; gold = "
                "graded qrels relevance, 0 for unjudged negatives"
            ),
            "primitive": "score",
            "metric": "ndcg10",
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
