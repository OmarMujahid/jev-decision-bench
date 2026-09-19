"""squad_sentence: pick the one sentence of a SQuAD context that contains the answer."""

import random
import re

from score_rank_extract_common import rows_many, squash, write_task

N = 200
SEED = 7

ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "no.", "vs.", "etc.", "e.g.", "i.e.",
    "jr.", "sr.", "inc.", "ltd.", "co.", "u.s.", "u.k.", "approx.", "fig.", "al.",
}

INSTRUCTIONS = (
    "`question` is a question about a passage, and `sentences` is that passage split into its "
    "sentences, in order. Exactly one of the sentences states the answer to `question`. Pick the "
    "option whose sentence contains the answer. Each option key is a position in `sentences`: "
    "S0 is `sentences[0]`, S1 is `sentences[1]`, and so on, and the option's description is that "
    "sentence. Pick the sentence that states the answer itself, not a sentence that merely "
    "mentions the same names or topic."
)


def split_sentences(text):
    """Split on sentence-final punctuation followed by whitespace + a capital/digit."""
    parts, buf = [], []
    tokens = re.split(r"(\s+)", text)
    for i, tok in enumerate(tokens):
        buf.append(tok)
        if tok.strip() and re.search(r"[.!?][\"')\]]*$", tok):
            word = tok.strip().lower().strip("\"')]")
            nxt = "".join(tokens[i + 1 : i + 3]).strip()
            if word in ABBREV or re.fullmatch(r"[a-z]\.", word):
                continue
            if nxt and not re.match(r"[\"'(\[]?[A-Z0-9]", nxt):
                continue
            parts.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def main():
    data = rows_many("rajpurkar/squad", "plain_text", "validation", 3000)
    rng = random.Random(SEED)

    by_ctx = {}
    for r in data:
        ans = r["answers"]
        if not ans["text"]:
            continue
        start = int(ans["answer_start"][0])
        text = ans["text"][0]
        ctx = r["context"]
        if ctx[start : start + len(text)] != text:
            continue

        sents = split_sentences(ctx)
        if not (4 <= len(sents) <= 10):
            continue
        # Map each sentence back to its char span in the context.
        spans, cur = [], 0
        ok = True
        for s in sents:
            j = ctx.find(s, cur)
            if j < 0:
                ok = False
                break
            spans.append((j, j + len(s)))
            cur = j + len(s)
        if not ok:
            continue

        hits = [
            k for k, (a, b) in enumerate(spans) if a <= start and start + len(text) <= b
        ]
        if len(hits) != 1:
            continue
        by_ctx.setdefault(ctx, []).append((r, sents, hits[0]))

    # SQuAD lists a paragraph's questions roughly in paragraph order, so keeping the first
    # qualifying question per context would put the answer in sentence 0 most of the time.
    # Keep one question per context, drawn at random.
    usable = [rng.choice(v) for _, v in sorted(by_ctx.items())]
    rng.shuffle(usable)
    picked = usable[:N]
    assert len(picked) == N, len(picked)

    items = []
    for i, (r, sents, gi) in enumerate(picked):
        sents = [squash(s) for s in sents]
        keys = ["S%d" % k for k in range(len(sents))]
        items.append(
            {
                "id": "squad_sentence-%d" % i,
                "state": {"question": squash(r["question"]), "sentences": sents},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": dict(zip(keys, sents)),
                    }
                },
                "gold": {"q": keys[gi]},
            }
        )

    write_task(
        {
            "id": "squad_sentence",
            "title": "Answer-sentence selection (SQuAD v1.1)",
            "category": "extraction",
            "source": "rajpurkar/squad plain_text validation, 200 sampled seed 7 from the first 3000 rows; contexts split into sentences by code, kept when they give 4-10 sentences and the gold answer span falls inside exactly one; one randomly chosen question per context; gold = the sentence holding the dataset's answer span",
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
