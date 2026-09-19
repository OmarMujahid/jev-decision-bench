"""value_selection: pick the right pre-extracted number/date span for a SQuAD question."""

import random
import re

from score_rank_extract_common import rows_many, squash, write_task

N = 200
SEED = 7
MAX_OPTS = 8
CTX_WORDS = 6

MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
PATTERNS = [
    r"%s\s+\d{1,2},\s*\d{4}" % MONTH,          # February 7, 2016
    r"\d{1,2}\s+%s\s+\d{4}" % MONTH,           # 7 February 2016
    r"%s\s+\d{4}" % MONTH,                     # February 2016
    r"%s\s+\d{1,2}" % MONTH,                   # February 7
    r"\d{4}s",                                 # 1990s
    r"\d{1,2}/\d{1,2}/\d{2,4}",
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion))?",
    r"\d[\d,]*(?:\.\d+)?\s?(?:percent|%)",
    r"\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion)",
    r"\d[\d,]*(?:\.\d+)?",
]
SPAN_RE = re.compile("|".join("(?:%s)" % p for p in PATTERNS))

INSTRUCTIONS = (
    "`question` is a question about a document. The options are values that were already pulled "
    "out of that document by a program: each option's description is one number or date from the "
    "document, shown with the few words that surround it there. Exactly one option is the answer "
    "to `question`. Pick that option. Use the words shown around each value to decide what the "
    "value refers to."
)


def spans(text):
    out = []
    for m in SPAN_RE.finditer(text):
        s, e = m.start(), m.end()
        # keep whole words only
        if s > 0 and (text[s - 1].isalnum() or text[s - 1] in "-/"):
            continue
        if e < len(text) and (text[e].isalnum() or text[e] in "/"):
            continue
        out.append((s, e, text[s:e]))
    return out


def with_context(text, s, e):
    left = " ".join(text[:s].split()[-CTX_WORDS:])
    right = " ".join(text[e:].split()[:CTX_WORDS])
    mid = squash(text[s:e])
    return squash(("%s [ %s ] %s" % (left, mid, right)).strip())


def main():
    data = rows_many("rajpurkar/squad", "plain_text", "validation", 3000)
    rng = random.Random(SEED)

    usable, seen = [], set()
    for r in data:
        ans = r["answers"]
        if not ans["text"]:
            continue
        gold_text = ans["text"][0].strip()
        start = int(ans["answer_start"][0])
        ctx = r["context"]
        if ctx[start : start + len(ans["text"][0])] != ans["text"][0]:
            continue

        sp = spans(ctx)
        gold_sp = [x for x in sp if x[2] == gold_text]
        # gold must be one of the extracted spans, and unambiguous in the context
        if len(gold_sp) != 1 or gold_sp[0][0] != start:
            continue
        others = [x for x in sp if x[2] != gold_text]
        # distinct by surface text
        seen_txt, distinct = set(), []
        for x in others:
            if x[2] in seen_txt:
                continue
            seen_txt.add(x[2])
            distinct.append(x)
        if len(distinct) < 3:
            continue
        key = (r["context"], gold_text)
        if key in seen:
            continue
        seen.add(key)
        usable.append((r, ctx, gold_sp[0], distinct))

    rng.shuffle(usable)
    picked = usable[:N]
    assert len(picked) >= 150, len(picked)  # spec allows 150-200

    items = []
    for i, (r, ctx, gold_sp, distinct) in enumerate(picked):
        rng.shuffle(distinct)
        chosen = [gold_sp] + distinct[: MAX_OPTS - 1]
        rng.shuffle(chosen)
        crit, gold_key = {}, None
        for k, (s, e, txt) in enumerate(chosen):
            key = "V%d" % k
            crit[key] = with_context(ctx, s, e)
            if (s, e) == (gold_sp[0], gold_sp[1]):
                gold_key = key
        assert gold_key is not None
        items.append(
            {
                "id": "value_selection-%d" % i,
                "state": {"question": squash(r["question"])},
                "questions": {
                    "q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}
                },
                "gold": {"q": gold_key},
            }
        )

    write_task(
        {
            "id": "value_selection",
            "title": "Value selection from pre-extracted spans (SQuAD v1.1)",
            "category": "extraction",
            "source": "rajpurkar/squad plain_text validation, 155 items (all that qualify) from the first 3000 rows where the gold answer is a number or date-like span (regex), occurs exactly once in the context, and the context holds at least 3 other distinct numeric/date spans; options = up to 8 distinct spans including the gold one, each shown with 6 words of context either side; gold = the key of the dataset's answer span",
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
