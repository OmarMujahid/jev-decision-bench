"""Group B - long context: needle_{1k,4k,12k,24k}.

Synthetic needle-in-a-haystack. Filler paragraphs are real SQuAD v1.1 validation
contexts (deduplicated); the needle sentence and the gold label are code-derived.
The same 60 needles / questions / depths are reused at every haystack size, so the
only thing that changes between the four tasks is how much filler surrounds them.
"""

import json
import random

from probes_common import SEED, rows_blocks, write_task

CHARS_PER_TOKEN = 4
# validate.py caps one serialized item at 90k chars, so the 24k-token haystack is
# built to a char budget just under that cap rather than the nominal 96k chars.
SIZES = [("needle_1k", 1000), ("needle_4k", 4000), ("needle_12k", 12000), ("needle_24k", 24000)]
MAX_ITEM_CHARS = 88000

COLORS = ["amber", "crimson", "emerald", "indigo", "ivory", "olive", "scarlet",
          "slate", "teal", "violet"]
OBJECTS = ["anchor", "beacon", "compass", "falcon", "harbor", "kestrel", "lantern",
           "meridian", "obsidian", "quartz"]

QUESTION = {
    "type": "noul",
    "instructions": (
        "`documents` is a long collection of unrelated text passages. Somewhere in it "
        "there is exactly one sentence of the form \"The access code for the <color> "
        "<object> locker is <four-digit code>.\" Does `documents` state that the access "
        "code for the `color` `object` locker is `code`?"
    ),
    "criteria": {
        "true": "`documents` contains a sentence saying the access code for the `color` `object` locker is exactly the four digits in `code`.",
        "false": "`documents` states a different access code for the `color` `object` locker, or states no access code for it at all.",
    },
}


def filler_paragraphs():
    rs = rows_blocks("rajpurkar/squad", "plain_text", "validation", range(0, 4000, 100))
    seen, out = set(), []
    for r in rs:
        c = " ".join(r["context"].split())
        if len(c) < 300 or c in seen:
            continue
        seen.add(c)
        out.append(c)
    assert len(out) > 400, len(out)
    return out


def make_needles():
    rng = random.Random(SEED)
    pairs = [(c, o) for c in COLORS for o in OBJECTS]
    rng.shuffle(pairs)
    needles = []
    for i in range(60):
        color, obj = pairs[i]
        true_code = f"{rng.randint(1000, 9999)}"
        ask_true = i % 2 == 0
        if ask_true:
            asked = true_code
        else:
            asked = true_code
            while asked == true_code:
                asked = f"{rng.randint(1000, 9999)}"
        needles.append({
            "i": i,
            "color": color,
            "object": obj,
            "true_code": true_code,
            "asked_code": asked,
            "gold": ask_true,
            "depth": round(rng.uniform(0.03, 0.97), 3),
            "seed": rng.randrange(10**9),
        })
    return needles


def build_haystack(pool, needle, target_chars, budget_chars):
    rng = random.Random(needle["seed"])
    order = list(range(len(pool)))
    rng.shuffle(order)
    paras, total, k = [], 0, 0
    while total < target_chars:
        p = pool[order[k % len(order)]]
        k += 1
        if total + len(p) + 2 > budget_chars:
            break
        paras.append(p)
        total += len(p) + 2
    sentence = (f"The access code for the {needle['color']} {needle['object']} locker "
                f"is {needle['true_code']}.")
    pos = min(len(paras), max(0, round(needle["depth"] * len(paras))))
    paras.insert(pos, sentence)
    return "\n\n".join(paras), (pos / max(1, len(paras) - 1))


def build():
    pool = filler_paragraphs()
    needles = make_needles()
    for task_id, tokens in SIZES:
        target = tokens * CHARS_PER_TOKEN
        items = []
        for n in needles:
            budget = MAX_ITEM_CHARS
            while True:
                docs, actual_depth = build_haystack(pool, n, target, budget)
                item = {
                    "id": f"{task_id}-{n['i']}",
                    "state": {
                        "documents": docs,
                        "color": n["color"],
                        "object": n["object"],
                        "code": n["asked_code"],
                    },
                    "gold": n["gold"],
                    "meta": {
                        "needle": n["i"],
                        "depth_fraction": round(actual_depth, 3),
                        "requested_depth": n["depth"],
                        "target_tokens": tokens,
                        "approx_tokens": round(len(docs) / CHARS_PER_TOKEN),
                    },
                }
                if len(json.dumps(item, ensure_ascii=False)) < 89000:
                    break
                budget -= 4000
            items.append(item)
        write_task({
            "id": task_id,
            "title": f"Needle in a haystack (~{tokens // 1000}k tokens)",
            "category": "long_context",
            "source": (
                "synthetic; filler paragraphs are deduplicated rajpurkar/squad plain_text "
                "validation contexts, needle sentence and gold label generated by code "
                "(seed 7). Same 60 needles, questions and depths across needle_1k/4k/12k/24k; "
                f"haystack sized to ~{tokens} tokens at 4 chars/token"
                + (" (capped at ~88k chars to stay under validate.py's 90k per-item limit, "
                   "so the effective size is ~22k tokens)" if tokens == 24000 else "")
            ),
            "primitive": "noul",
            "metric": "accuracy",
            "question": QUESTION,
            "items": items,
        })


if __name__ == "__main__":
    build()
