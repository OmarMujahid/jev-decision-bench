"""Group A - multilingual tasks: xnli_{en,ar,fr,zh,hi,sw}, arabic_sentiment, lang_id."""

import random

from probes_common import SEED, rows_blocks, total_rows, write_task

XNLI_LANGS = [("en", "English"), ("ar", "Arabic"), ("fr", "French"),
              ("zh", "Chinese"), ("hi", "Hindi"), ("sw", "Swahili")]
XNLI_LABELS = ["entailment", "neutral", "contradiction"]

# Identical English wording in every language - only the data changes.
XNLI_QUESTION = {
    "type": "choice",
    "instructions": (
        "`premise` and `hypothesis` are two sentences written in the same language. "
        "Read them literally and decide the relationship of `hypothesis` to `premise`. "
        "Answer with exactly one option: entailment, neutral, or contradiction."
    ),
    "criteria": {
        "entailment": "If `premise` is true, then `hypothesis` must also be true.",
        "neutral": "`hypothesis` may be true or false when `premise` is true; `premise` neither guarantees nor rules it out.",
        "contradiction": "If `premise` is true, then `hypothesis` must be false.",
    },
}


def xnli_indices():
    """Pick the same 150 parallel row indices (50 per label) used in every language."""
    rng = random.Random(SEED)
    n = total_rows("facebook/xnli", "en", "test")
    block_offsets = sorted(rng.sample(range(0, n - n % 100, 100), 15))
    en = rows_blocks("facebook/xnli", "en", "test", block_offsets)
    # flat index -> (block offset, position) so the same rows can be re-derived per language
    by_label = {0: [], 1: [], 2: []}
    for i, r in enumerate(en):
        by_label[r["label"]].append(i)
    picked = []
    for lab in (0, 1, 2):
        picked.extend(sorted(random.Random(SEED + lab).sample(by_label[lab], 50)))
    picked.sort()
    return block_offsets, picked


def build_xnli(block_offsets, picked):
    for code, name in XNLI_LANGS:
        rs = rows_blocks("facebook/xnli", code, "test", block_offsets)
        items = []
        for i in picked:
            r = rs[i]
            items.append({
                "id": f"xnli_{code}-{i}",
                "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                "gold": XNLI_LABELS[r["label"]],
                "meta": {"row": i, "language": code},
            })
        write_task({
            "id": f"xnli_{code}",
            "title": f"Natural language inference - XNLI ({name})",
            "category": "multilingual",
            "source": (
                f"facebook/xnli config {code} split test; 15 random 100-row blocks (seed 7), "
                "150 rows balanced 50/50/50 over the three labels; XNLI is parallel so the "
                "same 150 row indices are used in all six language tasks"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "question": XNLI_QUESTION,
            "items": items,
        })


def build_arabic_sentiment():
    rs = rows_blocks("komari6/ajgt_twitter_ar", "plain_text", "train",
                     range(0, 1800, 100))
    pos = [(i, r) for i, r in enumerate(rs) if r["label"] == 1]
    neg = [(i, r) for i, r in enumerate(rs) if r["label"] == 0]
    rng = random.Random(SEED)
    sel = rng.sample(pos, 100) + rng.sample(neg, 100)
    sel.sort(key=lambda p: p[0])
    items = [{
        "id": f"arabic_sentiment-{i}",
        "state": {"tweet": r["text"].strip()},
        "gold": r["label"] == 1,
    } for i, r in sel]
    write_task({
        "id": "arabic_sentiment",
        "title": "Arabic tweet sentiment (AJGT)",
        "category": "multilingual",
        "source": (
            "komari6/ajgt_twitter_ar (Arabic Jordanian General Tweets, human-labelled) "
            "config plain_text split train - the dataset ships only one split; "
            "200 sampled seed 7, balanced 100 positive / 100 negative"
        ),
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`tweet` is a tweet written in Arabic. Does `tweet` express a positive "
                "sentiment or attitude?"
            ),
            "criteria": {
                "true": "The tweet expresses approval, praise, satisfaction, happiness or another positive attitude.",
                "false": "The tweet expresses criticism, complaint, dissatisfaction, anger or another negative attitude.",
            },
        },
        "items": items,
    })


LANGS20 = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
    "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


def build_lang_id():
    rng = random.Random(SEED)
    offsets = sorted(rng.sample(range(0, 10000, 100), 30))
    rs = rows_blocks("papluca/language-identification", "default", "test", offsets)
    by_lang = {}
    for i, r in enumerate(rs):
        by_lang.setdefault(r["labels"], []).append((i, r))
    assert set(by_lang) == set(LANGS20), sorted(set(by_lang) ^ set(LANGS20))
    sel = []
    for lang in sorted(LANGS20):
        sel.extend(rng.sample(by_lang[lang], 10))
    sel.sort(key=lambda p: p[0])
    items = [{
        "id": f"lang_id-{i}",
        "state": {"text": r["text"]},
        "gold": r["labels"],
    } for i, r in sel]
    write_task({
        "id": "lang_id",
        "title": "Language identification (20 languages)",
        "category": "classification",
        "source": (
            "papluca/language-identification split test; 30 random 100-row blocks (seed 7), "
            "200 sampled seed 7, balanced 10 per language over all 20 languages"
        ),
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "In which language is `text` written? Answer with the option for that "
                "language, using only `text` itself."
            ),
            "criteria": {k: f"`text` is written in {v}." for k, v in sorted(LANGS20.items())},
        },
        "items": items,
    })


if __name__ == "__main__":
    offs, picked = xnli_indices()
    build_xnli(offs, picked)
    build_arabic_sentiment()
    build_lang_id()
