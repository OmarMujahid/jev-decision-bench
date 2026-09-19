"""ner_typing: type a marked entity mention as PER / ORG / LOC / MISC (CoNLL-2003)."""

import random

from score_rank_extract_common import rows_many, write_task

N = 200
SEED = 7

# tner/conll2003 label ids (dataset card label2id).
ID2TAG = {
    0: "O",
    1: "B-ORG",
    2: "B-MISC",
    3: "B-PER",
    4: "I-PER",
    5: "B-LOC",
    6: "I-ORG",
    7: "I-MISC",
    8: "I-LOC",
}

TYPES = ["PER", "ORG", "LOC", "MISC"]

CRITERIA = {
    "PER": "A person: a named human being, such as an individual's name, or a first or last name used on its own to refer to a person.",
    "ORG": "An organisation: a company, government body, political party, sports club, team, university, agency or other named group of people acting as one body.",
    "LOC": "A location: a named place on the map, such as a country, city, region, continent, river, mountain or street address.",
    "MISC": "A named thing that is none of the above: nationalities and other adjectives derived from a place (for example a demonym), named events, wars, competitions, titles of works, product names and similar proper names.",
}

INSTRUCTIONS = (
    "`sentence` is a sentence from a Reuters news story, and `mention` is a name that appears in "
    "it. Pick the option that says what kind of thing `mention` refers to in this sentence. "
    "Decide from how `mention` is used in `sentence`: the same name can refer to different kinds "
    "of thing in different sentences. For example a country name used as the name of that "
    "country's national team refers to an organisation, and a country name used as an adjective "
    "for people or things from there is a nationality."
)


def entities(tokens, tags):
    out, cur, typ, start = [], [], None, 0
    for i, (tok, tid) in enumerate(zip(tokens, tags)):
        tag = ID2TAG[int(tid)]
        if tag == "O":
            if cur:
                out.append((typ, start, cur))
            cur, typ = [], None
            continue
        pre, t = tag.split("-", 1)
        if pre == "B" or typ != t or not cur:
            if cur:
                out.append((typ, start, cur))
            cur, typ, start = [tok], t, i
        else:
            cur.append(tok)
    if cur:
        out.append((typ, start, cur))
    return out


def detok(tokens):
    s = ""
    for t in tokens:
        if s and (t in {",", ".", "!", "?", ";", ":", "'s", "''", ")", "%"} or t.startswith("'")):
            s += t
        elif s and s.endswith(("(", "``", "$")):
            s += t
        elif s:
            s += " " + t
        else:
            s = t
    return s


def main():
    data = rows_many("tner/conll2003", "conll2003", "test", 2500)
    rng = random.Random(SEED)

    by_type = {t: [] for t in TYPES}
    for r in data:
        tokens, tags = r["tokens"], r["tags"]
        if not (5 <= len(tokens) <= 60):
            continue
        # Headline-style all-caps sentences give the model no casing signal; skip them.
        letters = [t for t in tokens if t.isalpha()]
        if letters and all(t.isupper() for t in letters):
            continue
        ents = entities(tokens, tags)
        sent = detok(tokens)
        for typ, _, toks in ents:
            if typ not in by_type:
                continue
            mention = detok(toks)
            if len(mention) < 2 or sent.count(mention) != 1:
                continue
            by_type[typ].append({"sentence": sent, "mention": mention, "type": typ})

    for v in by_type.values():
        rng.shuffle(v)

    per = N // len(TYPES)
    picked = []
    for t in TYPES:
        assert len(by_type[t]) >= per, (t, len(by_type[t]))
        picked.extend(by_type[t][:per])
    rng.shuffle(picked)

    items = [
        {
            "id": "ner_typing-%d" % i,
            "state": {"sentence": e["sentence"], "mention": e["mention"]},
            "gold": e["type"],
        }
        for i, e in enumerate(picked)
    ]

    write_task(
        {
            "id": "ner_typing",
            "title": "Entity typing (CoNLL-2003)",
            "category": "extraction",
            "source": "tner/conll2003 test, 200 entity mentions sampled seed 7 (50 each of PER/ORG/LOC/MISC) from sentences of 5-60 tokens where the mention string occurs exactly once; all-caps headline sentences skipped; gold = the dataset's entity type",
            "primitive": "choice",
            "metric": "accuracy",
            "question": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
            },
            "items": items,
        }
    )


if __name__ == "__main__":
    main()
