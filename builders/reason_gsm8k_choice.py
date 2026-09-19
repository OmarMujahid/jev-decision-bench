"""gsm8k_choice: 200 GSM8K test problems as 4-way multiple choice.

The gold answer is the number after "####" in the dataset's answer field. The three distractors
are generated in code (percentage perturbation, adjacent-digit swap, double-or-halve), so no
label is authored by hand.
"""

from decimal import Decimal

from reason_common import all_rows, choice_item, new_rng, write_task

INSTRUCTIONS = (
    "`problem` is a grade-school arithmetic word problem. Each option below is a number. "
    "Exactly one option is the correct final answer to `problem`; the other three are wrong "
    "numbers of a similar size. Select the option that is the correct final answer."
)


def parse_gold(answer_field):
    tail = answer_field.split("####")[-1].strip()
    tail = tail.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return Decimal(tail)
    except Exception:
        return None


def fmt(dec, places):
    if places == 0:
        return str(int(dec))
    q = Decimal(1).scaleb(-places)
    return str(dec.quantize(q))


def digit_swap(gold, places, rng):
    s = fmt(gold, places)
    digits = [i for i, c in enumerate(s) if c.isdigit()]
    pairs = [
        (digits[i], digits[i + 1])
        for i in range(len(digits) - 1)
        if s[digits[i]] != s[digits[i + 1]]
    ]
    rng.shuffle(pairs)
    for a, b in pairs:
        ch = list(s)
        ch[a], ch[b] = ch[b], ch[a]
        if ch[0] == "0" and len(ch) > 1 and ch[1] != ".":
            continue
        try:
            v = Decimal("".join(ch))
        except Exception:
            continue
        if v != gold:
            return v
    return None


def make_options(gold, places, rng):
    """Return [gold, d1, d2, d3] as formatted strings, all distinct."""
    seen = {fmt(gold, places)}
    out = []

    def add(v):
        if v is None or v < 0 or (v == 0 and gold != 0):
            return False
        s = fmt(v, places)
        if s in seen:
            return False
        seen.add(s)
        out.append(s)
        return True

    # 1. percentage perturbation, gold +/- 1..10%
    order = [(sign, p) for p in range(1, 11) for sign in (1, -1)]
    rng.shuffle(order)
    for sign, p in order:
        v = gold * (Decimal(100 + sign * p) / Decimal(100))
        if add(v.quantize(Decimal(1).scaleb(-places))):
            break

    # 2. adjacent-digit swap
    add(digit_swap(gold, places, rng))

    # 3. double or halve
    for v in rng.sample(
        [gold * 2, (gold / 2).quantize(Decimal(1).scaleb(-places))], 2
    ):
        if add(v):
            break

    # top up with small offsets if any of the above collided
    k = 1
    while len(out) < 3 and k < 400:
        add(gold + Decimal(rng.choice([k, -k])))
        k += 1
    return [fmt(gold, places)] + out[:3]


def main():
    rng = new_rng()
    pool = all_rows("openai/gsm8k", "main", "test")
    usable = []
    for r in pool:
        g = parse_gold(r["answer"])
        if g is not None:
            usable.append((r["question"], g))
    picked = rng.sample(usable, 200)
    items = []
    for k, (question, gold) in enumerate(picked):
        places = -gold.as_tuple().exponent if gold.as_tuple().exponent < 0 else 0
        options = make_options(gold, places, rng)
        assert len(set(options)) == 4, (question, options)
        items.append(
            choice_item(f"gsm8k-{k}", {"problem": question}, options, 0, INSTRUCTIONS, rng)
        )
    write_task(
        {
            "id": "gsm8k_choice",
            "title": "Grade-school math as multiple choice (GSM8K)",
            "category": "known_weakness",
            "source": (
                "openai/gsm8k main test split, 200 sampled seed 7. Gold is the number after "
                "'####' in the dataset's answer field. The three distractors are code-generated "
                "with seed 7 (gold +/- 1-10%, an adjacent-digit swap, and double-or-halve, topped "
                "up with small integer offsets when a candidate collides); all four options are "
                "distinct and share the gold's integer/decimal format. Options shuffled seed 7"
            ),
            "primitive": "choice",
            "metric": "accuracy",
            "items": items,
        }
    )


main()
