import random

from classify_common import SEED, balanced, fetch_all, write

rng = random.Random(SEED)
# sms_spam ships a single `train` split only; there is no test/validation split to use.
data = fetch_all("ucirvine/sms_spam", "plain_text", "train")
by = {1: [], 0: []}
for r in data:
    sms = r["sms"].strip()
    if sms:
        by[r["label"]].append(sms)

picked = balanced(rng, by, 100, [1, 0])

write(
    {
        "id": "sms_spam",
        "title": "SMS spam detection",
        "category": "classification",
        "source": "ucirvine/sms_spam plain_text train (the dataset's only split), 200 sampled seed 7, balanced 100 spam / 100 ham",
        "primitive": "noul",
        "metric": "auroc",
        "question": {
            "type": "noul",
            "instructions": (
                "`message` is a text message received on a mobile phone. "
                "`message` is spam: an unsolicited commercial or fraudulent message "
                "sent in bulk by someone the recipient does not know."
            ),
            "criteria": {
                "true": "Spam. Unsolicited bulk content such as advertising, a prize or competition claim, a premium-rate number, a subscription service, a scam, or a link the recipient never asked for.",
                "false": "Not spam. An ordinary personal or business message from someone the recipient knows or has an existing relationship with, including short chat, arrangements, and replies.",
            },
        },
        "items": [
            {"id": f"sms_spam-{i}", "state": {"message": text}, "gold": label == 1}
            for i, (label, text) in enumerate(picked)
        ],
    }
)
