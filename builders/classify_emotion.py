import random

from classify_common import SEED, balanced, fetch_all, write

rng = random.Random(SEED)

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion ids 0..5

data = fetch_all("dair-ai/emotion", "split", "test")
by = {c: [] for c in LABELS}
for r in data:
    if r["text"].strip():
        by[LABELS[r["label"]]].append(r["text"].strip())

per = min(33, min(len(v) for v in by.values()))
picked = balanced(rng, by, per, LABELS)
# top up to 200 from the classes that still have spare rows, largest pools first
used = {id(x) for _, x in picked}
spare = []
for c in LABELS:
    for t in by[c]:
        if id(t) not in used:
            spare.append((c, t))
rng.shuffle(spare)
picked.extend(spare[: 200 - len(picked)])
rng.shuffle(picked)

write(
    {
        "id": "emotion",
        "title": "Emotion in a tweet (dair-ai/emotion)",
        "category": "classification",
        "source": (
            "dair-ai/emotion split=test, 200 sampled seed 7, "
            f"{per} per emotion then topped up to 200 at random from the remaining rows"
        ),
        "primitive": "choice",
        "metric": "accuracy",
        "question": {
            "type": "choice",
            "instructions": (
                "`message` is a short post in which someone describes how they feel. "
                "Which one of these six emotions is the writer of `message` expressing? "
                "Pick the single emotion that the message is mainly about."
            ),
            "criteria": {
                "sadness": "Sadness: feeling unhappy, low, hurt, lonely, discouraged, or sorry.",
                "joy": "Joy: feeling happy, pleased, content, grateful, excited in a good way, or proud.",
                "love": "Love: feeling affection, tenderness, caring, or romantic attachment towards someone.",
                "anger": "Anger: feeling annoyed, irritated, resentful, furious, or bitter.",
                "fear": "Fear: feeling afraid, anxious, nervous, worried, threatened, or overwhelmed.",
                "surprise": "Surprise: feeling startled, amazed, shocked, or caught off guard by something unexpected.",
            },
        },
        "items": [
            {"id": f"emotion-{i}", "state": {"message": text}, "gold": cls}
            for i, (cls, text) in enumerate(picked)
        ],
    }
)
