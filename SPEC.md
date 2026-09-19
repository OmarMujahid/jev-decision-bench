# jevbench task file spec

A benchmark for *decision models* (typed question in → probability/choice/score out), run against TypeSafe Jev and an LLM baseline with identical wording.
One JSON file per task in `tasks/<id>.json`. Pure data — the runner owns all model calls. Python stdlib only (no pandas/numpy/datasets installed).

## Getting data
Public, established, human-labelled datasets only (gold labels must come from the dataset, never from you), fetched via the Hugging Face datasets-server, e.g.
`https://datasets-server.huggingface.co/rows?dataset=nyu-mll/glue&config=sst2&split=validation&offset=0&length=100` (max length 100 per call; `/splits?dataset=...` lists configs/splits; `/info?dataset=` gives label names). Use test/validation splits. Sample with `random.Random(7)` and BALANCE classes where the task is classification. If a dataset is gated/unavailable, pick an equivalent well-known one and say so in `source`.
Synthetic tasks are allowed only where the label is computed by code (counting, arithmetic, dates, needle-in-haystack, injected instructions).

## File format
```json
{
  "id": "sst2", "title": "Sentiment (SST-2)",
  "category": "classification | understanding | reasoning | knowledge | scoring | ranking | extraction | multilingual | long_context | robustness | known_weakness",
  "source": "nyu-mll/glue sst2 validation, 200 sampled seed 7, balanced",
  "primitive": "noul | choice | score",
  "metric": "accuracy | auroc | spearman | mrr | ndcg10",
  "question": { "type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."} },
  "items": [ { "id": "sst2-0", "state": {"review": "..."}, "gold": true } ]
}
```
- `question` (task level) is asked of every item under question id `q`; then item `gold` is a scalar.
- If questions differ per item (multiple-choice options, one question per ranking candidate), omit the task-level `question` and give each item `"questions": {qid: question}` and `"gold": {qid: gold}`.
- Question shapes (TypeSafe API): noul → `criteria` optional `{"true","false"}`; choice → `criteria` = map option→description-or-null (option keys are the labels; keep keys short ascii like "A","B" or "entailment"); score → `criteria` = ordered array of ≥2 level descriptions, lowest first.
- Gold: noul → true/false. choice → the option key. score → a number on the level-index scale (0 = first level); floats allowed (e.g. STS-B 0–5 with 6 levels). ranking → per-candidate score questions with gold relevance number per qid; metric mrr/ndcg10 is computed across the item's questions.
- `state`: a JSON object with named fields; instructions reference fields in backticks, e.g. `premise`. Keep state to what the question needs.
- Write instructions the way a careful engineer would after reading https://docs.typesafe.ai/primitives.md and https://docs.typesafe.ai/model-jaggedness/jev-1.13.md (literal, explicit, criteria aligned with the instruction). No per-dataset prompt tuning against results — you never call any model. The same wording goes to both models.
- Sizes: 200 items per task (ranking: 100 queries × their candidates; long_context: 60). Each item's state+questions must stay under ~25k tokens.

## Validate
`python3 validate.py tasks/<id>.json` must print OK.
