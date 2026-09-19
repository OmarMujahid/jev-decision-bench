import json, sys
CATS = {"classification","understanding","reasoning","knowledge","scoring","ranking","extraction","multilingual","long_context","robustness","known_weakness"}
def check_q(q, where):
    assert q.get("type") in ("noul","choice","score"), f"{where}: bad type"
    assert q.get("instructions"), f"{where}: no instructions"
    c = q.get("criteria")
    if q["type"] == "choice": assert isinstance(c, dict) and len(c) >= 2, f"{where}: choice criteria must be a map of >=2"
    if q["type"] == "score": assert isinstance(c, list) and len(c) >= 2, f"{where}: score criteria must be a list of >=2"
    if q["type"] == "noul" and c is not None: assert set(c) <= {"true","false"}, f"{where}: noul criteria keys"
def check_gold(q, g, where):
    if q["type"] == "noul": assert isinstance(g, bool), f"{where}: noul gold must be bool"
    if q["type"] == "choice": assert g in q["criteria"], f"{where}: gold {g!r} not an option"
    if q["type"] == "score": assert isinstance(g, (int, float)) and not isinstance(g, bool) and 0 <= g <= len(q["criteria"]) - 1 or q.get("_ranking"), f"{where}: score gold out of range"
for path in sys.argv[1:]:
    t = json.load(open(path))
    for k in ("id","title","category","source","primitive","metric","items"): assert k in t, f"missing {k}"
    assert t["category"] in CATS, "bad category"
    assert t["metric"] in ("accuracy","auroc","spearman","mrr","ndcg10")
    if "question" in t: check_q(t["question"], "task.question")
    ids = set()
    for it in t["items"]:
        assert it["id"] not in ids, f"dup id {it['id']}"; ids.add(it["id"])
        assert isinstance(it["state"], (dict, list, str))
        if "questions" in it:
            assert isinstance(it["gold"], dict) and set(it["gold"]) == set(it["questions"]), f"{it['id']}: gold keys != question keys"
            for qid, q in it["questions"].items():
                check_q(q, f"{it['id']}.{qid}")
                if t["metric"] in ("mrr","ndcg10"): assert isinstance(it["gold"][qid], (int, float))
                else: check_gold(q, it["gold"][qid], f"{it['id']}.{qid}")
        else:
            assert "question" in t, f"{it['id']}: no questions and no task question"
            check_gold(t["question"], it["gold"], it["id"])
        size = len(json.dumps(it, ensure_ascii=False))
        assert size < 90000, f"{it['id']}: item too large ({size} chars)"
    golds = [json.dumps(it["gold"]) for it in t["items"] if "questions" not in it]
    dist = {g: golds.count(g) for g in sorted(set(golds))} if len(set(golds)) < 100 else "continuous"
    print(f"OK {t['id']}: {len(t['items'])} items, {t['primitive']}/{t['metric']}, gold dist {dist}")
