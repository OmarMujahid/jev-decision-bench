"""Scores results/<model>/*.jsonl against tasks/*.json. Writes report.json and RESULTS.md, prints a table.
Usage: python3 metrics.py"""
import glob, json, math, os, random, statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/results/*") if os.path.isdir(d))
# $ per million tokens (input, output), September 2026 list prices. Models not listed are costed at 0.
PRICE = {"jev": (0.042, 0.0), "luna": (0.2, 1.2), "luna-low": (0.2, 1.2)}


def auroc(pairs):  # (score, bool)
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(pairs, key=lambda x: x[0])
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    rsum = sum(ranks[k] for k, (_, y) in enumerate(ranked) if y)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        for k in range(i, j):
            r[order[k]] = (i + j + 1) / 2
        i = j
    return r


def spearman(pairs):
    a, b = rank([p for p, _ in pairs]), rank([g for _, g in pairs])
    ma, mb = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float("nan")


def ece(pairs, bins=10):  # (confidence, correct)
    total, n = 0.0, len(pairs)
    for b in range(bins):
        grp = [(c, y) for c, y in pairs if (b / bins <= c < (b + 1) / bins) or (b == bins - 1 and c == 1.0)]
        if grp:
            total += len(grp) / n * abs(statistics.mean(c for c, _ in grp) - statistics.mean(1.0 if y else 0.0 for _, y in grp))
    return total


def boot(units, fn, n=600):
    rnd = random.Random(7)
    vals = []
    for _ in range(n):
        v = fn([units[rnd.randrange(len(units))] for _ in units])
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"),) * 2


def score_task(task, recs):
    items = {it["id"]: it for it in task["items"]}
    units, extra = [], {}
    for r in recs:
        it = items.get(r["id"])
        if not it:
            continue
        qs = it.get("questions") or {"q": task["question"]}
        gold = it["gold"] if isinstance(it["gold"], dict) and "questions" in it else {"q": it["gold"]}
        units.append((it, qs, gold, r["preds"]))
    metric = task["metric"]

    def primary(us):
        if metric == "accuracy":
            flat = [(p[q], g[q], qs[q]["type"]) for _, qs, g, p in us for q in g]
            return statistics.mean((1.0 if ((x["p"] > 0.5) == gg if t == "noul" else x.get("choice") == gg) else 0.0) for x, gg, t in flat)
        if metric == "auroc":
            return auroc([(p[q]["p"], g[q]) for _, _, g, p in us for q in g])
        if metric == "spearman":
            return spearman([(p[q]["score"], g[q]) for _, _, g, p in us for q in g])
        vals = []
        for _, _, g, p in us:
            order = sorted(g, key=lambda q: -p[q]["score"])
            if metric == "mrr":
                vals.append(next((1 / (i + 1) for i, q in enumerate(order) if g[q] > 0), 0.0))
            else:
                dcg = sum((2 ** g[q] - 1) / math.log2(i + 2) for i, q in enumerate(order[:10]))
                ideal = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(sorted(g.values(), reverse=True)[:10]))
                vals.append(dcg / ideal if ideal else 0.0)
        return statistics.mean(vals)

    if not units:
        return None
    value = primary(units)
    lo, hi = boot(units, primary)
    out = {"n": len(units), "metric": metric, "value": value, "ci": [lo, hi]}
    if task["primitive"] == "noul":
        flat = [(p[q]["p"], g[q]) for _, _, g, p in units for q in g]
        out["accuracy"] = statistics.mean(1.0 if (s > 0.5) == y else 0.0 for s, y in flat)
        out["brier"] = statistics.mean((s - (1.0 if y else 0.0)) ** 2 for s, y in flat)
        out["ece"] = ece([(max(s, 1 - s), (s > 0.5) == y) for s, y in flat])
        conf = sorted(flat, key=lambda x: -abs(x[0] - 0.5))
    elif task["primitive"] == "choice" and metric == "accuracy":
        flat = [(p[q].get("conf", 0.0), p[q].get("choice") == g[q]) for _, _, g, p in units for q in g]
        out["ece"] = ece(flat)
        conf = sorted(flat, key=lambda x: -x[0])
        conf = [(c, ok) for c, ok in conf]
    else:
        conf = None
    if conf:
        for cov in (0.8, 0.5):
            top = conf[: max(1, int(len(conf) * cov))]
            out[f"acc_at_{int(cov * 100)}"] = statistics.mean(
                (1.0 if ((s > 0.5) == y) else 0.0) if task["primitive"] == "noul" else (1.0 if y else 0.0) for s, y in top)
    # special diagnostics
    if task["id"] == "negation_pairs":
        by = {}
        for it, _, _, p in units:
            by.setdefault(str(it.get("meta", {}).get("pair")), []).append(p["q"]["p"])
        sums = [sum(v) for v in by.values() if len(v) == 2]
        out["mean_abs_dev_from_1"] = statistics.mean(abs(s - 1) for s in sums) if sums else None
    if task["id"] == "paraphrase_consistency":
        by = {}
        for it, _, _, p in units:
            by.setdefault(str(it.get("meta", {}).get("base")), []).append(p["q"].get("choice"))
        out["all_variants_agree"] = statistics.mean(1.0 if len(set(v)) == 1 else 0.0 for v in by.values())
    return out


def main():
    report = {"tasks": {}, "models": {}}
    for path in sorted(glob.glob(f"{ROOT}/tasks/*.json")):
        task = json.load(open(path))
        row = {"title": task["title"], "category": task["category"], "primitive": task["primitive"], "metric": task["metric"], "source": task["source"], "n_items": len(task["items"]), "scores": {}}
        for m in MODELS:
            rp = f"{ROOT}/results/{m}/{task['id']}.jsonl"
            if not os.path.exists(rp):
                continue
            recs = [json.loads(l) for l in open(rp)]
            s = score_task(task, recs)
            if not s:
                continue
            s["server_ms_median"] = statistics.median(r["server_ms"] for r in recs)
            s["wall_ms_median"] = statistics.median(r["wall_ms"] for r in recs)
            tin, tout = sum(r["in_tok"] for r in recs), sum(r["out_tok"] for r in recs)
            s["in_tok"], s["out_tok"] = tin, tout
            s["cost_per_1k_items"] = (tin * PRICE.get(m, (0, 0))[0] + tout * PRICE.get(m, (0, 0))[1]) / 1e6 / len(recs) * 1000
            row["scores"][m] = s
            agg = report["models"].setdefault(m, {"in_tok": 0, "out_tok": 0, "items": 0, "cost": 0.0})
            agg["in_tok"] += tin; agg["out_tok"] += tout; agg["items"] += len(recs); agg["cost"] += (tin * PRICE.get(m, (0, 0))[0] + tout * PRICE.get(m, (0, 0))[1]) / 1e6
        report["tasks"][task["id"]] = row
    json.dump(report, open(f"{ROOT}/report.json", "w"), indent=1)
    cat = None
    for tid, row in sorted(report["tasks"].items(), key=lambda kv: (kv[1]["category"], kv[0])):
        if row["category"] != cat:
            cat = row["category"]
            print(f"\n== {cat}")
        cells = []
        for m in MODELS:
            s = row["scores"].get(m)
            cells.append(f"{m} {s['value']:.3f} [{s['ci'][0]:.2f},{s['ci'][1]:.2f}] {s['server_ms_median']:.0f}ms" if s else f"{m} -")
        print(f"{tid:26s} {row['metric']:9s} | " + " | ".join(cells))
    print("\ntotals:", json.dumps(report["models"]))
    lines = ["# Results", "", "Scores run 0 to 1, higher is better. Brackets are 95% bootstrap intervals. `ms` is median server-side time per item.", "",
             "| Category | Task | Metric | n | " + " | ".join(MODELS) + " |", "|---|---|---|---|" + "---|" * len(MODELS)]
    for tid, row in sorted(report["tasks"].items(), key=lambda kv: (kv[1]["category"], kv[0])):
        cells = []
        for m in MODELS:
            s = row["scores"].get(m)
            cells.append(f"{s['value']:.3f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] · {s['server_ms_median']:.0f} ms" if s else "–")
        lines.append(f"| {row['category']} | {tid} | {row['metric']} | {row['n_items']} | " + " | ".join(cells) + " |")
    open(f"{ROOT}/RESULTS.md", "w").write("\n".join(lines) + "\n")


main()
