"""Runs one model over the tasks in tasks/.

Usage: python3 run.py <model> [task ids...]        (no ids = every task)
  jev                        TypeSafe jev-1.13.0            needs TYPESAFE_API_KEY
  luna | luna-low            gpt-5.6-luna, reasoning none / low   needs OPENAI_API_KEY
  openai:<model>[:<effort>]  any OpenAI chat model with JSON-schema output, e.g. openai:gpt-5-mini:low

Each answer is appended to results/<model>/<task>.jsonl as it arrives, so an interrupted run resumes where it stopped."""
import concurrent.futures as cf, glob, http.client, json, os, sys, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()


def post(host, path, key, body, tries=6):
    for attempt in range(tries):
        try:
            c = http.client.HTTPSConnection(host, timeout=180)
            t = time.time()
            c.request("POST", path, json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.0"})
            r = c.getresponse()
            raw = r.read()
            wall = round((time.time() - t) * 1000)
            if r.status == 200:
                return json.loads(raw), wall, int(r.getheader("x-envoy-upstream-service-time") or r.getheader("openai-processing-ms") or 0)
            if r.status in (429, 500, 502, 503, 504, 529):
                time.sleep(min(30, 2 ** attempt + float(r.getheader("retry-after") or 0)))
                continue
            raise RuntimeError(f"{r.status} {raw[:400]!r}")
        except (OSError, http.client.HTTPException) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("retries exhausted")


def run_jev(state, questions):
    d, wall, srv = post("api.typesafe.ai", "/v1/systemone", os.environ["TYPESAFE_API_KEY"], {"model": "jev-1.13.0", "state": state, "questions": questions})
    preds = {}
    for qid, a in d["answers"].items():
        if a["type"] == "noul":
            preds[qid] = {"p": a["noul"]}
        elif a["type"] == "choice":
            preds[qid] = {"choice": a["choice"], "conf": max(a["probabilities"].values()), "vendor_conf": a["confidence"]}
        else:
            preds[qid] = {"score": a["score"], "vendor_conf": a["confidence"]}
    return preds, wall, srv, d["usage"]["input_tokens"], d["usage"].get("output_tokens", 0)


def render(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def run_openai(state, questions, model, effort):
    props, lines = {}, []
    for qid, q in questions.items():
        lines.append(f"### {qid} ({q['type']})\nInstructions: {render(q['instructions'])}")
        c = q.get("criteria")
        if q["type"] == "noul":
            if c:
                lines.append(f"Yes means: {c.get('true', '')}\nNo means: {c.get('false', '')}")
            props[qid] = {"type": "object", "additionalProperties": False, "required": ["p_yes"],
                          "properties": {"p_yes": {"type": "number", "description": "probability from 0 to 1 that the answer is yes"}}}
        elif q["type"] == "choice":
            lines.append("Options:\n" + "\n".join(f"- {k}" + (f": {render(d)}" if d else "") for k, d in c.items()))
            props[qid] = {"type": "object", "additionalProperties": False, "required": ["choice", "confidence"],
                          "properties": {"choice": {"type": "string", "enum": list(c)}, "confidence": {"type": "number", "description": "probability from 0 to 1 that this choice is correct"}}}
        else:
            lines.append("Levels (index: description):\n" + "\n".join(f"- {i}: {render(d)}" for i, d in enumerate(c)))
            props[qid] = {"type": "object", "additionalProperties": False, "required": ["level"],
                          "properties": {"level": {"type": "number", "description": f"position on the level scale from 0 to {len(c) - 1}; decimals allowed when between levels"}}}
    system = ("You answer typed questions about the STATE given by the user. Backticked names in instructions refer to fields of the STATE. "
              "Answer every question. Return only the JSON object.\n\n" + "\n\n".join(lines))
    body = {"model": model, **({"reasoning_effort": effort} if effort else {}),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": "STATE:\n" + render(state)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "strict": True,
                                "schema": {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}}
    d, wall, srv = post("api.openai.com", "/v1/chat/completions", os.environ["OPENAI_API_KEY"], body)
    out = json.loads(d["choices"][0]["message"]["content"])
    preds = {}
    for qid, q in questions.items():
        a = out[qid]
        if q["type"] == "noul":
            preds[qid] = {"p": min(1.0, max(0.0, a["p_yes"]))}
        elif q["type"] == "choice":
            preds[qid] = {"choice": a["choice"], "conf": min(1.0, max(0.0, a["confidence"]))}
        else:
            preds[qid] = {"score": a["level"]}
    u = d["usage"]
    return preds, wall, srv, u["prompt_tokens"], u["completion_tokens"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model = sys.argv[1]
    wanted = set(sys.argv[2:])
    if model == "jev":
        fn = run_jev
    elif model in ("luna", "luna-low"):
        fn = lambda s, q: run_openai(s, q, "gpt-5.6-luna", "none" if model == "luna" else "low")
    elif model.startswith("openai:"):
        _, name, *effort = model.split(":")
        fn = lambda s, q: run_openai(s, q, name, effort[0] if effort else None)
        model = model.replace(":", "_")
    else:
        sys.exit(__doc__)
    if not glob.glob(f"{ROOT}/tasks/*.json"):
        sys.exit("tasks/ is empty. Run: python3 build_tasks.py")
    os.makedirs(f"{ROOT}/results/{model}", exist_ok=True)
    for path in sorted(glob.glob(f"{ROOT}/tasks/*.json")):
        task = json.load(open(path))
        if wanted and task["id"] not in wanted:
            continue
        out_path = f"{ROOT}/results/{model}/{task['id']}.jsonl"
        done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
        todo = [it for it in task["items"] if it["id"] not in done]
        if not todo:
            continue
        big = len(json.dumps(todo[0], ensure_ascii=False)) > 20000
        workers = 3 if big else 8
        errors = 0
        t0 = time.time()

        def work(it):
            questions = it.get("questions") or {"q": task["question"]}
            try:
                preds, wall, srv, tin, tout = fn(it["state"], questions)
                rec = {"id": it["id"], "preds": preds, "wall_ms": wall, "server_ms": srv, "in_tok": tin, "out_tok": tout}
            except Exception as e:  # recorded, retried on the next run
                return str(e)[:300]
            with LOCK:
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return None

        with cf.ThreadPoolExecutor(workers) as ex:
            for err in ex.map(work, todo):
                if err:
                    errors += 1
                    if errors <= 2:
                        print(f"   error: {err}")
        print(f"{model} {task['id']}: {len(todo) - errors}/{len(todo)} done, {errors} errors, {time.time() - t0:.0f}s", flush=True)


main()
