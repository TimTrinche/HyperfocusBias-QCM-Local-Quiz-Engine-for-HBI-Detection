
import os, json, uuid, math, random, datetime, csv, pathlib
from typing import List, Dict, Any
from flask import Flask, jsonify, request, render_template, send_from_directory
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = pathlib.Path(__file__).parent.resolve()
EXPORT_DIR = BASE_DIR / "exports" / "sessions"
ANALYTICS_DIR = BASE_DIR / "exports" / "analytics"
ITEMS_PATH = BASE_DIR / "items_bank.json"
HBI_PATH = BASE_DIR / "HBI_QCM_bank_FULL_balanced.json" 

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_POOL = [m.strip() for m in os.getenv("LLM_POOL", "qwen:latest,mistral,qwen2:latest,llama3:latest").split(",")]
LLM_SPEED_MODE = os.getenv("LLM_SPEED_MODE", "balanced")  
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "200"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "1536"))

app = Flask(__name__)

RUNTIME = {"sessions": {}}

def now_iso():
    return datetime.datetime.utcnow().isoformat()

def load_bank() -> List[Dict[str, Any]]:
    if HBI_PATH.exists():
        try:
            data = json.loads(HBI_PATH.read_text())
            if isinstance(data, list) and data and "question" in data[0]:
                return data
        except Exception:
            pass
    return json.loads(ITEMS_PATH.read_text())

def stratified_sample(items: List[Dict[str, Any]], n=12) -> List[Dict[str, Any]]:
    '''Distribute difficulties (3PL b) across bins to avoid monotonic order.'''
    if not items:
        return []
    with_b = [it for it in items if "b" in it]
    base = with_b if len(with_b) >= n else items[:]
    bs = [it.get("b", 0.0) for it in base]
    q1, q2 = pd.Series(bs).quantile([0.33, 0.66]).tolist()
    low = [it for it in base if it.get("b", 0.0) <= q1]
    mid = [it for it in base if q1 < it.get("b", 0.0) <= q2]
    high = [it for it in base if it.get("b", 0.0) > q2]
    random.shuffle(low); random.shuffle(mid); random.shuffle(high)
    out = []
    while len(out) < n and any([low, mid, high]):
        for bucket in (low, mid, high):
            if bucket and len(out) < n:
                out.append(bucket.pop())
    random.shuffle(out)
    for it in out:
        if "choices" in it and isinstance(it["choices"], list) and "answer_index" in it:
            k = random.randrange(len(it["choices"]))
            it["choices"] = it["choices"][k:] + it["choices"][:k]
            it["answer_index"] = (it["answer_index"] - k) % len(it["choices"])
    return out

def three_pl_p(a, b, c, theta):
    return c + (1 - c) / (1 + math.exp(-1.7 * a * (theta - b)))

def estimate_theta_grid(responses):
    grid = [x/10 for x in range(-30, 31)]
    best_t, best_ll = 0.0, -1e9
    for t in grid:
        ll = 0.0
        for r, a, b, c in responses:
            p = three_pl_p(a, b, c, t)
            p = min(max(p,1e-6), 1-1e-6)
            ll += math.log(p if r else (1-p))
        if ll > best_ll:
            best_ll, best_t = ll, t
    return best_t

def ensure_csv_headers(path, headers):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def update_aggregates():
    all_path = ANALYTICS_DIR / "all_sessions.csv"
    if not all_path.exists():
        return
    df = pd.read_csv(all_path)
    out = {}
    if not df.empty:
        out["avg_by_level"] = df.groupby("study_level")["final_score"].mean().round(3).to_dict()
        out["avg_by_sector"] = df.groupby("sector")["final_score"].mean().round(3).to_dict()
        if {"sector","qcm_type"}.issubset(df.columns):
            ct = pd.crosstab(df["sector"], df["qcm_type"])
            try:
                from scipy.stats import chi2_contingency as _chi2
                chi2, p, dof, exp = _chi2(ct.values)
                n = ct.values.sum()
                r, k = ct.shape
                v = (chi2 / (n * (min(r-1, k-1) if min(r-1, k-1)>0 else 1))) ** 0.5
                out["cramers_v_sector_theme"] = round(float(v), 4)
            except Exception:
                out["cramers_v_sector_theme"] = None
    (ANALYTICS_DIR / "aggregates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/llm_ping")
def api_llm_ping():
    have = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5); r.raise_for_status()
        have = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    pool = [m for m in LLM_POOL if m in have]
    return jsonify({"ok": bool(pool), "base_url": OLLAMA_BASE_URL, "have": have, "pool": LLM_POOL, "speed_mode": LLM_SPEED_MODE})

@app.post("/api/consent")
def api_consent():
    j = request.get_json(force=True)
    consent = bool(j.get("consent")); session_id = j.get("session_id") or uuid.uuid4().hex[:12]; version = "v1"
    cons_path = ANALYTICS_DIR / "consents.csv"
    ensure_csv_headers(cons_path, ["ts","session_id","consent","version"])
    with cons_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([now_iso(), session_id, int(consent), version])
    return jsonify({"ok": True})

@app.post("/api/start_session")
def api_start_session():
    j = request.get_json(force=True)
    session_id = uuid.uuid4().hex[:12]
    qcm_type = j.get("qcm_type"); study_level = j.get("study_level"); sector = j.get("sector")
    llm_allowed = random.random() < 0.5
    have = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5); r.raise_for_status()
        have = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    pool = [m for m in LLM_POOL if m in have]
    size_order = {"qwen:latest":1,"qwen2:latest":2,"mistral":2,"llama3:latest":3}
    model = (sorted(pool, key=lambda m: size_order.get(m,9))[0] if (llm_allowed and pool) else None)
    RUNTIME["sessions"][session_id] = {
        "session_id": session_id, "qcm_type": qcm_type, "study_level": study_level, "sector": sector,
        "llm_allowed": llm_allowed, "llm_model": model, "started_at": now_iso(),
        "predicted": None, "items": [], "responses": {}
    }
    (EXPORT_DIR / f"session_{session_id}.jsonl").write_text("")
    return jsonify({"ok": True, "session_id": session_id, "llm_allowed": llm_allowed, "llm_model": model})

@app.post("/api/predict")
def api_predict():
    j = request.get_json(force=True)
    sid = j["session_id"]; pred = float(j["predicted"])
    RUNTIME["sessions"][sid]["predicted"] = pred
    return jsonify({"ok": True})

@app.get("/api/items")
def api_items():
    sid = request.args.get("session_id"); qcm_type = request.args.get("qcm_type")
    study_level = request.args.get("study_level"); sector = request.args.get("sector")
    bank = load_bank()
    cand = [it for it in bank if it.get("theme")==qcm_type and it.get("level")==study_level and it.get("sector")==sector]
    if len(cand) < 12: cand = [it for it in bank if it.get("theme")==qcm_type and it.get("level")==study_level]
    if len(cand) < 12: cand = [it for it in bank if it.get("theme")==qcm_type]
    items = stratified_sample(cand, n=12)
    RUNTIME["sessions"][sid]["items"] = [it["id"] for it in items]
    for idx, it in enumerate(items): it["order_index"] = idx
    return jsonify({"ok": True, "items": items})

@app.post("/api/record_response")
def api_record_response():
    j = request.get_json(force=True)
    sid = j["session_id"]; qid = j["question_id"]
    RUNTIME["sessions"][sid]["responses"][qid] = {
        "order_index": j["order_index"], "selected_index": j["selected_index"],
        "is_correct": bool(j["is_correct"]), "time_spent_ms": int(j.get("time_spent_ms",0)),
        "revisits": int(j.get("revisits",0)), "switches": int(j.get("switches",0)), "ts": now_iso()
    }
    return jsonify({"ok": True})

@app.post("/api/llm_chat")
def api_llm_chat():
    j = request.get_json(force=True)
    sid = j["session_id"]; prompt = j["prompt"]
    sess = RUNTIME["sessions"].get(sid, {})
    if not sess.get("llm_allowed") or not sess.get("llm_model"):
        return jsonify({"ok": False, "error": "LLM indisponible pour cette session."}), 403
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": sess["llm_model"], "prompt": prompt,
            "options": {"num_predict": LLM_NUM_PREDICT, "num_ctx": LLM_NUM_CTX, "temperature": 0.2},
            "stream": False
        }, timeout=60)
        r.raise_for_status()
        text = r.json().get("response","").strip()
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Erreur LLM: {e}"}), 500

@app.post("/api/end_session")
def api_end_session():
    j = request.get_json(force=True); sid = j["session_id"]
    sess = RUNTIME["sessions"].get(sid)
    if not sess: return jsonify({"ok": False, "error":"session inconnue"}), 404
    bank = load_bank(); bank_map = {it["id"]:it for it in bank}
    rows = []; corrects = 0; resp_list = []
    for qid in sess["items"]:
        it = bank_map[qid]; r = sess["responses"].get(qid, {})
        corr = bool(r.get("is_correct", False)); corrects += int(corr)
        rows.append({
            "session_id": sid, "ts": now_iso(), "qcm_type": sess["qcm_type"],
            "study_level": sess["study_level"], "sector": sess["sector"],
            "llm_allowed": int(sess["llm_allowed"]), "llm_model": sess["llm_model"] or "",
            "question_id": qid, "order_index": r.get("order_index"),
            "selected_index": r.get("selected_index"), "answer_index": it.get("answer_index"),
            "is_correct": int(corr), "time_spent_ms": r.get("time_spent_ms",0),
            "revisits": r.get("revisits",0), "switches": r.get("switches",0),
            "a": it.get("a",1.0), "b": it.get("b",0.0), "c": it.get("c",0.2)
        })
        resp_list.append((corr, it.get("a",1.0), it.get("b",0.0), it.get("c",0.2)))
    final_score = corrects
    theta_hat = estimate_theta_grid(resp_list) if resp_list else 0.0
    diff_rows = []
    for qid in sess["items"]:
        it = bank_map[qid]; a,b,c = it.get("a",1.0), it.get("b",0.0), it.get("c",0.2)
        diff_rows.append({"question_id": qid, "a":a,"b":b,"c":c, "theta_hat": theta_hat, "p_correct_theta": three_pl_p(a,b,c,theta_hat)})
    pd.DataFrame(rows).to_csv(EXPORT_DIR / f"session_{sid}.csv", index=False)
    (EXPORT_DIR / f"session_{sid}_difficulty.json").write_text(json.dumps(diff_rows, ensure_ascii=False, indent=2))
    all_path = ANALYTICS_DIR / "all_sessions.csv"
    ensure_csv_headers(all_path, ["session_id","qcm_type","study_level","sector","llm_allowed","llm_model","predicted_score","final_score"])
    with all_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([sid, sess["qcm_type"], sess["study_level"], sess["sector"],
                                int(sess["llm_allowed"]), sess["llm_model"] or "", sess.get("predicted", None), final_score])
    try: update_aggregates()
    except Exception: pass
    return jsonify({"ok": True, "final_score": int(final_score), "theta_hat": theta_hat, "items_modeled": len(diff_rows)})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR / "static", filename)

if __name__ == "__main__":
    app.run(debug=True)
