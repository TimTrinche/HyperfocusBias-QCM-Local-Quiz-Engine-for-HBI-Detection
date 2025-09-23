
import os, json, uuid, time, random, datetime, csv
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory, render_template
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE/".env")

app = Flask(__name__, static_folder=str(BASE/"static"), template_folder=str(BASE/"templates"))
app.config["JSON_AS_ASCII"] = False

# ----------- ENV -----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_POOL = [s.strip() for s in os.getenv("LLM_POOL","qwen2:latest,mistral:latest,llama3:latest").split(",") if s.strip()]
LLM_SPEED_MODE = os.getenv("LLM_SPEED_MODE","balanced")
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT","180"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX","1024"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT","45"))

EXPORTS_DIR = BASE/"exports"
SESSION_DIR = EXPORTS_DIR/"sessions"
ANALYTICS_DIR = EXPORTS_DIR/"analytics"
for p in [EXPORTS_DIR, SESSION_DIR, ANALYTICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ----------- BANK LOADER -----------
BANK_FILE = BASE/"HBI_QCM_bank_FULL_balanced.json"

def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _normalize_item(it: Dict[str,Any]) -> Dict[str,Any]:
    # question
    q = it.get("question") or it.get("stem") or it.get("prompt") or ""
    # options
    options = it.get("options") or it.get("choices") or it.get("answers") or []
    if isinstance(options, dict):
        ksorted = sorted(options.keys())
        options = [options[k] for k in ksorted]
    # answer -> index
    ans = it.get("answer") or it.get("correct") or it.get("correct_answer") or it.get("solution")
    answer_index = None
    if isinstance(ans, (int, float)):
        idx = int(ans)
        if 0 <= idx < len(options):
            answer_index = idx
    elif isinstance(ans, str):
        s = ans.strip()
        letters = {"A":0,"B":1,"C":2,"D":3,"E":4}
        if s in letters and letters[s] < len(options):
            answer_index = letters[s]
        else:
            for i,opt in enumerate(options):
                if str(opt).strip().lower() == s.lower():
                    answer_index = i; break
    if answer_index is None and options:
        answer_index = 0

    theme = it.get("theme") or it.get("qcm_type") or it.get("topic") or it.get("domain") or "général"
    level = it.get("level") or it.get("study_level") or it.get("niveau") or "Licence"
    sector = it.get("sector") or it.get("secteur") or it.get("field") or "général"

    def _f(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    return {
        "id": it.get("id") or f"item-{uuid.uuid4().hex[:8]}",
        "question": str(q),
        "options": [str(x) for x in options],
        "answer_index": answer_index,
        "theme": str(theme),
        "level": str(level),
        "sector": str(sector),
        "a": _f(it.get("a")),
        "b": _f(it.get("b")),
        "c": _f(it.get("c")),
    }

def load_bank() -> List[Dict[str,Any]]:
    if not BANK_FILE.exists():
        return []
    text = BANK_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = obj.get("items") or obj.get("data") or []
        else:
            items = []
    except Exception:
        items = list(_iter_jsonl(BANK_FILE))

    norm = []
    for it in items:
        if isinstance(it, dict):
            try:
                norm.append(_normalize_item(it))
            except Exception:
                continue
    return norm

BANK = load_bank()
if not BANK:
    # fallback already written in the packaged json
    BANK = load_bank()

# ----------- Utilities -----------
def interleave_by_difficulty(items: List[Dict[str,Any]], n: int = 12) -> List[Dict[str,Any]]:
    pool = items[:]
    if not pool:
        return []
    for it in pool:
        it["_bscore"] = it["b"] if isinstance(it.get("b"), (int,float)) else random.uniform(-0.5,0.5)
    pool.sort(key=lambda x: x["_bscore"])
    n = min(n, len(pool))
    q1 = pool[:max(1, len(pool)//3)]
    q2 = pool[max(1, len(pool)//3): max(2, 2*len(pool)//3)]
    q3 = pool[max(2, 2*len(pool)//3):]
    seq = []
    while (q1 or q2 or q3) and len(seq) < n:
        if q1: seq.append(q1.pop(0))
        if q3 and len(seq) < n: seq.append(q3.pop(0))
        if q2 and len(seq) < n: seq.append(q2.pop(0))
    for it in seq:
        it.pop("_bscore", None)
    return seq[:n]

def select_items(theme, level, sector, n=12):
    def eq(a,b): return a and b and a.strip().lower() == b.strip().lower()
    cand = [it for it in BANK if (eq(it["theme"], theme) if theme else True)
                             and (eq(it["level"], level) if level else True)
                             and (eq(it["sector"], sector) if sector else True)]
    if len(cand) < n:
        cand = [it for it in BANK if (eq(it["theme"], theme) if theme else True)]
        if len(cand) < n:
            cand = BANK[:]
    return interleave_by_difficulty(cand, n)

SESSIONS: Dict[str, Dict[str,Any]] = {}

# ----------- Routes -----------
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/api/llm_ping")
def llm_ping():
    have = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        tags = r.json().get("models", [])
        have = [m.get("name") for m in tags if m.get("name")]
    except Exception:
        pass
    return jsonify({"ok": True, "base_url": OLLAMA_BASE_URL, "pool": LLM_POOL, "have": have, "speed_mode": LLM_SPEED_MODE})

@app.post("/api/consent")
def record_consent():
    data = request.json or {}
    consent = bool(data.get("consent"))
    sid = data.get("session_id") or uuid.uuid4().hex[:12]
    ts = datetime.datetime.utcnow().isoformat()
    path = ANALYTICS_DIR/"consents.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp_utc","session_id","consent"])
        w.writerow([ts, sid, int(consent)])
    return jsonify({"ok": True, "session_id": sid})

@app.post("/api/start_session")
def start_session():
    data = request.json or {}
    theme = (data.get("qcm_type") or "").strip()
    level = (data.get("study_level") or "").strip()
    sector = (data.get("sector") or "").strip()
    session_id = uuid.uuid4().hex[:12]

    llm_allowed = random.random() < 0.5
    chosen_model = None
    dice_face = None
    if llm_allowed and LLM_POOL:
        idx = random.randrange(len(LLM_POOL))
        chosen_model = LLM_POOL[idx]
        dice_face = idx+1

    items = select_items(theme, level, sector, n=12)
    SESSIONS[session_id] = {
        "id": session_id,
        "started_at": datetime.datetime.utcnow().isoformat(),
        "profile": {"theme": theme, "level": level, "sector": sector},
        "llm_allowed": llm_allowed,
        "llm_model": chosen_model,
        "dice_face": dice_face,
        "items": items,
        "answers": {},
        "order": [],
        "llm_events": [],
        "predicted": None,
    }
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "llm_allowed": llm_allowed,
        "llm_model": chosen_model,
        "dice_face": dice_face,
        "items": [{"id": it["id"], "question": it["question"], "options": it["options"]} for it in items]
    })

@app.post("/api/predict")
def set_prediction():
    data = request.json or {}
    session_id = data.get("session_id")
    try:
        score12 = int(data.get("score12", 0))
    except Exception:
        score12 = 0
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"ok": False, "error": "session not found"}), 404
    s["predicted"] = score12
    return jsonify({"ok": True})

@app.get("/api/items")
def get_items():
    theme = request.args.get("qcm_type","")
    level = request.args.get("study_level","")
    sector = request.args.get("sector","")
    items = select_items(theme, level, sector, n=12)
    return jsonify({"ok": True, "items": [{"id": it["id"], "question": it["question"], "options": it["options"]} for it in items]})

@app.post("/api/record_response")
def record_response():
    data = request.json or {}
    session_id = data.get("session_id")
    try:
        qidx = int(data.get("qidx", -1))
        selected = int(data.get("selected", -1))
    except Exception:
        return jsonify({"ok": False, "error": "bad indices"}), 400
    ts = datetime.datetime.utcnow().isoformat()
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"ok": False, "error": "session not found"}), 404
    revisits = 0
    if qidx in s["answers"]:
        revisits = s["answers"][qidx].get("revisits", 0) + 1
    s["answers"][qidx] = {"selected": selected, "ts": ts, "revisits": revisits}
    s["order"].append(qidx)
    return jsonify({"ok": True})

@app.post("/api/llm_chat")
def llm_chat():
    data = request.json or {}
    session_id = data.get("session_id")
    prompt = (data.get("prompt") or "").strip()
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"ok": False, "error": "session not found"}), 404
    if not s.get("llm_allowed"):
        return jsonify({"ok": False, "error": "LLM non autorisé pour cette session."}), 403

    model = s.get("llm_model") or (LLM_POOL[0] if LLM_POOL else "mistral:latest")
    payload = {
        "model": model,
        "stream": False,
        "options": {"num_predict": LLM_NUM_PREDICT, "num_ctx": LLM_NUM_CTX, "temperature": 0.7},
        "messages": [
            {"role":"system","content":"Tu es un assistant concis. Donne des indices et explications simples sans révéler la réponse brute si possible."},
            {"role":"user","content": prompt}
        ]
    }
    started = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        out = r.json()
        content = (out.get("message") or {}).get("content") or out.get("response") or ""
    except requests.exceptions.Timeout:
        content = "⏳ Le modèle met trop de temps à répondre. Réessayez (timeout)."
    except Exception as e:
        content = f"Erreur LLM: {str(e)}"
    elapsed = time.time() - started
    s["llm_events"].append({"prompt": prompt, "response": content, "elapsed": elapsed, "model": model, "ts": datetime.datetime.utcnow().isoformat()})
    return jsonify({"ok": True, "model": model, "response": content, "elapsed": elapsed})

def save_session(s: Dict[str,Any]):
    sid = s["id"]
    items = s["items"]
    df_rows = []
    corrects = 0
    for i,it in enumerate(items):
        ans = s["answers"].get(i, {})
        sel = ans.get("selected", -1)
        is_correct = int(sel == it["answer_index"])
        corrects += is_correct
        df_rows.append({
            "session_id": sid,
            "qidx": i,
            "item_id": it["id"],
            "theme": it["theme"],
            "level": it["level"],
            "sector": it["sector"],
            "b": it.get("b"),
            "selected_index": sel,
            "correct_index": it["answer_index"],
            "is_correct": is_correct,
            "revisits": ans.get("revisits", 0),
            "answered_at_utc": ans.get("ts")
        })
    df = pd.DataFrame(df_rows)
    session_csv = SESSION_DIR/f"session_{sid}.csv"
    df.to_csv(session_csv, index=False)

    diff = [{"qidx": i, "item_id": it["id"], "b": it.get("b")} for i,it in enumerate(items)]
    (SESSION_DIR/f"session_{sid}_difficulty.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    mean_score = 12.0*corrects/len(items) if items else 0.0
    agg_path = ANALYTICS_DIR/"scores_by_level_and_sector.csv"
    new = not agg_path.exists()
    with open(agg_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc","session_id","level","sector","theme","score_over_12"])
        prof = s["profile"]
        w.writerow([datetime.datetime.utcnow().isoformat(), sid, prof.get("level"), prof.get("sector"), prof.get("theme"), mean_score])

    corr_path = ANALYTICS_DIR/"correlation_field_theme.csv"
    new2 = not corr_path.exists()
    with open(corr_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new2:
            w.writerow(["timestamp_utc","session_id","sector","theme","score_over_12"])
        w.writerow([datetime.datetime.utcnow().isoformat(), sid, s["profile"].get("sector"), s["profile"].get("theme"), mean_score])

    return corrects, mean_score, session_csv

@app.post("/api/end_session")
def end_session():
    data = request.json or {}
    session_id = data.get("session_id")
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"ok": False, "error": "session not found"}), 404
    corrects, mean_score, _ = save_session(s)
    total = len(s["items"])
    return jsonify({
        "ok": True,
        "score_over_12": float(12.0*corrects/total if total else 0.0),
        "corrects": int(corrects),
        "total": int(total),
        "predicted": s.get("predicted"),
        "llm_allowed": s.get("llm_allowed"),
        "llm_model": s.get("llm_model")
    })

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
