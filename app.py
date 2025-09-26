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

# ----------- LLM assignment & concurrency tracking -----------
# probabilité uniforme 16,667%
DICE_MODEL_MAP = {
    1: {"name":"mistral:latest",  "kind":"local"},
    2: {"name":"gpt-3.5",         "kind":"api"},   # API (insérer URL)
    3: {"name":"llama3:latest",   "kind":"local"},
    4: {"name":"gpt-o4",          "kind":"api"},   # API (insérer URL)
    5: {"name":"qwen2.7b:latest", "kind":"local"},
    6: {"name":"claude",          "kind":"api"},   # API (insérer URL)
}
LOCAL_MODELS = {v["name"] for v in DICE_MODEL_MAP.values() if v["kind"]=="local"}
API_MODELS   = {v["name"] for v in DICE_MODEL_MAP.values() if v["kind"]=="api"}

ACTIVE_SESSIONS = set()  # session_ids en cours

# Quotas de sélection LLM (pour la règle 16,667% sur les sessions LLM)
LLM_ASSIGN_COUNTS = {m["name"]: 0 for m in DICE_MODEL_MAP.values()}
LLM_ASSIGN_TOTAL = 0  # nb total de sessions avec LLM autorisé et modèle assigné

# Compteur global d'appels API par fournisseur (visibles/éditables)
API_CALL_COUNTS = {"gpt-3.5":0, "gpt-o4":0, "claude":0}

def server_at_max_load() -> bool:
    """
    Placeholder de détection de charge GPU/CPU.
    Retourne True quand le serveur est à pleine charge.
    TODO: brancher ici une vraie télémétrie (GPU util, VRAM, CPU load).
    """
    return False

def get_simultaneous_llm_sessions():
    """Sessions actives avec LLM autorisé."""
    return [SESSIONS[sid] for sid in ACTIVE_SESSIONS if SESSIONS.get(sid,{}).get("llm_allowed")]

def pick_model_with_rules(session_id: str) -> (int, str):
    """
    Règles d’assignation du modèle :
    - Dé 1..6 → mapping fixe (16,667% chacun).
    - Si >2 sessions en simultané et ≥2 avec LLM, on force la diversité :
        • Sur 2, si les 2 sont du même type (local/api), on choisit l’autre type.
        • Sur ≥6, on essaie d’attribuer des modèles tous différents.
    - Si 1 seule session et server_at_max_load() == True → choisir un local "le plus gourmand" :
        priorité: llama3 > mistral > qwen (si disponibles et sous quota).
    - Cap 16,667% par modèle dans l’ensemble des sessions LLM (notre dénominateur = LLM_ASSIGN_TOTAL).
    """
    global LLM_ASSIGN_TOTAL

    candidates = list(range(1,7))
    def model_name(face): return DICE_MODEL_MAP[face]["name"]
    def model_kind(face): return DICE_MODEL_MAP[face]["kind"]

    if LLM_ASSIGN_TOTAL >= 6:
        for face in candidates[:]:
            name = model_name(face)
            share = (LLM_ASSIGN_COUNTS.get(name,0) / max(1, LLM_ASSIGN_TOTAL))
            if share >= (1.0/6.0):
                candidates.remove(face)

    # Charge max + 1 session → local le plus "gourmand"
    if len(ACTIVE_SESSIONS) <= 1 and server_at_max_load():
        for p in ["llama3:latest", "mistral:latest", "qwen2.7b:latest"]:
            face = next((f for f in candidates if model_name(f)==p), None)
            if face is not None:
                return face, model_name(face)

    # Diversité en simultané
    concurrent_llm = get_simultaneous_llm_sessions()
    if len(concurrent_llm) >= 2:
        kinds = [("api" if s.get("llm_model") in API_MODELS else "local") if s.get("llm_model") else None
                 for s in concurrent_llm if s.get("llm_model")]
        if len(concurrent_llm) == 2:
            if kinds and all(k=="local" for k in kinds):
                candidates = [f for f in candidates if model_kind(f)=="api"]
            elif kinds and all(k=="api" for k in kinds):
                candidates = [f for f in candidates if model_kind(f)=="local"]
        if len(concurrent_llm) >= 6:
            in_use = {s.get("llm_model") for s in concurrent_llm if s.get("llm_model")}
            unused_faces = [f for f in candidates if model_name(f) not in in_use]
            if unused_faces:
                candidates = unused_faces

    if not candidates:
        candidates = list(range(1,7))
    face = random.choice(candidates)
    return face, model_name(face)

# ----------- BANK -----------
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
    # output -> index
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
                try:
                    if isinstance(opt,str) and opt.strip() == s:
                        answer_index = i
                        break
                except Exception: pass
    if answer_index is None: answer_index = 0

    def _f(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    return {
        "id": it.get("id") or str(uuid.uuid4()),
        "question": q,
        "options": list(options),
        "answer_index": answer_index,
        "theme": it.get("theme") or "général",
        "level": it.get("level") or "Licence",
        "sector": it.get("sector") or "général",
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

def interleave_by_difficulty(items: List[Dict[str,Any]], n=12):
    cand = items[:]
    for it in cand:
        b = it.get("b")
        if b is None:
            it["_bscore"] = random.uniform(-0.5,0.5)
        else:
            try:
                it["_bscore"] = float(b)
            except Exception:
                it["_bscore"] = 0.0
    cand.sort(key=lambda x: x["_bscore"])
    if not cand: return []
    k = max(1, len(cand)//3)
    q1 = cand[:k]
    q2 = cand[k:2*k]
    q3 = cand[2*k:]
    seq = []
    while len(seq) < n and (q1 or q2 or q3):
        if q1 and len(seq) < n: seq.append(q1.pop(0))
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

BANK = load_bank()
SESSIONS: Dict[str, Dict[str,Any]] = {}

# ----------- Config. LLM -----------
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/api/llm_ping")
def llm_ping():
    have = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        out = r.json()
        have = [m.get("name") or m.get("model") for m in (out.get("models") or [])]
    except Exception:
        have = []
    return jsonify({
        "ok": True,
        "base_url": OLLAMA_BASE_URL,
        "pool": LLM_POOL,
        "have": have,
        "speed_mode": LLM_SPEED_MODE
    })

@app.post("/api/consent")
def consent():
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

    # 1) Pièce pile/face → LLM autorisé ?
    llm_allowed = random.random() < 0.5

    # 2) Dé 1..6 avec mapping fixe
    chosen_model = None
    dice_face = None
    if llm_allowed:
        dice_face, chosen_model = pick_model_with_rules(session_id)
        # 5/6) Quota 16,667% sur les sessions LLM
        LLM_ASSIGN_COUNTS[chosen_model] = LLM_ASSIGN_COUNTS.get(chosen_model,0)+1
        LLM_ASSIGN_TOTAL += 1

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
        "llm_use_times": [],  # temps (s) 
        "ended": False,
    }
    ACTIVE_SESSIONS.add(session_id)
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
    qidx = data.get("qidx")
    try:
        qidx = int(qidx) if qidx is not None else None
    except Exception:
        qidx = None

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
        content = ""
        if model in API_MODELS:
            # === API CALL PLACEHOLDER ===
            # URL API ICI !!!!!!
            # Exemples :
            # if model == "gpt-3.5":
            #     resp = requests.post("INSERT_URL_HERE_FOR_GPT35", json={"model": "gpt-3.5", "messages": payload["messages"]}, timeout=LLM_TIMEOUT)
            # elif model == "gpt-o4":
            #     resp = requests.post("INSERT_URL_HERE_FOR_GPT_O4", json={...}, timeout=LLM_TIMEOUT)
            # elif model == "claude":
            #     resp = requests.post("INSERT_URL_HERE_FOR_CLAUDE", json={...}, timeout=LLM_TIMEOUT)
            # resp.raise_for_status()
            # content = resp.json().get("choices",[{}])[0].get("message",{}).get("content","")
            
            API_CALL_COUNTS[model] = API_CALL_COUNTS.get(model,0)+1
            content = "(Réponse API — insérez votre appel ici)"
        else:
            # model lcoal
            r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            out = r.json()
            content = (out.get("message") or {}).get("content") or out.get("response") or ""
    except requests.exceptions.Timeout:
        content = "⏳ Le modèle met trop de temps à répondre. Réessayez (timeout)."
    except Exception as e:
        content = f"Erreur LLM: {str(e)}"
    elapsed = time.time() - started

    # 4) On enregistre le délai 1ère/2e/... utilisation LLM (écart depuis start)
    if s.get("started_at"):
        try:
            t0 = datetime.datetime.fromisoformat(s["started_at"])
        except Exception:
            t0 = None
        if t0 is not None:
            s["llm_use_times"].append((datetime.datetime.utcnow()-t0).total_seconds())

    # 1) Tout logger : inputs, outputs, délai
    s["llm_events"].append({"qidx": qidx, "prompt": prompt, "response": content, "elapsed": elapsed, "model": model, "ts": datetime.datetime.utcnow().isoformat()})
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

    # --- Persistance des événements LLM par session ---
    llm_events = s.get("llm_events") or []
    if llm_events:
        ev_path = SESSION_DIR/f"session_{sid}_llm_events.csv"
        newev = not ev_path.exists()
        with open(ev_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newev:
                w.writerow(["timestamp_utc","qidx","prompt","response","elapsed_s","model"])
            for ev in llm_events:
                w.writerow([ev.get("ts"), ev.get("qidx"), ev.get("prompt"), ev.get("response"), ev.get("elapsed"), ev.get("model")])

    # --- Fichiers aux. existants ---
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

    # --- (HHI, Gini, switches, etc.) ---
    def hhi(counts):
        tot = sum(counts)
        if tot == 0: return 0.0
        return sum((c/tot)**2 for c in counts)

    def gini(arr):
        x = [a for a in arr if a is not None]
        n = len(x)
        if n == 0: return 0.0
        x = sorted(x)
        cum = 0.0
        for i,xi in enumerate(x, start=1):
            cum += i*xi
        return (2*cum)/(n*sum(x)) - (n+1)/n if sum(x)>0 else 0.0

    # Approx temps par item via différences
    times = []
    order = s.get("order", [])
    ans = s.get("answers", {})
    prev_ts = None
    for q in order:
        ts = ans.get(q,{}).get("ts")
        if ts and prev_ts:
            try:
                t_prev = datetime.datetime.fromisoformat(prev_ts)
                t_cur = datetime.datetime.fromisoformat(ts)
                times.append((t_cur - t_prev).total_seconds())
            except Exception:
                pass
        prev_ts = ts
    time_mean_overall = sum(times)/len(times) if times else 0.0
    gini_time = gini(times)

    # Switch de difficulté
    def tier(b):
        if b is None: return "mid"
        try:
            b = float(b)
        except Exception:
            return "mid"
        return "adv" if b>=1 else ("nonadv" if b<=-1 else "mid")
    tiers = [tier(items[q].get("b")) for q in order if 0<=q<len(items)]
    switch_difficulty_rate = 0.0
    if len(tiers) >= 2:
        switches = sum(1 for i in range(1,len(tiers)) if tiers[i]!=tiers[i-1])
        switch_difficulty_rate = switches/(len(tiers)-1)

    # LLM usage distribution across items
    llm_counts_per_item = {}
    for ev in llm_events:
        q = ev.get("qidx")
        llm_counts_per_item[q] = llm_counts_per_item.get(q,0)+1
    counts = list(llm_counts_per_item.values())
    hhi_llm = hhi(counts) if counts else 0.0
    hhi_llm_norm = (hhi_llm - (1/len(counts) if counts else 0)) / (1 - (1/len(counts) if counts else 1)) if counts and len(counts)>1 else 0.0
    effective_items_llm = (1.0/hhi_llm) if hhi_llm>0 else 0.0

    # Time distribution per item (approx)
    times_per_item = {}
    for i,q in enumerate(order):
        if i < len(times):
            times_per_item[q] = times[i]
    tcounts = list(times_per_item.values())
    hhi_time = hhi(tcounts) if tcounts else 0.0
    hhi_time_norm = (hhi_time - (1/len(tcounts) if tcounts else 0)) / (1 - (1/len(tcounts) if tcounts else 1)) if tcounts and len(tcounts)>1 else 0.0
    effective_items_time = (1.0/hhi_time) if hhi_time>0 else 0.0

    # Proxies “trust/switch” (faute de logs plus fins)
    switch_llm_items = 0
    revised_items = 0
    for i,it in enumerate(items):
        if i in llm_counts_per_item and s["answers"].get(i):
            revised = s["answers"][i].get("revisits",0)>0
            revised_items += 1 if revised else 0
            switch_llm_items += 1 if revised else 0
    switch_llm_rate = (switch_llm_items / revised_items) if revised_items else 0.0

    # Longest run d'items “adv”
    longest_adv_run = 0
    cur = 0
    for q in order:
        if tier(items[q].get("b"))=="adv":
            cur += 1
            longest_adv_run = max(longest_adv_run, cur)
        else:
            cur = 0

    # Variables demandées (non observables finement ici → 0.0/placeholder)
    p_switch_after_error = 0.0
    p_switch_after_success = 0.0
    p_switch_out_of_adv_after_llm = 0.0
    p_switch_out_of_adv_after_no_llm = 0.0
    delta_tunnel_adv = 0.0
    delta_time_after_error = 0.0

    score_main_12 = mean_score
    score_total_with_bonus = mean_score
    time_mean = time_mean_overall

    llm_use_times = s.get("llm_use_times", [])
    k_llm = len(llm_events)
    k_time = len(times)
    k_items_x = len(items)
    k_ref_x = len(order)

    # Z-scores (nécessitent stats de cohorte) → 0.0
    z_fields = {k: 0.0 for k in [
        "z_gini_time","z_p_switch_out_of_adv_after_llm","z_p_switch_out_of_adv_after_no_llm",
        "z_delta_tunnel_adv","z_delta_time_after_error","z_hhi_llm",
        "z_switch_difficulty_rate","z_p_switch_after_error"
    ]}

    HBI = 0.0  

    adv_path = ANALYTICS_DIR/"advanced_metrics.csv"
    is_new = not adv_path.exists()
    with open(adv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "timestamp_utc","session_id",
                "gini_time","hhi_llm","hhi_llm_norm","effective_items_llm",
                "hhi_time","hhi_time_norm","effective_items_time",
                "time_mean_overall",
                "p_switch_out_of_adv_after_llm","p_switch_out_of_adv_after_no_llm",
                "p_switch_after_error","p_switch_after_success",
                "switch_llm_rate","switch_difficulty_rate",
                "delta_tunnel_adv","delta_time_after_error",
                "longest_adv_run",
                "k_llm","k_items.x","k_ref.x","k_time",
                "score_main_12","score_total_with_bonus","time_mean",
                "z_gini_time","z_p_switch_out_of_adv_after_llm","z_p_switch_out_of_adv_after_no_llm",
                "z_delta_tunnel_adv","z_delta_time_after_error","z_hhi_llm","z_switch_difficulty_rate","z_p_switch_after_error",
                "HBI"
            ])
        w.writerow([
            datetime.datetime.utcnow().isoformat(), sid,
            gini_time, hhi_llm, hhi_llm_norm, effective_items_llm,
            hhi_time, hhi_time_norm, effective_items_time,
            time_mean_overall,
            p_switch_out_of_adv_after_llm, p_switch_out_of_adv_after_no_llm,
            p_switch_after_error, p_switch_after_success,
            switch_llm_rate, switch_difficulty_rate,
            delta_tunnel_adv, delta_time_after_error,
            longest_adv_run,
            k_llm, k_items_x, k_ref_x, k_time,
            score_main_12, score_total_with_bonus, time_mean,
            z_fields["z_gini_time"], z_fields["z_p_switch_out_of_adv_after_llm"], z_fields["z_p_switch_out_of_adv_after_no_llm"],
            z_fields["z_delta_tunnel_adv"], z_fields["z_delta_time_after_error"], z_fields["z_hhi_llm"], z_fields["z_switch_difficulty_rate"], z_fields["z_p_switch_after_error"],
            HBI
        ])

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
    s["ended"] = True
    ACTIVE_SESSIONS.discard(session_id)
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
