
let STATE = {
  session_id: null,
  llm_allowed: false,
  llm_model: null,
  items: [],
  answers: {},  // qid -> {selected_index, is_correct, order_index, revisits, switches, time_spent_ms}
  visitCount: {}, // qid -> visits
  switchCount: {}, // qid -> count of choice changes
  timers: {}, // qid -> start timestamp
};

function show(id){
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('visible'));
  const el = document.getElementById(id);
  el.classList.add('visible', 'fade-in');
}

async function pingLLM(){
  const box = document.getElementById('llm-ping');
  try{
    const r = await fetch('/api/llm_ping'); const j = await r.json();
    if(j.ok){
      box.textContent = `LLM prêt (${j.pool.filter(m => j.have.includes(m)).join(', ')})`;
    }else{
      box.textContent = `LLM non disponible`;
    }
  }catch(e){
    box.textContent = `LLM non disponible`;
  }
}

function coinAndDice(llm_allowed, model){
  const coin = document.getElementById('coin');
  const coinRes = document.getElementById('coin-result');
  const dice = document.getElementById('dice');
  const diceRes = document.getElementById('dice-result');
  const next = document.getElementById('to-predict');

  coin.classList.remove('hidden');
  coinRes.textContent = 'Tirage en cours…';
  dice.classList.add('hidden');
  diceRes.textContent = '';

  setTimeout(() => {
    coinRes.textContent = llm_allowed ? "Résultat: accès LLM" : "Résultat: sans LLM";
    // Après la pièce, on lance le dé (modèle)
    setTimeout(() => {
      if(llm_allowed){
        dice.classList.remove('hidden');
        diceRes.textContent = `Modèle sélectionné: ${model}`;
      }else{
        dice.classList.remove('hidden');
        diceRes.textContent = `Pas de modèle (session sans LLM)`;
      }
      next.disabled = false;
    }, 900);
  }, 1000);
}

async function startSession(){
  const qcm_type = document.getElementById('qcm_type').value;
  const study_level = document.getElementById('study_level').value;
  const sector = document.getElementById('sector').value;
  const r = await fetch('/api/start_session', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({qcm_type, study_level, sector})
  });
  const j = await r.json();
  STATE.session_id = j.session_id;
  STATE.llm_allowed = j.llm_allowed;
  STATE.llm_model = j.llm_model;
  show('screen-draw');
  coinAndDice(j.llm_allowed, j.llm_model);
}

async function loadItems(){
  const params = new URLSearchParams({
    session_id: STATE.session_id,
    qcm_type: document.getElementById('qcm_type').value,
    study_level: document.getElementById('study_level').value,
    sector: document.getElementById('sector').value
  });
  const r = await fetch('/api/items?'+params.toString());
  const j = await r.json();
  STATE.items = j.items;
  renderItems();
}

function renderItems(){
  const track = document.getElementById('qcm-track');
  track.innerHTML = '';
  STATE.items.forEach((it, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h4>${idx+1}. ${it.question}</h4>
      <div class="choices"></div>
    `;
    const choicesBox = card.querySelector('.choices');
    STATE.visitCount[it.id] = 0;
    STATE.switchCount[it.id] = 0;
    it.choices.forEach((c, ci) => {
      const btn = document.createElement('div');
      btn.className = 'choice';
      btn.textContent = c;
      btn.addEventListener('click', () => {
        const prev = STATE.answers[it.id]?.selected_index;
        if(prev !== undefined && prev !== ci){ STATE.switchCount[it.id]++; }
        STATE.answers[it.id] = {
          selected_index: ci,
          is_correct: (ci === it.answer_index),
          order_index: it.order_index,
          revisits: STATE.visitCount[it.id],
          switches: STATE.switchCount[it.id],
          time_spent_ms: (STATE.timers[it.id] ? (Date.now()-STATE.timers[it.id]) : 0)
        };
        // UI mark
        choicesBox.querySelectorAll('.choice').forEach(el => el.classList.remove('selected'));
        btn.classList.add('selected');
        // send to backend
        fetch('/api/record_response', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            session_id: STATE.session_id,
            question_id: it.id,
            order_index: it.order_index,
            selected_index: ci,
            is_correct: (ci === it.answer_index),
            time_spent_ms: STATE.answers[it.id].time_spent_ms,
            revisits: STATE.visitCount[it.id],
            switches: STATE.switchCount[it.id]
          })
        });
      });
      choicesBox.appendChild(btn);
    });
    // track visits/time
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if(e.isIntersecting){
          STATE.visitCount[it.id] = (STATE.visitCount[it.id]||0) + 1;
          STATE.timers[it.id] = Date.now();
        }
      });
    }, {root: document.querySelector('#qcm-wrapper'), threshold: 0.8});
    observer.observe(card);
    track.appendChild(card);
  });
}

async function sendPrediction(){
  const v = parseFloat(document.getElementById('pred-input').value || '0');
  await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: STATE.session_id, predicted: v})});
  await loadItems();
  const status = document.getElementById('llm-status');
  if(STATE.llm_allowed){
    status.textContent = `LLM autorisé (${STATE.llm_model})`;
  }else{
    status.textContent = `LLM non autorisé sur cette session`;
  }
  show('screen-qcm');
}

async function finishSession(){
  const r = await fetch('/api/end_session', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: STATE.session_id})
  });
  const j = await r.json();
  const box = document.getElementById('result-box');
  if(j.ok){
    box.innerHTML = `
      <p><strong>Score final :</strong> ${j.final_score} / 12</p>
      <p><strong>θ estimé :</strong> ${j.theta_hat.toFixed(2)}</p>
      <p><strong>Items modélisés :</strong> ${j.items_modeled}</p>
    `;
  }else{
    box.textContent = 'Erreur lors de la finalisation.';
  }
  show('screen-result');
}

async function sendConsent(consent){
  await fetch('/api/consent', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: STATE.session_id || null, consent})});
}

async function llmSend(){
  if(!STATE.llm_allowed){ return; }
  const input = document.getElementById('llm-input');
  const msg = input.value.trim();
  if(!msg) return;
  input.value = '';
  const log = document.getElementById('llm-chat-log');
  const user = document.createElement('div'); user.className = 'chat-msg user'; user.textContent = msg;
  const llm = document.createElement('div'); llm.className = 'chat-msg llm'; llm.textContent = 'Le modèle réfléchit…';
  log.appendChild(user); log.appendChild(llm); log.scrollTop = log.scrollHeight;
  try{
    const r = await fetch('/api/llm_chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: STATE.session_id, prompt: msg})
    });
    const j = await r.json();
    if(j.ok){ llm.textContent = j.text; }
    else { llm.textContent = j.error || 'LLM indisponible.'; }
  }catch(_){
    llm.textContent = 'Erreur réseau LLM.';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('go-consent').addEventListener('click', () => show('screen-consent'));
  const cb = document.getElementById('consent-box');
  const cc = document.getElementById('consent-continue');
  cb.addEventListener('change', () => { cc.disabled = !cb.checked; });
  cc.addEventListener('click', async () => {
    if(!STATE.session_id){ STATE.session_id = Math.random().toString(36).slice(2,14); }
    await sendConsent(!!cb.checked);
    show('screen-params');
    pingLLM();
  });

  document.getElementById('btn-begin').addEventListener('click', startSession);
  document.getElementById('to-predict').addEventListener('click', () => show('screen-predict'));
  document.getElementById('pred-go').addEventListener('click', sendPrediction);
  document.getElementById('finish').addEventListener('click', finishSession);
  document.getElementById('llm-send').addEventListener('click', llmSend);
  document.getElementById('llm-input').addEventListener('keydown', (e)=>{ if(e.key==='Enter') llmSend(); });
});
