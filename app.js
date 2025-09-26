let STATE = {
  session_id: null,
  llm_allowed: false,
  llm_model: null,
  items: [],            // [{id, question, options}]
  answers: {},          // qidx -> selected index
  visitCount: {},       // qidx -> visits
  switchCount: {},      // qidx -> count of choice changes
  timers: {},           // qidx -> start timestamp (ms)
  lastQidx: null        // pour lier le chat LLM à la dernière question manipulée
};

function show(id){
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('visible'));
  const el = document.getElementById(id);
  el.classList.add('visible', 'fade-in');
}

async function pingLLM(){
  const box = document.getElementById('llm-ping');
  try{
    const r = await fetch('/api/llm_ping'); 
    const j = await r.json();
    if(j.ok){
      const present = (j.have || []);
      const pool = (j.pool || []).filter(m => present.includes(m));
      box.textContent = pool.length ? `LLM prêt (${pool.join(', ')})` : `LLM local non détecté`;
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
  next.disabled = true;

  setTimeout(() => {
    coinRes.textContent = llm_allowed ? "Résultat: accès LLM" : "Résultat: sans LLM";
    setTimeout(() => {
      dice.classList.remove('hidden');
      diceRes.textContent = llm_allowed ? `Modèle sélectionné: ${model || '—'}` : `Pas de modèle (session sans LLM)`;
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
  if(!j.ok){ alert('Erreur démarrage session'); return; }

  STATE.session_id = j.session_id;
  STATE.llm_allowed = j.llm_allowed;
  STATE.llm_model = j.llm_model || null;
  STATE.items = j.items || [];

  show('screen-draw');
  coinAndDice(STATE.llm_allowed, STATE.llm_model);
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
  STATE.items = j.items || [];
  renderItems();
}

function renderItems(){
  const track = document.getElementById('qcm-track');
  track.innerHTML = '';

  STATE.items.forEach((it, qidx) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h4>${qidx+1}. ${it.question}</h4>
      <div class="choices"></div>
    `;
    const choicesBox = card.querySelector('.choices');

    // init compteurs par qidx
    STATE.visitCount[qidx] = 0;
    STATE.switchCount[qidx] = 0;

    (it.options || []).forEach((opt, optIdx) => {
      const btn = document.createElement('div');
      btn.className = 'choice';
      btn.textContent = opt;
      btn.addEventListener('click', async () => {
        const prev = STATE.answers[qidx];
        if(prev !== undefined && prev !== optIdx){ STATE.switchCount[qidx]++; }

        STATE.answers[qidx] = optIdx;
        STATE.lastQidx = qidx;
        choicesBox.querySelectorAll('.choice').forEach(el => el.classList.remove('selected'));
        btn.classList.add('selected');

        await fetch('/api/record_response', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            session_id: STATE.session_id,
            qidx: qidx,
            selected: optIdx
          })
        }).catch(()=>{});
      });
      choicesBox.appendChild(btn);
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if(e.isIntersecting){
          STATE.visitCount[qidx] = (STATE.visitCount[qidx] || 0) + 1;
          STATE.timers[qidx] = Date.now();
          STATE.lastQidx = qidx;
        }
      });
    }, {root: document.querySelector('#qcm-wrapper'), threshold: 0.8});
    observer.observe(card);

    track.appendChild(card);
  });
}

async function sendPrediction(){
  const v = parseInt(document.getElementById('pred-input').value || '0', 10);
  await fetch('/api/predict', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: STATE.session_id, score12: v})
  });
  await loadItems();

  const status = document.getElementById('llm-status');
  status.textContent = STATE.llm_allowed ? `LLM autorisé (${STATE.llm_model || '—'})` : `LLM non autorisé sur cette session`;

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
      <p><strong>Score final :</strong> ${(j.score_over_12 || 0).toFixed(1)} / 12</p>
      <p><strong>Détails :</strong> ${j.corrects}/${j.total} corrects</p>
      ${j.predicted!=null ? `<p><strong>Votre prédiction :</strong> ${j.predicted}/12</p>` : ''}
      <p><strong>LLM :</strong> ${j.llm_allowed ? `autorisé (${j.llm_model||'—'})` : 'non autorisé'}</p>
    `;
  }else{
    box.textContent = 'Erreur lors de la finalisation.';
  }
  show('screen-result');
}

async function sendConsent(consent){
  await fetch('/api/consent', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id: STATE.session_id || null, consent})
  });
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
      body: JSON.stringify({session_id: STATE.session_id, prompt: msg, qidx: STATE.lastQidx})
    });
    const j = await r.json();
    if(j.ok){ llm.textContent = j.response; }
    else { llm.textContent = j.error || 'LLM indisponible.'; }
  }catch(_){
    llm.textContent = 'Erreur réseau LLM.';
  }
}
