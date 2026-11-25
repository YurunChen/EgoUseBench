(() => {
  let record = JSON.parse(document.getElementById('initial-record').textContent);
  const form = document.getElementById('answer-form');
  const status = document.getElementById('save-status');
  const token = document.querySelector('meta[name=validation-session]').content;
  let pending = Promise.resolve();
  let timer;
  let dirty = false;
  let editVersion = 0;
  const cards = Array.from(document.querySelectorAll('[data-option]'));
  const pageSize = 24;
  let candidatePage = 0;
  function renderCandidates() {
    const pageCount = Math.max(1, Math.ceil(cards.length / pageSize));
    cards.forEach((card, index) => { card.hidden = Math.floor(index / pageSize) !== candidatePage; });
    document.getElementById('page-label').textContent = `第 ${candidatePage + 1} / ${pageCount} 页`;
    document.getElementById('previous-page').disabled = candidatePage === 0;
    document.getElementById('next-page').disabled = candidatePage >= pageCount - 1;
    const choosing = form.elements.judgment.value === 'selected';
    cards.forEach(card => card.querySelectorAll('input').forEach(input => { input.disabled = !choosing || !!record.submitted_at; }));
    const selected = choosing ? cards.filter(card => card.querySelector('[type=checkbox]').checked) : [];
    document.getElementById('selected-summary').textContent = selected.length ? '已选：' + selected.map(card =>
      card.querySelector('h3').textContent + ' × ' + card.querySelector('[type=number]').value
    ).join('；') : '';
  }
  document.getElementById('previous-page').addEventListener('click', () => { candidatePage--; renderCandidates(); });
  document.getElementById('next-page').addEventListener('click', () => { candidatePage++; renderCandidates(); });
  renderCandidates();
  function show(message, error = false) { status.textContent = message; status.classList.toggle('error', error); }
  function choices() {
    const judgment = form.elements.judgment.value;
    const selected = {};
    if (judgment === 'selected') {
      document.querySelectorAll('[data-option]').forEach(card => {
        if (card.querySelector('[type=checkbox]').checked) {
          selected[card.dataset.option] = Number(card.querySelector('[type=number]').value);
        }
      });
    }
    const fields = {};
    if (judgment === 'selected') document.querySelectorAll('[data-field]').forEach(input => {
      if (input.value) fields[input.dataset.field] = input.value;
    });
    return {judgment, choices: selected, fields, note: document.getElementById('note').value};
  }
  function send(action, data = {}) {
    pending = pending.catch(() => {}).then(async () => {
      const response = await fetch(location.pathname, {
        method:'POST', headers:{'Content-Type':'application/json', 'X-Validation-Session':token},
        body:JSON.stringify({...data, action, revision:record.revision})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '保存失败');
      record = result;
      return result;
    });
    return pending;
  }
  async function saveDraft() {
    const version = editVersion;
    try { await send('save', choices()); if (version === editVersion) { dirty = false; show('已保存到本机'); } }
    catch (error) { show(error.message + '；请重试保存或提交，不要关闭页面。', true); }
  }
  form.addEventListener('input', () => {
    dirty = true; editVersion++; renderCandidates(); clearTimeout(timer); show('正在保存…'); timer = setTimeout(saveDraft, 350);
  });
  form.addEventListener('submit', async event => {
    event.preventDefault(); clearTimeout(timer);
    if (!form.reportValidity()) return;
    const button = document.getElementById('submit'); button.disabled = true;
    const submission = choices();
    document.getElementById('answer-fields').disabled = true;
    if (document.getElementById('ask')) document.getElementById('ask').disabled = true;
    try {
      await send('submit', submission); dirty = false;
      document.getElementById('answer-fields').disabled = true;
      if (document.getElementById('ask')) document.getElementById('ask').disabled = true;
      show('首次提交已保存');
      const next = document.getElementById('next'); if (next) next.hidden = false;
    } catch(error) {
      show(error.message, true); button.disabled = false;
      document.getElementById('answer-fields').disabled = false;
      if (document.getElementById('ask')) document.getElementById('ask').disabled = record.questions.length > 0;
      renderCandidates();
    }
  });
  const ask = document.getElementById('ask');
  if (ask) ask.addEventListener('click', async () => {
    clearTimeout(timer);
    const question = document.getElementById('question');
    ask.disabled = true;
    document.getElementById('answer-fields').disabled = true;
    document.getElementById('submit').disabled = true;
    try {
      await send('save', choices()); await send('ask', {question:question.value});
      const answer = record.questions.at(-1);
      const paragraph = document.createElement('p');
      paragraph.textContent = answer.question + '\n佩戴者：' + answer.answer_event.payload;
      document.getElementById('answers').append(paragraph); question.disabled = true;
      dirty = false; show('问题和回答已保存到本机');
    } catch(error) { show(error.message, true); ask.disabled = false; }
    finally {
      document.getElementById('answer-fields').disabled = false;
      document.getElementById('submit').disabled = false;
      renderCandidates();
    }
  });
  window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
})();
