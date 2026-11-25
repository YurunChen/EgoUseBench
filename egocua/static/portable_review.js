(function () {
  'use strict';
  const bundle = window.reviewPackage;
  const prefix = 'egousebench-quality:' + bundle.package_id + ':';
  const content = document.getElementById('content');
  const status = document.getElementById('saved-status');
  const warning = document.getElementById('storage-error');
  const dialog = document.getElementById('reviewer-dialog');
  let state = { reviewer: '', records: {} };
  let currentItem = 0;
  let loader;

  function fail(message) { warning.textContent = message; warning.hidden = false; }
  function escape(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function validReviewer(value) { return typeof value === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(value); }
  function validateRecord(record) {
    if (!record || !Number.isInteger(record.item) || record.item < 1 || record.item > bundle.tasks.length ||
        record.task_id !== bundle.tasks[record.item-1].task_id || typeof record.note !== 'string' || record.note.length > 10000 ||
        !(record.decision === '' || Object.hasOwn(bundle.decisions, record.decision)) ||
        typeof record.started_at !== 'string' || !Number.isFinite(Date.parse(record.started_at)) ||
        typeof record.updated_at !== 'string' || !Number.isFinite(Date.parse(record.updated_at)) ||
        !(record.submitted_at === null || (typeof record.submitted_at === 'string' && Number.isFinite(Date.parse(record.submitted_at))))) {
      throw new Error('记录内容不完整或与本包任务不符');
    }
    if (record.submitted_at && (!record.decision || (record.decision !== 'pass' && !record.note.trim()))) {
      throw new Error('已提交记录缺少判断或原因');
    }
    return { item: record.item, task_id: record.task_id, decision: record.decision,
      note: record.note, started_at: record.started_at, updated_at: record.updated_at, submitted_at: record.submitted_at };
  }
  function storedState(reviewer) {
    const raw = localStorage.getItem(prefix + reviewer);
    if (!raw) return { reviewer: reviewer, records: {} };
    const saved = JSON.parse(raw);
    if (saved.reviewer !== reviewer || !saved.records || typeof saved.records !== 'object') throw new Error('本机记录异常，请先导入备份');
    Object.entries(saved.records).forEach(([item, record]) => {
      saved.records[item] = validateRecord(record);
      if (String(record.item) !== item) throw new Error('本机任务序号不一致');
    });
    return saved;
  }
  function persist() {
    try {
      localStorage.setItem(prefix + state.reviewer, JSON.stringify(state));
      localStorage.setItem(prefix + 'active', state.reviewer);
      warning.hidden = true;
      status.textContent = '已保存到本机';
      return true;
    } catch (error) {
      status.textContent = '未保存';
      fail('浏览器无法保存记录。请立即点击“导出记录”备份，不要关闭页面。原因：' + error.message);
      return false;
    }
  }
  function chooseReviewer(reviewer) {
    if (!validReviewer(reviewer)) throw new Error('审核者编号只能使用字母、数字、下划线和短横线（1–64位）');
    state = storedState(reviewer);
    if (!persist()) return;
    dialog.close();
    document.getElementById('reviewer-label').textContent = '审核者：' + reviewer;
    navigate();
  }
  function renderHome() {
    currentItem = 0;
    const records = Object.values(state.records);
    const submitted = records.filter(r => r.submitted_at);
    content.innerHTML = '<main class="page"><section class="hero"><div><p class="eyebrow">人工任务质量核查</p>' +
      '<h1>40题，逐题判断是否成立</h1><p>Graft 14题 · Refresh 10题 · Clarify 8题 · 组合变化8题</p>' +
      '<p>查看画面、请求和预设答案，选择通过、有问题或无法判断。无需搜索或执行网页操作。</p></div>' +
      '<div class="progress-number"><strong>' + submitted.length + '/' + bundle.tasks.length + '</strong><span>已核查</span></div></section>' +
      '<section class="panel offline-note"><h2>离线使用</h2><p>全部材料已在此包中，无需联网、安装软件或使用API密钥。填写意见后会保存到当前浏览器；点击“保存并进入下一题”后保留首次判断。</p>' +
      '<p><b>完成后点击顶部“导出记录”，把下载的JSON文件发回组织者。</b>中途也可导出备份。若更换电脑、浏览器、文件位置，或清理浏览器数据，请先导出，再在新位置导入；不要使用无痕窗口。</p>' +
      '<div class="review-home-actions"><button id="change-reviewer" type="button">切换审核者编号</button><span>当前编号：' + escape(state.reviewer) + '</span></div></section>' +
      '<div class="task-summary">' + Object.entries(bundle.decisions).map(([key,label]) => '<span>' + escape(label) + ' ' + submitted.filter(r=>r.decision===key).length + '</span>').join('') + '</div>' +
      '<section class="task-list">' + bundle.tasks.map((row,index) => {
        const record = state.records[index+1];
        return '<a class="task-row" href="#task=' + (index+1) + '"><span><b>任务 ' + String(index+1).padStart(2,'0') + '</b> · ' + escape(row.stratum) + '</span><span>' +
          (record && record.submitted_at ? escape(bundle.decisions[record.decision]) : record ? '草稿已保存' : '待核查') + '</span></a>';
      }).join('') + '</section></main>';
    document.getElementById('change-reviewer').addEventListener('click', function () { dialog.showModal(); });
  }
  function navigate() {
    if (!state.reviewer) { dialog.showModal(); return; }
    const match = /^#task=(\d+)$/.exec(location.hash);
    if (!match) { renderHome(); return; }
    const item = Number(match[1]);
    if (item < 1 || item > bundle.tasks.length) { fail('任务序号不在这40题清单中'); return; }
    currentItem = item;
    content.innerHTML = '<main class="page"><p>正在打开任务 ' + item + '…</p></main>';
    if (loader) loader.remove();
    loader = document.createElement('script');
    loader.src = 'materials/' + String(item).padStart(2,'0') + '.js';
    loader.onerror = function () { fail('材料读取失败。请先完整解压压缩包，再打开“打开审核.html”。'); };
    document.body.appendChild(loader);
  }
  window.showReviewTask = function (page) {
    if (page.item !== currentItem) return;
    content.innerHTML = page.html;
    const form = content.querySelector('form');
    const saved = state.records[currentItem];
    if (saved && saved.submitted_at) {
      const panel = document.createElement('div');
      const label = document.createElement('p'); label.textContent = '已保存：' + bundle.decisions[saved.decision];
      const note = document.createElement('p'); note.textContent = saved.note;
      panel.append(label, note); form.replaceWith(panel);
    } else if (saved) {
      if (saved.decision) form.querySelector('[value="' + saved.decision + '"]').checked = true;
      form.querySelector('[name=note]').value = saved.note;
    }
    window.initializeReviewImages();
    window.scrollTo(0, 0);
  };
  function formRecord(form) {
    const old = state.records[currentItem];
    if (old && old.submitted_at) throw new Error('首次提交已保存，不能覆盖');
    const now = new Date().toISOString();
    return { item: currentItem, task_id: bundle.tasks[currentItem-1].task_id,
      decision: form.querySelector('[name=decision]:checked')?.value || '', note: form.querySelector('[name=note]').value,
      started_at: old?.started_at || now, updated_at: now, submitted_at: null };
  }
  content.addEventListener('input', function (event) {
    const form = event.target.closest('form');
    if (!form) return;
    try { state.records[currentItem] = validateRecord(formRecord(form)); persist(); }
    catch (error) { fail(error.message); }
  });
  content.addEventListener('submit', function (event) {
    event.preventDefault();
    try {
      const record = formRecord(event.target);
      if (!record.decision || (record.decision !== 'pass' && !record.note.trim())) throw new Error('请选择判断；有问题或无法判断时请填写原因');
      record.submitted_at = new Date().toISOString();
      state.records[currentItem] = validateRecord(record);
      if (persist()) location.hash = currentItem < bundle.tasks.length ? 'task=' + (currentItem+1) : '';
    } catch (error) { fail(error.message); }
  });
  document.getElementById('reviewer-form').addEventListener('submit', function (event) {
    event.preventDefault();
    try { chooseReviewer(document.getElementById('reviewer-id').value.trim()); }
    catch (error) { fail(error.message); }
  });
  document.getElementById('export').addEventListener('click', function () {
    if (!state.reviewer) { fail('请先填写审核者编号'); return; }
    const payload = { protocol: bundle.protocol, package_id: bundle.package_id, reviewer: state.reviewer,
      sample: bundle.sample, sample_tasks: bundle.tasks, exported_at: new Date().toISOString(), records: Object.values(state.records) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = '质量核查-' + state.reviewer + '-' + new Date().toISOString().replace(/[:.]/g,'-') + '.json';
    link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.getElementById('import').addEventListener('change', async function (event) {
    try {
      const file = event.target.files[0]; if (!file) return;
      if (file.size > 2000000) throw new Error('记录文件过大，请选择本包导出的JSON文件');
      const data = JSON.parse(await file.text());
      if (data.package_id !== bundle.package_id || data.protocol !== bundle.protocol || !validReviewer(data.reviewer) || !Array.isArray(data.records)) throw new Error('记录不属于当前审核包');
      const next = storedState(data.reviewer);
      const seen = new Set();
      data.records.forEach(value => {
        const record = validateRecord(value);
        if (seen.has(record.item)) throw new Error('导入记录存在重复题号');
        seen.add(record.item);
        const old = next.records[record.item];
        if (old?.submitted_at && JSON.stringify(old) !== JSON.stringify(record)) throw new Error('第' + record.item + '题与本机首次提交不同，未覆盖任何记录');
        if (!old || old.updated_at <= record.updated_at) next.records[record.item] = record;
      });
      state = next;
      if (persist()) {
        dialog.close(); document.getElementById('reviewer-label').textContent = '审核者：' + state.reviewer;
        warning.hidden = true; location.hash = ''; renderHome(); status.textContent = '记录已导入并保存';
      }
    } catch (error) { fail('导入失败：' + error.message); }
    finally { event.target.value = ''; }
  });
  window.addEventListener('hashchange', navigate);
  window.addEventListener('storage', function(event) {
    if (event.key === prefix + state.reviewer) {
      try { state = storedState(state.reviewer); navigate(); }
      catch(error) { fail(error.message); }
    }
  });
  dialog.addEventListener('cancel', event => { if (!state.reviewer) event.preventDefault(); });
  try {
    const reviewer = localStorage.getItem(prefix + 'active');
    if (reviewer) chooseReviewer(reviewer);
    else dialog.showModal();
  } catch (error) { fail('无法恢复本机记录：' + error.message + '。请检查浏览器设置或导入备份。'); }
})();
