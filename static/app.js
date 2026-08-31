/* Paper Edit -- browser side.
   Vanilla on purpose: no build step, so this runs the moment the server does.
   The player never re-encodes; it seeks over deleted ranges (see seekPastCuts). */

const $ = s => document.querySelector(s);
// A 401 means the session lapsed -- most likely the desktop restarted. Go
// straight to the sign-in page rather than showing a stack of failures.
const signedOut = () => { location.href = '/login'; };

const api = async (path, opts) => {
  const r = await fetch('/api' + path, opts);
  if (r.status === 401) { signedOut(); throw new Error('signed out'); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.status === 204 ? null : r.json();
};
const jpost = (p, body) => api(p, {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(body || {})
});
const fmt = s => {
  if (s == null || isNaN(s)) return '-';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
};
const toast = msg => {
  const t = document.createElement('div');
  t.className = 'toast'; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3200);
};

let state = {pid: null, words: [], plan: null, sel: new Set(), undo: [], presets: {},
             duration: 0};

/* ---------------------------------------------------------------- projects */

async function loadProjects() {
  const list = await api('/projects');
  const box = $('#projects');
  if (!list.length) { box.textContent = 'Nothing yet. Upload something above.'; return; }
  box.innerHTML = '';
  for (const p of list) {
    const el = document.createElement('div');
    el.className = 'proj';
    const st = p.status === 'ready' ? 'ready' : (p.status === 'failed' ? 'failed' : '');
    el.innerHTML = `<b>${p.name}</b>
      <span class="badge ${st}">${p.status}</span>
      <span class="dim small">${p.duration ? fmt(p.duration) : ''}</span>
      <span style="flex:1"></span>
      <button class="danger small" data-del="${p.id}">Delete</button>`;
    el.onclick = e => {
      if (e.target.dataset.del) return;
      openProject(p.id);
    };
    el.querySelector('[data-del]').onclick = async e => {
      e.stopPropagation();
      if (!confirm('Delete "' + p.name + '" and its media?')) return;
      await api('/projects/' + p.id, {method: 'DELETE'});
      loadProjects();
    };
    box.appendChild(el);
  }
}

/* ------------------------------------------------------------------ upload */

const CHUNK = 8 * 1024 * 1024;   // 8 MB: big enough to be fast, small enough to retry

async function uploadResumable(pid, file, onProgress) {
  const name = 'source' + (file.name.match(/\.[a-z0-9]+$/i) || ['.bin'])[0];
  let sent = (await api(`/projects/${pid}/upload?name=${name}`)).received;
  if (sent > file.size) sent = 0;               // different file reusing the slot

  while (sent < file.size) {
    const end = Math.min(sent + CHUNK, file.size);
    const final = end === file.size ? 1 : 0;
    let tries = 0;
    for (;;) {
      try {
        const r = await fetch(
          `/api/projects/${pid}/upload?name=${name}&offset=${sent}&final=${final}`,
          {method: 'PUT', body: file.slice(sent, end)});
        // Retrying a lapsed session just burns the retry budget five times.
        if (r.status === 401) { signedOut(); throw new Error('signed out'); }
        if (!r.ok) throw new Error('chunk failed');
        sent = (await r.json()).received;
        break;
      } catch (err) {
        // A flaky Wi-Fi hop should cost a retry, not the whole upload.
        if (++tries > 5) throw err;
        onProgress(sent / file.size, 'connection dropped, retrying (' + tries + ')');
        await new Promise(r => setTimeout(r, 1500 * tries));
        sent = (await api(`/projects/${pid}/upload?name=${name}`)).received;
      }
    }
    onProgress(sent / file.size, fmt(0) && `${(sent / 1e6).toFixed(0)} of ${(file.size / 1e6).toFixed(0)} MB`);
  }
}

$('#createBtn').onclick = async () => {
  const file = $('#newFile').files[0];
  if (!file) { toast('Choose a video or audio file first'); return; }
  const name = $('#newName').value.trim() || file.name.replace(/\.[^.]+$/, '');
  $('#createBtn').disabled = true;
  $('#upRow').classList.remove('hide');
  try {
    const {id} = await jpost('/projects', {name});
    await uploadResumable(id, file, (frac, msg) => {
      $('#upBar').style.width = (frac * 100).toFixed(1) + '%';
      $('#upTxt').textContent = msg;
    });
    $('#upTxt').textContent = 'transcribing...';
    await jpost(`/projects/${id}/ingest`);
    openProject(id);
  } catch (e) {
    toast('Upload failed: ' + e.message);
  } finally {
    $('#createBtn').disabled = false;
    $('#upRow').classList.add('hide');
    $('#upBar').style.width = '0';
  }
};

/* -------------------------------------------------------------- job polling */

async function pollJob(jid, onTick) {
  for (;;) {
    const j = await api('/jobs/' + jid);
    onTick(j);
    if (['done', 'failed', 'interrupted'].includes(j.status)) return j;
    await new Promise(r => setTimeout(r, 1200));
  }
}

/* ------------------------------------------------------------------ editor */

async function openProject(pid) {
  state.pid = pid; state.sel.clear(); state.undo = [];
  $('#listView').classList.add('hide');
  $('#editView').classList.remove('hide');
  $('#backBtn').classList.remove('hide');

  const p = await api('/projects/' + pid);
  $('#sOrig').textContent = fmt(p.duration);

  if (p.job && ['queued', 'running'].includes(p.job.status)) {
    $('#jobCard').classList.remove('hide');
    await pollJob(p.job.id, j => {
      $('#jobMsg').textContent = j.message || j.status;
      $('#jobBar').style.width = (j.progress * 100).toFixed(0) + '%';
    });
  }
  const fresh = await api('/projects/' + pid);
  $('#jobCard').classList.toggle('hide', fresh.status === 'ready');
  $('#sOrig').textContent = fmt(fresh.duration);
  // Blank the edit stats until this project's plan arrives. The first open of a
  // project runs a full silence scan, and for those seconds the panel would
  // otherwise still be showing the last project's numbers.
  $('#sEdit').textContent = $('#sCut').textContent = $('#sSeg').textContent = '-';
  if (fresh.status === 'failed') {
    $('#jobMsg').textContent = 'Failed: ' + (fresh.job ? fresh.job.message : '');
    return;
  }
  if (fresh.status !== 'ready') return;

  state.duration = fresh.duration;
  // Restore the project's own dead-air settings rather than the last project's.
  $('#silToggle').checked = !!fresh.silence_on;
  $('#silKeep').value = fresh.silence_keep ?? 0.3;
  $('#silMin').value = fresh.silence_min ?? 0.6;
  $('#silKeepVal').textContent = parseFloat($('#silKeep').value).toFixed(2);
  $('#silMinVal').textContent = parseFloat($('#silMin').value).toFixed(2);
  $('#silBody').classList.toggle('hide', !fresh.silence_on);
  $('#sSil').textContent = '0:00';
  $('#fillInfo').textContent = FILL_DEFAULT;
  $('#soundPreset').value = fresh.sound_preset || 'auto';
  $('#capStyle').value = fresh.caption_style || 'off';
  pushSound();
  pushCaptions();
  $('#player').src = '/api/projects/' + pid + '/proxy';
  state.words = await api(`/projects/${pid}/words`);
  renderTranscript();
  await refreshPlan();
}

$('#backBtn').onclick = () => {
  $('#player').pause(); $('#player').removeAttribute('src');
  $('#editView').classList.add('hide');
  $('#backBtn').classList.add('hide');
  $('#listView').classList.remove('hide');
  loadProjects();
};

function renderTranscript() {
  const box = $('#transcript');
  box.innerHTML = '';
  const frag = document.createDocumentFragment();
  state.words.forEach(w => {
    const s = document.createElement('span');
    s.className = 'w' + (w.deleted ? ' del' : '');
    s.textContent = w.text.trim() + ' ';
    s.dataset.idx = w.idx;
    frag.appendChild(s);
  });
  box.appendChild(frag);
  paintSelection();
}

/* click to seek, drag to select a phrase */
let dragging = false, dragStart = null;
$('#transcript').addEventListener('pointerdown', e => {
  const el = e.target.closest('.w'); if (!el) return;
  dragging = true; dragStart = +el.dataset.idx;
  state.sel.clear(); state.sel.add(dragStart); paintSelection();
});
$('#transcript').addEventListener('pointermove', e => {
  if (!dragging) return;
  const el = e.target.closest('.w'); if (!el) return;
  const a = Math.min(dragStart, +el.dataset.idx), b = Math.max(dragStart, +el.dataset.idx);
  state.sel = new Set();
  for (let i = a; i <= b; i++) state.sel.add(i);
  paintSelection();
});
$('#transcript').addEventListener('pointerup', e => {
  const wasDrag = state.sel.size > 1;
  dragging = false;
  const el = e.target.closest('.w');
  if (el && !wasDrag) {                       // a plain click means "play here"
    const w = state.words[+el.dataset.idx];
    if (w) seekSource(w.start);
  }
});
$('#transcript').addEventListener('keydown', e => {
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); setDeleted(true); }
});

function paintSelection() {
  document.querySelectorAll('#transcript .w').forEach(el => {
    el.classList.toggle('sel', state.sel.has(+el.dataset.idx));
  });
  $('#selInfo').textContent = state.sel.size ? state.sel.size + ' words selected' : '';
}

async function setDeleted(deleted) {
  const idx = [...state.sel];
  if (!idx.length) { toast('Select some words first'); return; }
  state.undo.push({idx, deleted: !deleted});
  await jpost(`/projects/${state.pid}/words`, {indices: idx, deleted});
  idx.forEach(i => { const w = state.words.find(x => x.idx === i); if (w) w.deleted = deleted ? 1 : 0; });
  renderTranscript();
  await refreshPlan();
}

$('#delBtn').onclick = () => setDeleted(true);
$('#restoreBtn').onclick = () => setDeleted(false);
$('#undoBtn').onclick = async () => {
  const last = state.undo.pop();
  if (!last) { toast('Nothing to undo'); return; }
  await jpost(`/projects/${state.pid}/words`, {indices: last.idx, deleted: last.deleted});
  last.idx.forEach(i => { const w = state.words.find(x => x.idx === i); if (w) w.deleted = last.deleted ? 1 : 0; });
  renderTranscript();
  await refreshPlan();
};

async function refreshPlan() {
  state.plan = await api(`/projects/${state.pid}/plan`);
  // Measure against the media length, not the last word: trailing room tone is
  // still part of the file, and the two numbers must agree on screen.
  $('#sEdit').textContent = fmt(state.plan.duration);
  $('#sCut').textContent = fmt(Math.max(0, state.duration - state.plan.duration));
  $('#sSeg').textContent = state.plan.cuts.length;
}

/* ------------------------------------------------------------------ player

   The trick that makes editing feel instant: we never re-encode for preview.
   The <video> plays the untouched proxy and we jump the playhead over any
   range the edit removed. Export is the only thing that runs ffmpeg. */

const video = $('#player');

function seekSource(t) {
  video.currentTime = t;
  if (video.paused) video.play().catch(() => {});
}

function nextCutStart(t) {
  if (!state.plan) return null;
  for (const c of state.plan.cuts) if (c.start > t) return c.start;
  return null;
}

function inKeptRange(t) {
  if (!state.plan) return true;
  return state.plan.cuts.some(c => t >= c.start && t < c.end);
}

video.addEventListener('timeupdate', () => {
  const t = video.currentTime;
  if (!inKeptRange(t)) {
    const nxt = nextCutStart(t);
    if (nxt == null) { video.pause(); return; }
    video.currentTime = nxt;
    return;
  }
  highlightWord(t);
});

let lastWordEl = null;
function highlightWord(t) {
  const w = state.words.find(x => !x.deleted && t >= x.start && t < x.end);
  if (!w) return;
  const el = document.querySelector(`#transcript .w[data-idx="${w.idx}"]`);
  if (el === lastWordEl) return;
  if (lastWordEl) lastWordEl.classList.remove('now');
  if (el) {
    el.classList.add('now');
    const box = $('#transcript'), r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
    if (r.top < b.top || r.bottom > b.bottom) el.scrollIntoView({block: 'center'});
  }
  lastWordEl = el;
}

/* ------------------------------------------------------------------ export */

$('#expBtn').onclick = async () => {
  $('#expBtn').disabled = true;
  $('#expRow').classList.remove('hide');
  try {
    const {job} = await jpost(`/projects/${state.pid}/export`, {preset: $('#preset').value});
    const done = await pollJob(job, j => {
      $('#expBar').style.width = (j.progress * 100).toFixed(0) + '%';
      $('#expTxt').textContent = j.message || j.status;
    });
    if (done.status === 'done') {
      const a = document.createElement('a');
      a.className = 'dl';
      a.href = `/api/projects/${state.pid}/file/${done.result}`;
      a.textContent = 'Save ' + done.result + '  (' + done.message + ')';
      a.setAttribute('download', done.result);
      $('#downloads').prepend(a);
      toast('Export ready');
    } else {
      toast('Export failed: ' + done.message);
    }
  } catch (e) {
    toast('Export failed: ' + e.message);
  } finally {
    $('#expBtn').disabled = false;
    $('#expRow').classList.add('hide');
    $('#expBar').style.width = '0';
  }
};

/* -------------------------------------------------------------------- init */

(async function init() {
  $('#where').textContent = location.host;
  try {
    const h = await api('/health');
    state.presets = h.presets;
    $('#preset').innerHTML = Object.entries(h.presets)
      .map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  } catch (e) {
    toast('Cannot reach the editor server. Is the desktop awake?');
  }
  loadProjects();
})();

/* ---------------------------------------------------------- dead air removal

   Non-destructive: this only changes how the cut list is derived on the server,
   so turning it off puts every pause straight back. */

async function pushSilence() {
  const on = $('#silToggle').checked;
  $('#silBody').classList.toggle('hide', !on);
  const body = {
    enabled: on,
    keep: parseFloat($('#silKeep').value),
    min_remove: parseFloat($('#silMin').value),
  };
  const r = await jpost(`/projects/${state.pid}/silence`, body);
  $('#silInfo').textContent = on
    ? `${r.pauses_found} pauses found, removing ${fmt(r.seconds_removable)} of dead air`
    : 'Trims long pauses out of the middle, leaving a beat at each end.';
  $('#sSil').textContent = on ? fmt(r.seconds_removable) : '0:00';
  await refreshPlan();
}

$('#silToggle').onchange = pushSilence;
let silTimer = null;
for (const id of ['#silKeep', '#silMin']) {
  $(id).oninput = () => {
    $('#silKeepVal').textContent = parseFloat($('#silKeep').value).toFixed(2);
    $('#silMinVal').textContent = parseFloat($('#silMin').value).toFixed(2);
    // Debounced: dragging a slider should not fire a request per pixel.
    clearTimeout(silTimer);
    silTimer = setTimeout(pushSilence, 350);
  };
}

/* ------------------------------------------------------------- studio sound

   Applied at export only; the source file is never touched. "Auto" measures the
   recording's speech-to-noise ratio and denoises only when there is noise --
   denoising a clean take measurably damages it. */

async function pushSound() {
  const r = await jpost(`/projects/${state.pid}/sound`,
                        {preset: $('#soundPreset').value});
  const snr = r.snr ? ` — measured ${r.snr.toFixed(0)} dB signal-to-noise` : '';
  if (r.preset === 'auto') {
    $('#soundInfo').textContent = r.reason
      ? `Auto: ${r.reason}${snr}`
      : `Auto${snr}`;
  } else if (r.preset === 'off') {
    $('#soundInfo').textContent = 'Audio exported untouched.';
  } else {
    $('#soundInfo').textContent =
      `Using ${r.using}. Levelled to −16 LUFS on export.${snr}`;
  }
}

$('#soundPreset').onchange = pushSound;

/* --------------------------------------------------------------- captions

   Burned in at export. Timings are generated against the EDITED timeline, so
   they stay in step no matter how much has been cut. */

async function pushCaptions() {
  const r = await jpost(`/projects/${state.pid}/captions`,
                        {style: $('#capStyle').value});
  $('#capInfo').textContent = r.style === 'off'
    ? 'Burned in at export, with the spoken word highlighted.'
    : `${r.words} words will be captioned, in step with your edit.`;
}

$('#capStyle').onchange = pushCaptions;

/* ------------------------------------------------------------ filler words

   The transcript-side half of "take out the ums". Whisper writes down only
   7-17% of the fillers a person hears (FINDINGS.md), so the copy says so
   rather than promising a clean take -- dead air removal does the rest.

   Routed through the same undo stack as a hand-made edit, so one click of
   Undo reverses it and nothing about it is a special case. */

const FILL_DEFAULT = $('#fillInfo').textContent;

async function pushFillers(deleted) {
  const btn = deleted ? $('#fillBtn') : $('#fillUndoBtn');
  btn.disabled = true;
  try {
    const r = await jpost(`/projects/${state.pid}/fillers`, {deleted});
    if (!r.marked) {
      $('#fillInfo').textContent = 'No fillers written down in this transcript. '
        + 'Whisper tidies most of them away - use Remove dead air for the gaps they leave.';
      return;
    }
    state.undo.push({idx: r.indices, deleted: !deleted});
    r.indices.forEach(i => {
      const w = state.words.find(x => x.idx === i);
      if (w) w.deleted = deleted ? 1 : 0;
    });
    renderTranscript();
    await refreshPlan();
    $('#fillInfo').textContent = deleted
      ? `${r.marked} filler words taken out. Put back or Undo reverses it.`
      : `${r.marked} filler words put back.`;
  } catch (e) {
    toast('Filler words: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

$('#fillBtn').onclick = () => pushFillers(true);
$('#fillUndoBtn').onclick = () => pushFillers(false);

/* ----------------------------------------------------------------- sign out */

$('#signOutBtn').onclick = async () => {
  await api('/session', {method: 'DELETE'}).catch(() => {});
  location.href = '/login';
};
