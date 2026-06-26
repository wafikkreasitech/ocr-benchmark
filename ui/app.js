/* OCR Benchmark dashboard — vanilla JS, calm design */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const fmt = (n, d = 3) => (n === null || n === undefined || Number.isNaN(n)) ? "–" : Number(n).toFixed(d);

const LOCAL_TZ = Intl.DateTimeFormat().resolvedOptions().timeZone;

let summaryData = null;
let currentSort = { key: "f1", dir: "asc" };
let modelConfig = null;
let selectedModel = { ocr_version: null, model_type: null };
let historyRuns = [];
let selectedHistoryIds = new Set();

/* ─── API helpers ────────────────────────────────────────── */

async function fetchModels() {
  try {
    const r = await fetch("/api/models");
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

async function fetchSummary() {
  try {
    const r = await fetch("/api/summary");
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

/* ─── Model Selector (card-based) ────────────────────────── */

function initModelSelector(data) {
  modelConfig = data;
  selectedModel = { ...data.current };
  const grid = $("#model-grid");
  grid.innerHTML = "";

  for (const [ver, info] of Object.entries(data.available)) {
    const card = document.createElement("div");
    card.className = "model-card" + (ver === selectedModel.ocr_version ? " selected" : "");
    card.dataset.version = ver;

    const chips = info.model_types.map(t =>
      `<span class="size-chip${t === selectedModel.model_type && ver === selectedModel.ocr_version ? " active" : ""}" data-type="${t}">${t}</span>`
    ).join("");

    card.innerHTML = `
      <div class="model-card-head">
        <span class="model-card-name">${ver}</span>
        ${ver === "PP-OCRv6" ? '<span class="model-card-badge">latest</span>' : ""}
      </div>
      <div class="model-card-desc">${info.desc}</div>
      <div class="model-card-sizes">${chips}</div>
    `;
    grid.appendChild(card);
  }

  grid.addEventListener("click", (e) => {
    const chip = e.target.closest(".size-chip");
    const card = e.target.closest(".model-card");
    if (!card) return;

    const ver = card.dataset.version;

    if (chip) {
      selectedModel.ocr_version = ver;
      selectedModel.model_type = chip.dataset.type;
    } else {
      selectedModel.ocr_version = ver;
      const firstChip = card.querySelector(".size-chip");
      if (firstChip) selectedModel.model_type = firstChip.dataset.type;
    }

    // Update UI
    $$(".model-card").forEach(c => c.classList.toggle("selected", c.dataset.version === selectedModel.ocr_version));
    $$(".model-card").forEach(c => {
      c.querySelectorAll(".size-chip").forEach(ch => {
        ch.classList.toggle("active",
          c.dataset.version === selectedModel.ocr_version && ch.dataset.type === selectedModel.model_type
        );
      });
    });

    // Update badge
    updateModelBadge();
  });

  updateModelBadge();
}

function updateModelBadge() {
  const badge = $("#model-badge-text");
  if (badge && selectedModel.ocr_version) {
    badge.textContent = `${selectedModel.ocr_version} · ${selectedModel.model_type}`;
  }
}

/* ─── Status ─────────────────────────────────────────────── */

function setStatus(msg) {
  const el = $("#status");
  if (el) el.textContent = msg;
}

/* ─── Score Ring ─────────────────────────────────────────── */

function updateRing(f1) {
  const circ = 2 * Math.PI * 50; // r=50
  const offset = circ * (1 - (f1 || 0));
  const fill = $("#ring-fill");
  if (fill) fill.style.strokeDashoffset = offset;
  const val = $("#ring-value");
  if (val) val.textContent = fmt(f1);
  const sub = $("#ring-sub");
  if (sub) sub.textContent = `P ${fmt(summaryData?.overall?.detection_precision)} · R ${fmt(summaryData?.overall?.detection_recall)}`;
}

/* ─── Render Overview Metrics ────────────────────────────── */

function renderOverall(o) {
  updateRing(o.detection_f1);

  $("#m-cer").textContent = fmt(o.cer_mean);
  $("#m-wer").textContent = fmt(o.wer_mean);
  $("#m-pr").textContent = `${fmt(o.detection_precision, 2)} / ${fmt(o.detection_recall, 2)}`;
  $("#m-pr-hint").textContent = "precision · recall";
  $("#m-imgs").textContent = o.n_images;
  $("#m-lines-hint").textContent = `${o.n_lines} GT lines · ${o.n_categories} categories`;

  // Bars — CER/WER max at 1.0
  setBar("bar-cer", o.cer_mean);
  setBar("bar-wer", o.wer_mean);

  const correctorOn = o.corrector_enabled === true;
  if (correctorOn) {
    $("#m-cer-c").textContent = fmt(o.cer_corrected_mean);
    $("#m-wer-c").textContent = fmt(o.wer_corrected_mean);
    setBar("bar-cer-c", o.cer_corrected_mean);
    setBar("bar-wer-c", o.wer_corrected_mean);
    $("#corrector-banner").classList.remove("hidden");
    $("#cb-state").textContent = "ON";
    $("#cb-sub").textContent = "SymSpell + KBBI post-processing active";
    const dCER = (o.cer_corrected_mean ?? 0) - (o.cer_mean ?? 0);
    const el = $("#cb-delta");
    if (Math.abs(dCER) < 1e-4) {
      el.textContent = `CER Δ ${dCER >= 0 ? "+" : ""}${dCER.toFixed(4)} (neutral)`;
      el.className = "banner-delta delta-zero";
    } else if (dCER > 0) {
      el.textContent = `CER Δ +${dCER.toFixed(4)} (worse)`;
      el.className = "banner-delta delta-up";
    } else {
      el.textContent = `CER Δ ${dCER.toFixed(4)} (better)`;
      el.className = "banner-delta delta-down";
    }
  } else {
    $("#m-cer-c").textContent = "off";
    $("#m-wer-c").textContent = "off";
    setBar("bar-cer-c", 0);
    setBar("bar-wer-c", 0);
    $("#corrector-banner").classList.remove("hidden");
    $("#cb-state").textContent = "OFF";
    $("#cb-sub").textContent = "Enable in .env (ENABLE_SYMSPELL_CORRECTION=true)";
    $("#cb-delta").textContent = "–";
    $("#cb-delta").className = "banner-delta delta-zero";
  }
}

function setBar(id, val) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.min(100, (val || 0) * 100).toFixed(1)}%`;
}

/* ─── Category Bar Chart ─────────────────────────────────── */

function renderChart(cats) {
  const area = $("#chart-area");
  if (!cats || !cats.length) {
    area.innerHTML = '<div class="muted">run a benchmark to see results</div>';
    return;
  }

  const maxF1 = 1;
  const rows = cats.map(c => {
    const f1 = c.detection?.f1 ?? 0;
    const cer = c.cer_mean ?? 0;
    const wer = c.wer_mean ?? 0;
    return `<div class="chart-row">
      <div class="chart-cat" data-cat="${escapeHtml(c.category)}">${escapeHtml(c.category)}</div>
      <div class="chart-bars">
        <div class="chart-bar-row">
          <span class="chart-bar-label">F1</span>
          <div class="chart-bar-bg"><div class="chart-bar-fill f1" style="width:${(f1/maxF1*100).toFixed(1)}%"></div></div>
        </div>
        <div class="chart-bar-row">
          <span class="chart-bar-label">CER</span>
          <div class="chart-bar-bg"><div class="chart-bar-fill cer" style="width:${Math.min(100, cer*100).toFixed(1)}%"></div></div>
        </div>
        <div class="chart-bar-row">
          <span class="chart-bar-label">WER</span>
          <div class="chart-bar-bg"><div class="chart-bar-fill wer" style="width:${Math.min(100, wer*100).toFixed(1)}%"></div></div>
        </div>
      </div>
      <div class="chart-f1-val">${fmt(f1)}</div>
    </div>`;
  });
  area.innerHTML = rows.join("");

  area.querySelectorAll(".chart-cat").forEach(el => {
    el.addEventListener("click", () => openCategory(el.dataset.cat));
  });
}

/* ─── Category Table ─────────────────────────────────────── */

function renderTable(cats) {
  const tbody = $("#cat-tbody");
  tbody.innerHTML = "";
  const rows = cats.map((c) => ({
    category: c.category,
    n_images: c.n_images,
    n_lines: c.n_lines,
    f1: c.detection?.f1 ?? 0,
    cer: c.cer_mean,
    wer: c.wer_mean,
    conf: c.mean_confidence,
    ms: c.mean_ms_per_image,
  }));
  const { key, dir } = currentSort;
  rows.sort((a, b) => {
    const va = a[key] ?? -Infinity;
    const vb = b[key] ?? -Infinity;
    if (typeof va === "string") return dir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
    return dir === "asc" ? va - vb : vb - va;
  });
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.dataset.category = r.category;
    tr.innerHTML = `
      <td>${escapeHtml(r.category)}</td>
      <td class="num">${r.n_images}</td>
      <td class="num">${r.n_lines}</td>
      <td class="num"><span class="mini-bar f1" style="width:${Math.round(r.f1 * 30)}px"></span>${fmt(r.f1)}</td>
      <td class="num"><span class="mini-bar cer" style="width:${Math.round(r.cer * 30)}px"></span>${fmt(r.cer)}</td>
      <td class="num">${fmt(r.wer)}</td>
      <td class="num">${fmt(r.conf, 2)}</td>
      <td class="num">${fmt(r.ms, 0)}</td>
      <td class="num muted">›</td>
    `;
    tr.addEventListener("click", () => openCategory(r.category));
    tbody.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ─── Category Detail ────────────────────────────────────── */

async function openCategory(category) {
  setStatus(`loading ${category}…`);
  const r = await fetch(`/api/results/${encodeURIComponent(category)}`);
  if (!r.ok) { setStatus("failed to load category"); return; }
  const data = await r.json();
  const images = data.images || [];
  if (!images.length) { setStatus("no images in category"); return; }
  showImageDetail(category, images[0], data.summary);
  $("#detail").classList.remove("hidden");
  $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  window.__categoryData = data;
  window.__currentImageIdx = 0;
  setStatus(`${category} · ${images.length} images`);
}

function showImageDetail(category, img, summary) {
  $("#detail-title").textContent = img.image;
  $("#detail-sub").textContent = `${category} · F1 ${fmt(summary.detection?.f1)}`;
  $("#d-gt").textContent = img.n_gt;
  $("#d-pr").textContent = img.n_pred;
  $("#d-det").textContent = `${img.detection.tp} / ${img.detection.fp} / ${img.detection.fn}`;
  $("#d-f1").textContent = fmt(_f1(img.detection));
  $("#d-cer").textContent = img.matched_cer_mean === null ? "–" : fmt(img.matched_cer_mean);
  $("#d-cer-c").textContent = img.matched_cer_corrected_mean == null ? "off" : fmt(img.matched_cer_corrected_mean);
  $("#d-jcer").textContent = fmt(img.joined_cer);
  $("#d-jcer-c").textContent = img.joined_cer_corrected == null ? "off" : fmt(img.joined_cer_corrected);
  $("#d-conf").textContent = img.mean_confidence === null ? "–" : fmt(img.mean_confidence, 2);
  $("#d-ms").textContent = `${Math.round(img.elapsed_ms)} ms`;

  const cs = img.correction_status || {};
  const csEl = $("#d-cstatus");
  if (Object.keys(cs).length) {
    csEl.textContent = `${cs.unchanged || 0} ok · ${cs.corrected || 0} fixed · ${cs.not_found || 0} miss`;
  } else {
    csEl.textContent = "off";
  }

  const imgEl = $("#detail-img");
  imgEl.src = `/api/image/${encodeURIComponent(category)}/${encodeURIComponent(img.image)}`;
  imgEl.onload = () => drawOverlay(img);
  if (imgEl.complete) drawOverlay(img);

  const samples = img.overlays || [];
  const ul = $("#samples");
  ul.innerHTML = "";
  const correctorOn = img.correction_enabled === true;
  for (const o of samples) {
    const li = document.createElement("li");
    if (o.status === "matched") {
      const ok = _normalize(o.gt_text) === _normalize(o.pr_text);
      const corrected = o.pr_text_corrected;
      const correctedOk = corrected !== undefined && _normalize(o.gt_text) === _normalize(corrected);
      let cls = ok ? "match" : "diff";
      if (correctedOk && !ok) cls = "match";
      if (!correctedOk && ok) cls = "diff";
      const mark = correctedOk && !ok ? "✓*" : (ok ? "✓" : "≠");
      const correctedLine = (correctorOn && corrected !== undefined && corrected !== o.pr_text)
        ? `<span><span class="lbl">FIX</span><span class="pr" style="color:var(--accent)">${escapeHtml(corrected)}</span></span>`
        : "";
      li.className = cls;
      const cerVal = o.line_cer != null ? o.line_cer : null;
      const cerLabel = cerVal !== null
        ? `<span class="lbl">CER</span><span class="line-cer ${cerVal < 0.01 ? "cer-ok" : cerVal < 0.1 ? "cer-low" : "cer-high"}">${(cerVal * 100).toFixed(0)}%</span>`
        : "";
      li.innerHTML = `
        <span class="mark">${mark}</span>
        <span>
          <span><span class="lbl">GT</span><span class="gt">${escapeHtml(o.gt_text || "")}</span></span>
          <span><span class="lbl">PR</span><span class="pr">${escapeHtml(o.pr_text || "")}${o.pr_score ? ` · conf ${(o.pr_score*100).toFixed(0)}%` : ""}</span></span>
          ${cerLabel}
          ${correctedLine}
        </span>`;
    } else if (o.status === "missed") {
      li.className = "miss";
      li.innerHTML = `<span class="mark">✗</span><span><span class="lbl">MISSED</span><span class="gt">${escapeHtml(o.gt_text || "")}</span></span>`;
    } else if (o.status === "spurious") {
      const corrected = o.pr_text_corrected;
      const correctedLine = (correctorOn && corrected !== undefined && corrected !== o.pr_text)
        ? `<span><span class="lbl">FIX</span><span class="pr" style="color:var(--accent)">${escapeHtml(corrected)}</span></span>`
        : "";
      li.className = "spur";
      li.innerHTML = `<span class="mark">+</span><span><span class="lbl">EXTRA</span><span class="pr">${escapeHtml(o.pr_text || "")}</span></span>${correctedLine ? `<span></span>` : ""}`;
      if (correctedLine) li.innerHTML += correctedLine;
    }
    ul.appendChild(li);
  }
}

function _normalize(s) { return (s || "").normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim(); }
function _f1(d) {
  const p = d.tp / (d.tp + d.fp) || 0;
  const r = d.tp / (d.tp + d.fn) || 0;
  return 2 * p * r / (p + r) || 0;
}

function drawOverlay(img) {
  const svg = $("#overlay");
  const imgEl = $("#detail-img");
  const w = imgEl.clientWidth, h = imgEl.clientHeight;
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth || w} ${img.naturalHeight || h}`);
  svg.innerHTML = "";
  if (!img.overlays) return;
  for (const o of img.overlays) {
    const poly = o.gt_polygon || o.pr_polygon;
    if (!poly) continue;
    const pts = poly.map((p) => `${p[0]},${p[1]}`).join(" ");
    const cls = o.status === "matched" ? (o.gt_polygon && o.pr_polygon ? "box-pr" : "box-gt")
              : o.status === "missed" ? "box-missed" : "box-spurious";
    const el = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    el.setAttribute("points", pts);
    el.setAttribute("class", cls);
    svg.appendChild(el);
  }
}

/* ─── Run Benchmark ──────────────────────────────────────── */

async function runBenchmark(force = false) {
  const btn = $("#btn-run");
  btn.disabled = true;
  setStatus(force ? "restarting…" : "starting…");
  try {
    const params = new URLSearchParams();
    if (selectedModel.ocr_version) params.set("ocr_version", selectedModel.ocr_version);
    if (selectedModel.model_type) params.set("model_type", selectedModel.model_type);
    if (force) params.set("force", "true");
    const url = "/api/run" + (params.toString() ? "?" + params : "");
    const r = await fetch(url, { method: "POST" });
    if (!r.ok) throw new Error("run failed");
    const started = await r.json();
    if (!started.started) {
      setStatus("already running");
      $("#progress-panel").classList.remove("hidden");
    } else {
      $("#progress-panel").classList.remove("hidden");
      renderProgress({ running: true, total: 0, completed: [], current: null });
    }
    await pollProgress();
  } catch (e) {
    setStatus(`error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

async function pollProgress() {
  while (true) {
    const r = await fetch("/api/progress");
    if (!r.ok) break;
    const p = await r.json();
    renderProgress(p);
    if (p.stale) {
      setStatus("run looks stuck — restart below");
      return; // leave panel visible with the stalled notice
    }
    if (!p.running) break;
    await loadAndRender();
    await new Promise((res) => setTimeout(res, 800));
  }
  setStatus("done");
  $("#progress-panel").classList.add("hidden");
  await loadAndRender();
}

function renderProgress(p) {
  const panel = $("#progress-panel");
  const fill = $("#progress-fill");
  const total = p.total || 0;
  const done = (p.completed || []).length;
  const inFlight = p.current ? 1 : 0;
  const fraction = total ? (done + inFlight * 0.5) / total : 0;
  fill.style.width = `${Math.min(100, fraction * 100).toFixed(1)}%`;

  panel.classList.toggle("stalled", !!p.stale);
  renderStaleNotice(p);

  $("#progress-summary").textContent = p.stale
    ? `stalled at ${done} / ${total} categories`
    : total
      ? `${done} / ${total} categories${inFlight ? " · running…" : ""}`
      : "preparing…";

  if (p.current) {
    const cur = p.current;
    const pctImg = cur.total_images ? `${Math.round((cur.done_images / cur.total_images) * 100)}%` : "0%";
    $("#progress-current").textContent = `${cur.name} · ${cur.done_images}/${cur.total_images} (${pctImg})`;
  } else {
    $("#progress-current").textContent = done >= total && total ? "finalizing…" : "waiting…";
  }

  const avg = (p.completed || []).reduce((a, c) => a + (c.elapsed_s || 0), 0) / Math.max(1, done);
  const remaining = total - done - inFlight;
  const eta = remaining > 0 && avg > 0 ? Math.round(remaining * avg + (inFlight ? avg * 0.5 : 0)) : 0;
  $("#progress-eta").textContent = eta > 0 ? `ETA ≈ ${fmtDuration(eta)}` : "";
  $("#progress-elapsed").textContent = p.started_at ? `started ${fmtDate(p.started_at)}` : "";

  const list = $("#progress-list");
  list.innerHTML = "";
  for (const c of (p.completed || [])) {
    const li = document.createElement("li");
    li.className = "prog-done";
    li.textContent = `✓ ${c.name} · ${c.elapsed_s.toFixed(1)}s`;
    list.appendChild(li);
  }
  if (p.current) {
    const li = document.createElement("li");
    li.className = "prog-now";
    li.textContent = `… ${p.current.name} · ${p.current.done_images}/${p.current.total_images}`;
    list.appendChild(li);
  }
}

function renderStaleNotice(p) {
  let notice = $("#progress-stale");
  if (!p.stale) { if (notice) notice.remove(); return; }
  if (!notice) {
    notice = document.createElement("div");
    notice.id = "progress-stale";
    notice.className = "stale-notice";
    notice.innerHTML = `
      <span>This run hasn't reported progress in a while — it likely stalled. Partial results are kept.</span>
      <button id="btn-restart" class="btn-restart">Restart run</button>`;
    $("#progress-panel").appendChild(notice);
    notice.querySelector("#btn-restart").addEventListener("click", () => runBenchmark(true));
  }
}

function fmtDuration(s) {
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s % 60).toString().padStart(2, "0")}s`;
}

function fmtDate(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const stamp = new Intl.DateTimeFormat(undefined, {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false, timeZone: LOCAL_TZ,
  }).format(d);
  const tzShort = new Intl.DateTimeFormat(undefined, {
    timeZone: LOCAL_TZ, timeZoneName: "short",
  }).formatToParts(d).find((p) => p.type === "timeZoneName")?.value || "";
  const ageS = (Date.now() - d.getTime()) / 1000;
  let rel = "";
  if (ageS < 60) rel = " (now)";
  else if (ageS < 3600) rel = ` (${Math.floor(ageS / 60)}m ago)`;
  else if (ageS < 86400) rel = ` (${Math.floor(ageS / 3600)}h ago)`;
  return `${stamp}${tzShort ? " " + tzShort : ""}${rel}`;
}

function renderLastRun(iso) {
  const el = $("#last-run");
  el.textContent = iso ? fmtDate(iso) : "–";
  const ts = iso ? Date.parse(iso) : NaN;
  if (ts && Date.now() - ts < 5 * 60 * 1000) el.classList.add("fresh");
  else el.classList.remove("fresh");
}

/* ─── History ─────────────────────────────────────────────── */

async function loadHistory() {
  try {
    const r = await fetch("/api/history");
    if (!r.ok) return;
    const data = await r.json();
    historyRuns = data.runs || [];
    renderHistory();
  } catch {}
}

function renderHistory() {
  const tbody = $("#history-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const countEl = $("#history-count");
  if (countEl) countEl.textContent = `${historyRuns.length} runs`;

  for (const run of historyRuns) {
    const tr = document.createElement("tr");
    tr.dataset.id = run.id;
    const checked = selectedHistoryIds.has(run.id) ? "checked" : "";
    const ts = run.timestamp ? fmtDateShort(run.timestamp) : run.id;
    tr.innerHTML = `
      <td><input type="checkbox" data-id="${run.id}" ${checked} /></td>
      <td>${ts}</td>
      <td><span class="history-model">${run.ocr_version || "?"} · ${run.model_type || "?"}</span></td>
      <td class="num history-f1">${fmt(run.f1)}</td>
      <td class="num">${fmt(run.cer)}</td>
      <td class="num">${fmt(run.wer)}</td>
      <td class="num">${run.n_images || 0}</td>
      <td class="num">${run.total_elapsed_s ? fmtDuration(Math.round(run.total_elapsed_s)) : "–"}</td>
      <td class="num muted">${run.timestamp ? fmtDateAge(run.timestamp) : "–"}</td>
      <td class="num muted">›</td>
    `;
    tbody.appendChild(tr);
  }

  // Checkbox handlers (stop row-click when toggling)
  tbody.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      if (cb.checked) selectedHistoryIds.add(cb.dataset.id);
      else selectedHistoryIds.delete(cb.dataset.id);
      updateCompareBtn();
    });
  });

  // Row click → open run detail
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => openRunDetail(tr.dataset.id));
  });

  updateCompareBtn();
}

function updateCompareBtn() {
  const btn = $("#btn-compare");
  if (btn) btn.disabled = selectedHistoryIds.size < 2;
}

function fmtDateShort(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false, timeZone: LOCAL_TZ,
  }).format(d);
}

function fmtDateAge(iso) {
  const ageS = (Date.now() - new Date(iso).getTime()) / 1000;
  if (ageS < 60) return "now";
  if (ageS < 3600) return `${Math.floor(ageS / 60)}m ago`;
  if (ageS < 86400) return `${Math.floor(ageS / 3600)}h ago`;
  return `${Math.floor(ageS / 86400)}d ago`;
}

async function showComparison() {
  const ids = [...selectedHistoryIds];
  const runs = [];
  for (const id of ids) {
    try {
      const r = await fetch(`/api/history/${encodeURIComponent(id)}`);
      if (r.ok) runs.push(await r.json());
    } catch {}
  }
  if (runs.length < 2) return;

  const panel = $("#compare-panel");
  const content = $("#compare-content");
  panel.classList.remove("hidden");

  // Overall comparison table
  const metrics = [
    { key: "detection_f1", label: "F1", fmt: fmt, better: "up" },
    { key: "cer_mean", label: "CER", fmt: fmt, better: "down" },
    { key: "wer_mean", label: "WER", fmt: fmt, better: "down" },
    { key: "n_images", label: "Images", fmt: (v) => v, better: null },
    { key: "total_elapsed_s", label: "Time (s)", fmt: (v) => v?.toFixed(1), better: "down" },
  ];

  // Find best/worst for each metric
  const best = {};
  const worst = {};
  for (const m of metrics) {
    const vals = runs.map(r => r.overall?.[m.key] ?? 0);
    if (m.better === "up") { best[m.key] = Math.max(...vals); worst[m.key] = Math.min(...vals); }
    else if (m.better === "down") { best[m.key] = Math.min(...vals.filter(v => v > 0)); worst[m.key] = Math.max(...vals); }
  }

  let html = `<table class="compare-table"><thead><tr><th>Metric</th>`;
  for (const r of runs) {
    html += `<th class="num">${r.ocr_version || "?"} ${r.model_type || "?"}<br><span style="font-weight:400;color:var(--text-muted)">${fmtDateShort(r.timestamp)}</span></th>`;
  }
  html += `<th class="num">Δ</th></tr></thead><tbody>`;

  for (const m of metrics) {
    html += `<tr><td>${m.label}</td>`;
    const vals = runs.map(r => r.overall?.[m.key] ?? null);
    for (const v of vals) {
      const isBest = m.better && v !== null && v === best[m.key];
      const isWorse = m.better && v !== null && v === worst[m.key] && best[m.key] !== worst[m.key];
      const cls = isBest ? "best" : isWorse ? "worse" : "";
      html += `<td class="num ${cls}">${v !== null ? m.fmt(v) : "–"}</td>`;
    }
    // Delta column (first vs last)
    if (vals.length >= 2 && vals[0] !== null && vals[1] !== null) {
      const d = vals[0] - vals[1];
      const pct = vals[1] !== 0 ? ((d / Math.abs(vals[1])) * 100).toFixed(1) : "–";
      const arrow = d > 0 ? (m.better === "up" ? "down" : "up") : (m.better === "up" ? "up" : "down");
      const cls = (m.better === "up" && d > 0) || (m.better === "down" && d < 0) ? "down" : "up";
      html += `<td class="num"><span class="compare-delta ${cls}">${d > 0 ? "+" : ""}${pct}%</span></td>`;
    } else {
      html += `<td class="num">–</td>`;
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;

  // Per-category comparison
  const cats0 = runs[0].per_category || [];
  const cats1 = runs[1].per_category || [];
  if (cats0.length && cats1.length) {
    html += `<div style="margin-top:16px;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:500">Per-category F1</div>`;
    html += `<table class="compare-table" style="margin-top:6px"><thead><tr><th>Category</th><th class="num">${runs[0].model_type}</th><th class="num">${runs[1].model_type}</th><th class="num">Δ</th></tr></thead><tbody>`;
    const catMap1 = Object.fromEntries(cats1.map(c => [c.category, c]));
    for (const c0 of cats0) {
      const c1 = catMap1[c0.category];
      if (!c1) continue;
      const d = (c0.f1 || 0) - (c1.f1 || 0);
      const cls = d > 0.01 ? "down" : d < -0.01 ? "up" : "";
      html += `<tr><td>${escapeHtml(c0.category)}</td><td class="num">${fmt(c0.f1)}</td><td class="num">${fmt(c1.f1)}</td><td class="num"><span class="compare-delta ${cls}">${d > 0 ? "+" : ""}${d.toFixed(3)}</span></td></tr>`;
    }
    html += `</tbody></table>`;
  }

  content.innerHTML = html;
}

async function openRunDetail(id) {
  const panel = $("#run-detail-panel");
  const content = $("#run-detail-content");
  panel.classList.remove("hidden");
  content.innerHTML = '<div class="muted">loading…</div>';
  panel.scrollIntoView({ behavior: "smooth", block: "start" });

  let run;
  try {
    const r = await fetch(`/api/history/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error("not found");
    run = await r.json();
  } catch {
    content.innerHTML = '<div class="muted">failed to load run</div>';
    return;
  }

  const o = run.overall || {};
  const dur = run.total_elapsed_s ? fmtDuration(Math.round(run.total_elapsed_s)) : "–";
  const cards = [
    ["F1", fmt(o.detection_f1)],
    ["CER", fmt(o.cer_mean)],
    ["WER", fmt(o.wer_mean)],
    ["Precision", fmt(o.detection_precision, 2)],
    ["Recall", fmt(o.detection_recall, 2)],
    ["Images", o.n_images ?? "–"],
    ["GT lines", o.n_lines ?? "–"],
    ["Duration", dur],
  ].map(([k, v]) => `<div class="rd-card"><div class="rd-k">${k}</div><div class="rd-v">${v}</div></div>`).join("");

  $("#run-detail-title").textContent = `${run.ocr_version || "?"} · ${run.model_type || "?"}`;
  $("#run-detail-sub").textContent = `${run.timestamp ? fmtDate(run.timestamp) : run.id} · corrector ${run.corrector_enabled ? "ON" : "OFF"}`;

  const cats = run.per_category || [];
  let catRows = cats.map(c => `
    <tr>
      <td>${escapeHtml(c.category)}</td>
      <td class="num">${c.n_images ?? "–"}</td>
      <td class="num"><span class="mini-bar f1" style="width:${Math.round((c.f1 || 0) * 30)}px"></span>${fmt(c.f1)}</td>
      <td class="num">${fmt(c.cer)}</td>
      <td class="num">${fmt(c.wer)}</td>
      <td class="num">${fmt(c.mean_conf, 2)}</td>
      <td class="num">${fmt(c.ms_per_img, 0)}</td>
    </tr>`).join("");

  content.innerHTML = `
    <div class="rd-cards">${cards}</div>
    <table class="compare-table" style="margin-top:16px">
      <thead><tr><th>Category</th><th class="num">Imgs</th><th class="num">F1</th><th class="num">CER</th><th class="num">WER</th><th class="num">Conf</th><th class="num">ms/img</th></tr></thead>
      <tbody>${catRows || '<tr><td colspan="7" class="muted">no per-category data</td></tr>'}</tbody>
    </table>`;
}

/* ─── Load & Render ──────────────────────────────────────── */

async function loadAndRender() {
  const data = await fetchSummary();
  if (!data) {
    // No summary yet — check if benchmark is running
    try {
      const r = await fetch("/api/progress");
      if (r.ok) {
        const p = await r.json();
        if (p.running) {
          $("#progress-panel").classList.remove("hidden");
          renderProgress(p);
          setStatus("benchmark running — waiting for results…");
        }
      }
    } catch {}
    return;
  }
  summaryData = data;
  renderOverall(data.overall);
  renderChart(data.per_category);
  renderTable(data.per_category);
  renderLastRun(data.overall.last_run);
  const ov = data.overall;
  const ml = $("#model-label");
  if (ml && ov.ocr_version) {
    ml.textContent = `${ov.ocr_version} · ${ov.model_type || ""}`.trim();
  }
  const mb = $("#model-badge-text");
  if (mb && ov.ocr_version) {
    mb.textContent = `${ov.ocr_version} · ${ov.model_type || ""}`;
  }
  setStatus(`loaded · ${data.overall.n_images} imgs · ${data.overall.n_lines} lines`);
  loadHistory();
}

/* ─── Init ───────────────────────────────────────────────── */

fetchModels().then((d) => { if (d) initModelSelector(d); });

$$("thead th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (currentSort.key === key) currentSort.dir = currentSort.dir === "asc" ? "desc" : "asc";
    else { currentSort.key = key; currentSort.dir = "asc"; }
    if (summaryData) renderTable(summaryData.per_category);
  });
});

$("#btn-run").addEventListener("click", runBenchmark);
$("#back").addEventListener("click", () => {
  $("#detail").classList.add("hidden");
  setStatus("idle");
});
$("#btn-compare").addEventListener("click", showComparison);
$("#btn-compare-close").addEventListener("click", () => {
  $("#compare-panel").classList.add("hidden");
});
$("#btn-run-detail-close").addEventListener("click", () => {
  $("#run-detail-panel").classList.add("hidden");
});

loadAndRender().then(() => {
  // If no summary, start polling progress in case a run is active
  if (!summaryData) pollProgressOnLoad();
});

async function pollProgressOnLoad() {
  try {
    const r = await fetch("/api/progress");
    if (!r.ok) return;
    const p = await r.json();
    if (p.running) {
      $("#progress-panel").classList.remove("hidden");
      renderProgress(p);
      setStatus("benchmark running…");
      await pollProgress();
      await loadAndRender();
    }
  } catch {}
}

loadHistory();
