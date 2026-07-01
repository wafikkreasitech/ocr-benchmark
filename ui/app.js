/* OCR Benchmark dashboard — vanilla JS, calm design */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const fmt = (n, d = 3) => (n === null || n === undefined || Number.isNaN(n)) ? "–" : Number(n).toFixed(d);

const LOCAL_TZ = Intl.DateTimeFormat().resolvedOptions().timeZone;

let summaryData = null;
let currentSort = { key: "f1", dir: "asc" };
let modelConfig = null;
let selectedModel = { ocr_version: null, model_type: null };
let selectedDataset = { key: null, category: null };  // category = null → "All"
let datasetsCache = [];
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

/* ─── Dataset Selector ─────────────────────────────────────── */

async function fetchDatasets() {
  try {
    const r = await fetch("/api/datasets");
    if (!r.ok) return [];
    const j = await r.json();
    return j.datasets || [];
  } catch { return []; }
}

async function fetchDatasetCategories(key) {
  try {
    const r = await fetch(`/api/datasets/${encodeURIComponent(key)}/categories`);
    if (!r.ok) return [];
    const j = await r.json();
    return j.categories || [];
  } catch { return []; }
}

function updateDatasetBadge() {
  const el = $("#dataset-badge-text");
  if (!el) return;
  const ds = datasetsCache.find(d => d.key === selectedDataset.key);
  el.textContent = ds ? ds.label : (selectedDataset.key || "–");
}

function setDatasetSummary() {
  const el = $("#dataset-summary");
  if (!el) return;
  const ds = datasetsCache.find(d => d.key === selectedDataset.key);
  if (!ds) { el.textContent = "–"; return; }
  el.textContent = `${ds.n_categories} categories · ${ds.n_images} images · ${ds.n_lines.toLocaleString()} GT lines · ${ds.format}`;
}

async function initDatasetSelector() {
  // 1. Pull the registry once.
  datasetsCache = await fetchDatasets();
  const sel = $("#dataset-select");
  const catSel = $("#category-select");
  if (!sel || !catSel) return;

  sel.innerHTML = "";
  for (const ds of datasetsCache) {
    const opt = document.createElement("option");
    opt.value = ds.key;
    opt.textContent = ds.label;
    sel.appendChild(opt);
  }
  // Honour the active dataset from the server (falls back to first).
  const active = datasetsCache.find(d => d.active) || datasetsCache[0];
  if (active) selectedDataset.key = active.key;
  sel.value = selectedDataset.key || "";

  // 2. Populate the Category dropdown for the active dataset.
  await refreshCategories();

  // 3. Update the badge + summary line.
  updateDatasetBadge();
  setDatasetSummary();

  // 4. Listeners.
  sel.addEventListener("change", async () => {
    selectedDataset.key = sel.value;
    selectedDataset.category = null;  // reset on dataset change
    await refreshCategories();
    updateDatasetBadge();
    setDatasetSummary();
    // Fire a global event so the Datasets page (if open) can sync its card.
    window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { key: selectedDataset.key } }));
  });
  catSel.addEventListener("change", () => {
    selectedDataset.category = catSel.value || null;
  });

  // 5. Listen for changes initiated from the Datasets page.
  window.addEventListener("dataset-changed", async (e) => {
    if (!e.detail || e.detail.key === selectedDataset.key) return;
    sel.value = e.detail.key;
    selectedDataset.key = e.detail.key;
    selectedDataset.category = null;
    await refreshCategories();
    updateDatasetBadge();
    setDatasetSummary();
  });
}

async function refreshCategories() {
  const catSel = $("#category-select");
  if (!catSel || !selectedDataset.key) return;
  const cats = await fetchDatasetCategories(selectedDataset.key);
  catSel.innerHTML = "";
  for (const c of cats) {
    const opt = document.createElement("option");
    opt.value = c.name === "All" ? "" : c.name;
    opt.textContent = c.name === "All"
      ? `All · ${c.n_images} images`
      : `${c.name} · ${c.n_images} images`;
    catSel.appendChild(opt);
  }
  catSel.value = "";  // default to All
  selectedDataset.category = null;
}

/* ─── Datasets page ────────────────────────────────────────── */

async function initDatasetsPage() {
  const grid = $("#datasets-grid");
  if (!grid) return;  // not the Datasets page
  datasetsCache = await fetchDatasets();
  renderDatasetCards(datasetsCache);
  // Re-render cards when the dataset changes from any source (incl. main page).
  window.addEventListener("dataset-changed", async () => {
    datasetsCache = await fetchDatasets();
    renderDatasetCards(datasetsCache);
  });
}

function renderDatasetCards(datasets) {
  const grid = $("#datasets-grid");
  if (!grid) return;
  grid.innerHTML = "";
  for (const ds of datasets) {
    const card = document.createElement("div");
    card.className = "dataset-card" + (ds.active ? " is-active" : "");
    card.dataset.key = ds.key;
    card.innerHTML = `
      <div class="dataset-card-head">
        <div>
          <div class="dataset-card-name">${escapeHtml(ds.label)}</div>
          <div class="dataset-card-key">${escapeHtml(ds.key)} · ${escapeHtml(ds.format)}</div>
        </div>
        ${ds.active ? '<span class="dataset-card-badge">Active</span>' : ""}
      </div>
      <div class="dataset-card-stats">
        <div><span>${ds.n_categories}</span><label>categories</label></div>
        <div><span>${ds.n_images}</span><label>images</label></div>
        <div><span>${ds.n_lines.toLocaleString()}</span><label>GT lines</label></div>
      </div>
      <div class="dataset-card-actions">
        <button type="button" class="btn-primary btn-run-dataset" data-key="${escapeHtml(ds.key)}">▶ Run</button>
        <button type="button" class="btn-ghost btn-view-dataset" data-key="${escapeHtml(ds.key)}">📊 View last results</button>
      </div>
      <div class="dataset-card-progress" hidden>
        <div class="progress-track"><div class="progress-fill"></div></div>
        <div class="progress-label muted">idle</div>
      </div>
    `;
    grid.appendChild(card);
  }
  grid.addEventListener("click", (e) => {
    const runBtn = e.target.closest(".btn-run-dataset");
    const viewBtn = e.target.closest(".btn-view-dataset");
    if (runBtn) runFromCard(runBtn.dataset.key);
    else if (viewBtn) viewFromCard(viewBtn.dataset.key);
  });
}

async function runFromCard(key) {
  selectedDataset.key = key;
  selectedDataset.category = null;
  // Sync the main page (if open in another tab it won't matter; if same tab
  // the dropdown will pick this up next time it's bound).
  window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { key } }));
  setStatus(`running benchmark on ${key}…`);
  try {
    const r = await fetch(`/api/run?dataset=${encodeURIComponent(key)}`, { method: "POST" });
    if (!r.ok) throw new Error("run failed");
    const started = await r.json();
    if (started.started) {
      // Show inline progress in the card; the global SSE will also reflect it.
      const card = document.querySelector(`.dataset-card[data-key="${key}"]`);
      if (card) {
        card.classList.add("is-running");
        card.querySelector(".dataset-card-progress").hidden = false;
        card.querySelector(".progress-label").textContent = "starting…";
      }
    } else {
      setStatus("already running — wait for it to finish");
    }
  } catch (e) {
    setStatus(`error: ${e.message}`);
  }
}

async function viewFromCard(key) {
  // Jump to the per-category detail of the most recent run for this dataset.
  // We don't know which category was last — fetch /api/summary which now
  // carries the active dataset's last overall dict.
  try {
    const r = await fetch("/api/summary");
    if (!r.ok) { setStatus("no results yet"); return; }
    const data = await r.json();
    const cats = (data.per_category || []).map(c => c.category);
    const cat = cats[0];
    if (!cat) { setStatus("no categories in latest run"); return; }
    window.location.href = `index.html#cat=${encodeURIComponent(cat)}`;
  } catch (e) {
    setStatus(`error: ${e.message}`);
  }
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

let drView = "latest"; // "latest" | "all" | "best"
let drDatasetKey = null; // dataset key the Detailed Results table is showing
let drModelData = null;  // { cats: Set, latestByModel: Map(model -> run) } for the active dataset

async function fetchDrModelData() {
  if (!drDatasetKey) return null;
  try {
    const r = await fetch("/api/history");
    if (!r.ok) return null;
    const data = await r.json();
    const runs = (data.runs || []).filter(run => {
      const ds = run.dataset || run.config?.dataset || run.overall?.dataset || "";
      return ds === drDatasetKey;
    });
    // index.json entries only carry top-level metrics; per_category lives in
    // the per-run file. Fetch those in parallel and keep only the latest run
    // per (model) on this dataset.
    const detailed = await Promise.all(runs.map(async run => {
      try {
        const d = await fetch(`/api/history/${encodeURIComponent(run.id)}`);
        if (!d.ok) return null;
        const full = await d.json();
        // Splice the index row's timestamp + dataset hint onto the detail so
        // the rest of the code can find them in one place.
        full._index = run;
        return full;
      } catch { return null; }
    }));
    const latestByModel = new Map();
    for (const run of detailed.filter(Boolean)) {
      const ver = run.ocr_version || run.config?.ocr_version || "?";
      const mtype = run.model_type || run.config?.model_type || "?";
      const key = `${ver} · ${mtype}`;
      const prev = latestByModel.get(key);
      if (!prev || (run.timestamp || "") > (prev.timestamp || "")) {
        latestByModel.set(key, run);
      }
    }
    const cats = new Set();
    for (const run of latestByModel.values()) {
      for (const c of (run.per_category || [])) cats.add(c.category);
    }
    drModelData = { cats, latestByModel };
    return drModelData;
  } catch { return null; }
}

async function renderTable(cats) {
  const tbody = $("#cat-tbody");
  const thead = $("#cat-thead");
  tbody.innerHTML = "";
  thead.innerHTML = "";
  // Always recompute model data when the dataset changes.
  if (!drModelData || drModelData.datasetKey !== drDatasetKey) {
    drModelData = null;
    await fetchDrModelData();
    drModelData = { ...drModelData, datasetKey: drDatasetKey };
  }

  if (drView === "latest") {
    renderTableLatest(thead, tbody, cats);
  } else if (drView === "all") {
    renderTableAllModels(thead, tbody);
  } else if (drView === "best") {
    renderTableBestPerModel(thead, tbody);
  }
  updateDrToolbarMeta();
  updateDrDatasetSummary();
}

function renderTableLatest(thead, tbody, cats) {
  // If the user picked a dataset that summary.json doesn't cover (because the
  // most recent OCR run was on the other dataset), fall back to the latest
  // history run for that dataset so the table still shows real per-category
  // rows instead of an empty hint.
  const summaryDs = summaryData?.overall?.dataset || "";
  let sourceCats = cats;
  let sourceVer = summaryData?.overall?.ocr_version || "?";
  let sourceMtype = summaryData?.overall?.model_type || "?";
  let sourceNote = null;
  // sourceVer/sourceMtype reserved for a future header-row "viewing run"
  // label so when summary != drDatasetKey the user sees which run they're
  // actually looking at, not just "model · dataset".
  if (drDatasetKey && summaryDs && drDatasetKey !== summaryDs) {
    // Find the latest history run for drDatasetKey.
    let fallback = null;
    if (drModelData?.latestByModel) {
      for (const run of drModelData.latestByModel.values()) {
        if (!fallback || (run.timestamp || "") > (fallback.timestamp || "")) {
          fallback = run;
        }
      }
    }
    if (fallback && (fallback.per_category || []).length) {
      const o = fallback.overall || {};
      sourceCats = (fallback.per_category || []).map(c => ({
        category: c.category,
        n_images: c.n_images,
        n_lines: c.n_lines,
        detection: { f1: c.f1 },
        cer_mean: c.cer,
        wer_mean: c.wer,
        mean_confidence: c.mean_conf,
        mean_ms_per_image: c.ms_per_img,
      }));
      sourceVer = o.ocr_version || fallback.ocr_version || "?";
      sourceMtype = o.model_type || fallback.model_type || "?";
      sourceNote = `latest summary is for ${datasetLabelFor(summaryDs)} — showing the most recent history run for ${datasetLabelFor(drDatasetKey)} instead`;
    } else {
      thead.innerHTML = `
        <tr>
          <th class="sortable" data-key="category">category</th>
          <th class="num sortable" data-key="n_images">imgs</th>
          <th class="num sortable" data-key="n_lines">lines</th>
          <th class="num sortable" data-key="f1">F1</th>
          <th class="num sortable" data-key="cer">CER</th>
          <th class="num sortable" data-key="wer">WER</th>
          <th class="num sortable" data-key="conf">conf</th>
          <th class="num sortable" data-key="ms">ms/img</th>
          <th></th>
        </tr>`;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="9" class="muted dr-hint">
        No runs yet for <strong>${escapeHtml(datasetLabelFor(drDatasetKey))}</strong> —
        click <em>Run benchmark</em> at the top of the page to populate it.
      </td>`;
      tbody.appendChild(tr);
      return;
    }
  }
  thead.innerHTML = `
    <tr>
      <th class="sortable" data-key="category">category</th>
      <th class="num sortable" data-key="n_images">imgs</th>
      <th class="num sortable" data-key="n_lines">lines</th>
      <th class="num sortable" data-key="f1">F1</th>
      <th class="num sortable" data-key="cer">CER</th>
      <th class="num sortable" data-key="wer">WER</th>
      <th class="num sortable" data-key="conf">conf</th>
      <th class="num sortable" data-key="ms">ms/img</th>
      <th></th>
    </tr>`;
  if (sourceNote) {
    const noteTr = document.createElement("tr");
    noteTr.className = "dr-note-row";
    noteTr.innerHTML = `<td colspan="9" class="dr-note">${escapeHtml(sourceNote)}</td>`;
    tbody.appendChild(noteTr);
  }
  const rows = sourceCats.map((c) => ({
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

/* "All models" view: one row per (category, model). Same metric columns as
   Latest; the first cell shows category + sub-line with the model chip so the
   row reads cleanly when sorted by category or F1. */
function renderTableAllModels(thead, tbody) {
  if (!drModelData || !drModelData.latestByModel.size) {
    tbody.innerHTML = `<tr><td colspan="9" class="muted">no history yet — run a benchmark first</td></tr>`;
    return;
  }
  thead.innerHTML = `
    <tr>
      <th class="sortable" data-key="category">category</th>
      <th>model</th>
      <th class="num sortable" data-key="f1">F1</th>
      <th class="num sortable" data-key="cer">CER</th>
      <th class="num sortable" data-key="wer">WER</th>
      <th class="num sortable" data-key="conf">conf</th>
      <th class="num sortable" data-key="ms">ms/img</th>
    </tr>`;

  // Build flat rows then sort by current sort key (default F1 asc → worst first).
  const rows = [];
  for (const [model, run] of drModelData.latestByModel) {
    for (const c of (run.per_category || [])) {
      rows.push({
        category: c.category,
        model,
        f1: c.f1 ?? 0,
        cer: c.cer ?? 0,
        wer: c.wer ?? 0,
        conf: c.mean_conf ?? 0,
        ms: c.ms_per_img ?? 0,
      });
    }
  }
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
      <td><span class="dr-model-chip">${escapeHtml(r.model)}</span></td>
      <td class="num"><span class="mini-bar f1" style="width:${Math.round((r.f1 || 0) * 30)}px"></span>${fmt(r.f1)}</td>
      <td class="num"><span class="mini-bar cer" style="width:${Math.round((r.cer || 0) * 30)}px"></span>${fmt(r.cer)}</td>
      <td class="num">${fmt(r.wer)}</td>
      <td class="num">${fmt(r.conf, 2)}</td>
      <td class="num">${fmt(r.ms, 0)}</td>`;
    tr.addEventListener("click", () => openCategory(r.category));
    tbody.appendChild(tr);
  }
}

/* "Best per model" view: one row per category, one column per model, value is
   F1 (best F1 wins). Empty cell means that model has no result for that category.
   Empty categories get "no runs" placeholder text. */
function renderTableBestPerModel(thead, tbody) {
  if (!drModelData || !drModelData.latestByModel.size) {
    tbody.innerHTML = `<tr><td colspan="3" class="muted">no history yet</td></tr>`;
    return;
  }
  const models = [...drModelData.latestByModel.keys()].sort();
  thead.innerHTML = `
    <tr>
      <th class="sortable" data-key="category">category</th>
      ${models.map(m => `<th class="num">${escapeHtml(m)}</th>`).join("")}
      <th class="num">best model</th>
    </tr>`;
  const cats = [...drModelData.cats].sort();
  if (!cats.length) {
    tbody.innerHTML = `<tr><td colspan="${models.length + 2}" class="muted">no per-category data in history</td></tr>`;
    return;
  }
  for (const cat of cats) {
    const cells = [];
    let best = null;
    for (const m of models) {
      const run = drModelData.latestByModel.get(m);
      const c = (run?.per_category || []).find(x => x.category === cat);
      if (!c) { cells.push(`<td class="num muted">–</td>`); continue; }
      if (best === null || (c.f1 ?? 0) > best.f1) best = { model: m, ...c };
      cells.push(`<td class="num">${fmt(c.f1)}</td>`);
    }
    const bestCell = best
      ? `<td class="num"><span class="dr-best-chip" title="F1 ${fmt(best.f1)} · CER ${fmt(best.cer)}">${escapeHtml(best.model)}</span></td>`
      : `<td class="num muted">–</td>`;
    const tr = document.createElement("tr");
    tr.dataset.category = cat;
    tr.innerHTML = `<td>${escapeHtml(cat)}</td>${cells.join("")}${bestCell}`;
    tr.addEventListener("click", () => openCategory(cat));
    tbody.appendChild(tr);
  }
}

function updateDrToolbarMeta() {
  const el = $("#dr-toolbar-meta");
  if (!el) return;
  const dsLabel = drDatasetKey ? datasetLabelFor(drDatasetKey) : "(no dataset)";
  if (drView === "latest") {
    const summaryDs = summaryData?.overall?.dataset || "";
    if (drDatasetKey && summaryDs && drDatasetKey !== summaryDs) {
      el.textContent = `latest summary covers ${datasetLabelFor(summaryDs)} — falling back to history for ${dsLabel}`;
    } else {
      const ver = summaryData?.overall?.ocr_version || "?";
      const mtype = summaryData?.overall?.model_type || "?";
      el.textContent = `${ver} · ${mtype} · ${dsLabel}`;
    }
  } else if (drView === "all") {
    const n = drModelData?.latestByModel?.size || 0;
    el.textContent = `${n} model${n === 1 ? "" : "s"} on ${dsLabel}`;
  } else if (drView === "best") {
    const n = drModelData?.cats?.size || 0;
    el.textContent = `best F1 across ${n} categor${n === 1 ? "y" : "ies"} on ${dsLabel}`;
  }
}

function updateDrDatasetSummary() {
  const el = $("#dr-dataset-summary");
  if (!el) return;
  if (!drDatasetKey) { el.textContent = "–"; return; }
  const ds = (datasetsCache || []).find(d => d.key === drDatasetKey);
  if (!ds) { el.textContent = drDatasetKey; return; }
  el.textContent = `${ds.n_categories} categories · ${ds.n_images} images · ${ds.n_lines.toLocaleString()} GT lines · ${ds.format}`;
}

function initDrDatasetSelector() {
  const sel = $("#dr-dataset-select");
  if (!sel) return;
  // Honour the page's main dataset selector: pre-select the active key.
  const active = (datasetsCache || []).find(d => d.active) || (datasetsCache || [])[0];
  if (active) drDatasetKey = active.key;

  function populate() {
    sel.innerHTML = "";
    for (const ds of datasetsCache) {
      const opt = document.createElement("option");
      opt.value = ds.key;
      opt.textContent = ds.label;
      sel.appendChild(opt);
    }
    if (drDatasetKey) sel.value = drDatasetKey;
  }
  populate();

  sel.addEventListener("change", () => {
    drDatasetKey = sel.value || null;
    drModelData = null;  // force re-fetch on next render
    if (summaryData) renderTable(summaryData.per_category);
  });

  // Stay in sync with the main dataset selector (top of page).
  window.addEventListener("dataset-changed", (e) => {
    if (!e?.detail?.key || e.detail.key === drDatasetKey) return;
    sel.value = e.detail.key;
    drDatasetKey = e.detail.key;
    drModelData = null;
    if (summaryData) renderTable(summaryData.per_category);
  });
}

function initDrTabs() {
  const tabs = $$(".dr-tab");
  tabs.forEach((t) => {
    t.addEventListener("click", () => {
      drView = t.dataset.drView;
      tabs.forEach((x) => x.classList.toggle("dr-tab-active", x === t));
      if (summaryData) renderTable(summaryData.per_category);
    });
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Mirror of the API's _DATASET_LABELS — keeps the History table readable
// without an extra round-trip to /api/datasets.
const DATASET_LABELS = {
  ind_cn: "IMG_OCR_IND_CN",
  new:    "FUNSD-form",
};
function datasetLabelFor(key) {
  return DATASET_LABELS[key] || key || "";
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

  // Compose a short interpretation line so the user doesn't have to read every
  // sample to know how this image went. Counts come from the overlays — same
  // source the table renders from.
  const counts = { matched: 0, missed: 0, spurious: 0, corrected_to_match: 0 };
  for (const o of samples) {
    if (o.status === "matched") counts.matched++;
    else if (o.status === "missed") counts.missed++;
    else if (o.status === "spurious") counts.spurious++;
    if (o.status === "matched" && o.pr_text_corrected !== undefined
        && _normalize(o.pr_text_corrected) === _normalize(o.gt_text)
        && _normalize(o.pr_text) !== _normalize(o.gt_text)) {
      counts.corrected_to_match++;
    }
  }
  const interpEl = $("#d-interpretation");
  if (interpEl) {
    const parts = [];
    const tpRatio = img.detection?.tp != null && img.n_gt
      ? (img.detection.tp / img.n_gt) : 0;
    if (tpRatio >= 0.9) parts.push("Detection: nearly all GT lines found.");
    else if (tpRatio >= 0.7) parts.push("Detection: most GT lines found; misses are usually small.");
    else if (tpRatio > 0) parts.push(`Detection: ${Math.round(tpRatio * 100)}% of GT lines matched.`);
    if (counts.spurious > 0) parts.push(`${counts.spurious} extra region${counts.spurious > 1 ? "s" : ""} the engine invented.`);
    if (counts.missed > 0) parts.push(`${counts.missed} region${counts.missed > 1 ? "s" : ""} missed entirely.`);
    if (correctorOn && counts.corrected_to_match > 0) {
      parts.push(`Corrector salvaged ${counts.corrected_to_match} line${counts.corrected_to_match > 1 ? "s" : ""} that were wrong before.`);
    }
    const cerJoined = img.joined_cer;
    if (typeof cerJoined === "number") {
      if (cerJoined <= 0.1) parts.push(`Joined CER ${(cerJoined * 100).toFixed(0)}% — readable.`);
      else if (cerJoined <= 0.4) parts.push(`Joined CER ${(cerJoined * 100).toFixed(0)}% — partial errors; readable with effort.`);
      else parts.push(`Joined CER ${(cerJoined * 100).toFixed(0)}% — many errors; hard to read without correction.`);
    }
    interpEl.textContent = parts.length ? parts.join(" ") : "No regions to score on this image.";
  }

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
          <span><span class="lbl">GT</span><span class="gt">${escapeHtml(o.gt_text || "")}</span>${_listenBtn(o.gt_text)}</span>
          <span><span class="lbl">PR</span><span class="pr">${escapeHtml(o.pr_text || "")}${o.pr_score ? ` · conf ${(o.pr_score*100).toFixed(0)}%` : ""}</span>${_listenBtn(o.pr_text)}</span>
          ${cerLabel}
          ${correctedLine}
        </span>`;
    } else if (o.status === "missed") {
      li.className = "miss";
      li.innerHTML = `<span class="mark">✗</span><span><span class="lbl">MISSED</span><span class="gt">${escapeHtml(o.gt_text || "")}</span>${_listenBtn(o.gt_text)}</span>`;
    } else if (o.status === "spurious") {
      const corrected = o.pr_text_corrected;
      const correctedLine = (correctorOn && corrected !== undefined && corrected !== o.pr_text)
        ? `<span><span class="lbl">FIX</span><span class="pr" style="color:var(--accent)">${escapeHtml(corrected)}</span></span>`
        : "";
      li.className = "spur";
      li.innerHTML = `<span class="mark">+</span><span><span class="lbl">EXTRA</span><span class="pr">${escapeHtml(o.pr_text || "")}</span>${_listenBtn(o.pr_text)}</span>${correctedLine ? `<span></span>` : ""}`;
      if (correctedLine) li.innerHTML += correctedLine;
    }
    ul.appendChild(li);
  }
}

// TTS: speaker button per line. Speaks via /api/tts (Piper). One shared
// <audio> so a new click cancels the previous playback. ponytail: native
// Audio element, no library.
function _listenBtn(text) {
  const t = (text || "").trim();
  if (!t) return "";
  return `<button class="listen" title="Listen (Piper TTS)" data-tts="${escapeHtml(t)}">🔊</button>`;
}
let _ttsAudio = null;
document.addEventListener("click", (e) => {
  const btn = e.target.closest("button.listen");
  if (!btn) return;
  const text = btn.getAttribute("data-tts");
  if (!text) return;
  if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; }
  btn.classList.add("playing");
  _ttsAudio = new Audio("/api/tts?text=" + encodeURIComponent(text));
  const clear = () => btn.classList.remove("playing");
  _ttsAudio.addEventListener("ended", clear);
  _ttsAudio.addEventListener("error", () => {
    clear();
    btn.title = "TTS unavailable — run scripts/download_voice";
  });
  _ttsAudio.play().catch(clear);
});

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

let runActive = false;   // true while a run is in flight (drives button state)
let lastRunning = false; // edge-detect run→done to refresh results once

/* Tuning knobs — populated from /api/config on load. When a knob differs from
   the server default, we send it as a query param to /api/run. */
const KNOB_FIELDS = [
  // Detection
  { id: "knob-box-thresh",  param: "det_box_thresh",        float: true,  min: 0.10, max: 0.90, step: 0.05, dec: 2 },
  { id: "knob-det-thresh",  param: "det_thresh",            float: true,  min: 0.10, max: 0.70, step: 0.05, dec: 2 },
  { id: "knob-unclip",      param: "det_unclip_ratio",      float: true,  min: 0.80, max: 2.60, step: 0.10, dec: 2 },
  { id: "knob-limit-side",  param: "det_limit_side_len",    float: false, min: 320,  max: 2048, step: 32,   dec: 0 },
  // Recognition
  { id: "knob-rec-batch",   param: "rec_batch_num",         float: false, min: 1,    max: 32,   step: 1,    dec: 0 },
  { id: "knob-rec-width",   param: "rec_img_width",         float: false, min: 64,   max: 1024, step: 32,   dec: 0 },
  // Pipeline / runtime
  { id: "knob-angle-cls",   param: "use_angle_cls",         bool: true },
  { id: "knob-preprocessing", param: "enable_preprocessing", bool: true },
  { id: "knob-iou",         param: "iou_threshold",         float: true,  min: 0.10, max: 0.90, step: 0.05, dec: 2 },
];
let knobDefaults = null; // server-provided .env defaults

function readKnobs() {
  const out = {};
  for (const f of KNOB_FIELDS) {
    const el = document.getElementById(f.id);
    if (!el) continue;
    if (f.bool) {
      out[f.param] = el.checked;
    } else if (f.float) {
      const v = parseFloat(el.value);
      if (!Number.isNaN(v)) out[f.param] = v;
    } else {
      const v = parseInt(el.value, 10);
      if (!Number.isNaN(v)) out[f.param] = v;
    }
  }
  return out;
}

function applyKnobDefaults(d) {
  knobDefaults = d;
  const set = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined && val !== null) el.value = val; };
  set("knob-box-thresh", d.det_box_thresh);
  set("knob-det-thresh", d.det_thresh);
  set("knob-unclip",     d.det_unclip_ratio);
  set("knob-limit-side", d.det_limit_side_len);
  set("knob-rec-batch",  d.rec_batch_num);
  set("knob-rec-width",  d.rec_img_width);
  set("knob-iou",        d.iou_threshold);
  const angleEl = document.getElementById("knob-angle-cls");
  if (angleEl) angleEl.checked = !!d.use_angle_cls;
  const preEl = document.getElementById("knob-preprocessing");
  if (preEl) preEl.checked = !!d.enable_preprocessing;
  updateKnobsSummary();
  refreshKnobVisuals();
}

function updateKnobsSummary() {
  const el = document.getElementById("knobs-summary");
  if (!el) return;
  if (!knobDefaults) { el.textContent = "defaults"; el.classList.remove("has-overrides"); return; }
  const k = readKnobs();
  const diff = KNOB_FIELDS.filter(f => k[f.param] !== knobDefaults[f.param]);
  if (diff.length === 0) {
    el.textContent = "defaults";
    el.classList.remove("has-overrides");
  } else {
    el.textContent = `${diff.length} override${diff.length > 1 ? "s" : ""}`;
    el.classList.add("has-overrides");
  }
}

/* Live-update knob visuals: numeric value readout, range-track fill,
   override-highlight border, delta-from-default pill, and the config
   preview line. Cheap — runs on every input event. */
function refreshKnobVisuals() {
  for (const f of KNOB_FIELDS) {
    const el = document.getElementById(f.id);
    if (!el) continue;
    const card = document.querySelector(`.knob[data-knob="${f.param}"]`);
    if (!card) continue;

    const valEl = document.getElementById(`knob-val-${f.param}`);
    const deltaEl = document.getElementById(`knob-delta-${f.param}`);

    let cur;
    let isOverridden = false;
    if (f.bool) {
      cur = el.checked;
      if (valEl) valEl.textContent = cur ? "on" : "off";
      if (knobDefaults) isOverridden = cur !== !!knobDefaults[f.param];
      if (deltaEl) {
        if (isOverridden) {
          const def = !!knobDefaults[f.param] ? "on" : "off";
          deltaEl.textContent = `was ${def}`;
        } else {
          deltaEl.textContent = "";
        }
      }
    } else {
      cur = f.float ? parseFloat(el.value) : parseInt(el.value, 10);
      if (valEl && Number.isFinite(cur)) valEl.textContent = cur.toFixed(f.dec ?? 2);
      if (el.type === "range") {
        const min = parseFloat(el.min);
        const max = parseFloat(el.max);
        const pct = ((cur - min) / (max - min)) * 100;
        el.style.setProperty("--p", `${Math.max(0, Math.min(100, pct))}%`);
      }
      if (knobDefaults) {
        const def = knobDefaults[f.param];
        isOverridden = Number.isFinite(def) && Math.abs(cur - def) > 1e-9;
        if (deltaEl) {
          if (isOverridden) {
            const d = cur - def;
            const sign = d > 0 ? "+" : "−";
            deltaEl.textContent = `${sign}${Math.abs(d).toFixed(f.dec ?? 2)}`;
          } else {
            deltaEl.textContent = "";
          }
        }
      }
    }

    card.classList.toggle("is-overridden", isOverridden);
  }
  refreshConfigPreview();
}

/* Render the effective config block (env-style lines). Override rows
   show first and bold so the user sees what would change vs. .env. */
function refreshConfigPreview() {
  const body = document.getElementById("knobs-preview-body");
  if (!body) return;
  if (!knobDefaults) { body.textContent = "(loading defaults…)"; return; }
  const k = readKnobs();
  const lines = KNOB_FIELDS.map(f => {
    const def = knobDefaults[f.param];
    const cur = k[f.param];
    const overridden = f.bool
      ? cur !== def
      : Math.abs(Number(cur) - Number(def)) > 1e-9;
    const display = f.bool ? (cur ? "true" : "false") : cur;
    const tag = overridden ? "★ " : "  ";
    return `${tag}<span class="knobs-preview-key">${f.param.toUpperCase()}</span><span class="knobs-preview-eq">=</span><span class="knobs-preview-val">${display}</span>`;
  });
  body.innerHTML = lines.join("\n");
}

function wireKnobInputs() {
  for (const f of KNOB_FIELDS) {
    const el = document.getElementById(f.id);
    if (!el) continue;
    el.addEventListener("input", () => { refreshKnobVisuals(); updateKnobsSummary(); });
  }
}

/* ─── Recent runs strip ─────────────────────────────────────────
   Shows the last 5 history rows with the knobs they actually used,
   so users can spot what changed between A/B runs without scrolling. */
async function refreshKnobsHistory() {
  const list = document.getElementById("knobs-history-list");
  if (!list) return;
  try {
    const r = await fetch("/api/history");
    if (!r.ok) return;
    const data = await r.json();
    const runs = (data.runs || []).slice(0, 5);
    if (runs.length === 0) {
      list.innerHTML = `<div class="knobs-history-row muted">No runs yet.</div>`;
      return;
    }
    list.innerHTML = runs.map(run => {
      const cfg = run.config || {};
      const ml = `${run.ocr_version ?? cfg.ocr_version ?? "?"} · ${run.model_type ?? cfg.model_type ?? "?"}`;
      const ts = fmtDateShort(run.timestamp);
      // Show only the knobs that differ from the current server defaults —
      // the rest are noise. Falls back to listing box_thresh + limit_side_len
      // when we don't have defaults yet (first paint).
      const compareAgainst = knobDefaults || {
        det_box_thresh: cfg.det_box_thresh, det_unclip_ratio: cfg.det_unclip_ratio,
        det_limit_side_len: cfg.det_limit_side_len, rec_batch_num: cfg.rec_batch_num,
        rec_img_width: cfg.rec_img_width, det_thresh: cfg.det_thresh,
        use_angle_cls: cfg.use_angle_cls, enable_preprocessing: cfg.enable_preprocessing,
        iou_threshold: cfg.iou_threshold,
      };
      const interesting = [];
      for (const [k, v] of Object.entries(cfg)) {
        if (["ocr_version", "model_type"].includes(k)) continue;
        if (compareAgainst[k] !== v && v !== undefined && v !== null) {
          const display = typeof v === "boolean" ? (v ? "on" : "off") : v;
          interesting.push(`<span class="knobs-history-delta">${k}=<strong>${display}</strong></span>`);
        }
      }
      const deltas = interesting.length
        ? `<span class="knobs-history-deltas">${interesting.slice(0, 4).join("")}${interesting.length > 4 ? `<span class="knobs-history-delta">+${interesting.length - 4}</span>` : ""}</span>`
        : `<span class="knobs-history-delta muted">defaults</span>`;
      return `<div class="knobs-history-row" data-run-id="${run.id}">
        <span class="knobs-history-time">${ts}</span>
        <span class="knobs-history-model">${ml}</span>
        ${deltas}
      </div>`;
    }).join("");
    // Click a recent run row → load those knob values into the form.
    list.querySelectorAll(".knobs-history-row[data-run-id]").forEach(row => {
      row.addEventListener("click", async () => {
        const id = row.dataset.runId;
        try {
          const r = await fetch(`/api/history/${encodeURIComponent(id)}`);
          if (!r.ok) return;
          const run = await r.json();
          const cfg = run.config || {};
          const set = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined && val !== null) el.value = val; };
          const setBool = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined && val !== null) el.checked = !!val; };
          set("knob-box-thresh", cfg.det_box_thresh);
          set("knob-det-thresh", cfg.det_thresh);
          set("knob-unclip", cfg.det_unclip_ratio);
          set("knob-limit-side", cfg.det_limit_side_len);
          set("knob-rec-batch", cfg.rec_batch_num);
          set("knob-rec-width", cfg.rec_img_width);
          set("knob-iou", cfg.iou_threshold);
          setBool("knob-angle-cls", cfg.use_angle_cls);
          setBool("knob-preprocessing", cfg.enable_preprocessing);
          refreshKnobVisuals();
          updateKnobsSummary();
        } catch {}
      });
    });
  } catch {}
}

async function fetchConfig() {
  try {
    const r = await fetch("/api/config");
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

async function runBenchmark(force = false) {
  setStatus(force ? "restarting…" : "starting…");
  try {
    const params = new URLSearchParams();
    if (selectedDataset.key) params.set("dataset", selectedDataset.key);
    if (selectedDataset.category) params.set("category", selectedDataset.category);
    if (selectedModel.ocr_version) params.set("ocr_version", selectedModel.ocr_version);
    if (selectedModel.model_type) params.set("model_type", selectedModel.model_type);
    // Send knob overrides only when they differ from server defaults — keeps
    // run history clean when the user hasn't tweaked anything.
    if (knobDefaults) {
      const k = readKnobs();
      for (const f of KNOB_FIELDS) {
        if (k[f.param] !== knobDefaults[f.param]) {
          params.set(f.param, String(k[f.param]));
        }
      }
    }
    if (force) params.set("force", "true");
    const url = "/api/run" + (params.toString() ? "?" + params : "");
    const r = await fetch(url, { method: "POST" });
    if (!r.ok) throw new Error("run failed");
    const started = await r.json();
    $("#progress-panel").classList.remove("hidden");
    if (!started.started) setStatus("already running");
    else renderProgress({ running: true, total: 0, completed: [], current: null });
    // SSE stream (already connected) will drive progress + button from here.
    setRunActive(true);
  } catch (e) {
    setStatus(`error: ${e.message}`);
  }
}

function setRunActive(active) {
  runActive = active;
  const btn = $("#btn-run");
  if (btn) {
    btn.disabled = active;
    btn.classList.toggle("running", active);
  }
}

/* Persistent SSE subscription — survives page refresh (EventSource reconnects),
   so a background run is always reflected and the Run button stays disabled. */
async function subscribeProgress() {
  // 1) Immediate plain GET — always reflects current state on (re)load, even
  //    behind a proxy that buffers SSE. This is what makes refresh-mid-run work.
  try {
    const r = await fetch("/api/progress");
    if (r.ok) handleProgress(await r.json());
  } catch {}

  // 2) Live updates via SSE.
  let es;
  try {
    es = new EventSource("/api/progress/stream");
  } catch {
    return pollProgressFallback(); // EventSource unsupported
  }
  es.onmessage = (ev) => {
    let p;
    try { p = JSON.parse(ev.data); } catch { return; }
    sseSeen = true;
    handleProgress(p);
  };
  es.onerror = () => {
    // If SSE never delivered a message (proxy strips/buffers it), fall back to
    // polling so progress still updates live.
    if (!sseSeen && !polling) { polling = true; es.close(); pollProgressFallback(); }
  };
}

let sseSeen = false;
let polling = false;

function handleProgress(p) {
  if (p.running) {
    $("#progress-panel").classList.remove("hidden");
    renderProgress(p);
    setRunActive(true);
    setStatus(p.stale ? "run looks stuck — restart below"
                      : "benchmark running — leave this open");
  } else {
    if (lastRunning) {
      // just finished — refresh results + history once
      $("#progress-panel").classList.add("hidden");
      setStatus("done");
      loadAndRender();
      loadHistory();
      refreshKnobsHistory();
    } else if (runActive) {
      $("#progress-panel").classList.add("hidden");
    }
    setRunActive(false);
  }
  lastRunning = !!p.running;
}

async function pollProgressFallback() {
  // Minimal polling for browsers without EventSource.
  while (true) {
    try {
      const r = await fetch("/api/progress");
      if (r.ok) handleProgress(await r.json());
    } catch {}
    await new Promise((res) => setTimeout(res, 1500));
  }
}

function renderProgress(p) {
  const panel = $("#progress-panel");
  const fill = $("#progress-fill");
  const total = p.total || 0;
  const done = (p.completed || []).length;
  const cur = p.current;
  // Per-image granularity: a slow category still advances the bar image by image.
  const curFrac = cur && cur.total_images ? (cur.done_images / cur.total_images) : 0;
  const fraction = total ? (done + curFrac) / total : 0;
  fill.style.width = `${Math.min(100, fraction * 100).toFixed(1)}%`;

  panel.classList.toggle("stalled", !!p.stale);
  renderStaleNotice(p);

  $("#progress-summary").textContent = p.stale
    ? `stalled at ${done} / ${total} categories`
    : total
      ? `${done} / ${total} categories · ${Math.round(fraction * 100)}%${cur ? " · running…" : ""}`
      : "preparing…";

  if (cur) {
    const pctImg = cur.total_images ? `${Math.round((cur.done_images / cur.total_images) * 100)}%` : "0%";
    $("#progress-current").textContent = `${cur.name} · ${cur.done_images}/${cur.total_images} (${pctImg})`;
  } else {
    $("#progress-current").textContent = done >= total && total ? "finalizing…" : "waiting…";
  }

  const avg = (p.completed || []).reduce((a, c) => a + (c.elapsed_s || 0), 0) / Math.max(1, done);
  const remaining = total - done - curFrac;
  const eta = remaining > 0 && avg > 0 ? Math.round(remaining * avg) : 0;
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
    const isSelected = selectedHistoryIds.has(run.id);
    if (isSelected) tr.classList.add("selected-for-compare");
    const checked = isSelected ? "checked" : "";
    const ts = run.timestamp ? fmtDateShort(run.timestamp) : run.id;
    const datasetKey = run.dataset || run.config?.dataset || run.overall?.dataset || "";
    const datasetLabel = datasetKey ? datasetLabelFor(datasetKey) : "";
    const correctorOn = !!(run.corrector_enabled ?? run.config?.enable_symspell_correction);
    tr.innerHTML = `
      <td><input type="checkbox" data-id="${run.id}" ${checked} /></td>
      <td>${ts}</td>
      <td>
        <span class="history-model">${modelLabel(run)}</span>
        ${datasetLabel ? `<span class="history-dataset" title="Dataset used">${escapeHtml(datasetLabel)}</span>` : ""}
        ${correctorOn ? '<span class="history-tag history-tag-corrector" title="SymSpell corrector was ON for this run">corrector</span>' : ""}
      </td>
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

  // Checkbox handlers — checked runs feed the Compare button. Row click
  // opens the detail panel; the checkbox click must NOT also trigger that.
  tbody.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      if (cb.checked) selectedHistoryIds.add(cb.dataset.id);
      else selectedHistoryIds.delete(cb.dataset.id);
      updateCompareBtn();
      // Re-render to apply the selected-for-compare row highlight.
      renderHistory();
    });
  });

  // Click the first cell (where the checkbox lives) — toggle compare
  // selection without opening detail. The actual <input> change event still
  // fires because we don't preventDefault; this is belt-and-braces so the
  // checkbox cell is a generous click target.
  tbody.querySelectorAll("tr").forEach((tr) => {
    const firstCell = tr.querySelector("td:first-child");
    if (!firstCell) return;
    firstCell.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return; // native change already fires
      const cb = firstCell.querySelector("input[type=checkbox]");
      if (!cb) return;
      e.stopPropagation();
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change", { bubbles: true }));
    });
  });

  // Row click → open run detail (skip if the click target is the checkbox
  // cell or the chevron column).
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", (e) => {
      if (e.target.closest("td:first-child") || e.target.closest('input[type="checkbox"]')) return;
      openRunDetail(tr.dataset.id);
    });
  });

  updateCompareBtn();
}

function updateCompareBtn() {
  const btn = $("#btn-compare");
  if (!btn) return;
  const n = selectedHistoryIds.size;
  btn.disabled = n < 2;
  // Reflect selection state in the label so a disabled button is informative.
  if (n === 0) btn.textContent = "Compare selected";
  else if (n === 1) btn.textContent = "Compare selected (pick one more)";
  else btn.textContent = `Compare ${n} selected`;
}

// Old history rows store ocr_version/model_type at the top level; new rows
// nest them under ``config``. Read both shapes so nothing renders "?".
function modelLabel(run) {
  // Run files come in three flavors: top-level (old), nested config (recent
  // runs with overrides), and overall (always). Try them all.
  const o = run.overall || {};
  const cfg = run.config || {};
  const v = run.ocr_version ?? cfg.ocr_version ?? o.ocr_version ?? null;
  const t = run.model_type ?? cfg.model_type ?? o.model_type ?? null;
  return `${v || "?"} · ${t || "?"}`;
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
  console.log("[compare] clicked, selectedHistoryIds =", ids);
  const runs = [];
  for (const id of ids) {
    try {
      const r = await fetch(`/api/history/${encodeURIComponent(id)}`);
      if (r.ok) runs.push(await r.json());
      else console.warn("[compare] fetch failed for", id, r.status);
    } catch (e) {
      console.error("[compare] fetch error", id, e);
    }
  }
  console.log("[compare] fetched runs =", runs.length);

  const panel = $("#compare-panel");
  const content = $("#compare-content");
  if (runs.length < 2) {
    panel.classList.remove("hidden");
    content.innerHTML = `<div class="muted">Pick at least 2 runs to compare (currently ${runs.length}).</div>`;
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
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
    html += `<th class="num">${modelLabel(r)}<br><span style="font-weight:400;color:var(--text-muted)">${fmtDateShort(r.timestamp)}</span></th>`;
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

  // Per-category F1 — works for 2+ runs. Rows sorted by biggest absolute delta
  // first so the meaningful wins/losses are at the top, not in arbitrary order.
  const allCats = new Set();
  for (const r of runs) for (const c of (r.per_category || [])) allCats.add(c.category);
  if (allCats.size) {
    const headers = runs.map(r => modelLabel(r)).map(s => `<th class="num">${s}</th>`).join("");
    html += `<div style="margin-top:16px;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:500">Per-category F1</div>`;
    html += `<table class="compare-table" style="margin-top:6px"><thead><tr><th>Category</th>${headers}<th class="num">Δ (max−min)</th></tr></thead><tbody>`;
    const rows = [];
    for (const cat of allCats) {
      const vals = runs.map(r => (r.per_category || []).find(c => c.category === cat)?.f1 ?? null);
      const numeric = vals.filter(v => v !== null);
      if (numeric.length < 2) continue;
      const d = Math.max(...numeric) - Math.min(...numeric);
      rows.push({ cat, vals, d });
    }
    rows.sort((a, b) => b.d - a.d);
    for (const { cat, vals, d } of rows) {
      html += `<tr><td>${escapeHtml(cat)}</td>`;
      for (const v of vals) {
        html += `<td class="num">${v !== null ? fmt(v) : "–"}</td>`;
      }
      const cls = d > 0.01 ? "down" : d < -0.01 ? "up" : "";
      html += `<td class="num"><span class="compare-delta ${cls}">${d.toFixed(3)}</span></td></tr>`;
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

  // Keep the Detailed Results dropdown in sync with this run's dataset, so the
  // per-category tables below show the same categories the run produced.
  const runDs = run.dataset || run.config?.dataset || run.overall?.dataset || "";
  if (runDs && runDs !== drDatasetKey) {
    drDatasetKey = runDs;
    drModelData = null;  // force re-fetch in next renderTable call
    const sel = $("#dr-dataset-select");
    if (sel) sel.value = runDs;
    if (summaryData) renderTable(summaryData.per_category);
    // Mirror the change in the top dataset selector + main badge so the rest
    // of the page (model badge, footer) reflects the same dataset.
    window.dispatchEvent(new CustomEvent("dataset-changed", { detail: { key: runDs } }));
  }

  const o = run.overall || {};
  const cfg = run.config || {};
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

  // Header — model + dataset + corrector + age, all in one row so the user
  // can tell at a glance *which* run they're looking at without reading JSON.
  const datasetKey = run.dataset || cfg.dataset || o.dataset || "";
  const datasetLabel = datasetKey ? datasetLabelFor(datasetKey) : "–";
  const ver = o.ocr_version || cfg.ocr_version || run.ocr_version || "?";
  const mtype = o.model_type || cfg.model_type || run.model_type || "?";
  const correctorOn = !!(run.corrector_enabled ?? cfg.enable_symspell_correction ?? o.corrector_enabled);
  const iou = (cfg.iou_threshold ?? o.iou_threshold ?? 0.5);

  $("#run-detail-title").textContent = `${ver} · ${mtype}`;
  $("#run-detail-sub").innerHTML = `
    <span class="rd-pill rd-pill-dataset">${escapeHtml(datasetLabel)}</span>
    <span class="rd-pill ${correctorOn ? "rd-pill-on" : "rd-pill-off"}">corrector ${correctorOn ? "ON" : "OFF"}</span>
    <span class="rd-pill">IoU ≥ ${Number(iou).toFixed(2)}</span>
    <span class="muted">·</span>
    <span>${run.timestamp ? fmtDate(run.timestamp) : run.id}</span>
  `;

  // Config block — show only values that differ from the live .env defaults so
  // the section answers "what was different about this run?" not "all 14 knobs".
  let cfgBlock = "";
  if (knobDefaults) {
    const lines = [];
    for (const f of KNOB_FIELDS) {
      const cf = cfg[f.param];
      const cur = cf !== undefined && cf !== null ? cf : o[f.param];
      const def = knobDefaults[f.param];
      if (cur === undefined || cur === null) continue;
      const diff = typeof cur === "boolean"
        ? cur !== !!def
        : Math.abs(Number(cur) - Number(def)) > 1e-9;
      const display = typeof cur === "boolean" ? (cur ? "true" : "false") : cur;
      const tag = diff
        ? `<span class="history-tag history-tag-override">override</span>`
        : "";
      lines.push(`<div class="rd-cfg-row">
        <span class="rd-cfg-k">${f.param}</span>
        <span class="rd-cfg-v">${display}</span>
        <span class="rd-cfg-d muted">default ${typeof def === "boolean" ? (def ? "true" : "false") : def}</span>
        ${tag}
      </div>`);
    }
    cfgBlock = `
      <div class="rd-section">
        <div class="rd-section-head">Knobs used by this run <span class="muted">— overrides highlighted</span></div>
        <div class="rd-cfg">${lines.join("") || '<div class="muted">no config recorded</div>'}</div>
      </div>`;
  }

  // Per-category table — added Joined CER + corrected columns when the
  // corrector was on for this run, so a comparison of "raw vs corrected" is
  // visible without opening the per-image detail page.
  const cats = run.per_category || [];
  const correctorCol = correctorOn
    ? `<th class="num">CER (c)</th><th class="num">WER (c)</th>`
    : "";
  const catRows = cats.map(c => {
    const cer_c = correctorOn ? `<td class="num">${fmt(c.cer_corrected)}</td><td class="num">${fmt(c.wer_corrected)}</td>` : "";
    return `<tr>
      <td>${escapeHtml(c.category)}</td>
      <td class="num">${c.n_images ?? "–"}</td>
      <td class="num">${c.n_lines ?? "–"}</td>
      <td class="num"><span class="mini-bar f1" style="width:${Math.round((c.f1 || 0) * 30)}px"></span>${fmt(c.f1)}</td>
      <td class="num">${fmt(c.cer)}</td>
      <td class="num">${fmt(c.wer)}</td>
      ${cer_c}
      <td class="num">${fmt(c.mean_conf, 2)}</td>
      <td class="num">${fmt(c.ms_per_img, 0)}</td>
    </tr>`;
  }).join("");

  const tableHead = `
    <tr>
      <th>Category</th>
      <th class="num">Imgs</th>
      <th class="num">Lines</th>
      <th class="num">F1</th>
      <th class="num">CER</th>
      <th class="num">WER</th>
      ${correctorCol}
      <th class="num">Conf</th>
      <th class="num">ms/img</th>
    </tr>`;

  // Quick interpretation — a single sentence that says what these numbers
  // mean for this run, so the user doesn't have to compare to the glossary.
  const f1 = o.detection_f1 ?? 0;
  let interpretation;
  if (f1 >= 0.9) interpretation = "Excellent detection — the engine is finding almost every text region.";
  else if (f1 >= 0.75) interpretation = "Good detection — most regions are found; misses cluster on small or rotated text.";
  else if (f1 >= 0.5) interpretation = "Moderate detection — half the regions are missed or over-segmented.";
  else interpretation = "Weak detection — many text regions are missed.";
  const cer = o.cer_mean ?? 0;
  let cerInterpretation;
  if (cer <= 0.05) cerInterpretation = "Characters are read almost perfectly.";
  else if (cer <= 0.15) cerInterpretation = "Minor recognition errors; readable.";
  else cerInterpretation = "Frequent character errors; readable only with manual correction.";
  if (correctorOn) {
    const cerC = o.cer_corrected_mean ?? 0;
    const dCER = cerC - cer;
    if (Math.abs(dCER) < 0.01) cerInterpretation += ` Corrector: no meaningful change (Δ ${(dCER).toFixed(3)}).`;
    else if (dCER < 0) cerInterpretation += ` Corrector helps: CER ${(dCER * 100).toFixed(1)}% lower.`;
    else cerInterpretation += ` Corrector hurts here: CER ${(dCER * 100).toFixed(1)}% higher.`;
  }

  content.innerHTML = `
    <div class="rd-cards">${cards}</div>
    <div class="rd-section">
      <div class="rd-section-head">Interpretation</div>
      <div class="rd-interpretation">
        <div>${interpretation}</div>
        <div>${cerInterpretation}</div>
      </div>
    </div>
    ${cfgBlock}
    <div class="rd-section">
      <div class="rd-section-head">Per-category breakdown</div>
      <table class="compare-table">
        <thead>${tableHead}</thead>
        <tbody>${catRows || '<tr><td colspan="9" class="muted">no per-category data</td></tr>'}</tbody>
      </table>
    </div>`;
}

/* ─── Load & Render ──────────────────────────────────────── */

async function loadAndRender() {
  const data = await fetchSummary();
  if (!data) return;  // no results yet; SSE drives the progress panel
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
fetchConfig().then((d) => { if (d) applyKnobDefaults(d); });
initDatasetSelector();
initDatasetsPage();  // no-op on non-datasets pages

document.addEventListener("input", (e) => {
  if (e.target.closest(".knobs")) updateKnobsSummary();
});
document.addEventListener("click", (e) => {
  if (e.target && e.target.id === "knobs-reset") {
    fetchConfig().then((d) => { if (d) applyKnobDefaults(d); });
  }
});

$$("thead th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (currentSort.key === key) currentSort.dir = currentSort.dir === "asc" ? "desc" : "asc";
    else { currentSort.key = key; currentSort.dir = "asc"; }
    if (summaryData) renderTable(summaryData.per_category);
  });
});

$("#btn-run").addEventListener("click", () => runBenchmark(false));
initDrTabs();
initDrDatasetSelector();
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

loadAndRender();
loadHistory();
subscribeProgress();  // always-on: reflects any background run, survives refresh
wireKnobInputs();
refreshKnobsHistory();
