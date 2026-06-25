/* OCR Benchmark dashboard — vanilla JS, no build step */

const $ = (s) => document.querySelector(s);
const fmt = (n, d = 3) => (n === null || n === undefined || Number.isNaN(n)) ? "–" : Number(n).toFixed(d);
const pct = (n) => n === null || n === undefined ? "–" : `${(Number(n) * 100).toFixed(1)}%`;

let summaryData = null;
let currentSort = { key: "f1", dir: "asc" };

async function fetchSummary() {
  const r = await fetch("/api/summary");
  if (!r.ok) {
    setStatus("no reports — click Run benchmark");
    return null;
  }
  return await r.json();
}

function setStatus(msg) {
  $("#status").textContent = msg;
}

function renderOverall(o) {
  $("#m-f1").textContent = fmt(o.detection_f1);
  $("#m-f1-hint").textContent = `P ${fmt(o.detection_precision)} · R ${fmt(o.detection_recall)}`;
  $("#m-cer").textContent = fmt(o.cer_mean);
  $("#m-cer-hint").textContent = "raw OCR";
  $("#m-wer").textContent = fmt(o.wer_mean);
  $("#m-wer-hint").textContent = "raw OCR";
  $("#m-imgs").textContent = o.n_images;
  $("#m-lines-hint").textContent = `${o.n_lines} GT lines · ${o.n_categories} categories`;

  const correctorOn = o.corrector_enabled === true;
  if (correctorOn) {
    $("#m-cer-c").textContent = fmt(o.cer_corrected_mean);
    $("#m-wer-c").textContent = fmt(o.wer_corrected_mean);
    $("#m-cer-c-hint").textContent = "after corrector";
    $("#m-wer-c-hint").textContent = "after corrector";
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
    $("#m-cer-c-hint").textContent = "set ENABLE_SYMSPELL_CORRECTION=true in .env";
    $("#m-wer-c-hint").textContent = "set ENABLE_SYMSPELL_CORRECTION=true in .env";
    $("#corrector-banner").classList.remove("hidden");
    $("#cb-state").textContent = "OFF";
    $("#cb-sub").textContent = "Enable in .env (ENABLE_SYMSPELL_CORRECTION=true) and restart.";
    $("#cb-delta").textContent = "no delta";
    $("#cb-delta").className = "banner-delta delta-zero";
  }
}

function renderTable(cats) {
  const tbody = $("#cat-tbody");
  tbody.innerHTML = "";
  const rows = cats.map((c) => ({
    category: c.category,
    n_images: c.n_images,
    n_lines: c.n_lines,
    f1: c.detection.f1,
    cer: c.cer_mean,
    wer: c.wer_mean,
    conf: c.mean_confidence,
    ms: c.mean_ms_per_image,
    raw: c,
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
      <td class="num"><span class="bar" style="width:${Math.round(r.f1 * 40)}px"></span>${fmt(r.f1)}</td>
      <td class="num"><span class="bar cer" style="width:${Math.round(r.cer * 40)}px"></span>${fmt(r.cer)}</td>
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

async function openCategory(category) {
  setStatus(`loading ${category}…`);
  const r = await fetch(`/api/results/${encodeURIComponent(category)}`);
  if (!r.ok) {
    setStatus("failed to load category");
    return;
  }
  const data = await r.json();
  const images = data.images || [];
  if (!images.length) {
    setStatus("no images in category");
    return;
  }
  // First image by default
  showImageDetail(category, images[0], data.summary);
  $("#detail").classList.remove("hidden");
  $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });

  // Stash for drill-down navigation
  window.__categoryData = data;
  window.__currentImageIdx = 0;
  setStatus(`${category} · ${images.length} images`);
}

function showImageDetail(category, img, summary) {
  $("#detail-title").textContent = img.image;
  $("#detail-sub").textContent = `${category} · ${summary.detection.f1 >= 0 ? "category F1 " + fmt(summary.detection.f1) : ""}`;
  $("#d-gt").textContent = img.n_gt;
  $("#d-pr").textContent = img.n_pred;
  $("#d-det").textContent = `${img.detection.tp} / ${img.detection.fp} / ${img.detection.fn}`;
  $("#d-f1").textContent = fmt(_f1(img.detection));
  $("#d-cer").textContent = img.matched_cer_mean === null ? "–" : fmt(img.matched_cer_mean);
  $("#d-cer-c").textContent = img.matched_cer_corrected_mean === null || img.matched_cer_corrected_mean === undefined ? "off" : fmt(img.matched_cer_corrected_mean);
  $("#d-jcer").textContent = fmt(img.joined_cer);
  $("#d-jcer-c").textContent = img.joined_cer_corrected === undefined ? "off" : fmt(img.joined_cer_corrected);
  $("#d-conf").textContent = img.mean_confidence === null ? "–" : fmt(img.mean_confidence, 2);
  $("#d-ms").textContent = `${Math.round(img.elapsed_ms)} ms`;
  $("#d-when").textContent = summaryData?.overall?.last_run ? formatRunDate(summaryData.overall.last_run) : "–";

  // Render correction status counts
  const cs = img.correction_status || {};
  const csEl = $("#d-cstatus");
  if (Object.keys(cs).length) {
    csEl.textContent = `${cs.unchanged || 0} unchanged · ${cs.corrected || 0} corrected · ${cs.not_found || 0} not_found`;
  } else {
    csEl.textContent = "off";
  }

  // Image + overlay
  const imgEl = $("#detail-img");
  imgEl.src = `/api/image/${encodeURIComponent(category)}/${encodeURIComponent(img.image)}`;
  imgEl.onload = () => drawOverlay(img);
  // If cached, onload may not fire
  if (imgEl.complete) drawOverlay(img);

  // Sample comparison
  const samples = (img.overlays || []).slice(0, 8);
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
      if (correctedOk && !ok) cls = "match";  // correction fixed it
      if (!correctedOk && ok) cls = "diff";    // correction broke it
      const mark = correctedOk && !ok ? "✓*" : (ok ? "✓" : "≠");
      const correctedLine = (correctorOn && corrected !== undefined && corrected !== o.pr_text)
        ? `<span><span class="lbl">FIX</span><span class="pr" style="color: var(--accent)">${escapeHtml(corrected)}${o.correction_status === "corrected" ? " ⚡" : ""}</span></span>`
        : "";
      li.className = cls;
      li.innerHTML = `
        <span class="mark">${mark}</span>
        <span>
          <span><span class="lbl">GT</span><span class="gt">${escapeHtml(o.gt_text || "")}</span></span>
          <span><span class="lbl">PR</span><span class="pr">${escapeHtml(o.pr_text || "")}${o.pr_score ? ` · ${(o.pr_score*100).toFixed(0)}%` : ""}</span></span>
          ${correctedLine}
        </span>`;
    } else if (o.status === "missed") {
      li.className = "miss";
      li.innerHTML = `<span class="mark">✗</span><span><span class="lbl">MISSED</span><span class="gt">${escapeHtml(o.gt_text || "")}</span></span>`;
    } else if (o.status === "spurious") {
      const corrected = o.pr_text_corrected;
      const correctedLine = (correctorOn && corrected !== undefined && corrected !== o.pr_text)
        ? `<span><span class="lbl">FIX</span><span class="pr" style="color: var(--accent)">${escapeHtml(corrected)}</span></span>`
        : "";
      li.className = "spur";
      li.innerHTML = `<span class="mark">+</span><span><span class="lbl">EXTRA</span><span class="pr">${escapeHtml(o.pr_text || "")}</span></span>${correctedLine ? `<span></span>` : ""}`;
      if (correctedLine) {
        li.innerHTML += correctedLine;
      }
    }
    ul.appendChild(li);
  }
}

function _normalize(s) {
  return (s || "").normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();
}

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
              : o.status === "missed" ? "box-missed"
              : "box-spurious";
    const el = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    el.setAttribute("points", pts);
    el.setAttribute("class", cls);
    svg.appendChild(el);
  }
}

async function runBenchmark() {
  const btn = $("#btn-run");
  btn.disabled = true;
  setStatus("running benchmark…");
  try {
    const r = await fetch("/api/run", { method: "POST" });
    if (!r.ok) throw new Error("run failed");
    setStatus("done");
    await loadAndRender();
  } catch (e) {
    setStatus(`error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

function formatRunDate(iso) {
  if (!iso) return "–";
  // "2026-06-25T14:32:07Z" → "25 Jun 2026 · 14:32:07 UTC"
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$/);
  if (!m) return iso;
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const [, y, mo, d, hh, mm, ss] = m;
  const ageMs = Date.now() - Date.UTC(+y, +mo - 1, +d, +hh, +mm, +ss);
  const ageS = ageMs / 1000;
  let rel = "";
  if (ageS < 60) rel = " (just now)";
  else if (ageS < 3600) rel = ` (${Math.floor(ageS / 60)}m ago)`;
  else if (ageS < 86400) rel = ` (${Math.floor(ageS / 3600)}h ago)`;
  else if (ageS < 7 * 86400) rel = ` (${Math.floor(ageS / 86400)}d ago)`;
  return `${d} ${months[+mo - 1]} ${y} · ${hh}:${mm}:${ss} UTC${rel}`;
}

function renderLastRun(iso) {
  const el = $("#last-run");
  el.textContent = `last run: ${formatRunDate(iso)}`;
  // Highlight "fresh" if within last 5 minutes
  const ts = iso ? Date.parse(iso) : NaN;
  if (ts && Date.now() - ts < 5 * 60 * 1000) el.classList.add("fresh");
  else el.classList.remove("fresh");
}

async function loadAndRender() {
  const data = await fetchSummary();
  if (!data) return;
  summaryData = data;
  renderOverall(data.overall);
  renderTable(data.per_category);
  renderLastRun(data.overall.last_run);
  setStatus(`loaded · ${data.overall.n_images} imgs · ${data.overall.n_lines} lines`);
}

// Sort handlers
document.querySelectorAll("thead th.sortable").forEach((th) => {
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

loadAndRender();