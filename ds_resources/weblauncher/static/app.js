// 데이터 스트림 웹 런처 (2026-11-25, Docker 계획 D7) — tools/launcher/launcher.ps1 과 같은 화면 · 같은 규칙.
// 화면 상태(ui)는 런처 settings.json 과 같은 모양이다. 프로필은 서버(ds_resources/weblauncher/profile.py)가 만든다.
// 바꾸면 1초 뒤 원하는 상태를 다시 쓴다(실행 중이면 감독자가 반영). 실행 · 끄기는 running 칸만 바꾼다.
"use strict";

let cat = null;          // /api/catalog
let ui = null;           // 화면 상태
let running = false;
let timer = null;
let lastError = "";      // 저장 실패 이유 (상태 줄 앞에)
let profileNote = "";    // 비전이 거절할 프로필이면 그 이유

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, kids = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c != null) e.append(c);
  return e;
};
function check(text, checked, onChange, opts = {}) {
  const box = el("input", { type: "checkbox" });
  box.checked = !!checked;
  box.disabled = !!opts.disabled;
  box.addEventListener("change", () => onChange(box.checked));
  const l = el("label", { title: opts.title || null }, [box, opts.node || text]);
  if (opts.disabled) l.classList.add("dis");
  if (opts.bold) l.classList.add("bold");
  return l;
}
async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {},
                                body: body ? JSON.stringify(body) : undefined });
  let j = {};
  try { j = await r.json(); } catch (e) { j = { errors: ["응답을 읽지 못했습니다 (" + r.status + ")"] }; }
  return { ok: r.ok, status: r.status, body: j };
}

// ---------------------------------------------------------------- 런처 함수 (profile.py 와 같은 규칙)
const modelKeys = () => cat.models.map((m) => m.key);
const shown = (m) => (cat.models.find((x) => x.key === m) || { shown: m }).shown;
const chosenModels = () => modelKeys().filter((m) => ui.models.includes(m));
const depthModels = () => Object.keys(cat.overlay).filter((m) => cat.overlay[m].pose === true);
const distanceOn = () => !!ui.distance && !!cat.distance_available;
const getDirect = (n) => ui.direct.find((d) => d.name === n) || null;
const isDirect = (n) => getDirect(n) !== null;
function targetNames() {
  const n = cat.targets.map((t) => t.name);
  for (const d of ui.direct) if (!n.includes(d.name)) n.push(d.name);
  return n;
}
function checkedTargets() {
  const keep = ui.targets === null ? cat.targets.filter((t) => t.default).map((t) => t.name) : ui.targets;
  return targetNames().filter((n) => keep.includes(n));
}
function touch() { if (ui.targets === null) ui.targets = checkedTargets(); }
function cueList(name) {
  const d = getDirect(name);
  if (!d) { const t = cat.targets.find((x) => x.name === name); return t ? t.cues : []; }
  const out = cat.direct.always.map((c) => ({ ...c, default: true }));
  if (d.size) out.push(...cat.direct.with_size.map((c) => ({ ...c, default: true })));
  if (d.colour) out.push(...cat.direct.with_colour.map((c) => ({ ...c, default: true })));
  return out;
}
function cueChecked(name, c) {
  if (c.key === "class") return true;                                // 후보를 내는 근거 — 끌 수 없다
  const mode = ui.cueMode[name] || "default";
  const chosen = ui.cues[name];
  if (mode === "custom" && Array.isArray(chosen)) return chosen.includes(c.key);
  return !!c.default;
}
const cueSelection = (name) => cueList(name).filter((c) => cueChecked(name, c)).map((c) => c.key);
function overlayValue(m, key) {
  const o = ui.overlay[m];
  if (o && key in o) return o[key];
  return (cat.overlay_defaults[m] || {})[key];
}
function overlayAllowed(m, key) {
  if ((cat.overlay[m] || {})[key] !== true) return false;
  if (m === "yoloe" && (key === "dist" || key === "gps")) return chosenModels().some((x) => depthModels().includes(x));
  if (m === "targeted" && (key === "dist" || key === "gps")) return distanceOn();
  return true;
}
function labelHas(m, tok) {
  const lab = [].concat(overlayValue(m, "label") || []);
  if (tok === "name") return lab.includes("name") || lab.includes("lead");
  if (tok === "dist") return lab.includes("dist") || lab.includes("range") || lab.includes("depth");
  return lab.includes(tok);
}
function markOverlay(m, k, v) { (ui.overlay[m] = ui.overlay[m] || {})[k] = v; }
function optionTip(m) {
  const lines = ["단서 → 모델"];
  for (const r of (cat.options || {})[m] || []) {
    const ms = r.models.map((x) => (x.available ? x.key : x.key + " (없음)"));
    lines.push(r.label + ": " + (ms.length ? ms.join(", ") : "규칙 (모델 없음)"));
  }
  lines.push(m === "targeted" ? "GPU 어림: 고른 단서의 모델 합" : "GPU 어림: +" + ((cat.gpu_mb_option || {})[m] || 0) + " MB");
  return lines.join("\n");
}

// ---------------------------------------------------------------- 그리기
function renderStream() {
  const src = $("sources"); src.replaceChildren();
  for (const n of cat.sources) {
    const dot = el("span", { class: "dot", id: "dot-" + n });
    src.append(check(n, ui.sources.includes(n), (on) => {
      ui.sources = cat.sources.filter((x) => (x === n ? on : ui.sources.includes(x)));
      changed();
    }, { node: el("span", {}, [n, " ", dot]) }));
  }
  const cam = $("robotCam"); cam.replaceChildren();
  for (const c of cat.robot_cams) cam.append(el("option", { value: c, text: String(c) }));
  cam.value = String(ui.robotCam || 1);
  $("slice").checked = !!ui.slice;
  $("sliceU").value = ui.sliceU || "";
  $("sliceV").textContent = "v " + cat.slice_v;
  $("vision").checked = !!ui.vision;
  $("orchestrate").checked = ui.orchestrate !== false;
  const ms = $("models"); ms.replaceChildren();
  for (const m of modelKeys()) {
    ms.append(check(shown(m), ui.models.includes(m), (on) => {
      ui.models = modelKeys().filter((x) => (x === m ? on : ui.models.includes(x)));
      renderOverlay(); changed();
    }, { title: optionTip(m), cls: "model" }));
  }
  const fs = $("features"); fs.replaceChildren();
  for (const f of cat.features || []) {
    fs.append(check(f.shown, f.available && !!ui.features[f.key], (on) => { ui.features[f.key] = on; changed(); },
                    { disabled: !f.available }));
  }
}

function renderTargets() {
  const box = $("targets"); box.replaceChildren();
  const on = checkedTargets();
  for (const n of targetNames()) {
    box.append(check(n, on.includes(n), (c) => {
      touch();
      ui.targets = targetNames().filter((x) => (x === n ? c : ui.targets.includes(x)));
      if (!c && isDirect(n)) ui.direct = ui.direct.filter((d) => d.name !== n);   // 직접 입력은 체크를 풀면 사라진다
      renderTargets(); renderCues(); renderOverlay(); changed();
    }));
  }
  const dist = $("distance");
  dist.checked = distanceOn();
  dist.disabled = !cat.distance_available;
}

function renderCues() {
  const box = $("cues"); box.replaceChildren();
  for (const name of checkedTargets()) {
    const mode = ui.cueMode[name] || "default";
    const g = el("div", { class: "box" }, [el("h3", { text: name })]);
    const radios = el("div", { class: "row" });
    for (const [key, text] of [["default", "기본값"], ["custom", "직접 고르기"]]) {
      const r = el("input", { type: "radio", name: "mode-" + name });
      r.checked = mode === key;
      r.addEventListener("change", () => {
        if (!r.checked) return;
        if (key === "custom" && !Array.isArray(ui.cues[name])) ui.cues[name] = cueSelection(name);
        ui.cueMode[name] = key;
        touch(); renderCues(); changed();
      });
      radios.append(el("label", {}, [r, text]));
    }
    g.append(radios);
    const grid = el("div", { class: "grid4" });
    for (const c of cueList(name)) {
      const on = cueChecked(name, c);
      const enabled = mode === "custom" && !(c.needs_distance && !distanceOn()) && c.key !== "class";
      grid.append(check(c.label, on, (v) => {
        const sel = cueList(name).filter((x) => (x.key === c.key ? v : cueChecked(name, x))).map((x) => x.key);
        ui.cues[name] = sel;
        touch(); renderCues(); changed();
      }, { disabled: !enabled, bold: mode === "custom" && on !== !!c.default && c.key !== "class" }));
    }
    g.append(grid);
    const d = getDirect(name);
    if (d) {
      const sz = el("input", { type: "text", size: 14, value: d.size, placeholder: "300x300x700" });
      sz.addEventListener("change", () => {
        const v = sz.value.trim().toLowerCase();
        if (v === d.size) return;
        if (v && !/^\d+(\.\d+)?x\d+(\.\d+)?x\d+(\.\d+)?$/.test(v)) {
          alert("크기는 가로x세로x높이 mm (예: 300x300x700)"); sz.value = d.size; return;
        }
        d.size = v; renderCues(); changed();
      });
      const col = el("select", {}, [el("option", { value: "", text: "" })].concat(cat.colours.map((c) => el("option", { value: c, text: c }))));
      col.value = d.colour || "";
      col.addEventListener("change", () => { d.colour = col.value; renderCues(); changed(); });
      const reg = el("button", { type: "button", text: "등록으로 저장" });
      reg.addEventListener("click", async () => {
        const r = await api("POST", "/api/register_target", { name: d.name, size: d.size, colour: d.colour });
        alert(r.ok ? "저장했습니다. 다음 실행부터 목록에 나옵니다." : (r.body.errors || ["저장 실패"]).join("\n"));
      });
      g.append(el("div", { class: "row" }, [el("label", {}, ["크기", sz]), el("label", {}, ["색", col]), reg]));
    }
    box.append(g);
  }
}

function renderOverlay() {
  const box = $("overlay"); box.replaceChildren();
  const targets = checkedTargets();
  for (const m of chosenModels()) {
    const o = cat.overlay[m] || {};
    const g = el("div", { class: "box" }, [el("h3", { text: shown(m) })]);
    const labels = el("div", { class: "grid4" });
    const items = cat.label_items.filter(([tok]) => o[tok] === true);
    for (const [tok, text] of items) {
      const allowed = overlayAllowed(m, tok);
      labels.append(check(text, allowed && labelHas(m, tok), (v) => {
        // 보이는 라벨 칸 중 체크된 것 (river 의 클래스명은 lead)
        const toks = [];
        for (const [t2] of items) {
          const on = t2 === tok ? v : overlayAllowed(m, t2) && labelHas(m, t2);
          if (on) toks.push(t2 === "name" && m === "river" ? "lead" : t2);
        }
        markOverlay(m, "label", toks); renderOverlay(); changed();
      }, { disabled: !allowed }));
    }
    g.append(labels);
    const flags = el("div", { class: "grid4" });
    for (const [k, text] of cat.flag_items) {
      if (o[k] !== true) continue;
      if (k === "show_rejected" && !targets.includes("demo_box")) continue;
      flags.append(check(text, !!overlayValue(m, k), (v) => { markOverlay(m, k, v); changed(); }));
    }
    g.append(flags);
    if (o.cards === true) {
      const row = el("div", { class: "row" }, [el("span", { class: "lab", text: "카드" })]);
      const cur = overlayValue(m, "cards");
      for (const cm of cat.card_modes) {
        const r = el("input", { type: "radio", name: "cards-" + m });
        r.checked = cur === cm.key;
        r.addEventListener("change", () => { if (r.checked) { markOverlay(m, "cards", cm.key); changed(); } });
        row.append(el("label", {}, [r, cm.shown]));
      }
      g.append(row);
    }
    box.append(g);
  }
}

function syncEnabled() {
  const vis = !!ui.vision;
  const tOn = vis && ui.models.includes("targeted");
  $("targetGroup").disabled = !tOn;
  $("overlayGroup").disabled = !vis;
  for (const l of $("models").querySelectorAll("input")) l.disabled = !vis;
  const feats = cat.features || [];
  $("features").querySelectorAll("input").forEach((b, i) => { b.disabled = !vis || !feats[i].available; });
  $("orchestrate").disabled = !vis;
  $("planBtn").disabled = !vis;
  $("sliceU").disabled = !ui.slice;
  $("runBtn").hidden = running;
  $("stopBtn").hidden = !running;
}

function renderAll() { renderStream(); renderTargets(); renderCues(); renderOverlay(); syncEnabled(); }

// ---------------------------------------------------------------- 저장 · 실행 · 끄기
function changed() {
  syncEnabled();
  clearTimeout(timer);
  timer = setTimeout(() => save(running), 1000);       // 런처 Apply-Changes 와 같은 1초
}
async function save(run, interactive = false) {
  clearTimeout(timer);
  timer = null;
  const r = await api("PUT", "/api/desired", { ui, running: run });
  if (!r.ok) {
    lastError = (r.body.errors || ["저장 실패"]).join(" / ");
    if (interactive) alert((r.body.errors || ["저장 실패"]).join("\n"));
    showStatus(null);
    return false;
  }
  lastError = "";
  running = !!r.body.desired.running;
  profileNote = r.body.profile_check || "";
  $("profileView").textContent = JSON.stringify(r.body.desired.vision.profile, null, 1);
  syncEnabled();
  pollStatus();
  return true;
}
$("runBtn").addEventListener("click", () => save(true, true));
$("stopBtn").addEventListener("click", () => { if (confirm("데이터 스트림을 끕니다.")) save(false, true); });
$("robotCam").addEventListener("change", (e) => { ui.robotCam = parseInt(e.target.value, 10); changed(); });
$("slice").addEventListener("change", (e) => { ui.slice = e.target.checked; changed(); });
$("sliceU").addEventListener("change", (e) => { ui.sliceU = e.target.value.trim(); changed(); });
$("vision").addEventListener("change", (e) => { ui.vision = e.target.checked; renderOverlay(); changed(); });
$("orchestrate").addEventListener("change", (e) => { ui.orchestrate = e.target.checked; changed(); });
$("distance").addEventListener("change", (e) => { ui.distance = e.target.checked; touch(); renderCues(); renderOverlay(); changed(); });
function addDirect() {
  const n = $("directBox").value.trim().toLowerCase();
  if (!/^[a-z][a-z0-9 _\-]{0,40}$/.test(n)) { alert("영문 이름으로 적으세요 (예: traffic cone)."); return; }
  $("directBox").value = "";
  touch();
  if (!targetNames().includes(n)) ui.direct.push({ name: n, size: "", colour: "" });
  if (!ui.targets.includes(n)) ui.targets.push(n);
  renderTargets(); renderCues(); renderOverlay(); changed();
}
$("directAdd").addEventListener("click", addDirect);
$("directBox").addEventListener("keydown", (e) => { if (e.key === "Enter") addDirect(); });
$("planBtn").addEventListener("click", async () => {
  $("planText").textContent = "배치 계산 중";
  $("planJson").textContent = "";
  $("planDlg").showModal();
  const r = await api("POST", "/api/plan", { ui });
  $("planText").textContent = r.ok ? r.body.text : (r.body.errors || ["배치를 계산하지 못했습니다."]).join("\n");
  $("planJson").textContent = r.ok ? JSON.stringify(r.body.json, null, 1) : "";
});
for (const b of document.querySelectorAll(".tabs button")) {
  b.addEventListener("click", () => {
    for (const x of document.querySelectorAll(".tabs button")) x.classList.toggle("on", x === b);
    for (const t of document.querySelectorAll(".tab")) t.classList.toggle("on", t.id === "tab-" + b.dataset.tab);
  });
}

// ---------------------------------------------------------------- 상태 줄 (2초마다)
let lastStatus = null;
function showStatus(st) {
  if (st) lastStatus = st;
  st = lastStatus;
  const s = $("status");
  let line = st ? st.line : "";
  if (profileNote && running) line += " · 프로필 경고: " + profileNote;
  if (lastError) line = lastError + (line ? "\n" + line : "");
  s.textContent = line;
  s.classList.toggle("err", !!lastError || !!(st && st.problems && st.problems.length));
  s.classList.toggle("ok", !lastError && !!(st && st.converged && st.running));
  if (!st) return;
  for (const n of cat.sources) {
    const d = $("dot-" + n); if (!d) continue;
    const h = (st.health || {})[n];
    d.className = "dot" + (h ? (h.ok ? " ok" : " bad") : "");
    d.title = h ? (h.ok ? "/health 응답" : "응답 없음") : "";
  }
  const comp = $("components"); comp.replaceChildren();
  if (st.components && st.components.length) {
    const t = el("table", {}, [el("tr", {}, ["역할", "구성 요소", "상태", "pid", "재시작", "이유"].map((x) => el("th", { text: x })))]);
    for (const c of st.components) t.append(el("tr", {}, [c.role, c.name, c.shown, c.pid ?? "", c.restarts ?? "", c.reason].map((x) => el("td", { text: String(x) }))));
    comp.append(t);
  }
  const orch = $("orch"); orch.replaceChildren();
  if (st.orchestration && st.orchestration.length) {
    const t = el("table", {}, [el("tr", {}, ["시각", "실행", "이벤트", "이유"].map((x) => el("th", { text: x })))]);
    for (const e of st.orchestration) t.append(el("tr", {}, [(e.ts || "").slice(11, 19), e.execution_id || e.request_id, e.event, e.reason].map((x) => el("td", { text: String(x) }))));
    orch.append(t);
  }
}
async function pollStatus() {
  try {
    const r = await api("GET", "/api/status");
    if (r.ok) {
      if (r.body.running !== running && !timer) { running = r.body.running; syncEnabled(); }   // 다른 창에서 실행 · 끄기
      showStatus(r.body);
    }
  } catch (e) { lastStatus = { line: "웹 런처 서버에 닿지 않음" }; showStatus(null); }
}

// ---------------------------------------------------------------- 시작
(async function start() {
  try {
    cat = (await api("GET", "/api/catalog")).body;
    const d = (await api("GET", "/api/desired")).body;
    ui = d.ui;
    running = !!d.desired.running;                                    // 이어받기: 이미 실행 중이면 그대로
    $("profileView").textContent = JSON.stringify(d.desired.vision.profile || {}, null, 1);
  } catch (e) {
    $("status").textContent = "선택지를 못 읽었습니다.";
    return;
  }
  renderAll();
  pollStatus();
  setInterval(pollStatus, 2000);
})();
