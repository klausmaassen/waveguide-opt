const state = {
  best: null,
  refreshInFlight: false,
};

const BOUNDS = [
  ["length", "Tiefe", "mm", 32, 52, 0.5],
  ["throat_diameter", "Hals", "mm", 40, 41.5, 0.1],
  ["throat_angle", "Halswinkel", "deg", 0, 12, 0.25],
  ["coverage_angle", "Coverage", "deg", 45, 70, 0.5],
  ["os_k", "OS k", "", 0.65, 1.0, 0.01],
  ["term_s", "Term s", "", 0.6, 0.95, 0.01],
  ["term_q", "Term q", "", 0.985, 0.999, 0.001],
  ["term_n", "Term n", "", 3.5, 8.0, 0.1],
  ["target_mouth_diameter", "Mund real", "mm", 180, 205, 0.5],
];

const OBJECTIVE_WEIGHTS = [
  ["constant_directivity", "Konstante Directivity", 0.30],
  ["ptt8_crossover_match", "PTT8 Crossover", 0.25],
  ["off_axis_smoothness", "Off-axis Glaette", 0.20],
  ["response_ripple", "Frequenzgang-Ripple", 0.15],
  ["resonance_penalty", "Resonanz-Penalty", 0.10],
];

const DIRECTIVITY_DB_MIN = -50;
const DIRECTIVITY_DB_MAX = 0;
const DIRECTIVITY_DB_LEVELS = [0, -3, -6, -9, -12, -15, -18, -21, -24, -27, -30, -33, -36, -39, -42, -45, -48, -50];
const DIRECTIVITY_ANGLE_TICKS = [180, 150, 120, 90, 60, 30, 0];
const DIRECTIVITY_FREQ_TICKS = [1000, 2000, 5000, 10000, 20000];
const DIRECTIVITY_COLOR_STOPS = [
  [-50, [173, 28, 199]],
  [-40, [37, 60, 193]],
  [-30, [0, 185, 218]],
  [-21, [35, 210, 90]],
  [-12, [235, 220, 52]],
  [-6, [236, 92, 33]],
  [0, [205, 31, 31]],
];

const fmt = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
};

const escapeHtml = (text) =>
  String(text ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function renderBounds() {
  const rows = document.querySelector("#boundsRows");
  rows.innerHTML = BOUNDS.map(([key, label, unit, min, max, step]) => `
    <tr data-bound="${key}">
      <td>
        <span>${label}</span>
        ${unit ? `<small>${unit}</small>` : ""}
      </td>
      <td><input class="bound-min" type="number" step="${step}" value="${min}" /></td>
      <td><input class="bound-max" type="number" step="${step}" value="${max}" /></td>
    </tr>
  `).join("");
}

function renderWeights() {
  const rows = document.querySelector("#weightRows");
  rows.innerHTML = OBJECTIVE_WEIGHTS.map(([key, label, value]) => `
    <tr data-weight="${key}">
      <td>${label}</td>
      <td><input class="weight-value" type="number" min="0" step="0.05" value="${value}" /></td>
    </tr>
  `).join("");
}

function collectBounds() {
  const bounds = {};
  document.querySelectorAll("[data-bound]").forEach((row) => {
    const key = row.dataset.bound;
    const lo = Number(row.querySelector(".bound-min").value);
    const hi = Number(row.querySelector(".bound-max").value);
    if (Number.isFinite(lo) && Number.isFinite(hi)) bounds[key] = [lo, hi];
  });
  return bounds;
}

function collectWeights() {
  const weights = {};
  document.querySelectorAll("[data-weight]").forEach((row) => {
    const key = row.dataset.weight;
    const value = Number(row.querySelector(".weight-value").value);
    if (Number.isFinite(value)) weights[key] = Math.max(0, value);
  });
  return weights;
}

function setMetrics(best) {
  const metrics = document.querySelector("#metrics");
  const candidate = best?.candidate || {};
  const rows = [
    ["Score", best?.pre_score, "", 3],
    ["CD", best?.constant_directivity, "", 3],
    ["PTT8 Match", best?.ptt8_crossover_match, "dB", 2],
    ["Off-axis", best?.off_axis_smoothness, "", 3],
    ["Ripple", best?.response_ripple, "", 3],
    ["Resonanz", best?.resonance_penalty, "", 3],
    ["DI Smooth", best?.directivity_index_smoothness, "", 3],
    ["BW 2k", best?.beamwidth_2000, "deg", 0],
    ["BW 4k", best?.beamwidth_4000, "deg", 0],
    ["Hals", candidate.throat_diameter, "mm", 2],
    ["Tiefe", candidate.length, "mm", 1],
    ["Mund est.", best?.estimated_mouth_diameter, "mm", 1],
    ["Coverage", candidate.coverage_angle, "deg", 1],
  ];
  metrics.innerHTML = rows
    .map(([label, value, unit, digits]) => `<dt>${label}</dt><dd>${fmt(value, digits)} ${unit}</dd>`)
    .join("");
  document.querySelector("#bestName").textContent = candidate.name || "start_40mm";
}

function prepareCanvas(canvas, logicalHeight = 560, minWidth = 760) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(minWidth, Math.floor(canvas.clientWidth || minWidth));
  const height = logicalHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  if (inMax === inMin) return (outMin + outMax) / 2;
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

function colorForDb(value) {
  const quantized = Math.floor(value / 3) * 3;
  const clamped = Math.max(DIRECTIVITY_DB_MIN, Math.min(DIRECTIVITY_DB_MAX, quantized));
  for (let i = 0; i < DIRECTIVITY_COLOR_STOPS.length - 1; i += 1) {
    const [v0, c0] = DIRECTIVITY_COLOR_STOPS[i];
    const [v1, c1] = DIRECTIVITY_COLOR_STOPS[i + 1];
    if (clamped >= v0 && clamped <= v1) {
      const t = (clamped - v0) / (v1 - v0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  const [r, g, b] = DIRECTIVITY_COLOR_STOPS[DIRECTIVITY_COLOR_STOPS.length - 1][1];
  return `rgb(${r},${g},${b})`;
}

function line(ctx, x1, y1, x2, y2, color = "#d9e0e3", width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function dimensionArrow(ctx, x1, y1, x2, y2, label, color = "#365c8d") {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 7;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  for (const [x, y, dir] of [[x1, y1, angle + Math.PI], [x2, y2, angle]]) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - head * Math.cos(dir - 0.45), y - head * Math.sin(dir - 0.45));
    ctx.lineTo(x - head * Math.cos(dir + 0.45), y - head * Math.sin(dir + 0.45));
    ctx.closePath();
    ctx.fill();
  }
  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "center";
  ctx.fillText(label, (x1 + x2) / 2, (y1 + y2) / 2 - 8);
}

function drawParamBox(ctx, x, y, label, value) {
  ctx.fillStyle = "#f6f9fa";
  ctx.strokeStyle = "#d9e0e3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, 112, 42, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#61707a";
  ctx.font = "11px Segoe UI, Arial";
  ctx.textAlign = "left";
  ctx.fillText(label, x + 10, y + 16);
  ctx.fillStyle = "#1c2429";
  ctx.font = "600 13px Segoe UI, Arial";
  ctx.fillText(value, x + 10, y + 32);
}

function drawProfile(best) {
  const canvas = document.querySelector("#profileCanvas");
  const logicalHeight = Math.max(360, Math.floor(canvas.clientHeight || 380));
  const { ctx, width, height } = prepareCanvas(canvas, logicalHeight);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfc";
  ctx.fillRect(0, 0, width, height);

  if (!best?.profile?.length) return;

  const profile = best.profile;
  const candidate = best.candidate || {};
  const throat = profile[0];
  const mouth = profile[profile.length - 1];
  const maxZ = Math.max(candidate.length || mouth.z || 1, 1);
  const maxR = Math.max(...profile.map((p) => p.r), 60);

  const left = 78;
  const rightInset = width >= 980 ? 245 : 190;
  const top = 58;
  const bottom = height < 460 ? 76 : 112;
  const plotW = width - left - rightInset - 34;
  const plotH = height - top - bottom;
  const axisY = top + plotH / 2;
  const scale = Math.min(plotW / Math.max(maxZ + 42, 1), plotH / Math.max(maxR * 2.35, 1));
  const sectionW = maxZ * scale;
  const xStart = left + Math.max(0, (plotW - sectionW) / 2);
  const xEnd = xStart + sectionW;
  const gridLeft = Math.max(left, xStart - 56);
  const gridRight = Math.min(left + plotW, xEnd + 56);
  const xOf = (z) => xStart + z * scale;
  const yTop = (r) => axisY - r * scale;
  const yBot = (r) => axisY + r * scale;

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#d9e0e3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(18, 18, width - 36, height - 36, 8);
  ctx.fill();
  ctx.stroke();

  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "right";
  ctx.fillStyle = "#8a98a2";
  for (let r = 20; r <= Math.ceil(maxR / 20) * 20; r += 20) {
    const yt = yTop(r);
    const yb = yBot(r);
    line(ctx, gridLeft, yt, gridRight, yt, "#edf1f3", 1);
    line(ctx, gridLeft, yb, gridRight, yb, "#edf1f3", 1);
    ctx.fillText(`${r}`, gridLeft - 10, yt + 4);
    ctx.fillText(`${r}`, gridLeft - 10, yb + 4);
  }
  line(ctx, gridLeft, axisY, gridRight, axisY, "#b9c6cc", 1.2);
  ctx.fillStyle = "#61707a";
  ctx.textAlign = "left";
  ctx.fillText("Radius r [mm]", 24, top - 22);
  ctx.fillText("Achse", gridLeft + 8, axisY - 8);
  ctx.fillText(`Hauptansicht proportional, ${fmt(1 / scale, 2)} mm/px`, gridLeft, top - 4);

  const topPath = profile.map((p) => [xOf(p.z), yTop(p.r)]);
  const bottomPath = profile.slice().reverse().map((p) => [xOf(p.z), yBot(p.r)]);

  ctx.beginPath();
  topPath.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  bottomPath.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.closePath();
  const fill = ctx.createLinearGradient(xStart, 0, xEnd, 0);
  fill.addColorStop(0, "rgba(15, 118, 110, 0.08)");
  fill.addColorStop(1, "rgba(54, 92, 141, 0.13)");
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 3;
  ctx.beginPath();
  topPath.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.stroke();
  ctx.beginPath();
  bottomPath.slice().reverse().forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.stroke();

  const x0 = xOf(0);
  const xM = xOf(maxZ);
  line(ctx, x0, yTop(throat.r) - 18, x0, yBot(throat.r) + 18, "#24343b", 2);
  line(ctx, xM, yTop(mouth.r) - 18, xM, yBot(mouth.r) + 18, "#24343b", 2);

  ctx.setLineDash([6, 5]);
  line(ctx, x0, yTop(20), xM, yTop(20), "#ad5f00", 1.3);
  line(ctx, x0, yBot(20), xM, yBot(20), "#ad5f00", 1.3);
  ctx.setLineDash([]);

  const domeTop = [];
  for (let i = 0; i <= 28; i += 1) {
    const r = 17 * i / 28;
    const z = 7 * Math.sqrt(Math.max(0, 1 - (r / 17) ** 2));
    domeTop.push([xOf(z), yTop(r)]);
  }
  const domeBottom = domeTop.slice().reverse().map(([x, y]) => [x, axisY + (axisY - y)]);
  ctx.beginPath();
  domeTop.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  domeBottom.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.closePath();
  ctx.fillStyle = "rgba(198, 45, 45, 0.76)";
  ctx.fill();
  ctx.strokeStyle = "#8f1f1f";
  ctx.lineWidth = 1.6;
  ctx.stroke();
  line(ctx, x0 - 7, yTop(20), x0 + 7, yTop(20), "#ad5f00", 2);
  line(ctx, x0 - 7, yBot(20), x0 + 7, yBot(20), "#ad5f00", 2);

  dimensionArrow(ctx, x0, axisY + maxR * scale + 38, xM, axisY + maxR * scale + 38, `Tiefe ${fmt(maxZ, 1)} mm`);
  dimensionArrow(ctx, x0 - 24, yTop(throat.r), x0 - 24, yBot(throat.r), `Hals ${fmt(throat.r * 2, 1)} mm`);
  dimensionArrow(ctx, xM + 24, yTop(mouth.r), xM + 24, yBot(mouth.r), `Mund ${fmt(mouth.r * 2, 1)} mm`);

  ctx.fillStyle = "#ad5f00";
  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "left";
  ctx.fillText("T34B 40 mm inkl. Sicke", x0 + 10, yTop(20) - 9);
  ctx.fillStyle = "#8f1f1f";
  ctx.fillText("7 mm Kalotte", xOf(8) + 8, axisY - 10);

  const insetX = width - rightInset + 30;
  const insetY = height < 460 ? 72 : 84;
  const insetSize = Math.min(height < 460 ? 130 : 170, rightInset - 62);
  const cx = insetX + insetSize / 2;
  const cy = insetY + insetSize / 2;
  const insetScale = (insetSize / 2 - 10) / Math.max(mouth.r, 1);
  ctx.fillStyle = "#f6f9fa";
  ctx.strokeStyle = "#d9e0e3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(insetX - 14, insetY - 36, insetSize + 28, insetSize + 92, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#1c2429";
  ctx.font = "600 13px Segoe UI, Arial";
  ctx.textAlign = "left";
  ctx.fillText("Frontansicht", insetX - 2, insetY - 14);
  ctx.beginPath();
  ctx.arc(cx, cy, mouth.r * insetScale, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(54, 92, 141, 0.08)";
  ctx.fill();
  ctx.strokeStyle = "#365c8d";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 20 * insetScale, 0, Math.PI * 2);
  ctx.strokeStyle = "#ad5f00";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 17 * insetScale, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(198, 45, 45, 0.35)";
  ctx.fill();
  ctx.strokeStyle = "#8f1f1f";
  ctx.lineWidth = 1.3;
  ctx.stroke();
  line(ctx, cx - mouth.r * insetScale, cy, cx + mouth.r * insetScale, cy, "#c6d0d5", 1);
  line(ctx, cx, cy - mouth.r * insetScale, cx, cy + mouth.r * insetScale, "#c6d0d5", 1);
  ctx.fillStyle = "#61707a";
  ctx.font = "12px Segoe UI, Arial";
  ctx.textAlign = "left";
  ctx.fillText(`Mund ${fmt(mouth.r * 2, 1)} mm`, insetX - 2, insetY + insetSize + 22);
  ctx.fillText("Rot: T34B 34/40 mm", insetX - 2, insetY + insetSize + 40);

  const stripY = height - (height < 460 ? 52 : 72);
  const boxes = [
    ["Tiefe", `${fmt(candidate.length, 1)} mm`],
    ["Hals", `${fmt(candidate.throat_diameter, 2)} mm`],
    ["Mund", `${fmt(best.estimated_mouth_diameter, 1)} mm`],
    ["Coverage", `${fmt(candidate.coverage_angle, 1)} deg`],
    ["OS k", fmt(candidate.os_k, 3)],
    ["Term", `${fmt(candidate.term_s, 2)} / ${fmt(candidate.term_n, 1)}`],
  ];
  const visibleBoxes = height < 460 ? boxes.slice(0, 4) : boxes;
  visibleBoxes.forEach(([label, value], index) => drawParamBox(ctx, 32 + index * 124, stripY, label, value));
}

function drawAcoustics(best) {
  const canvas = document.querySelector("#acousticCanvas");
  const { ctx, width, height } = prepareCanvas(canvas, 560, 520);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfc";
  ctx.fillRect(0, 0, width, height);

  const freqs = best?.freqs_hz || [];
  const angles = best?.angles_deg || [];
  const polar = best?.polar_db || [];
  if (!freqs.length || !angles.length || !polar.length) {
    ctx.fillStyle = "#61707a";
    ctx.font = "14px Segoe UI, Arial";
    ctx.fillText("Noch keine akustischen Ergebnisdaten vorhanden.", 28, 42);
    return;
  }

  const freqMin = Math.min(...freqs);
  const freqMax = Math.max(...freqs);
  const logMin = Math.log2(freqMin);
  const logMax = Math.log2(freqMax);
  const xFreq = (f, x0, w) => mapRange(Math.log2(f), logMin, logMax, x0, x0 + w);
  const yAngle = (a, y0, h) => mapRange(a, 180, 0, y0, y0 + h);

  function panel(x, y, w, h, title) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#d9e0e3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1c2429";
    ctx.font = "600 13px Segoe UI, Arial";
    ctx.textAlign = "left";
    ctx.fillText(title, x + 14, y + 22);
  }

  function directivityAt(logFreq, angleDeg) {
    const clampedAngle = Math.max(0, Math.min(180, angleDeg));
    let fi = 0;
    while (fi < freqs.length - 2 && Math.log2(freqs[fi + 1]) < logFreq) fi += 1;
    let ai = 0;
    while (ai < angles.length - 2 && angles[ai + 1] < clampedAngle) ai += 1;
    const f0 = Math.log2(freqs[fi]);
    const f1 = Math.log2(freqs[Math.min(fi + 1, freqs.length - 1)]);
    const a0 = angles[ai];
    const a1 = angles[Math.min(ai + 1, angles.length - 1)];
    const ft = f1 === f0 ? 0 : (logFreq - f0) / (f1 - f0);
    const at = a1 === a0 ? 0 : (clampedAngle - a0) / (a1 - a0);
    const v00 = polar[fi][ai];
    const v01 = polar[fi][Math.min(ai + 1, angles.length - 1)];
    const v10 = polar[Math.min(fi + 1, freqs.length - 1)][ai];
    const v11 = polar[Math.min(fi + 1, freqs.length - 1)][Math.min(ai + 1, angles.length - 1)];
    const v0 = v00 + (v01 - v00) * at;
    const v1 = v10 + (v11 - v10) * at;
    return v0 + (v1 - v0) * ft;
  }

  function drawContour(level, x, y, w, h) {
    const cols = 130;
    const rows = 100;
    ctx.strokeStyle = "rgba(40, 50, 55, 0.38)";
    ctx.lineWidth = level % 6 === 0 ? 1.1 : 0.7;
    ctx.beginPath();
    for (let ix = 0; ix < cols; ix += 1) {
      for (let iy = 0; iy < rows; iy += 1) {
        const lf0 = mapRange(ix, 0, cols, logMin, logMax);
        const lf1 = mapRange(ix + 1, 0, cols, logMin, logMax);
        const a0 = mapRange(iy, 0, rows, 180, 0);
        const a1 = mapRange(iy + 1, 0, rows, 180, 0);
        const p = [
          [x + (ix / cols) * w, y + (iy / rows) * h, directivityAt(lf0, a0)],
          [x + ((ix + 1) / cols) * w, y + (iy / rows) * h, directivityAt(lf1, a0)],
          [x + ((ix + 1) / cols) * w, y + ((iy + 1) / rows) * h, directivityAt(lf1, a1)],
          [x + (ix / cols) * w, y + ((iy + 1) / rows) * h, directivityAt(lf0, a1)],
        ];
        const hits = [];
        for (const [i0, i1] of [[0, 1], [1, 2], [2, 3], [3, 0]]) {
          const v0 = p[i0][2];
          const v1 = p[i1][2];
          if ((level >= v0 && level <= v1) || (level >= v1 && level <= v0)) {
            if (v0 === v1) continue;
            const t = (level - v0) / (v1 - v0);
            hits.push([p[i0][0] + (p[i1][0] - p[i0][0]) * t, p[i0][1] + (p[i1][1] - p[i0][1]) * t]);
          }
        }
        if (hits.length === 2) {
          ctx.moveTo(hits[0][0], hits[0][1]);
          ctx.lineTo(hits[1][0], hits[1][1]);
        } else if (hits.length === 4) {
          ctx.moveTo(hits[0][0], hits[0][1]);
          ctx.lineTo(hits[1][0], hits[1][1]);
          ctx.moveTo(hits[2][0], hits[2][1]);
          ctx.lineTo(hits[3][0], hits[3][1]);
        }
      }
    }
    ctx.stroke();
  }

  const panelX = 28;
  const panelY = 24;
  const panelW = width - 56;
  const panelH = 364;
  panel(panelX, panelY, panelW, panelH, "Directivity (vertikal)");
  const barX = panelX + 18;
  const chartX = panelX + 68;
  const chartY = panelY + 38;
  const chartW = panelW - 112;
  const chartH = panelH - 82;

  const cols = 260;
  const rows = 180;
  for (let ix = 0; ix < cols; ix += 1) {
    const lf = mapRange(ix + 0.5, 0, cols, logMin, logMax);
    const x0 = chartX + (ix / cols) * chartW;
    const x1 = chartX + ((ix + 1) / cols) * chartW;
    for (let iy = 0; iy < rows; iy += 1) {
      const angle = mapRange(iy + 0.5, 0, rows, 180, 0);
      const y0 = chartY + (iy / rows) * chartH;
      const y1 = chartY + ((iy + 1) / rows) * chartH;
      ctx.fillStyle = colorForDb(directivityAt(lf, angle));
      ctx.fillRect(x0, y0, Math.ceil(x1 - x0) + 0.5, Math.ceil(y1 - y0) + 0.5);
    }
  }

  for (const level of DIRECTIVITY_DB_LEVELS) {
    drawContour(level, chartX, chartY, chartW, chartH);
  }

  ctx.strokeStyle = "#263238";
  ctx.lineWidth = 1;
  ctx.strokeRect(chartX, chartY, chartW, chartH);
  for (const angle of DIRECTIVITY_ANGLE_TICKS) {
    const y = yAngle(angle, chartY, chartH);
    line(ctx, chartX, y, chartX + chartW, y, angle === 90 ? "rgba(35,35,35,0.28)" : "rgba(255,255,255,0.22)", 1);
    ctx.fillStyle = "#334149";
    ctx.font = "11px Segoe UI, Arial";
    ctx.textAlign = "left";
    ctx.fillText(`${angle}`, chartX + chartW + 8, y + 4);
  }
  ctx.fillStyle = "#334149";
  ctx.textAlign = "left";
  ctx.fillText("deg", chartX + chartW + 8, chartY - 8);
  ctx.textAlign = "center";
  for (const f of DIRECTIVITY_FREQ_TICKS) {
    if (f < freqMin || f > freqMax) continue;
    const x = xFreq(f, chartX, chartW);
    line(ctx, x, chartY, x, chartY + chartH, "rgba(255,255,255,0.18)", 1);
    ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, x, chartY + chartH + 20);
  }
  ctx.fillText("Hz", chartX + chartW + 12, chartY + chartH + 20);

  const barH = chartH;
  for (let i = 0; i < barH; i += 1) {
    const value = mapRange(i, 0, barH, DIRECTIVITY_DB_MAX, DIRECTIVITY_DB_MIN);
    ctx.fillStyle = colorForDb(value);
    ctx.fillRect(barX, chartY + i, 16, 1);
  }
  ctx.strokeStyle = "#263238";
  ctx.strokeRect(barX, chartY, 16, barH);
  ctx.fillStyle = "#334149";
  ctx.font = "11px Segoe UI, Arial";
  ctx.textAlign = "right";
  for (const value of DIRECTIVITY_DB_LEVELS) {
    const y = mapRange(value, DIRECTIVITY_DB_MAX, DIRECTIVITY_DB_MIN, chartY, chartY + barH);
    ctx.fillText(`${value}`, barX - 5, y + 4);
  }
  ctx.fillText("dB", barX + 16, chartY - 8);

  const bottomY = 414;
  const bottomH = 118;
  const halfW = (panelW - 24) / 2;
  const bwX = panelX + 52;
  const bwY = bottomY + 34;
  const bwW = halfW - 78;
  const bwH = bottomH - 58;
  panel(panelX, bottomY, halfW, bottomH, "Beamwidth / DI");
  ctx.fillStyle = "#8a98a2";
  ctx.font = "11px Segoe UI, Arial";
  ctx.textAlign = "right";
  for (const v of [0, 90, 180]) {
    const y = mapRange(v, 0, 180, bwY + bwH, bwY);
    line(ctx, bwX, y, bwX + bwW, y, "#edf1f3", 1);
    ctx.fillText(`${v}`, bwX - 6, y + 4);
  }
  const bw = best.beamwidth_deg || [];
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  let started = false;
  freqs.forEach((freq, idx) => {
    if (bw[idx] === null || bw[idx] === undefined) return;
    const x = xFreq(freq, bwX, bwW);
    const y = mapRange(bw[idx], 0, 180, bwY + bwH, bwY);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  const di = best.directivity_index_db || [];
  ctx.strokeStyle = "#365c8d";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  di.forEach((value, idx) => {
    const x = xFreq(freqs[idx], bwX, bwW);
    const y = mapRange(value, 0, 15, bwY + bwH, bwY);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#0f766e";
  ctx.textAlign = "left";
  ctx.fillText("Beamwidth", bwX, bwY - 8);
  ctx.fillStyle = "#365c8d";
  ctx.fillText("DI gestrichelt", bwX + 82, bwY - 8);

  const scoreX = panelX + halfW + 24;
  panel(scoreX, bottomY, halfW, bottomH, "Score-Terme");
  const terms = [
    ["CD", best.constant_directivity],
    ["PTT8", best.ptt8_crossover_match],
    ["Off-axis", best.off_axis_smoothness],
    ["Ripple", best.response_ripple],
    ["Res.", best.resonance_penalty],
  ];
  const maxTerm = Math.max(...terms.map(([, value]) => Number(value) || 0), 1);
  terms.forEach(([label, value], idx) => {
    const y = bottomY + 38 + idx * 18;
    const x = scoreX + 72;
    const w = mapRange(Number(value) || 0, 0, maxTerm, 0, halfW - 130);
    ctx.fillStyle = "#61707a";
    ctx.textAlign = "right";
    ctx.fillText(label, x - 8, y + 10);
    ctx.fillStyle = idx === 0 ? "#0f766e" : "#365c8d";
    ctx.fillRect(x, y, w, 10);
    ctx.fillStyle = "#1c2429";
    ctx.textAlign = "left";
    ctx.fillText(fmt(value, 3), x + w + 8, y + 10);
  });
}

function setCandidates(rows) {
  const tbody = document.querySelector("#candidateRows");
  const bestName = state.best?.candidate?.name;
  const sorted = [...(rows || [])].sort((a, b) => (a.pre_score ?? 999) - (b.pre_score ?? 999)).slice(0, 12);
  tbody.innerHTML = sorted
    .map(
      (r) => `<tr class="${r.name === bestName ? "best-row" : ""}">
        <td>${escapeHtml(r.name)}</td>
        <td>${fmt(r.pre_score, 3)}</td>
        <td>${fmt(r.constant_directivity, 3)}</td>
        <td>${fmt(r.ptt8_crossover_match, 2)}</td>
        <td>${fmt(r.response_ripple, 3)}</td>
        <td>${fmt(r.throat_diameter, 2)}</td>
        <td>${fmt(r.length, 1)}</td>
        <td>${fmt(r.beamwidth_2000, 0)}</td>
      </tr>`,
    )
    .join("");
}

function setStatus(data) {
  document.querySelector("#phase").textContent = data.phase || "idle";
  document.querySelector("#progressText").textContent = `${data.progress || 0} / ${data.total || 0}`;
  document.querySelector("#runDir").textContent = data.run_dir ? data.run_dir.split(/[\\/]/).slice(-2).join("/") : "";
  const ratio = data.total ? Math.min(100, (data.progress / data.total) * 100) : 0;
  document.querySelector("#bar").style.width = `${ratio}%`;
  document.querySelector("#log").textContent = (data.logs || []).join("\n");
  state.best = data.best || state.best;
  setMetrics(state.best);
  drawProfile(state.best);
  drawAcoustics(state.best);
  setCandidates(data.candidates || []);
  document.querySelector("#startBtn").disabled = Boolean(data.running);
}

async function refresh() {
  if (state.refreshInFlight) return;
  state.refreshInFlight = true;
  try {
    const data = await api("/api/status");
    setStatus(data);
  } catch (error) {
    document.querySelector("#phase").textContent = "offline";
  } finally {
    state.refreshInFlight = false;
  }
}

renderBounds();
renderWeights();

document.querySelector("#useCma").addEventListener("change", (e) => {
  const cma = e.target.checked;
  document.querySelector("#countLabel").textContent = cma ? "Max. Evaluierungen" : "Kandidaten";
  const countInput = document.querySelector("#count");
  countInput.max = cma ? "2000" : "5000";
  if (!cma && Number(countInput.value) > 5000) countInput.value = "5000";
});

document.querySelector("#startBtn").addEventListener("click", async () => {
  const notice = document.querySelector("#notice");
  notice.textContent = "";
  const useCma = document.querySelector("#useCma").checked;
  try {
    await api("/api/start", {
      method: "POST",
      body: JSON.stringify({
        count: Number(document.querySelector("#count").value),
        seed: Number(document.querySelector("#seed").value),
        mode: useCma ? "cma" : "sample",
        runAth: false,
        bounds: collectBounds(),
        objectiveWeights: collectWeights(),
      }),
    });
  } catch (error) {
    notice.textContent = "Start nicht moeglich: Es laeuft bereits ein Job oder die Eingaben sind ungueltig.";
  }
  refresh();
});

document.querySelector("#stopBtn").addEventListener("click", async () => {
  await api("/api/stop", { method: "POST", body: "{}" });
  refresh();
});

window.addEventListener("resize", () => {
  drawProfile(state.best);
  drawAcoustics(state.best);
});
refresh();
setInterval(refresh, 250);
