
const METHODS = ['Spectral', 'SCORE', 'SCORE+', 'Louvain'];
const COLOR = { 'Spectral': '#7fb2d6', 'SCORE': '#5fcdb9', 'SCORE+': '#bda6e8', 'Louvain': '#f09a80' };
const PALETTE = ['#7fb2d6', '#f0b566', '#63c2a5', '#e8869a', '#b49be0', '#c79a7f', '#e7a8ce', '#a9b4bc', '#c6d176', '#6fd0de', '#a8c8f0'];
const RGB = PALETTE.map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = id => document.getElementById(id);
const fmt = (v, d = 4) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const SVGNS = 'http://www.w3.org/2000/svg';
function svgEl(name, attrs = {}, text) {
  const el = document.createElementNS(SVGNS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (text !== undefined) el.textContent = text;
  return el;
}

const STATION_TEXT = {
  'Spectral': 'We take the 11 leading eigenvectors of the mutual-follow adjacency matrix and cluster users by Euclidean distance. Degree is left unadjusted. The largest circle contains 6,830 of the 7,431 users.',
  'SCORE': 'Each user is placed by the ratios of eigenvectors 2 through 11 to the leading eigenvector. Ratios beyond ±log n = ±8.91 are clipped, and that bound applies to 39.3% of the coordinates.',
  'SCORE+': 'Follows that touch high-degree users are first downweighted with regularization τ = 7.28, the mean degree. We then keep 12 eigenvectors, weight them by their eigenvalues, form ratios, and clip 7.2% of the ratio coordinates.',
  'Louvain': 'Users are moved to raise modularity at resolution γ = 0.25. Across seeds 236 to 240, this resolution has the median circle count closest to 11. The partition shown uses seed 236.'
};

/* ------------ station state ------------ */
let station = 'SCORE';
const listeners = [];
function setStation(m) { station = m; listeners.forEach(f => f(m)); }

const presets = $('presets');
METHODS.forEach(m => {
  const b = document.createElement('button');
  b.className = 'preset'; b.id = 'preset-' + m.replace('+', 'p'); b.textContent = m; b.style.setProperty('--c', COLOR[m]);
  b.addEventListener('click', () => setStation(m));
  presets.append(b);
});
document.querySelectorAll('[data-station-pills]').forEach((box, i) => {
  const label = document.createElement('span'); label.className = 'ctl-label'; label.textContent = 'Station'; box.append(label);
  METHODS.forEach(m => {
    const b = document.createElement('button'); b.className = 'pill'; b.dataset.m = m; b.textContent = m;
    b.addEventListener('click', () => setStation(m)); box.append(b);
  });
});
listeners.push(m => {
  presets.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.textContent === m));
  document.querySelectorAll('.pill[data-m]').forEach(b => b.setAttribute('aria-pressed', b.dataset.m === m));
  const at = (12.5 + 25 * METHODS.indexOf(m)) + '%';
  $('needle').style.left = at;
  const fill = $('playFill'); if (fill) fill.style.width = at;
  const now = $('nowName'); if (now) now.textContent = m;
  $('status').textContent = STATION_TEXT[m];
  const B = D.blocks[m], sizes = B.sizes, n = D.users.id.length;
  $('readout').innerHTML = `<span>Largest group <b>${(100 * Math.max(...sizes) / n).toFixed(1)}%</b></span><span>Group sizes <b>${Math.min(...sizes).toLocaleString()} to ${Math.max(...sizes).toLocaleString()}</b></span><span>Inside vs. between <b>${B.ratio.toFixed(2)}×</b></span>`;
});

/* ------------ network ------------ */
const U = D.users, N = U.id.length;
const adj = Array.from({ length: N }, () => []);
for (let i = 0; i < D.edges.length; i += 2) { adj[D.edges[i]].push(D.edges[i + 1]); adj[D.edges[i + 1]].push(D.edges[i]); }
const net = $('net'), nctx = net.getContext('2d');
const edgeLayer = document.createElement('canvas');
const px = new Float32Array(N), py = new Float32Array(N), size = new Float32Array(N);
const col = new Float32Array(N * 3), from = new Float32Array(N * 3), delay = new Float32Array(N);
let W = 0, H = 0, DPR = 1, selected = U.d.indexOf(Math.max(...U.d)), hover = -1, animStart = 0, animating = false;
const xmax = Math.max(...U.x), ymax = Math.max(...U.y);
const grid = new Map();
function layoutNet() {
  const r = net.getBoundingClientRect();
  W = r.width; H = r.height; DPR = Math.min(devicePixelRatio || 1, 2);
  net.width = W * DPR; net.height = H * DPR;
  const pad = 14, s = Math.min((W - 2 * pad) / xmax, (H - 2 * pad) / ymax);
  const ox = (W - xmax * s) / 2, oy = (H - ymax * s) / 2;
  grid.clear();
  for (let i = 0; i < N; i++) {
    px[i] = ox + U.x[i] * s; py[i] = oy + U.y[i] * s;
    size[i] = Math.min(1 + Math.sqrt(U.d[i]) * .26, 5) * Math.max(.75, Math.min(W / 700, 1.15));
    const key = (px[i] >> 4) + ':' + (py[i] >> 4);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
    const dx = px[i] - W / 2, dy = py[i] - H / 2;
    delay[i] = Math.sqrt(dx * dx + dy * dy) / Math.max(W, H) * 900;
  }
  edgeLayer.width = net.width; edgeLayer.height = net.height;
  const e = edgeLayer.getContext('2d');
  e.setTransform(DPR, 0, 0, DPR, 0, 0);
  e.strokeStyle = 'rgba(150,190,205,.07)'; e.lineWidth = .5; e.beginPath();
  for (let i = 0; i < D.edges.length; i += 2) { const a = D.edges[i], b = D.edges[i + 1]; e.moveTo(px[a], py[a]); e.lineTo(px[b], py[b]); }
  e.stroke();
  drawNet();
}
function targetOf(i) { return RGB[D.colors[station][i] % RGB.length]; }
function drawNet() {
  nctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  nctx.fillStyle = '#1d1d1f'; nctx.fillRect(0, 0, W, H);
  nctx.setTransform(1, 0, 0, 1, 0, 0); nctx.drawImage(edgeLayer, 0, 0); nctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  for (let i = 0; i < N; i++) {
    nctx.fillStyle = `rgb(${col[i * 3] | 0},${col[i * 3 + 1] | 0},${col[i * 3 + 2] | 0})`;
    const s = size[i]; nctx.fillRect(px[i] - s / 2, py[i] - s / 2, s, s);
  }
  if (selected >= 0) {
    nctx.strokeStyle = 'rgba(255,255,255,.4)'; nctx.lineWidth = .8; nctx.beginPath();
    for (const j of adj[selected]) { nctx.moveTo(px[selected], py[selected]); nctx.lineTo(px[j], py[j]); }
    nctx.stroke();
    nctx.strokeStyle = '#0071e3'; nctx.lineWidth = 2; nctx.beginPath(); nctx.arc(px[selected], py[selected], 7, 0, 7); nctx.stroke();
  }
  if (hover >= 0 && hover !== selected) { nctx.strokeStyle = '#ffffff'; nctx.lineWidth = 1.2; nctx.beginPath(); nctx.arc(px[hover], py[hover], 5, 0, 7); nctx.stroke(); }
}
function tick(t) {
  const e = t - animStart; let done = true;
  for (let i = 0; i < N; i++) {
    let p = (Math.max(e, 0) - delay[i]) / 320; if (p < 1) done = false;
    p = Math.max(0, Math.min(1, p)); p = p * p * (3 - 2 * p);
    const c = targetOf(i);
    for (let k = 0; k < 3; k++) col[i * 3 + k] = from[i * 3 + k] + (c[k] - from[i * 3 + k]) * p;
  }
  drawNet();
  if (!done) requestAnimationFrame(tick); else animating = false;
}
function recolor(animate) {
  if (!animate || reduce) { for (let i = 0; i < N; i++) { const c = targetOf(i); col.set(c, i * 3); } drawNet(); return; }
  from.set(col); animStart = performance.now();
  if (!animating) { animating = true; requestAnimationFrame(tick); }
}
function nearest(x, y) {
  let best = -1, bd = 16 * 16; const cx = x >> 4, cy = y >> 4;
  for (let a = cx - 1; a <= cx + 1; a++) for (let b = cy - 1; b <= cy + 1; b++) {
    for (const i of grid.get(a + ':' + b) || []) { const d = (px[i] - x) ** 2 + (py[i] - y) ** 2; if (d < bd) { bd = d; best = i; } }
  }
  return best;
}
net.addEventListener('pointermove', e => {
  const r = net.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, h = nearest(x, y);
  if (h !== hover) { hover = h; if (!animating) drawNet(); }
  const tip = $('tip');
  if (h >= 0) { tip.hidden = false; tip.style.left = px[h] + 'px'; tip.style.top = py[h] + 'px'; tip.textContent = `Listener ${U.id[h]} · ${U.d[h]} friends`; } else tip.hidden = true;
});
net.addEventListener('pointerleave', () => { hover = -1; $('tip').hidden = true; if (!animating) drawNet(); });
net.addEventListener('click', e => {
  const r = net.getBoundingClientRect(), h = nearest(e.clientX - r.left, e.clientY - r.top);
  if (h >= 0) { selected = h; renderCard(); if (!animating) drawNet(); }
});
function renderCard() {
  const i = selected, m = station;
  const own = D.own[m][i], other = D.other[m][i], top = Math.max(own, other, .001);
  const rows = METHODS.map(x => {
    const l = D.labels[x][i], same = adj[i].filter(j => D.labels[x][j] === l).length;
    return `<tr class="${x === m ? 'on' : ''}"><td>${x}</td><td>${D.blocks[x].sizes[l].toLocaleString()}</td><td>${same} of ${adj[i].length}</td></tr>`;
  }).join('');
  $('card').innerHTML = `<div><h3>Listener ${U.id[i]}</h3><div class="dim">Anonymous LastFM Asia user</div></div>
    <div class="facts"><div class="fact"><b>${U.d[i]}</b><span>friends</span></div><div class="fact"><b>${U.a[i]}</b><span>liked artists</span></div></div>
    <div class="mini"><div class="dim">Music match on ${m}</div>
      <div class="mini-row"><span>Own group</span><div class="mini-track"><div class="mini-fill" style="width:${100 * own / top}%"></div></div><span class="mini-num">${own.toFixed(3)}</span></div>
      <div class="mini-row"><span>Everyone else</span><div class="mini-track"><div class="mini-fill" style="width:${100 * other / top}%;background:#8ea3aa"></div></div><span class="mini-num">${other.toFixed(3)}</span></div></div>
    <table class="stations"><thead><tr><th>Station</th><th>Group size</th><th>Friends in group</th></tr></thead><tbody>${rows}</tbody></table>
    <button class="ghost" id="randomUser">Pick a random listener</button>`;
  $('randomUser').addEventListener('click', () => { selected = Math.floor(Math.random() * N); renderCard(); if (!animating) drawNet(); });
}
listeners.push(() => { recolor(true); renderCard(); });

/* ------------ guess ------------ */
const scale = $('scale');
const scaleX = pct => pct + '%';
[0, 20, 40, 60, 80, 100].forEach(v => { const t = document.createElement('div'); t.className = 'tick'; t.style.left = v + '%'; t.innerHTML = `<span>+${v}%</span>`; scale.append(t); });
const line = document.createElement('div'); line.className = 'scale-line'; scale.prepend(line);
const you = document.createElement('div'); you.className = 'mark you'; scale.append(you);
const marks = METHODS.map((m, i) => {
  const el = document.createElement('div'); el.className = 'mark'; el.style.left = scaleX(100 * (D.blocks[m].ratio - 1)); el.style.background = COLOR[m]; el.style.transitionDelay = (i * 160) + 'ms';
  scale.append(el); return el;
});
function updateGuess() { const v = +$('guess').value; $('guessNum').textContent = '+' + v + '%'; you.style.left = scaleX(v); }
$('guess').addEventListener('input', updateGuess); updateGuess();
$('lock').addEventListener('click', () => {
  const v = +$('guess').value;
  const place = [
    { bottom: '34px', top: 'auto', x: '-50%' },
    { top: '32px', bottom: 'auto', x: '-100%' },
    { bottom: '14px', top: 'auto', x: '-50%' },
    { top: '32px', bottom: 'auto', x: '0%' }
  ];
  marks.forEach((el, i) => {
    el.classList.add('show');
    const up = 100 * (D.blocks[METHODS[i]].ratio - 1);
    el.innerHTML = `<em><b>${METHODS[i]}</b>+${up.toFixed(0)}%</em>`;
    const em = el.querySelector('em');
    em.style.color = COLOR[METHODS[i]];
    em.style.top = place[i].top;
    em.style.bottom = place[i].bottom;
    em.style.transform = 'translateX(' + place[i].x + ')';
  });
  const a = $('answer'); a.hidden = false;
  a.innerHTML = `You guessed <strong>+${v}%</strong>. SCORE, SCORE+, and Louvain fall between <strong>+25% and +36%</strong>. Spectral is <strong>+54%</strong>. Each mark says how much higher the inside average is than the outside average.`;
  $('lock').textContent = 'Answers shown';
});

/* ------------ equalizer ------------ */
const eq = $('eq');
const EQ = { x0: 56, x1: 750, y0: 18, y1: 270, max: .18, bars: [], marks: [], labels: [] };
const ey = v => EQ.y1 - (v / EQ.max) * (EQ.y1 - EQ.y0);
[0, .05, .1, .15].forEach(v => {
  eq.append(svgEl('line', { x1: EQ.x0, x2: EQ.x1, y1: ey(v), y2: ey(v), stroke: '#2a3c44', 'stroke-width': 1 }));
  eq.append(svgEl('text', { x: EQ.x0 - 8, y: ey(v) + 4, 'text-anchor': 'end' }, v.toFixed(2)));
});
eq.append(svgEl('text', { x: 14, y: 150, transform: 'rotate(-90 14 150)', 'text-anchor': 'middle' }, 'Mean Jaccard similarity'));
const slot = (EQ.x1 - EQ.x0) / 11;
for (let k = 0; k < 11; k++) {
  const x = EQ.x0 + slot * k + 6, w = slot - 12;
  const bar = svgEl('rect', { x, width: w, y: EQ.y1, height: 0, fill: COLOR.SCORE }); eq.append(bar); EQ.bars.push(bar);
}
const stripes = svgEl('g');
for (let y = EQ.y1 - 7; y > EQ.y0; y -= 7) stripes.append(svgEl('line', { x1: EQ.x0, x2: EQ.x1, y1: y, y2: y, stroke: '#1d1d1f', 'stroke-width': 2 }));
eq.append(stripes);
for (let k = 0; k < 11; k++) {
  const x = EQ.x0 + slot * k + 2, w = slot - 4;
  const mk = svgEl('rect', { x, width: w, y: EQ.y1, height: 3, fill: '#ffffff' }); eq.append(mk); EQ.marks.push(mk);
  const t = svgEl('text', { x: x + w / 2, y: EQ.y1 + 18, 'text-anchor': 'middle', class: 'strong' }, ''); eq.append(t); EQ.labels.push(t);
}
eq.append(svgEl('text', { x: (EQ.x0 + EQ.x1) / 2, y: EQ.y1 + 44, 'text-anchor': 'middle' }, 'Users in each community, largest first'));
let eqState = new Array(22).fill(0), eqAnim = 0;
function renderEq(m) {
  const B = D.blocks[m];
  const target = [...B.within];
  B.within.forEach((_, k) => { const row = B.between[k], o = row.reduce((s, v, j) => j === k ? s : s + v, 0) / 10; target.push(o); });
  const start = [...eqState], t0 = performance.now(), id = ++eqAnim;
  EQ.bars.forEach(b => b.setAttribute('fill', COLOR[m]));
  EQ.labels.forEach((t, k) => t.textContent = B.sizes[k].toLocaleString());
  const step = now => {
    if (id !== eqAnim) return;
    let p = reduce ? 1 : Math.max(0, Math.min(1, (now - t0) / 650)); const e = 1 - Math.pow(1 - p, 3);
    for (let i = 0; i < 22; i++) eqState[i] = start[i] + (target[i] - start[i]) * e;
    for (let k = 0; k < 11; k++) {
      const y = ey(eqState[k]); EQ.bars[k].setAttribute('y', y); EQ.bars[k].setAttribute('height', EQ.y1 - y);
      EQ.marks[k].setAttribute('y', ey(eqState[11 + k]) - 1.5);
    }
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  const above = B.within.filter((v, k) => v > target[11 + k]).length;
  const extra = { 'Spectral': ' The tall bars are tiny groups around popular users. The first bar holds 6,830 people, and it sits below its mark.', 'SCORE': '', 'SCORE+': '', 'Louvain': '' }[m];
  $('eqNote').innerHTML = `On <strong>${m}</strong>, <strong>${above} of 11</strong> communities sit above their mark. Mean similarity is <strong>${B.S_same.toFixed(3)}</strong> inside communities and <strong>${B.S_diff.toFixed(3)}</strong> between them, a ratio of <strong>${B.ratio.toFixed(2)}</strong>.${extra}`;
}
listeners.push(renderEq);

/* ------------ shuffle ------------ */
const RULES = { activity: 'People with similar activity', country: 'People in the same country', both: 'Same country and similar activity' };
const RULE_NOTE = {
  activity: 'Shuffling among people with a similar number of friends and artists removes the idea that active users simply look alike. This uses 1,999 shuffles.',
  country: 'Now any similarity from sharing a country stays in the shuffled data, so the pile moves right. The gap that is left cannot come from country alone. This uses 199 shuffles.',
  both: 'This is the strictest version: same country, and similar friend and artist counts. This uses 199 shuffles.'
};
let rule = 'activity';
const shuffled = {};
const keyOf = () => rule + '|' + station;
const stateOf = () => (shuffled[keyOf()] ??= { n: Math.min(40, D.nulls[rule][station].null.length) });
const rulesBox = $('rules');
Object.keys(RULES).forEach(r => {
  const b = document.createElement('button'); b.className = 'pill'; b.textContent = RULES[r]; b.dataset.r = r;
  b.addEventListener('click', () => { rule = r; refreshRule(); }); rulesBox.append(b);
});
function refreshRule() {
  rulesBox.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.r === rule));
  $('ruleNote').textContent = RULE_NOTE[rule];
  drawHist();
}
const hist = $('hist'), hctx = hist.getContext('2d');
let drop = null;
function drawHist() {
  const r = hist.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  if (hist.width !== Math.round(r.width * dpr)) { hist.width = Math.round(r.width * dpr); hist.height = Math.round(r.height * dpr); }
  const w = r.width, h = r.height, c = hctx; c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = '#1d1d1f'; c.fillRect(0, 0, w, h);
  const S = D.nulls[rule][station], obs = D.blocks[station].T, st = stateOf(), arr = S.null, n = st.n;
  const lo = Math.min(0, ...arr), hi = Math.max(obs, ...arr) * 1.1;
  const L = 30, R = w - 30, B = h - 44, Tp = 40, X = v => L + (v - lo) / (hi - lo) * (R - L);
  const bins = Math.max(40, Math.floor((R - L) / 5)), cnt = new Array(bins).fill(0), bw = (R - L) / bins;
  for (let i = 0; i < n; i++) cnt[Math.min(bins - 1, Math.floor((X(arr[i]) - L) / bw))]++;
  const top = Math.max(...cnt, 8);
  c.font = '12px "JetBrains Mono", monospace'; c.textBaseline = 'alphabetic';
  c.strokeStyle = '#2a3c44'; c.lineWidth = 1; c.beginPath(); c.moveTo(L, B + .5); c.lineTo(R, B + .5); c.stroke();
  c.fillStyle = '#8ea3aa'; c.textAlign = 'center';
  const stepV = hi > .03 ? .01 : .005;
  for (let v = Math.ceil(lo / stepV) * stepV; v <= hi; v += stepV) { c.fillText(v.toFixed(3), X(v), B + 18); c.fillRect(X(v), B, 1, 5); }
  c.fillText('Difference in music similarity (T_eq)', (L + R) / 2, h - 6);
  c.fillStyle = COLOR[station];
  cnt.forEach((v, i) => { if (v) { const bh = Math.max(3, v / top * (B - Tp)); c.fillRect(L + i * bw + .5, B - bh, Math.max(bw - 1, 2), bh); } });
  if (drop) { c.fillStyle = '#ffffff'; c.fillRect(X(drop.v) - 3, Tp + (B - Tp) * drop.p - 6, 6, 6); }
  const x = X(obs);
  c.strokeStyle = '#0071e3'; c.lineWidth = 2.5; c.beginPath(); c.moveTo(x, Tp - 18); c.lineTo(x, B); c.stroke();
  c.fillStyle = '#f5f5f7'; c.textAlign = x > w * .7 ? 'right' : 'left';
  c.fillText(`Real value ${obs.toFixed(4)}`, x + (x > w * .7 ? -8 : 8), Tp - 8);
  c.textAlign = 'left'; c.fillStyle = '#dfe9ea'; c.fillText(`${n.toLocaleString()} of ${arr.length.toLocaleString()} shuffles`, L, 22);
  let reached = 0, sum = 0; for (let i = 0; i < n; i++) { if (arr[i] >= obs) reached++; sum += arr[i]; }
  const mean = n ? sum / n : 0;
  $('shuffleNote').innerHTML = `<strong>${reached}</strong> of <strong>${n.toLocaleString()}</strong> shuffles reached the real value. The pile's mean is <strong>${mean.toFixed(4)}</strong>, so the real value is <strong>${(obs - mean).toFixed(4)}</strong> above it.` + (rule === 'country' ? ` Shuffling inside countries reproduces about ${Math.round(100 * D.country[station].share)}% of the real value.` : '');
  const finished = n >= arr.length;
  ['once', 'many', 'all'].forEach(id => $(id).disabled = finished);
}
function addShuffles(k) {
  const st = stateOf(), arr = D.nulls[rule][station].null;
  if (k === 1 && !reduce && st.n < arr.length) {
    const v = arr[st.n], t0 = performance.now();
    const run = now => {
      const p = Math.max(0, Math.min(1, (now - t0) / 260)); drop = { v, p: p * p };
      drawHist();
      if (p < 1) requestAnimationFrame(run); else { drop = null; st.n++; drawHist(); }
    };
    requestAnimationFrame(run); return;
  }
  const goal = Math.min(arr.length, st.n + k), speed = Math.max(1, Math.ceil((goal - st.n) / 45));
  const run = () => { st.n = Math.min(goal, st.n + speed); drawHist(); if (st.n < goal && !reduce) requestAnimationFrame(run); else { st.n = goal; drawHist(); } };
  run();
}
$('once').addEventListener('click', () => addShuffles(1));
$('many').addEventListener('click', () => addShuffles(100));
$('all').addEventListener('click', () => addShuffles(1e6));
listeners.push(() => drawHist());

/* ------------ crossfader ------------ */
function tAlpha(m, a) {
  const B = D.blocks[m], n = B.sizes; let ws = 0, wv = 0, bs = 0, bv = 0;
  for (let i = 0; i < n.length; i++) {
    const w = Math.pow(n[i] * (n[i] - 1), a); ws += w; wv += w * B.within[i];
    for (let j = i + 1; j < n.length; j++) { const b = Math.pow(n[i] * n[j], a); bs += b; bv += b * B.between[i][j]; }
  }
  return wv / ws - bv / bs;
}
const fc = $('fadeChart');
const F = { x0: 62, x1: 520, y0: 16, y1: 250, min: -.02, max: .05 };
const fx = a => F.x0 + a * (F.x1 - F.x0), fy = v => F.y1 - (v - F.min) / (F.max - F.min) * (F.y1 - F.y0);
[-.02, -.01, 0, .01, .02, .03, .04, .05].forEach(v => {
  fc.append(svgEl('line', { x1: F.x0, x2: F.x1, y1: fy(v), y2: fy(v), stroke: v === 0 ? '#f5f5f7' : '#2a3c44', 'stroke-width': v === 0 ? 1.5 : 1 }));
  fc.append(svgEl('text', { x: F.x0 - 8, y: fy(v) + 4, 'text-anchor': 'end', class: v === 0 ? 'strong' : '' }, fmt(v, 2)));
});
fc.append(svgEl('text', { x: F.x0, y: F.y1 + 30 }, 'Every community'));
fc.append(svgEl('text', { x: F.x1, y: F.y1 + 30, 'text-anchor': 'end' }, 'Every pair'));
const fLines = {};
METHODS.forEach(m => {
  let d = ''; for (let i = 0; i <= 60; i++) { const a = i / 60; d += (i ? 'L' : 'M') + fx(a).toFixed(1) + ' ' + fy(tAlpha(m, a)).toFixed(1); }
  fc.append(svgEl('path', { d, fill: 'none', stroke: COLOR[m], 'stroke-width': 2.2 }));
  const dot = svgEl('circle', { r: 6, fill: COLOR[m], stroke: '#1d1d1f', 'stroke-width': 2 });
  const label = svgEl('text', { x: F.x1 + 10, fill: COLOR[m], style: `fill:${COLOR[m]}` }, m);
  fLines[m] = { dot, label };
});
const fMarker = svgEl('line', { y1: F.y0, y2: F.y1, stroke: '#ffffff', 'stroke-width': 1, 'stroke-dasharray': '3 4' }); fc.append(fMarker);
METHODS.forEach(m => { fc.append(fLines[m].dot); fc.append(fLines[m].label); });
let crossing = 0; for (let i = 0; i <= 1000; i++) { if (tAlpha('Spectral', i / 1000) < 0) { crossing = i / 1000; break; } }
const faderInput = document.querySelector('input#fader');
function renderFade() {
  const a = +faderInput.value / 100, x = fx(a); fMarker.setAttribute('x1', x); fMarker.setAttribute('x2', x);
  const placed = METHODS.map(m => ({ m, v: tAlpha(m, a) })).sort((p, q) => q.v - p.v);
  let last = -Infinity;
  placed.forEach(({ m, v }) => {
    fLines[m].dot.setAttribute('cx', x); fLines[m].dot.setAttribute('cy', fy(v));
    const y = Math.max(fy(v) + 4, last + 18); last = y;
    fLines[m].label.setAttribute('y', y); fLines[m].label.textContent = `${m} ${fmt(v)}`;
  });
  const sp = tAlpha('Spectral', a);
  $('fadeNote').textContent = a === 0 ? 'With equal community weight, Spectral has the largest value of all four.'
    : sp >= 0 ? 'Spectral is falling fast. Its huge group is getting more of the say.'
    : a >= 1 ? 'With equal pair weight, Spectral ends at −0.0137. The other three stay positive.'
    : 'Spectral has crossed zero. For a typical pair of users, its groups no longer look like music circles.';
  const B = D.blocks.Spectral, w = B.sizes.map(n => Math.pow(n * (n - 1), a)), tot = w.reduce((p, q) => p + q, 0);
  $('sayLabel').textContent = `Spectral: how much say each circle gets. The largest circle has ${(100 * w[0] / tot).toFixed(w[0] / tot > .99 ? 2 : 1)}%.`;
  const bar = $('sayBar');
  if (bar.children.length !== w.length) { bar.innerHTML = ''; w.forEach(() => bar.append(document.createElement('i'))); }
  [...bar.children].forEach((el, k) => { el.style.flex = (w[k] / tot) + ' 1 0'; el.style.setProperty('--c', PALETTE[k]); });
}
faderInput.addEventListener('input', renderFade);
listeners.push(renderFade);

/* ------------ popularity toy ------------ */
const A0 = [2, 1], C0 = [1, 2.2];
function toyDraw() {
  const pop = +$('pop').value; $('popNum').textContent = pop.toFixed(1);
  const a = $('toyA'); a.innerHTML = '';
  const O = { x: 42, y: 228 }, sc = 8.4, P = (x, y) => [O.x + x * sc, O.y - y * sc];
  a.append(svgEl('text', { x: 14, y: 20, class: 'strong' }, 'Spectral · Bea × ' + pop.toFixed(1)));
  a.append(svgEl('line', { x1: O.x, y1: O.y, x2: 345, y2: O.y, stroke: '#2a3c44' }));
  a.append(svgEl('line', { x1: O.x, y1: O.y, x2: O.x, y2: 30, stroke: '#2a3c44' }));
  a.append(svgEl('text', { x: 345, y: 252, 'text-anchor': 'end' }, 'Coordinate 1'));
  a.append(svgEl('text', { x: 14, y: 130, transform: 'rotate(-90 14 130)', 'text-anchor': 'middle' }, 'Coordinate 2'));
  const ray = (v, col) => { const [x, y] = P(v[0] * 11, v[1] * 11); a.append(svgEl('line', { x1: O.x, y1: O.y, x2: x, y2: y, stroke: col, 'stroke-dasharray': '3 5', opacity: .55 })); };
  ray(A0, '#5fcdb9'); ray([C0[0] * .72, C0[1] * .72], '#f09a80');
  const dot = (v, col, name, dy = -10) => { const [x, y] = P(v[0], v[1]); a.append(svgEl('circle', { cx: x, cy: y, r: 7, fill: col, stroke: '#1d1d1f', 'stroke-width': 2 })); a.append(svgEl('text', { x, y: y + dy, 'text-anchor': 'middle', class: 'strong' }, name)); };
  dot(A0, '#5fcdb9', 'Ana', 22);
  dot([A0[0] * pop, A0[1] * pop], '#5fcdb9', 'Bea');
  dot([C0[0] * 3, C0[1] * 3], '#f09a80', 'Cal');
  const b = $('toyB'); b.innerHTML = '';
  b.append(svgEl('text', { x: 14, y: 20, class: 'strong' }, 'SCORE · Bea × ' + pop.toFixed(1)));
  const y0 = 150, X = v => 34 + (v - 0) / 2.6 * 292;
  b.append(svgEl('line', { x1: 34, x2: 326, y1: y0, y2: y0, stroke: '#2a3c44' }));
  [0, .5, 1, 1.5, 2, 2.5].forEach(v => { b.append(svgEl('line', { x1: X(v), x2: X(v), y1: y0, y2: y0 + 6, stroke: '#2a3c44' })); b.append(svgEl('text', { x: X(v), y: y0 + 22, 'text-anchor': 'middle' }, v.toFixed(1))); });
  b.append(svgEl('text', { x: 180, y: 206, 'text-anchor': 'middle' }, 'Coordinate 2 ÷ Coordinate 1'));
  const rA = A0[1] / A0[0], rC = C0[1] / C0[0];
  b.append(svgEl('circle', { cx: X(rA), cy: y0 - 12, r: 8, fill: '#5fcdb9', stroke: '#1d1d1f', 'stroke-width': 2 }));
  b.append(svgEl('text', { x: X(rA), y: y0 - 30, 'text-anchor': 'middle', class: 'strong' }, 'Ana & Bea'));
  b.append(svgEl('circle', { cx: X(rC), cy: y0 - 12, r: 8, fill: '#f09a80', stroke: '#1d1d1f', 'stroke-width': 2 }));
  b.append(svgEl('text', { x: X(rC), y: y0 - 30, 'text-anchor': 'middle', class: 'strong' }, 'Cal'));
  const dist = Math.hypot(A0[0] * (pop - 1), A0[1] * (pop - 1));
  const distC = Math.hypot(A0[0] * pop - C0[0] * 3, A0[1] * pop - C0[1] * 3);
  $('toyNote').innerHTML = pop === 1 ? 'Ana and Bea are in the same community. For now they sit together in both views.'
    : `Ana and Bea are still in the same community, but their coordinates are now <strong>${dist.toFixed(1)}</strong> apart${dist > distC ? `, while Bea is only <strong>${distC.toFixed(1)}</strong> from Cal in the other community. A method that groups by distance could put Bea with Cal` : ''}. Their ratio stays the same, so SCORE still sees them together.`;
}
$('pop').addEventListener('input', toyDraw); toyDraw();

/* ------------ start ------------ */
new ResizeObserver(() => { layoutNet(); }).observe(net);
new ResizeObserver(() => drawHist()).observe(hist);
layoutNet();
refreshRule();
setStation('SCORE');
recolor(false);
renderFade();
document.fonts && document.fonts.ready.then(() => drawHist());

function pulse(el) {
  if (!el || reduce) return;
  el.classList.remove('pulse');
  void el.offsetWidth;
  el.classList.add('pulse');
}
function mountScrolly(root, startNow) {
  const steps = [...root.querySelectorAll('.step')];
  if (!steps.length) return;
  let active = -1;
  let activate = function (i) {
    if (i === active || i < 0) return;
    active = i;
    steps.forEach((s, k) => s.classList.toggle('is-active', k === i));
    const d = steps[i].dataset;
    if (d.station) {
      setStation(d.station);
      pulse(document.getElementById('preset-' + d.station.replace('+', 'p')));
    }
    if (d.rule) {
      rule = d.rule;
      refreshRule();
      pulse(rulesBox.querySelector('[data-r="' + d.rule + '"]'));
    }
    if (d.fader !== undefined) {
      faderInput.value = d.fader;
      renderFade();
    }
    if (d.pop !== undefined) {
      $('pop').value = d.pop;
      toyDraw();
      pulse($('pop'));
    }
    if (d.sketch !== undefined) setSketch(+d.sketch);
    if (d.q !== undefined) setQuestion(+d.q);
    if (d.clean !== undefined && typeof setClean === 'function') setClean(+d.clean);
    if (d.shot !== undefined) setShot(+d.shot);
    if (d.pulse) pulse(document.querySelector(d.pulse));
  }
  // Desktop: the whole section (cards and graphic) pins at the centre of the
  // viewport. Scrolling only moves the highlight from card to card; nothing
  // drifts. After the last card the section scrolls away as one piece.
  // If the cards do not fit, the card column steps to the active card.
  const graphic = root.querySelector('.scrolly-graphic');
  const col = root.querySelector('.scrolly-steps');
  const narrow = matchMedia('(max-width: 820px)');
  const pin = document.createElement('div');
  pin.className = 'scrolly-pin';
  root.insertBefore(pin, col);
  pin.append(col, graphic);
  const track = document.createElement('div');
  track.className = 'scrolly-track';
  steps.forEach(st => track.append(st));
  col.append(track);
  root.classList.add('is-pinned');

  let maxH = 0, lastW = innerWidth, D = 0, pinTop = 0, pinH = 0;
  const n = steps.length;
  function layout() {
    if (narrow.matches) {
      root.style.height = pin.style.top = graphic.style.minHeight = col.style.height = track.style.transform = '';
      col.classList.remove('is-clipped');
      maxH = 0;
      return;
    }
    if (innerWidth !== lastW) { lastW = innerWidth; graphic.style.minHeight = ''; maxH = graphic.offsetHeight; }
    // Hold the graphic at the tallest height seen so it does not jitter.
    const gH = maxH = Math.max(maxH, graphic.offsetHeight);
    graphic.style.minHeight = gH + 'px';
    const vh = innerHeight;
    const stepsH = track.scrollHeight;
    pinH = Math.max(gH, Math.min(stepsH, vh - 20));
    col.style.height = pinH + 'px';
    col.classList.toggle('is-clipped', stepsH > pinH + 1);
    pinTop = Math.max(10, (vh - pinH) / 2);
    pin.style.top = pinTop + 'px';
    D = n * Math.round(vh * 0.3);
    root.style.height = (pinH + D) + 'px';
    place();
    onScroll();
  }
  function place() {
    if (narrow.matches || active < 0) return;
    const over = track.scrollHeight - pinH;
    if (over <= 1) { track.style.transform = ''; return; }
    const s = steps[active];
    const c = s.offsetTop + s.offsetHeight / 2 - pinH / 2;
    track.style.transform = 'translateY(' + (-Math.max(0, Math.min(over, c))) + 'px)';
  }
  function onScroll() {
    if (narrow.matches || !D) return;
    const r = root.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    const prog = (pinTop - r.top) / D;
    activate(Math.max(0, Math.min(n - 1, Math.floor(prog * n))));
  }
  const _activate = activate;
  activate = i => { const was = active; _activate(i); if (active !== was) place(); };
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', layout);
  new ResizeObserver(layout).observe(graphic);

  // Narrow screens keep the simple stacked layout with a reading line.
  const io = new IntersectionObserver(entries => {
    if (!narrow.matches) return;
    entries.forEach(e => { if (e.isIntersecting) activate(steps.indexOf(e.target)); });
  }, { rootMargin: '-70% 0px -30% 0px', threshold: 0 });
  // Measure the graphic at every step once, so its height (and the pinned
  // position) never changes while reading. Off-screen, so nothing is seen.
  if (!narrow.matches) {
    steps.forEach((_, k) => { _activate(k); maxH = Math.max(maxH, graphic.offsetHeight); });
    active = -1;
  }
  steps.forEach((s, k) => {
    io.observe(s);
    s.addEventListener('click', e => {
      if (s.classList.contains('is-active') || e.target.closest('a, button, input, label')) return;
      if (narrow.matches) { s.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' }); return; }
      const y = root.getBoundingClientRect().top + scrollY - pinTop + (k + 0.5) / n * D;
      scrollTo({ top: y, behavior: reduce ? 'auto' : 'smooth' });
    });
  });
  layout();
  if (startNow) activate(0);
}
function startScrolly() {
  document.querySelectorAll('[data-scrolly]').forEach((root, i) => mountScrolly(root, i === 0));
}
document.querySelectorAll('[data-skip]').forEach(b => b.addEventListener('click', () => {
  setStation(METHODS[(METHODS.indexOf(station) + Number(b.dataset.skip) + METHODS.length) % METHODS.length]);
}));
const playBtn = $('playBtn');
if (playBtn) playBtn.addEventListener('click', () => setStation(station));

function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296; };
}
function gauss(r) {
  const u = Math.max(r(), 1e-9), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}
function fitCanvas(cvs) {
  const r = cvs.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(1, r.width), h = Math.max(1, r.height || w * 0.58);
  cvs.width = w * dpr; cvs.height = h * dpr;
  const ctx = cvs.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const sketchRnd = mulberry(236);
const sketchDots = [];
for (let i = 0; i < 150; i++) {
  const g = i < 88 ? 0 : 1;
  const cx = g ? 0.67 : 0.33, s = g ? 0.105 : 0.12;
  sketchDots.push({
    x: Math.min(0.96, Math.max(0.04, cx + gauss(sketchRnd) * s)),
    y: Math.min(0.9, Math.max(0.1, 0.5 + gauss(sketchRnd) * s * 0.9)),
    g
  });
}
const sketchEdges = [];
for (let i = 0; i < sketchDots.length; i++) {
  for (let j = i + 1; j < sketchDots.length && sketchEdges.length < 160; j++) {
    if (sketchDots[i].g !== sketchDots[j].g) continue;
    const dx = sketchDots[i].x - sketchDots[j].x, dy = sketchDots[i].y - sketchDots[j].y;
    if (dx * dx + dy * dy < 0.0036) sketchEdges.push([i, j]);
  }
}
const sketchLines = [
  'Only people so far. No follows yet.',
  'Each line is two people who follow each other.',
  'The amber group is one circle, marked from the follows.'
];
function setSketch(phase) {
  const cvs = $('sketch'); if (!cvs) return;
  const { ctx, w, h } = fitCanvas(cvs);
  ctx.fillStyle = '#141416'; ctx.fillRect(0, 0, w, h);
  if (phase >= 1) {
    ctx.strokeStyle = 'rgba(255,255,255,.22)'; ctx.lineWidth = 1;
    sketchEdges.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(sketchDots[a].x * w, sketchDots[a].y * h);
      ctx.lineTo(sketchDots[b].x * w, sketchDots[b].y * h);
      ctx.stroke();
    });
  }
  sketchDots.forEach(p => {
    const on = phase >= 2 && p.g === 0;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, on ? 2.6 : 2.1, 0, 7);
    ctx.fillStyle = on ? '#ffb454' : 'rgba(223,223,226,.85)';
    ctx.fill();
  });
  const note = $('sketchNote'); if (note) note.textContent = sketchLines[phase] || '';
}
const sketchCvs = $('sketch');
if (sketchCvs) new ResizeObserver(() => setSketch(+(document.querySelector('.step.is-active[data-sketch]')?.dataset.sketch || 0))).observe(sketchCvs);

const cleanRnd = mulberry(7);
const cleanDots = [];
for (let i = 0; i < 180; i++) {
  cleanDots.push({
    x: Math.min(0.96, Math.max(0.04, 0.5 + gauss(cleanRnd) * 0.2)),
    y: Math.min(0.78, Math.max(0.16, 0.46 + gauss(cleanRnd) * 0.18)),
    kind: i < 14 ? 'empty' : i < 20 ? 'out' : 'keep'
  });
}
const cleanLines = [
  'The published network. 7,624 users and 27,806 mutual follows.',
  'Users with an empty artist list cannot be compared. 173 dots leave. 7,451 users and 27,048 edges remain.',
  'Twenty users sit outside the largest connected component. The sample is 7,431 users, 27,045 edges, 7,842 artists, and 18 countries.',
  'Median follows: 4. The most-followed user has 214. Median artist list: 408. Spearman correlation of the two is 0.246. A like is a mark, not play time.'
];
function setClean(phase) {
  const cvs = $('cleanCanvas'); if (!cvs) return;
  const { ctx, w, h } = fitCanvas(cvs);
  ctx.fillStyle = '#141416'; ctx.fillRect(0, 0, w, h);
  if (phase < 3) {
    cleanDots.forEach(p => {
      let alpha = 0.9, y = p.y;
      if (phase >= 1 && p.kind === 'empty') alpha = 0.08;
      if (phase >= 2 && p.kind === 'out') { alpha = 0.12; y = Math.min(0.94, p.y + 0.18); }
      ctx.beginPath();
      ctx.arc(p.x * w, y * h, p.kind === 'keep' ? 2.3 : 2, 0, 7);
      ctx.fillStyle = `rgba(223,223,226,${alpha})`;
      ctx.fill();
    });
    ctx.fillStyle = '#ffb454'; ctx.font = '600 28px -apple-system, sans-serif';
    ctx.fillText(['7,624', '7,451', '7,431'][phase], 28, h - 28);
    ctx.fillStyle = '#a1a1a6'; ctx.font = '13px -apple-system, sans-serif';
    ctx.fillText('users', 28, h - 12);
  } else {
    function boxplot(x, title, unit, lo, hi, min, q1, med, mean, q3, max) {
      const y = h * 0.46;
      const X = v => x + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (w * 0.38);
      ctx.fillStyle = '#f5f5f7'; ctx.font = '600 15px -apple-system, sans-serif';
      ctx.fillText(title, x, 48);
      ctx.strokeStyle = '#8e8e93'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(min), y); ctx.lineTo(X(max), y); ctx.stroke();
      [min, max].forEach(v => { ctx.beginPath(); ctx.moveTo(X(v), y - 10); ctx.lineTo(X(v), y + 10); ctx.stroke(); });
      ctx.fillStyle = 'rgba(255,180,84,.25)';
      ctx.fillRect(X(q1), y - 16, X(q3) - X(q1), 32);
      ctx.strokeStyle = '#ffb454'; ctx.strokeRect(X(q1), y - 16, X(q3) - X(q1), 32);
      ctx.strokeStyle = '#f5f5f7'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(med), y - 16); ctx.lineTo(X(med), y + 16); ctx.stroke();
      ctx.fillStyle = '#7fb2d6';
      ctx.beginPath(); ctx.arc(X(mean), y, 5, 0, 7); ctx.fill();
      ctx.fillStyle = '#a1a1a6'; ctx.font = '12px -apple-system, sans-serif';
      [[min, 'min ' + min], [med, 'median ' + med], [max, 'max ' + max]].forEach(([v, lab], i) => {
        ctx.fillText(lab, X(v) - (i === 2 ? 48 : 0), y + 40 + (i === 1 ? 18 : 0));
      });
      ctx.fillStyle = '#7fb2d6';
      ctx.fillText('mean ' + mean, X(mean) - 18, y - 28);
      ctx.fillStyle = '#8e8e93';
      ctx.fillText(unit, x, h - 28);
    }
    boxplot(36, 'Follows', 'Log scale. Most people have a handful of follows.', 1, 214, 1, 2, 4, 7.28, 8, 214);
    boxplot(w * 0.52, 'Artists liked', 'A like is a mark, not how often someone listens.', 1, 944, 1, 219, 408, 404.39, 586, 944);
  }
  const note = $('cleanNote'); if (note) note.textContent = cleanLines[phase] || '';
}
const cleanCvs = $('cleanCanvas');
if (cleanCvs) new ResizeObserver(() => setClean(+(document.querySelector('.step.is-active[data-clean]')?.dataset.clean || 0))).observe(cleanCvs);

const QUESTIONS = [
  ['Research question', 'Are users in the same social circle more similar in their music preferences, on average, than users in different circles?']
];
function setShot(i) {
  document.querySelectorAll('#dataStage .shot').forEach(el => {
    el.hidden = el.dataset.shotPanel !== String(i);
  });
}
function setQuestion(i) {
  const p = $('qpanel'); if (!p) return;
  const q = QUESTIONS[i] || QUESTIONS[0];
  p.innerHTML = `<p class="q-kicker">${q[0]}</p><p class="q-line">${q[1]}</p>`;
}
setQuestion(0);
setSketch(0);
setClean(0);
startScrolly();

/* Right-hand contents. Closed, it is a stack of dots. Open, the names can be chosen. */
(function railNav() {
  const rail = $('rail');
  if (!rail) return;
  const toggle = rail.querySelector('.rail-toggle');
  const links = [...rail.querySelectorAll('a')];
  toggle.addEventListener('click', () => {
    const open = rail.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', open);
  });
  function openRail(on) {
    rail.classList.toggle('is-open', on);
    toggle.setAttribute('aria-expanded', on);
  }
  rail.addEventListener('mouseenter', () => openRail(true));
  rail.addEventListener('mouseleave', () => openRail(false));
  links.forEach(a => a.addEventListener('click', e => {
    if (!rail.classList.contains('is-open')) {
      e.preventDefault();
      openRail(true);
    }
  }));
  const dots = [...toggle.querySelectorAll('i')];
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const id = '#' + e.target.id;
      links.forEach((a, i) => {
        const on = a.getAttribute('href') === id;
        a.classList.toggle('is-here', on);
        if (on && dots.length) dots.forEach((d, k) => d.classList.toggle('on', k === i % dots.length));
      });
    });
  }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
  links.forEach(a => { const s = document.querySelector(a.getAttribute('href')); if (s) io.observe(s); });
})();
