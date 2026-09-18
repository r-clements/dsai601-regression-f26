// ---------- random numbers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- data generation ----------

const BASE_B0 = 5, BASE_B1 = 1.5, SIGMA = 2;
const TRUE_DELTA0 = 2; // fixed true intercept difference (Group B - Group A)
const N_PER_GROUP = 30;

function generateData(delta1) {
  const pts = [];
  for (let g = 0; g < 2; g++) {
    const b0 = BASE_B0 + (g === 1 ? TRUE_DELTA0 : 0);
    const b1 = BASE_B1 + (g === 1 ? delta1 : 0);
    for (let i = 0; i < N_PER_GROUP; i++) {
      const x = Math.random() * 10;
      const y = b0 + b1 * x + randNormal() * SIGMA;
      pts.push({ x, y, group: g });
    }
  }
  return pts;
}

// ---------- general OLS ----------

function solveLinearSystemN(A, c) {
  const n = A.length;
  const M = A.map((row, i) => [...row, c[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    }
    if (Math.abs(M[maxRow][col]) < 1e-9) return null;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let cc = col; cc <= n; cc++) M[r][cc] -= factor * M[col][cc];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

function fitModel(points, cols) {
  const p = cols.length;
  const A = [], c = [];
  for (let i = 0; i < p; i++) {
    const row = [];
    for (let j = 0; j < p; j++) {
      row.push(d3.sum(points, (pt, idx) => cols[i][idx] * cols[j][idx]));
    }
    A.push(row);
    c.push(d3.sum(points, (pt, idx) => cols[i][idx] * pt.y));
  }
  const coefs = solveLinearSystemN(A, c);
  const fitted = points.map((pt, idx) => coefs.reduce((sum, b, j) => sum + b * cols[j][idx], 0));
  const SSE = d3.sum(points, (pt, idx) => (pt.y - fitted[idx]) ** 2);
  return { coefs, SSE };
}

function fitDummyOnly(points) {
  const ones = points.map(() => 1);
  const xs = points.map(p => p.x);
  const groups = points.map(p => p.group);
  const { coefs, SSE } = fitModel(points, [ones, xs, groups]);
  return { b0: coefs[0], b1: coefs[1], b2: coefs[2], SSE };
}

function fitInteraction(points) {
  const ones = points.map(() => 1);
  const xs = points.map(p => p.x);
  const groups = points.map(p => p.group);
  const xg = points.map(p => p.x * p.group);
  const { coefs, SSE } = fitModel(points, [ones, xs, groups, xg]);
  return { b0: coefs[0], b1: coefs[1], b2: coefs[2], b3: coefs[3], SSE };
}

// ---------- main panel ----------

const width = 760, height = 440;
const margin = { top: 20, right: 20, bottom: 45, left: 55 };

const svg = d3.select("#main-panel").append("svg")
  .attr("width", width).attr("height", height)
  .attr("role", "img")
  .attr("aria-label", "Scatter plot of two groups (circles and triangles), each with dashed parallel lines from a no-interaction model and solid separate-slope lines from an interaction model.");

const x = d3.scaleLinear().range([margin.left, width - margin.right]);
const y = d3.scaleLinear().range([height - margin.bottom, margin.top]);
const xAxisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
const yAxisG = svg.append("g").attr("transform", `translate(${margin.left},0)`);
const lineLayer = svg.append("g");
const pointLayer = svg.append("g");

function addAxisLabels(svgSel, w, h, xLabel, yLabel) {
  svgSel.selectAll("text.x-label").data([xLabel]).join("text")
    .attr("class", "axis-label x-label")
    .attr("text-anchor", "middle")
    .attr("x", w / 2).attr("y", h - 4)
    .text(d => d);
  svgSel.selectAll("text.y-label").data([yLabel]).join("text")
    .attr("class", "axis-label y-label")
    .attr("text-anchor", "middle")
    .attr("transform", `translate(12,${h / 2}) rotate(-90)`)
    .text(d => d);
}

const symbolGen = d3.symbol().size(70);
function symbolPath(group) {
  return symbolGen.type(group === 0 ? d3.symbolCircle : d3.symbolTriangle)();
}

function setDomain(points) {
  const xExt = d3.extent(points, p => p.x);
  const yExt = d3.extent(points, p => p.y);
  const xPad = (xExt[1] - xExt[0]) * 0.1 || 1;
  const yPad = (yExt[1] - yExt[0]) * 0.15 || 1;
  x.domain([xExt[0] - xPad, xExt[1] + xPad]);
  y.domain([yExt[0] - yPad, yExt[1] + yPad]);
}

function renderMain(points, dummyFit, interactionFit) {
  xAxisG.call(d3.axisBottom(x));
  yAxisG.call(d3.axisLeft(y));

  const [x0, x1] = x.domain();
  const lines = [
    { key: "dummy-a", cls: "line-dummy-a", x1: x0, y1: dummyFit.b0 + dummyFit.b1 * x0, x2: x1, y2: dummyFit.b0 + dummyFit.b1 * x1 },
    { key: "dummy-b", cls: "line-dummy-b", x1: x0, y1: dummyFit.b0 + dummyFit.b2 + dummyFit.b1 * x0, x2: x1, y2: dummyFit.b0 + dummyFit.b2 + dummyFit.b1 * x1 },
    { key: "int-a", cls: "line-interaction-a", x1: x0, y1: interactionFit.b0 + interactionFit.b1 * x0, x2: x1, y2: interactionFit.b0 + interactionFit.b1 * x1 },
    { key: "int-b", cls: "line-interaction-b",
      x1: x0, y1: interactionFit.b0 + interactionFit.b2 + (interactionFit.b1 + interactionFit.b3) * x0,
      x2: x1, y2: interactionFit.b0 + interactionFit.b2 + (interactionFit.b1 + interactionFit.b3) * x1 }
  ];

  lineLayer.selectAll("line").data(lines, d => d.key).join("line")
    .attr("class", d => d.cls)
    .attr("x1", d => x(d.x1)).attr("y1", d => y(d.y1))
    .attr("x2", d => x(d.x2)).attr("y2", d => y(d.y2));

  pointLayer.selectAll("path").data(points).join("path")
    .attr("d", d => symbolPath(d.group))
    .attr("class", d => d.group === 0 ? "point-a" : "point-b")
    .attr("transform", d => `translate(${x(d.x)},${y(d.y)})`);

  addAxisLabels(svg, width, height, "x", "y");
}

// ---------- state + wiring ----------

let points = [];

function formatNum(v) { return v.toFixed(2); }

function render() {
  const dummyFit = fitDummyOnly(points);
  const interactionFit = fitInteraction(points);

  renderMain(points, dummyFit, interactionFit);

  document.getElementById("d-b0").textContent = formatNum(dummyFit.b0);
  document.getElementById("d-b1").textContent = formatNum(dummyFit.b1);
  document.getElementById("d-b2").textContent = formatNum(dummyFit.b2);
  document.getElementById("d-sse").textContent = formatNum(dummyFit.SSE);
  document.getElementById("d-a-b0").textContent = formatNum(dummyFit.b0);
  document.getElementById("d-a-b1").textContent = formatNum(dummyFit.b1);
  document.getElementById("d-b-b0").textContent = formatNum(dummyFit.b0 + dummyFit.b2);
  document.getElementById("d-b-b1").textContent = formatNum(dummyFit.b1);

  document.getElementById("i-b0").textContent = formatNum(interactionFit.b0);
  document.getElementById("i-b1").textContent = formatNum(interactionFit.b1);
  document.getElementById("i-b2").textContent = formatNum(interactionFit.b2);
  document.getElementById("i-b3").textContent = formatNum(interactionFit.b3);
  document.getElementById("i-sse").textContent = formatNum(interactionFit.SSE);
  document.getElementById("i-a-b0").textContent = formatNum(interactionFit.b0);
  document.getElementById("i-a-b1").textContent = formatNum(interactionFit.b1);
  document.getElementById("i-b-b0").textContent = formatNum(interactionFit.b0 + interactionFit.b2);
  document.getElementById("i-b-b1").textContent = formatNum(interactionFit.b1 + interactionFit.b3);

  const ssReduction = dummyFit.SSE - interactionFit.SSE;
  document.getElementById("ss-reduction").textContent = formatNum(ssReduction);
  document.getElementById("sse-ratio-denom").textContent = formatNum(dummyFit.SSE);
}

function resample() {
  const delta1 = parseFloat(document.getElementById("delta1").value);
  points = generateData(delta1);
  setDomain(points);
  render();
}

const delta1Slider = document.getElementById("delta1");
delta1Slider.addEventListener("input", () => {
  document.getElementById("delta1-value").textContent = parseFloat(delta1Slider.value).toFixed(2);
  resample();
});
document.getElementById("resample").addEventListener("click", resample);

resample();
