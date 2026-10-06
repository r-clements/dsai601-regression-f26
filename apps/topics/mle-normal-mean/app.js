// ---------- random numbers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normalPDF(x, mu, sigma) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}

// ---------- model ----------

const TRUE_MU = 5;
const SIGMA = 2; // known
const MU_MIN = TRUE_MU - 3 * SIGMA;
const MU_MAX = TRUE_MU + 3 * SIGMA;
const X_MIN = TRUE_MU - 4 * SIGMA;
const X_MAX = TRUE_MU + 4 * SIGMA;
const PEAK_DENSITY = normalPDF(0, 0, SIGMA);
const GRID_N = 220;
const N = 30;

// A deliberately-off starting guess, randomized left/right so repeated
// resamples don't always place it (and its tall nearby stems) on the same side.
function randomStartMu() {
  const sign = Math.random() < 0.5 ? -1 : 1;
  return TRUE_MU + sign * 1.8 * SIGMA;
}

function generateSample(n) {
  const pts = [];
  for (let i = 0; i < n; i++) pts.push(TRUE_MU + randNormal() * SIGMA);
  return pts;
}

// Closed form: logL(mu) = -n/2 * log(2*pi*sigma^2) - sum((x-mu)^2) / (2*sigma^2)
function logLikelihood(mu, pts) {
  const n = pts.length;
  const ss = d3.sum(pts, x => (x - mu) ** 2);
  return -0.5 * n * Math.log(2 * Math.PI * SIGMA * SIGMA) - ss / (2 * SIGMA * SIGMA);
}

// ---------- panel helper ----------

function setupPanel(containerId, ariaLabel, width, height, margin) {
  const svg = d3.select(containerId).append("svg").attr("width", width).attr("height", height)
    .attr("role", "img")
    .attr("aria-label", ariaLabel);
  const x = d3.scaleLinear().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().range([height - margin.bottom, margin.top]);
  const xAxisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
  const yAxisG = svg.append("g").attr("transform", `translate(${margin.left},0)`);
  const decorLayer = svg.append("g");
  const curveLayer = svg.append("g");
  const pointLayer = svg.append("g");
  const markerLayer = svg.append("g");
  return { svg, x, y, xAxisG, yAxisG, decorLayer, curveLayer, pointLayer, markerLayer, width, height, margin };
}

function addAxisLabels(svg, width, height, xLabel, yLabel) {
  svg.selectAll("text.x-label").data([xLabel]).join("text")
    .attr("class", "axis-label x-label")
    .attr("text-anchor", "middle")
    .attr("x", width / 2).attr("y", height - 4)
    .text(d => d);
  svg.selectAll("text.y-label").data([yLabel]).join("text")
    .attr("class", "axis-label y-label")
    .attr("text-anchor", "middle")
    .attr("transform", `translate(14,${height / 2}) rotate(-90)`)
    .text(d => d);
}

// ---------- density panel ----------

const densityWidth = 900, densityHeight = 320;
const densityMargin = { top: 15, right: 20, bottom: 40, left: 55 };
const densityPanel = setupPanel(
  "#density-panel",
  "Normal density curve for the current guess of mu, with the sample points along the x-axis and dashed stems up to their density values on the curve. A draggable red handle above the curve sets the guess.",
  densityWidth, densityHeight, densityMargin
);
densityPanel.x.domain([X_MIN, X_MAX]);
densityPanel.y.domain([0, PEAK_DENSITY * 1.15]);

const densityDrag = d3.drag().on("drag", (event) => {
  setMu(densityPanel.x.invert(event.x));
});
densityPanel.svg.append("rect")
  .attr("x", densityMargin.left).attr("y", densityMargin.top)
  .attr("width", densityWidth - densityMargin.left - densityMargin.right)
  .attr("height", densityHeight - densityMargin.top - densityMargin.bottom)
  .attr("fill", "transparent")
  .style("pointer-events", "all")
  .style("cursor", "ew-resize")
  .on("click", (event) => setMu(densityPanel.x.invert(d3.pointer(event)[0])))
  .call(densityDrag);

function renderDensityPanel() {
  const curve = d3.range(160).map(i => {
    const x = X_MIN + (X_MAX - X_MIN) * i / 159;
    return { x, y: normalPDF(x, muHat, SIGMA) };
  });
  const lineGen = d3.line().x(d => densityPanel.x(d.x)).y(d => densityPanel.y(d.y));
  densityPanel.curveLayer.selectAll("path.guess-curve").data([curve]).join("path")
    .attr("class", "guess-curve")
    .attr("d", lineGen);

  densityPanel.xAxisG.call(d3.axisBottom(densityPanel.x));
  densityPanel.yAxisG.call(d3.axisLeft(densityPanel.y).ticks(5));
  addAxisLabels(densityPanel.svg, densityWidth, densityHeight, "x", "density");

  const stemData = points.map(xi => ({ x: xi, d: normalPDF(xi, muHat, SIGMA) }));
  densityPanel.pointLayer.selectAll("line.density-stem").data(stemData).join("line")
    .attr("class", "density-stem")
    .attr("x1", d => densityPanel.x(d.x)).attr("x2", d => densityPanel.x(d.x))
    .attr("y1", densityPanel.y(0)).attr("y2", d => densityPanel.y(d.d));
  densityPanel.pointLayer.selectAll("circle.density-dot").data(stemData).join("circle")
    .attr("class", "density-dot")
    .attr("r", 4)
    .attr("cx", d => densityPanel.x(d.x)).attr("cy", d => densityPanel.y(d.d));
  densityPanel.pointLayer.selectAll("circle.point").data(points).join("circle")
    .attr("class", "point")
    .attr("r", 4)
    .attr("cx", d => densityPanel.x(d)).attr("cy", densityPanel.y(0));

  const handleX = densityPanel.x(muHat);
  const top = densityMargin.top, bottom = densityHeight - densityMargin.bottom;
  densityPanel.markerLayer.selectAll("line.guess-handle-line").data([muHat]).join("line")
    .attr("class", "guess-handle-line")
    .attr("x1", handleX).attr("x2", handleX)
    .attr("y1", top).attr("y2", bottom);
  densityPanel.markerLayer.selectAll("path.guess-handle-grip").data([muHat]).join("path")
    .attr("class", "guess-handle-grip")
    .attr("d", `M${handleX - 6},${top} L${handleX + 6},${top} L${handleX},${top + 9} Z`);
}

// ---------- likelihood / log-likelihood panels ----------

const curveWidth = 440, curveHeight = 300;
const curveMargin = { top: 15, right: 15, bottom: 40, left: 65 };

const likPanel = setupPanel(
  "#likelihood-panel",
  "Likelihood L as a function of the candidate mean mu, a bell-shaped curve peaking at the sample mean. A red dot marks the current guess; a blue ring marks the maximum.",
  curveWidth, curveHeight, curveMargin
);
likPanel.x.domain([MU_MIN, MU_MAX]);

const loglikPanel = setupPanel(
  "#loglik-panel",
  "Log-likelihood as a function of the candidate mean mu, a downward parabola peaking at the sample mean. A red dot marks the current guess; a blue ring marks the maximum.",
  curveWidth, curveHeight, curveMargin
);
loglikPanel.x.domain([MU_MIN, MU_MAX]);

function renderRefLines(panel) {
  panel.decorLayer.selectAll("line.true-ref-line").data(showTrueMu ? [TRUE_MU] : []).join("line")
    .attr("class", "true-ref-line")
    .attr("x1", d => panel.x(d)).attr("x2", d => panel.x(d))
    .attr("y1", panel.margin.top).attr("y2", panel.height - panel.margin.bottom);
  panel.decorLayer.selectAll("text.true-ref-label").data(showTrueMu ? [TRUE_MU] : []).join("text")
    .attr("class", "ref-label true-ref-label")
    .attr("x", d => panel.x(d) + 4).attr("y", panel.margin.top + 10)
    .text("true μ");
}

function renderAllRefLines() {
  renderRefLines(densityPanel);
  renderRefLines(likPanel);
  renderRefLines(loglikPanel);
}

function renderStaticCurves() {
  const likExtent = [0, d3.max(grid, d => d.L) * 1.15];
  likPanel.y.domain(likExtent);
  const loglikExtent = d3.extent(grid, d => d.logL);
  const loglikPad = (loglikExtent[1] - loglikExtent[0]) * 0.08 || 1;
  loglikPanel.y.domain([loglikExtent[0] - loglikPad, loglikExtent[1] + loglikPad]);

  likPanel.xAxisG.call(d3.axisBottom(likPanel.x).ticks(6));
  likPanel.yAxisG.call(d3.axisLeft(likPanel.y).ticks(5).tickFormat(d3.format(".1e")));
  addAxisLabels(likPanel.svg, curveWidth, curveHeight, "μ", "L(μ)");

  loglikPanel.xAxisG.call(d3.axisBottom(loglikPanel.x).ticks(6));
  loglikPanel.yAxisG.call(d3.axisLeft(loglikPanel.y).ticks(5).tickFormat(d3.format(".1f")));
  addAxisLabels(loglikPanel.svg, curveWidth, curveHeight, "μ", "ℓ(μ)");

  const likLine = d3.line().x(d => likPanel.x(d.mu)).y(d => likPanel.y(d.L));
  likPanel.curveLayer.selectAll("path.fn-curve").data([grid]).join("path")
    .attr("class", "fn-curve")
    .attr("d", likLine);

  const loglikLine = d3.line().x(d => loglikPanel.x(d.mu)).y(d => loglikPanel.y(d.logL));
  loglikPanel.curveLayer.selectAll("path.fn-curve").data([grid]).join("path")
    .attr("class", "fn-curve")
    .attr("d", loglikLine);

  likPanel.markerLayer.selectAll("circle.optimum-marker").data([{ mu: xbar, v: xbarL }]).join("circle")
    .attr("class", "optimum-marker")
    .attr("r", 7)
    .attr("cx", d => likPanel.x(d.mu)).attr("cy", d => likPanel.y(d.v));

  loglikPanel.markerLayer.selectAll("circle.optimum-marker").data([{ mu: xbar, v: xbarLogL }]).join("circle")
    .attr("class", "optimum-marker")
    .attr("r", 7)
    .attr("cx", d => loglikPanel.x(d.mu)).attr("cy", d => loglikPanel.y(d.v));
}

function renderCurveMarkers() {
  likPanel.markerLayer.selectAll("circle.current-marker").data([{ mu: muHat, v: curL }]).join("circle")
    .attr("class", "current-marker")
    .attr("r", 6)
    .attr("cx", d => likPanel.x(d.mu)).attr("cy", d => likPanel.y(d.v));

  loglikPanel.markerLayer.selectAll("circle.current-marker").data([{ mu: muHat, v: curLogL }]).join("circle")
    .attr("class", "current-marker")
    .attr("r", 6)
    .attr("cx", d => loglikPanel.x(d.mu)).attr("cy", d => loglikPanel.y(d.v));
}

// ---------- state + wiring ----------

let points = [];
let xbar = 0, xbarL = 0, xbarLogL = 0;
let grid = [];
let muHat = randomStartMu();
let curL = 0, curLogL = 0;
let showTrueMu = true;

const muSlider = document.getElementById("mu-slider");
const muReadout = document.getElementById("mu-readout");
const showTrueCheckbox = document.getElementById("show-true");

muSlider.min = MU_MIN;
muSlider.max = MU_MAX;
muSlider.step = (MU_MAX - MU_MIN) / 400;

function rebuildGrid() {
  grid = d3.range(GRID_N).map(i => {
    const mu = MU_MIN + (MU_MAX - MU_MIN) * i / (GRID_N - 1);
    const logL = logLikelihood(mu, points);
    return { mu, logL, L: Math.exp(logL) };
  });
}

function setMu(mu) {
  muHat = clamp(mu, MU_MIN, MU_MAX);
  muSlider.value = muHat;
  render();
}

function render() {
  curLogL = logLikelihood(muHat, points);
  curL = Math.exp(curLogL);

  muReadout.textContent = muHat.toFixed(3);
  renderDensityPanel();
  renderCurveMarkers();

  document.getElementById("cur-n").textContent = points.length;
  document.getElementById("cur-sigma").textContent = SIGMA.toFixed(2);
  document.getElementById("cur-mu").textContent = muHat.toFixed(3);
  document.getElementById("cur-xbar").textContent = xbar.toFixed(3);
  document.getElementById("cur-true-mu").textContent = TRUE_MU.toFixed(2);
  document.getElementById("cur-L").textContent = curL.toExponential(3);
  document.getElementById("cur-logL").textContent = curLogL.toFixed(3);
}

function regenerate() {
  points = generateSample(N);
  xbar = d3.mean(points);
  xbarLogL = logLikelihood(xbar, points);
  xbarL = Math.exp(xbarLogL);
  rebuildGrid();

  renderStaticCurves();
  document.getElementById("true-mu-stat").style.display = showTrueMu ? "" : "none";
  renderAllRefLines();

  setMu(randomStartMu());
}

muSlider.addEventListener("input", () => setMu(parseFloat(muSlider.value)));
document.getElementById("new-sample").addEventListener("click", regenerate);
document.getElementById("snap-mle").addEventListener("click", () => setMu(xbar));
showTrueCheckbox.addEventListener("change", () => {
  showTrueMu = showTrueCheckbox.checked;
  document.getElementById("true-mu-stat").style.display = showTrueMu ? "" : "none";
  renderAllRefLines();
});

regenerate();
