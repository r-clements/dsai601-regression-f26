// ---------- random numbers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- data generation ----------

const TRUE_B0 = 1, TRUE_B1 = 2, TRUE_B2 = -1.5;
const SIGMA = 1.5;

function correlatedPredictors(n, r) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const z1 = randNormal();
    const z2 = randNormal();
    const x1std = z1;
    const x2std = r * z1 + Math.sqrt(1 - r * r) * z2;
    pts.push({ x1: 5 + 2 * x1std, x2: 5 + 2 * x2std });
  }
  return pts;
}

function generateSample(n, r) {
  return correlatedPredictors(n, r).map(p => ({
    x1: p.x1,
    x2: p.x2,
    y: TRUE_B0 + TRUE_B1 * p.x1 + TRUE_B2 * p.x2 + randNormal() * SIGMA
  }));
}

// ---------- 3x3 linear solve (Gaussian elimination, partial pivoting) ----------

function solveLinearSystem3(A, c) {
  const M = A.map((row, i) => [...row, c[i]]);
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    }
    if (Math.abs(M[maxRow][col]) < 1e-9) return null;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let cc = col; cc < 4; cc++) M[r][cc] -= factor * M[col][cc];
    }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

function computeFit(points) {
  const n = points.length;
  if (n < 4) return null;

  let Sx1 = 0, Sx2 = 0, Sy = 0, Sx1x1 = 0, Sx2x2 = 0, Sx1x2 = 0, Sx1y = 0, Sx2y = 0;
  points.forEach(p => {
    Sx1 += p.x1; Sx2 += p.x2; Sy += p.y;
    Sx1x1 += p.x1 * p.x1; Sx2x2 += p.x2 * p.x2; Sx1x2 += p.x1 * p.x2;
    Sx1y += p.x1 * p.y; Sx2y += p.x2 * p.y;
  });

  const A = [
    [n, Sx1, Sx2],
    [Sx1, Sx1x1, Sx1x2],
    [Sx2, Sx1x2, Sx2x2]
  ];
  const c = [Sy, Sx1y, Sx2y];
  const solved = solveLinearSystem3(A, c);
  if (!solved) return null;
  const [b0, b1, b2] = solved;
  return { b0, b1, b2 };
}

function runMonteCarlo(r, replicates, nEach) {
  const results = [];
  for (let i = 0; i < replicates; i++) {
    const fit = computeFit(generateSample(nEach, r));
    if (fit) results.push({ b1: fit.b1, b2: fit.b2 });
  }
  return results;
}

// ---------- generic small-multiple panel ----------

function setupPanel(containerId, ariaLabel) {
  const width = 440, height = 380;
  const margin = { top: 15, right: 15, bottom: 45, left: 55 };
  const svg = d3.select(containerId).append("svg").attr("width", width).attr("height", height)
    .attr("role", "img")
    .attr("aria-label", ariaLabel);
  const x = d3.scaleLinear().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().range([height - margin.bottom, margin.top]);
  const xAxisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
  const yAxisG = svg.append("g").attr("transform", `translate(${margin.left},0)`);
  const decorLayer = svg.append("g");
  const pointLayer = svg.append("g");
  return { svg, x, y, xAxisG, yAxisG, decorLayer, pointLayer, width, height, margin };
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
    .attr("transform", `translate(12,${height / 2}) rotate(-90)`)
    .text(d => d);
}

function renderScatterPanel(panel, data, xAcc, yAcc) {
  const xExt = d3.extent(data, xAcc);
  const yExt = d3.extent(data, yAcc);
  const xPad = (xExt[1] - xExt[0]) * 0.15 || 1;
  const yPad = (yExt[1] - yExt[0]) * 0.15 || 1;
  panel.x.domain([xExt[0] - xPad, xExt[1] + xPad]);
  panel.y.domain([yExt[0] - yPad, yExt[1] + yPad]);
  panel.xAxisG.call(d3.axisBottom(panel.x).ticks(5));
  panel.yAxisG.call(d3.axisLeft(panel.y).ticks(5));
  panel.pointLayer.selectAll("circle.diag-point")
    .data(data)
    .join("circle")
    .attr("class", "diag-point")
    .attr("r", 4)
    .attr("cx", d => panel.x(xAcc(d)))
    .attr("cy", d => panel.y(yAcc(d)));
}

const predictorPanel = setupPanel("#predictor-scatter", "Scatter plot of one simulated sample of the two predictors x1 and x2. As the correlation slider increases in magnitude, the points fall closer to a straight line.");
const coefPanel = setupPanel("#coef-cloud", "Scatter plot of estimated slope coefficients b1 and b2 from 150 independent resamples at the current correlation, with a red cross marking the true values. As correlation increases, the cloud stretches into a narrow diagonal ridge, showing the coefficients becoming unstable and anti-correlated.");

function renderPredictorPanel(sample) {
  renderScatterPanel(predictorPanel, sample, d => d.x1, d => d.x2);
  addAxisLabels(predictorPanel.svg, predictorPanel.width, predictorPanel.height, "x₁", "x₂");
}

function renderCoefPanel(results) {
  // ensure the true point is always within the domain even if estimates cluster tightly
  const domainData = results.concat([{ b1: TRUE_B1, b2: TRUE_B2 }]);
  renderScatterPanel(coefPanel, domainData, d => d.b1, d => d.b2);

  // remove the synthetic point from the plotted cloud (it was only used for domain padding)
  coefPanel.pointLayer.selectAll("circle.diag-point")
    .data(results)
    .join("circle")
    .attr("class", "diag-point")
    .attr("r", 4)
    .attr("cx", d => coefPanel.x(d.b1))
    .attr("cy", d => coefPanel.y(d.b2));

  const [x0, x1] = coefPanel.x.domain();
  const [y0, y1] = coefPanel.y.domain();

  coefPanel.decorLayer.selectAll("line.ref-line-v").data([0]).join("line")
    .attr("class", "ref-line ref-line-v")
    .attr("x1", coefPanel.x(TRUE_B1)).attr("x2", coefPanel.x(TRUE_B1))
    .attr("y1", coefPanel.y(y0)).attr("y2", coefPanel.y(y1));

  coefPanel.decorLayer.selectAll("line.ref-line-h").data([0]).join("line")
    .attr("class", "ref-line ref-line-h")
    .attr("x1", coefPanel.x(x0)).attr("x2", coefPanel.x(x1))
    .attr("y1", coefPanel.y(TRUE_B2)).attr("y2", coefPanel.y(TRUE_B2));

  const cx = coefPanel.x(TRUE_B1), cy = coefPanel.y(TRUE_B2);
  const armLen = 7;
  coefPanel.decorLayer.selectAll("line.true-marker").data([0, 1]).join("line")
    .attr("class", "true-marker")
    .attr("x1", d => d === 0 ? cx - armLen : cx)
    .attr("y1", d => d === 0 ? cy : cy - armLen)
    .attr("x2", d => d === 0 ? cx + armLen : cx)
    .attr("y2", d => d === 0 ? cy : cy + armLen);

  addAxisLabels(coefPanel.svg, coefPanel.width, coefPanel.height, "b̂₁", "b̂₂");
}

// ---------- state + wiring ----------

const rSlider = document.getElementById("correlation");
const rValueLabel = document.getElementById("r-value");

function run() {
  const r = parseFloat(rSlider.value);
  rValueLabel.textContent = r.toFixed(2);

  const sample = generateSample(60, r);
  const results = runMonteCarlo(r, 150, 30);

  renderPredictorPanel(sample);
  renderCoefPanel(results);

  const vif = 1 / (1 - r * r);
  const sdB1 = d3.deviation(results, d => d.b1);
  const sdB2 = d3.deviation(results, d => d.b2);

  document.getElementById("vif").textContent = isFinite(vif) ? vif.toFixed(2) : "∞";
  document.getElementById("sd-b1").textContent = sdB1 !== undefined ? sdB1.toFixed(3) : "–";
  document.getElementById("sd-b2").textContent = sdB2 !== undefined ? sdB2.toFixed(3) : "–";
}

rSlider.addEventListener("input", run);
document.getElementById("resample").addEventListener("click", run);

run();
