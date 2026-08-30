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

// ---------- data generation ----------

const TRUE_B0 = 2, TRUE_B1 = 1.3, SIGMA = 1.5;

function generateData(n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 10;
    const y = TRUE_B0 + TRUE_B1 * x + randNormal() * SIGMA;
    pts.push({ x, y });
  }
  return pts;
}

// ---------- OLS closed form ----------

function fitOLS(points) {
  const n = points.length;
  const meanX = d3.mean(points, d => d.x);
  const meanY = d3.mean(points, d => d.y);
  const Sxx = d3.sum(points, d => (d.x - meanX) ** 2);
  const slope = Sxx === 0 ? 0 : d3.sum(points, d => (d.x - meanX) * (d.y - meanY)) / Sxx;
  const intercept = meanY - slope * meanX;
  const resid = points.map(d => d.y - (intercept + slope * d.x));
  const SSR = d3.sum(resid, r => r * r);
  const s = Math.sqrt(SSR / Math.max(n - 2, 1));
  const seSlope = Sxx === 0 ? 1 : s / Math.sqrt(Sxx);
  const seIntercept = Sxx === 0 ? 1 : s * Math.sqrt(1 / n + meanX * meanX / Sxx);
  return { n, slope, intercept, SSR, s, seSlope, seIntercept };
}

function ssrAt(points, b0, b1) {
  return d3.sum(points, d => (d.y - b0 - b1 * d.x) ** 2);
}

// ---------- scatter + line panel ----------

const scatterWidth = 440, scatterHeight = 360;
const scatterMargin = { top: 20, right: 20, bottom: 40, left: 50 };

const scatterSvg = d3.select("#scatter-panel").append("svg")
  .attr("width", scatterWidth).attr("height", scatterHeight)
  .attr("role", "img")
  .attr("aria-label", "Scatter plot of the simulated data with the line for the currently selected beta-0 and beta-1, and dashed residual segments.");

const scatterX = d3.scaleLinear().range([scatterMargin.left, scatterWidth - scatterMargin.right]);
const scatterY = d3.scaleLinear().range([scatterHeight - scatterMargin.bottom, scatterMargin.top]);
const scatterXAxisG = scatterSvg.append("g").attr("transform", `translate(0,${scatterHeight - scatterMargin.bottom})`);
const scatterYAxisG = scatterSvg.append("g").attr("transform", `translate(${scatterMargin.left},0)`);
const scatterResidualLayer = scatterSvg.append("g");
const scatterLineLayer = scatterSvg.append("g");
const scatterPointLayer = scatterSvg.append("g");

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

function setScatterDomain(pts) {
  const xExt = d3.extent(pts, d => d.x);
  const yExt = d3.extent(pts, d => d.y);
  const xPad = (xExt[1] - xExt[0]) * 0.15 || 1;
  const yPad = (yExt[1] - yExt[0]) * 0.25 || 1;
  scatterX.domain([xExt[0] - xPad, xExt[1] + xPad]);
  scatterY.domain([yExt[0] - yPad, yExt[1] + yPad]);
}

function renderScatter() {
  scatterXAxisG.call(d3.axisBottom(scatterX));
  scatterYAxisG.call(d3.axisLeft(scatterY));

  const domain = scatterX.domain();
  const lineData = [
    { x: domain[0], y: curB0 + curB1 * domain[0] },
    { x: domain[1], y: curB0 + curB1 * domain[1] }
  ];
  scatterLineLayer.selectAll("line.fit-line").data([lineData]).join("line")
    .attr("class", "fit-line")
    .attr("x1", d => scatterX(d[0].x)).attr("y1", d => scatterY(d[0].y))
    .attr("x2", d => scatterX(d[1].x)).attr("y2", d => scatterY(d[1].y));

  scatterResidualLayer.selectAll("line.residual").data(points).join("line")
    .attr("class", "residual")
    .attr("x1", d => scatterX(d.x)).attr("x2", d => scatterX(d.x))
    .attr("y1", d => scatterY(d.y)).attr("y2", d => scatterY(curB0 + curB1 * d.x));

  scatterPointLayer.selectAll("circle").data(points).join("circle")
    .attr("class", "point")
    .attr("r", 4)
    .attr("cx", d => scatterX(d.x))
    .attr("cy", d => scatterY(d.y));

  addAxisLabels(scatterSvg, scatterWidth, scatterHeight, "x", "y");
}

// ---------- shared beta grid ----------

const GRID_N = 50;
let b0Min, b0Max, b1Min, b1Max;
let grid, ssrMin, ssrMax;

function rebuildGrid() {
  grid = new Float64Array(GRID_N * GRID_N);
  let mn = Infinity, mx = -Infinity;
  for (let row = 0; row < GRID_N; row++) {
    const b1v = b1Min + (b1Max - b1Min) * row / (GRID_N - 1);
    for (let col = 0; col < GRID_N; col++) {
      const b0v = b0Min + (b0Max - b0Min) * col / (GRID_N - 1);
      const v = ssrAt(points, b0v, b1v);
      grid[row * GRID_N + col] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  ssrMin = mn; ssrMax = mx;
}

function gridB0(col) { return b0Min + (b0Max - b0Min) * col / (GRID_N - 1); }
function gridB1(row) { return b1Min + (b1Max - b1Min) * row / (GRID_N - 1); }

// ---------- contour panel ----------

const contourWidth = 900, contourHeight = 420;
const contourMargin = { top: 15, right: 20, bottom: 40, left: 55 };

const contourSvg = d3.select("#contour-panel").append("svg")
  .attr("width", contourWidth).attr("height", contourHeight)
  .attr("role", "img")
  .attr("aria-label", "Contour map of the sum of squared residuals as a function of beta-0 and beta-1, viewed from above. A red dot marks the current selection; a blue ring marks the least-squares optimum. Click or drag anywhere on the map to change the selection.");

const contourX = d3.scaleLinear().range([contourMargin.left, contourWidth - contourMargin.right]);
const contourY = d3.scaleLinear().range([contourHeight - contourMargin.bottom, contourMargin.top]);
const contourXAxisG = contourSvg.append("g").attr("transform", `translate(0,${contourHeight - contourMargin.bottom})`);
const contourYAxisG = contourSvg.append("g").attr("transform", `translate(${contourMargin.left},0)`);
const contourFillLayer = contourSvg.append("g");
const contourMarkerLayer = contourSvg.append("g");

const contourCaptureRect = contourSvg.append("rect")
  .attr("fill", "transparent")
  .style("pointer-events", "all")
  .on("click", (event) => {
    const [px, py] = d3.pointer(event);
    setBeta(contourX.invert(px), contourY.invert(py));
  });

const contourDrag = d3.drag().on("drag", (event) => {
  setBeta(contourX.invert(event.x), contourY.invert(event.y));
});
contourCaptureRect.call(contourDrag);

function renderContourSurface() {
  contourX.domain([b0Min, b0Max]);
  contourY.domain([b1Min, b1Max]);
  contourCaptureRect
    .attr("x", contourMargin.left).attr("y", contourMargin.top)
    .attr("width", contourWidth - contourMargin.left - contourMargin.right)
    .attr("height", contourHeight - contourMargin.top - contourMargin.bottom);

  contourXAxisG.call(d3.axisBottom(contourX));
  contourYAxisG.call(d3.axisLeft(contourY));
  addAxisLabels(contourSvg, contourWidth, contourHeight, "β₀", "β₁");

  // Contour bands are computed on a sqrt-transformed copy of the grid so that
  // rings are evenly spaced (SSR grows quadratically, so raw thresholds would
  // bunch almost all of the visible detail into one huge outer band).
  const transformed = Float64Array.from(grid, v => Math.sqrt(Math.max(v, 0)));
  const contourGen = d3.contours().size([GRID_N, GRID_N]).thresholds(14);
  const features = contourGen(transformed);

  const transform = d3.geoTransform({
    point(x, y) {
      this.stream.point(contourX(gridB0(x)), contourY(gridB1(y)));
    }
  });
  const pathGen = d3.geoPath(transform);
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(transformed).reverse());

  contourFillLayer.selectAll("path").data(features).join("path")
    .attr("d", pathGen)
    .attr("fill", d => colorScale(d.value))
    .attr("stroke", "none");
}

function renderContourMarkers() {
  contourMarkerLayer.selectAll("circle.current-marker").data([0]).join("circle")
    .attr("class", "current-marker")
    .attr("r", 6)
    .attr("cx", contourX(curB0)).attr("cy", contourY(curB1));

  contourMarkerLayer.selectAll("circle.optimum-marker").data([0]).join("circle")
    .attr("class", "optimum-marker")
    .attr("r", 8)
    .attr("cx", contourX(fit.intercept)).attr("cy", contourY(fit.slope));
}

// ---------- 3D bowl panel ----------

const bowlWidth = 440, bowlHeight = 420;
const bowlCenter = { x: bowlWidth / 2, y: bowlHeight / 2 + 10 };
const bowlScale = 150;

let azimuth = -0.6, elevation = 0.4;

const bowlSvg = d3.select("#bowl-panel").append("svg")
  .attr("width", bowlWidth).attr("height", bowlHeight)
  .attr("role", "img")
  .attr("aria-label", "Rotatable 3D view of the sum of squared residuals surface over beta-0 and beta-1, shown as a bowl-shaped wireframe mesh. A red dot marks the current selection; a blue ring marks the least-squares optimum at the bottom of the bowl.");

bowlSvg.append("rect")
  .attr("x", 0).attr("y", 0).attr("width", bowlWidth).attr("height", bowlHeight)
  .attr("fill", "#fff");

const bowlAxisLayer = bowlSvg.append("g");
const bowlSurfaceLayer = bowlSvg.append("g");
const bowlMarkerLayer = bowlSvg.append("g");

function toBowlCube(b0, b1, ssr) {
  const ssrMid = (ssrMin + ssrMax) / 2;
  const ssrHalf = (ssrMax - ssrMin) / 2 || 1;
  return {
    u: (b0 - (b0Min + b0Max) / 2) / ((b0Max - b0Min) / 2),
    v: (ssr - ssrMid) / ssrHalf,
    w: (b1 - (b1Min + b1Max) / 2) / ((b1Max - b1Min) / 2)
  };
}

function projectBowl(cube) {
  const { u, v, w } = cube;
  const cosA = Math.cos(azimuth), sinA = Math.sin(azimuth);
  const xr = u * cosA + w * sinA;
  const zr = -u * sinA + w * cosA;
  const cosE = Math.cos(elevation), sinE = Math.sin(elevation);
  const yr = v * cosE - zr * sinE;
  const zr2 = v * sinE + zr * cosE;
  return { x: bowlCenter.x + xr * bowlScale, y: bowlCenter.y - yr * bowlScale, depth: zr2 };
}

function renderBowlAxes() {
  const b0Mid = (b0Min + b0Max) / 2, b1Mid = (b1Min + b1Max) / 2;
  const lines = [
    { from: toBowlCube(b0Min, b1Mid, ssrMin), to: toBowlCube(b0Max, b1Mid, ssrMin), label: "β₀" },
    { from: toBowlCube(b0Mid, b1Mid, ssrMin), to: toBowlCube(b0Mid, b1Mid, ssrMax), label: "SSR" },
    { from: toBowlCube(b0Mid, b1Min, ssrMin), to: toBowlCube(b0Mid, b1Max, ssrMin), label: "β₁" }
  ].map(d => ({ p1: projectBowl(d.from), p2: projectBowl(d.to), label: d.label }));

  bowlAxisLayer.selectAll("line").data(lines).join("line")
    .attr("stroke", "#999")
    .attr("x1", d => d.p1.x).attr("y1", d => d.p1.y)
    .attr("x2", d => d.p2.x).attr("y2", d => d.p2.y);

  bowlAxisLayer.selectAll("text").data(lines).join("text")
    .attr("class", "axis-label")
    .attr("x", d => d.p2.x + 6).attr("y", d => d.p2.y)
    .text(d => d.label);
}

function renderBowlSurface() {
  const meshLines = 11;
  const step = Math.max(1, Math.floor((GRID_N - 1) / (meshLines - 1)));
  const rowIdx = d3.range(0, GRID_N, step);
  if (rowIdx[rowIdx.length - 1] !== GRID_N - 1) rowIdx.push(GRID_N - 1);

  const segments = [];
  rowIdx.forEach(row => {
    const line = d3.range(GRID_N).map(col => toBowlCube(gridB0(col), gridB1(row), grid[row * GRID_N + col]));
    segments.push(line);
  });
  rowIdx.forEach(col => {
    const line = d3.range(GRID_N).map(row => toBowlCube(gridB0(col), gridB1(row), grid[row * GRID_N + col]));
    segments.push(line);
  });

  const lineGen = d3.line().x(d => d.x).y(d => d.y);
  const projectedSegments = segments.map(line => line.map(projectBowl));

  bowlSurfaceLayer.selectAll("path").data(projectedSegments).join("path")
    .attr("class", "surface-line")
    .attr("d", lineGen);
}

function renderBowlMarkers() {
  const curSsr = ssrAt(points, curB0, curB1);
  const curProj = projectBowl(toBowlCube(curB0, curB1, curSsr));
  const optProj = projectBowl(toBowlCube(fit.intercept, fit.slope, fit.SSR));

  bowlMarkerLayer.selectAll("circle.optimum-marker").data([optProj]).join("circle")
    .attr("class", "optimum-marker")
    .attr("r", 7)
    .attr("cx", d => d.x).attr("cy", d => d.y);

  bowlMarkerLayer.selectAll("circle.current-marker").data([curProj]).join("circle")
    .attr("class", "current-marker")
    .attr("r", 6)
    .attr("cx", d => d.x).attr("cy", d => d.y);
}

const rotateDrag = d3.drag().on("drag", (event) => {
  azimuth += event.dx * 0.01;
  elevation = clamp(elevation - event.dy * 0.01, -1.5, 1.5);
  renderBowlAxes();
  renderBowlSurface();
  renderBowlMarkers();
});
bowlSvg.call(rotateDrag);

// ---------- state + wiring ----------

let points = [];
let fit = null;
let curB0 = 0, curB1 = 0;

const b0Slider = document.getElementById("b0-slider");
const b1Slider = document.getElementById("b1-slider");
const b0Readout = document.getElementById("b0-readout");
const b1Readout = document.getElementById("b1-readout");

function setBeta(b0, b1) {
  curB0 = clamp(b0, b0Min, b0Max);
  curB1 = clamp(b1, b1Min, b1Max);
  b0Slider.value = curB0;
  b1Slider.value = curB1;
  render();
}

function render() {
  b0Readout.textContent = curB0.toFixed(2);
  b1Readout.textContent = curB1.toFixed(2);

  renderScatter();
  renderContourMarkers();
  renderBowlMarkers();

  const curSsr = ssrAt(points, curB0, curB1);
  document.getElementById("cur-b0").textContent = curB0.toFixed(2);
  document.getElementById("cur-b1").textContent = curB1.toFixed(2);
  document.getElementById("cur-ssr").textContent = curSsr.toFixed(2);
  document.getElementById("opt-b0").textContent = fit.intercept.toFixed(2);
  document.getElementById("opt-b1").textContent = fit.slope.toFixed(2);
  document.getElementById("opt-ssr").textContent = fit.SSR.toFixed(2);
}

function regenerate() {
  points = generateData(25);
  fit = fitOLS(points);

  const seB1 = Math.max(fit.seSlope, 0.05);
  const seB0 = Math.max(fit.seIntercept, 0.05);
  b1Min = fit.slope - 4 * seB1; b1Max = fit.slope + 4 * seB1;
  b0Min = fit.intercept - 4 * seB0; b0Max = fit.intercept + 4 * seB0;

  setScatterDomain(points);
  rebuildGrid();
  renderContourSurface();
  renderBowlAxes();
  renderBowlSurface();

  b0Slider.min = b0Min; b0Slider.max = b0Max; b0Slider.step = (b0Max - b0Min) / 300;
  b1Slider.min = b1Min; b1Slider.max = b1Max; b1Slider.step = (b1Max - b1Min) / 300;

  setBeta(b0Min, b1Min);
}

b0Slider.addEventListener("input", () => setBeta(parseFloat(b0Slider.value), parseFloat(b1Slider.value)));
b1Slider.addEventListener("input", () => setBeta(parseFloat(b0Slider.value), parseFloat(b1Slider.value)));
document.getElementById("regenerate").addEventListener("click", regenerate);
document.getElementById("snap-optimum").addEventListener("click", () => setBeta(fit.intercept, fit.slope));

regenerate();
