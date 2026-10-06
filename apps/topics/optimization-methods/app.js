// ---------- small math helpers ----------

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// Numerical gradient and Hessian via finite differences, so any smooth
// f(x,y) can be plugged in without hand-deriving its derivatives.
function gradient(f, x, y, h = 1e-4) {
  const gx = (f(x + h, y) - f(x - h, y)) / (2 * h);
  const gy = (f(x, y + h) - f(x, y - h)) / (2 * h);
  return { gx, gy };
}

function hessian(f, x, y, h = 1e-3) {
  const f0 = f(x, y);
  const fxx = (f(x + h, y) - 2 * f0 + f(x - h, y)) / (h * h);
  const fyy = (f(x, y + h) - 2 * f0 + f(x, y - h)) / (h * h);
  const fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h);
  return { fxx, fyy, fxy };
}

// Solve H * delta = grad for a symmetric 2x2 H = [[fxx,fxy],[fxy,fyy]].
function solveNewtonStep(hess, grad) {
  const det = hess.fxx * hess.fyy - hess.fxy * hess.fxy;
  if (Math.abs(det) < 1e-8) return null;
  const dx = (hess.fyy * grad.gx - hess.fxy * grad.gy) / det;
  const dy = (-hess.fxy * grad.gx + hess.fxx * grad.gy) / det;
  return { dx, dy };
}

function formatNum(v) {
  if (!isFinite(v)) return v > 0 ? "∞" : (v < 0 ? "−∞" : "NaN");
  const av = Math.abs(v);
  if (av !== 0 && (av < 0.001 || av >= 10000)) return v.toExponential(2);
  return v.toFixed(3);
}

// ---------- surfaces ----------

const SURFACES = {
  convex: {
    label: "Convex bowl (elongated ravine)",
    f: (x, y) => 0.5 * (1 * x * x + 20 * y * y),
    xMin: -5, xMax: 5, yMin: -2.5, yMax: 2.5,
    minima: [{ x: 0, y: 0 }],
  },
  multimodal: {
    label: "Multiple minima (Himmelblau)",
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    xMin: -5, xMax: 5, yMin: -5, yMax: 5,
    minima: [
      { x: 3, y: 2 },
      { x: -2.805118, y: 3.131312 },
      { x: -3.779310, y: -3.283186 },
      { x: 3.584428, y: -1.848126 },
    ],
  },
  saddle: {
    label: "Saddle point",
    f: (x, y) => x * x - y * y,
    xMin: -3, xMax: 3, yMin: -3, yMax: 3,
    minima: [],
    saddle: { x: 0, y: 0 },
  },
};

// ---------- algorithms ----------
// Each algorithm exposes: label, formula text, a small params spec (used to
// auto-build sliders), an initial auxiliary state, and a step function that
// takes (state, grad, hess, params) and returns the next state.

const ALGORITHMS = {
  gd: {
    label: "Gradient Descent",
    formula: "x<sub>t+1</sub> = x<sub>t</sub> − η ∇f(x<sub>t</sub>)",
    params: [{ key: "lr", label: "learning rate η", min: -3, max: 0, default: -1.3, log: true }],
    initState: (x, y) => ({ x, y }),
    step(state, grad, hess, p) {
      return { x: state.x - p.lr * grad.gx, y: state.y - p.lr * grad.gy };
    },
  },
  momentum: {
    label: "Gradient Descent with Momentum",
    formula: "v<sub>t+1</sub> = γv<sub>t</sub> + η∇f(x<sub>t</sub>);&nbsp;&nbsp; x<sub>t+1</sub> = x<sub>t</sub> − v<sub>t+1</sub>",
    params: [
      { key: "lr", label: "learning rate η", min: -3, max: 0, default: -1.5, log: true },
      { key: "momentum", label: "momentum γ", min: 0, max: 0.99, step: 0.01, default: 0.85, log: false },
    ],
    initState: (x, y) => ({ x, y, vx: 0, vy: 0 }),
    step(state, grad, hess, p) {
      const vx = p.momentum * state.vx + p.lr * grad.gx;
      const vy = p.momentum * state.vy + p.lr * grad.gy;
      return { x: state.x - vx, y: state.y - vy, vx, vy };
    },
  },
  adagrad: {
    label: "AdaGrad",
    formula: "G<sub>t+1</sub> = G<sub>t</sub> + g<sub>t</sub>²;&nbsp;&nbsp; x<sub>t+1</sub> = x<sub>t</sub> − η g<sub>t</sub> / (√G<sub>t+1</sub> + ε)",
    params: [{ key: "lr", label: "learning rate η", min: -2, max: 0.3, default: -0.3, log: true }],
    initState: (x, y) => ({ x, y, Gx: 0, Gy: 0 }),
    step(state, grad, hess, p) {
      const eps = 1e-8;
      const Gx = state.Gx + grad.gx * grad.gx;
      const Gy = state.Gy + grad.gy * grad.gy;
      return {
        x: state.x - (p.lr * grad.gx) / (Math.sqrt(Gx) + eps),
        y: state.y - (p.lr * grad.gy) / (Math.sqrt(Gy) + eps),
        Gx, Gy,
      };
    },
  },
  adam: {
    label: "Adam",
    formula: "m, v = bias-corrected EMAs of g, g²;&nbsp;&nbsp; x<sub>t+1</sub> = x<sub>t</sub> − η m̂ / (√v̂ + ε)",
    params: [{ key: "lr", label: "learning rate η", min: -3, max: 0.3, default: -0.7, log: true }],
    initState: (x, y) => ({ x, y, mx: 0, my: 0, vx: 0, vy: 0, t: 0 }),
    step(state, grad, hess, p) {
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
      const t = state.t + 1;
      const mx = beta1 * state.mx + (1 - beta1) * grad.gx;
      const my = beta1 * state.my + (1 - beta1) * grad.gy;
      const vx = beta2 * state.vx + (1 - beta2) * grad.gx * grad.gx;
      const vy = beta2 * state.vy + (1 - beta2) * grad.gy * grad.gy;
      const mxh = mx / (1 - Math.pow(beta1, t));
      const myh = my / (1 - Math.pow(beta1, t));
      const vxh = vx / (1 - Math.pow(beta2, t));
      const vyh = vy / (1 - Math.pow(beta2, t));
      return {
        x: state.x - (p.lr * mxh) / (Math.sqrt(vxh) + eps),
        y: state.y - (p.lr * myh) / (Math.sqrt(vyh) + eps),
        mx, my, vx, vy, t,
      };
    },
  },
  newton: {
    label: "Newton-Raphson",
    formula: "x<sub>t+1</sub> = x<sub>t</sub> − α H(x<sub>t</sub>)<sup>−1</sup> ∇f(x<sub>t</sub>)",
    params: [{ key: "alpha", label: "step size α", min: 0.1, max: 1, step: 0.05, default: 1, log: false }],
    initState: (x, y) => ({ x, y }),
    step(state, grad, hess, p) {
      const d = solveNewtonStep(hess, grad);
      if (!d) return { x: state.x, y: state.y, singular: true };
      return { x: state.x - p.alpha * d.dx, y: state.y - p.alpha * d.dy };
    },
  },
};

// ---------- panel helper (shared scale/axis/layer setup) ----------

function setupPanel(containerId, ariaLabel, width, height, margin) {
  const svg = d3.select(containerId).append("svg").attr("width", width).attr("height", height)
    .attr("role", "img")
    .attr("aria-label", ariaLabel);
  const x = d3.scaleLinear().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().range([height - margin.bottom, margin.top]);
  const xAxisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
  const yAxisG = svg.append("g").attr("transform", `translate(${margin.left},0)`);
  const fillLayer = svg.append("g");
  const decorLayer = svg.append("g");
  const pathLayer = svg.append("g");
  const markerLayer = svg.append("g");
  return { svg, x, y, xAxisG, yAxisG, fillLayer, decorLayer, pathLayer, markerLayer, width, height, margin };
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

// ---------- shared x,y grid over the current surface ----------

const GRID_N = 50;
let grid, fMin, fMax;

function rebuildGrid(surface) {
  grid = new Float64Array(GRID_N * GRID_N);
  let mn = Infinity, mx = -Infinity;
  for (let row = 0; row < GRID_N; row++) {
    const yv = surface.yMin + (surface.yMax - surface.yMin) * row / (GRID_N - 1);
    for (let col = 0; col < GRID_N; col++) {
      const xv = surface.xMin + (surface.xMax - surface.xMin) * col / (GRID_N - 1);
      const v = surface.f(xv, yv);
      grid[row * GRID_N + col] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  fMin = mn; fMax = mx;
}
function gridX(surface, col) { return surface.xMin + (surface.xMax - surface.xMin) * col / (GRID_N - 1); }
function gridY(surface, row) { return surface.yMin + (surface.yMax - surface.yMin) * row / (GRID_N - 1); }

// ---------- contour panel ----------

const contourWidth = 440, contourHeight = 420;
const contourMargin = { top: 15, right: 20, bottom: 40, left: 55 };
const contourPanel = setupPanel(
  "#contour-panel",
  "Contour map of the surface viewed from above, with the optimization path traced on it.",
  contourWidth, contourHeight, contourMargin
);

function renderContourSurface(surface) {
  contourPanel.x.domain([surface.xMin, surface.xMax]);
  contourPanel.y.domain([surface.yMin, surface.yMax]);
  contourPanel.xAxisG.call(d3.axisBottom(contourPanel.x));
  contourPanel.yAxisG.call(d3.axisLeft(contourPanel.y));
  addAxisLabels(contourPanel.svg, contourWidth, contourHeight, "x", "y");

  const contourGen = d3.contours().size([GRID_N, GRID_N]).thresholds(14);
  const features = contourGen(grid);
  const transform = d3.geoTransform({
    point(x, y) {
      this.stream.point(contourPanel.x(gridX(surface, x)), contourPanel.y(gridY(surface, y)));
    },
  });
  const pathGen = d3.geoPath(transform);
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(grid).reverse());

  contourPanel.fillLayer.selectAll("path").data(features).join("path")
    .attr("d", pathGen)
    .attr("fill", d => colorScale(d.value))
    .attr("stroke", "none");

  contourPanel.decorLayer.selectAll("circle.minima-marker").data(surface.minima || []).join("circle")
    .attr("class", "minima-marker")
    .attr("r", 6)
    .attr("cx", d => contourPanel.x(d.x)).attr("cy", d => contourPanel.y(d.y));
  const saddlePts = surface.saddle ? [surface.saddle] : [];
  contourPanel.decorLayer.selectAll("path.saddle-marker").data(saddlePts).join("path")
    .attr("class", "minima-marker")
    .attr("d", d => {
      const cx = contourPanel.x(d.x), cy = contourPanel.y(d.y), r = 7;
      return `M${cx - r},${cy} L${cx + r},${cy} M${cx},${cy - r} L${cx},${cy + r}`;
    });
}

function renderContourPath() {
  const lineGen = d3.line().x(d => contourPanel.x(d.x)).y(d => contourPanel.y(d.y));
  contourPanel.pathLayer.selectAll("path.trail-line").data([trajectory]).join("path")
    .attr("class", "trail-line")
    .attr("d", lineGen);
  contourPanel.pathLayer.selectAll("circle.trail-dot").data(trajectory.slice(0, -1)).join("circle")
    .attr("class", "trail-dot")
    .attr("r", 3)
    .attr("cx", d => contourPanel.x(d.x)).attr("cy", d => contourPanel.y(d.y));

  contourPanel.markerLayer.selectAll("circle.start-marker").data([trajectory[0]]).join("circle")
    .attr("class", "start-marker")
    .attr("r", 7)
    .attr("cx", d => contourPanel.x(d.x)).attr("cy", d => contourPanel.y(d.y));
  const cur = trajectory[trajectory.length - 1];
  contourPanel.markerLayer.selectAll("circle.ball").data([cur]).join("circle")
    .attr("class", "ball")
    .attr("r", 6)
    .attr("cx", d => contourPanel.x(d.x)).attr("cy", d => contourPanel.y(d.y));
}

// ---------- 3D surface panel ----------

const bowlWidth = 440, bowlHeight = 420;
const bowlCenter = { x: bowlWidth / 2, y: bowlHeight / 2 + 10 };
const bowlScale = 150;
let azimuth = -0.6, elevation = 0.4;

const bowlSvg = d3.select("#bowl-panel").append("svg")
  .attr("width", bowlWidth).attr("height", bowlHeight)
  .attr("role", "img")
  .attr("aria-label", "Rotatable 3D view of the surface f(x,y), with a ball marking the current point and a trail showing the path taken so far.");

bowlSvg.append("rect").attr("x", 0).attr("y", 0).attr("width", bowlWidth).attr("height", bowlHeight).attr("fill", "#fff");

const bowlAxisLayer = bowlSvg.append("g");
const bowlSurfaceLayer = bowlSvg.append("g");
const bowlTrailLayer = bowlSvg.append("g");
const bowlMarkerLayer = bowlSvg.append("g");

function toBowlCube(surface, x, y, fval) {
  const xMid = (surface.xMin + surface.xMax) / 2, xHalf = (surface.xMax - surface.xMin) / 2;
  const yMid = (surface.yMin + surface.yMax) / 2, yHalf = (surface.yMax - surface.yMin) / 2;
  const fMid = (fMin + fMax) / 2, fHalf = (fMax - fMin) / 2 || 1;
  return { u: (x - xMid) / xHalf, v: (fval - fMid) / fHalf, w: (y - yMid) / yHalf };
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

function renderBowlAxes(surface) {
  const xMid = (surface.xMin + surface.xMax) / 2, yMid = (surface.yMin + surface.yMax) / 2;
  const lines = [
    { from: toBowlCube(surface, surface.xMin, yMid, fMin), to: toBowlCube(surface, surface.xMax, yMid, fMin), label: "x" },
    { from: toBowlCube(surface, xMid, yMid, fMin), to: toBowlCube(surface, xMid, yMid, fMax), label: "f" },
    { from: toBowlCube(surface, xMid, surface.yMin, fMin), to: toBowlCube(surface, xMid, surface.yMax, fMin), label: "y" },
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

function renderBowlSurface(surface) {
  const meshLines = 11;
  const step = Math.max(1, Math.floor((GRID_N - 1) / (meshLines - 1)));
  const rowIdx = d3.range(0, GRID_N, step);
  if (rowIdx[rowIdx.length - 1] !== GRID_N - 1) rowIdx.push(GRID_N - 1);

  const segments = [];
  rowIdx.forEach(row => {
    segments.push(d3.range(GRID_N).map(col => toBowlCube(surface, gridX(surface, col), gridY(surface, row), grid[row * GRID_N + col])));
  });
  rowIdx.forEach(col => {
    segments.push(d3.range(GRID_N).map(row => toBowlCube(surface, gridX(surface, col), gridY(surface, row), grid[row * GRID_N + col])));
  });

  const lineGen = d3.line().x(d => d.x).y(d => d.y);
  const projectedSegments = segments.map(line => line.map(projectBowl));
  bowlSurfaceLayer.selectAll("path").data(projectedSegments).join("path")
    .attr("fill", "none").attr("stroke", "#1a5fb4").attr("stroke-opacity", 0.5)
    .attr("d", lineGen);
}

function renderBowlTrailAndBall(surface) {
  const projected = trajectory.map(p => projectBowl(toBowlCube(surface, p.x, p.y, p.f)));
  const lineGen = d3.line().x(d => d.x).y(d => d.y);
  bowlTrailLayer.selectAll("path.trail-line").data([projected]).join("path")
    .attr("class", "trail-line")
    .attr("d", lineGen);
  bowlTrailLayer.selectAll("circle.trail-dot").data(projected.slice(0, -1)).join("circle")
    .attr("class", "trail-dot")
    .attr("r", 2.5)
    .attr("cx", d => d.x).attr("cy", d => d.y);

  const cur = projected[projected.length - 1];
  bowlMarkerLayer.selectAll("circle.ball").data([cur]).join("circle")
    .attr("class", "ball")
    .attr("r", 7)
    .transition().duration(220)
    .attr("cx", d => d.x).attr("cy", d => d.y);
}

const rotateDrag = d3.drag().on("drag", (event) => {
  azimuth += event.dx * 0.01;
  elevation = clamp(elevation - event.dy * 0.01, -1.5, 1.5);
  renderBowlAxes(currentSurface());
  renderBowlSurface(currentSurface());
  renderBowlTrailAndBall(currentSurface());
});
bowlSvg.call(rotateDrag);

// ---------- state + wiring ----------

let surfaceKey = "convex";
let algoKey = "gd";
let params = {};
let trajectory = [];
let algoState = null;
let status = "ready"; // ready | running | converged | diverged
let timer = null;

const MAX_ITER = 200;
const CONVERGE_TOL = 1e-4;

function currentSurface() { return SURFACES[surfaceKey]; }
function currentAlgo() { return ALGORITHMS[algoKey]; }

function evalPoint(surface, x, y) {
  return { x, y, f: surface.f(x, y) };
}

function randomStart(surface) {
  const marginFrac = 0.15;
  const xLo = surface.xMin + marginFrac * (surface.xMax - surface.xMin);
  const xHi = surface.xMax - marginFrac * (surface.xMax - surface.xMin);
  const yLo = surface.yMin + marginFrac * (surface.yMax - surface.yMin);
  const yHi = surface.yMax - marginFrac * (surface.yMax - surface.yMin);
  const diag = Math.hypot(surface.xMax - surface.xMin, surface.yMax - surface.yMin);
  const critical = (surface.minima || []).concat(surface.saddle ? [surface.saddle] : []);

  for (let tries = 0; tries < 30; tries++) {
    const x = xLo + Math.random() * (xHi - xLo);
    const y = yLo + Math.random() * (yHi - yLo);
    const tooClose = critical.some(c => Math.hypot(x - c.x, y - c.y) < 0.1 * diag);
    if (!tooClose) return { x, y };
  }
  return { x: (xLo + xHi) / 2, y: yLo };
}

function buildAlgoControls() {
  const algo = currentAlgo();
  const container = document.getElementById("algo-controls");
  container.innerHTML = "";
  params = {};
  algo.params.forEach(spec => {
    params[spec.key] = spec.log ? Math.pow(10, spec.default) : spec.default;

    const group = document.createElement("div");
    group.className = "slider-group";
    const label = document.createElement("label");
    label.setAttribute("for", `param-${spec.key}`);
    label.textContent = spec.label;
    const input = document.createElement("input");
    input.type = "range";
    input.id = `param-${spec.key}`;
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step || 0.01;
    input.value = spec.default;
    const output = document.createElement("output");
    output.id = `param-${spec.key}-readout`;
    output.textContent = formatNum(params[spec.key]);

    input.addEventListener("input", () => {
      const raw = parseFloat(input.value);
      params[spec.key] = spec.log ? Math.pow(10, raw) : raw;
      output.textContent = formatNum(params[spec.key]);
    });

    group.appendChild(label);
    group.appendChild(input);
    group.appendChild(output);
    container.appendChild(group);
  });
}

function stepSizeOf(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function doStep() {
  if (status !== "running" && status !== "ready") return;
  if (trajectory.length - 1 >= MAX_ITER) { setStatus("stopped", `reached the ${MAX_ITER}-iteration limit without converging`); return; }

  const surface = currentSurface();
  const algo = currentAlgo();
  const last = algoState;
  const grad = gradient(surface.f, last.x, last.y);
  const hess = hessian(surface.f, last.x, last.y);
  const next = algo.step(last, grad, hess, params);

  if (next.singular) { setStatus("diverged", "Hessian is singular here — Newton's step is undefined."); return; }

  const xSpan = surface.xMax - surface.xMin, ySpan = surface.yMax - surface.yMin;
  const outOfBounds = !isFinite(next.x) || !isFinite(next.y) ||
    Math.abs(next.x - (surface.xMin + surface.xMax) / 2) > 2.5 * xSpan ||
    Math.abs(next.y - (surface.yMin + surface.yMax) / 2) > 2.5 * ySpan;
  if (outOfBounds) { setStatus("diverged"); return; }

  algoState = next;
  trajectory.push(evalPoint(surface, next.x, next.y));

  const moved = stepSizeOf({ x: last.x, y: last.y }, { x: next.x, y: next.y });
  render();
  if (moved < CONVERGE_TOL) setStatus("converged");
}

function setStatus(s, note) {
  status = s;
  if (timer) { clearInterval(timer); timer = null; }
  document.getElementById("play-pause").textContent = "Start";
  const el = document.getElementById("cur-status");
  el.textContent = s + (note ? ` — ${note}` : "");
  el.className = s === "converged" ? "stat-value status-converged" : (s === "diverged" ? "stat-value status-diverged" : "stat-value");
  render();
}

function render() {
  const surface = currentSurface();
  renderContourPath();
  renderBowlTrailAndBall(surface);

  const cur = trajectory[trajectory.length - 1];
  const grad = gradient(surface.f, cur.x, cur.y);
  document.getElementById("cur-iter").textContent = trajectory.length - 1;
  document.getElementById("cur-x").textContent = formatNum(cur.x);
  document.getElementById("cur-y").textContent = formatNum(cur.y);
  document.getElementById("cur-f").textContent = formatNum(cur.f);
  document.getElementById("cur-gradnorm").textContent = formatNum(Math.hypot(grad.gx, grad.gy));
  if (status === "ready" || status === "running" || status === "paused") {
    document.getElementById("cur-status").textContent = status;
    document.getElementById("cur-status").className = "stat-value";
  }
}

function resetTrajectory(startPoint) {
  if (timer) { clearInterval(timer); timer = null; }
  const surface = currentSurface();
  algoState = currentAlgo().initState(startPoint.x, startPoint.y);
  trajectory = [evalPoint(surface, startPoint.x, startPoint.y)];
  status = "ready";
  document.getElementById("play-pause").textContent = "Start";
  render();
}

let lastStart = null;

function regenerateSurface() {
  const surface = currentSurface();
  rebuildGrid(surface);
  renderContourSurface(surface);
  renderBowlAxes(surface);
  renderBowlSurface(surface);
  lastStart = randomStart(surface);
  resetTrajectory(lastStart);
}

function changeAlgorithm() {
  buildAlgoControls();
  document.getElementById("formula-text").innerHTML = currentAlgo().formula;
  resetTrajectory(lastStart);
}

function speedToInterval() {
  const v = parseInt(document.getElementById("speed-slider").value, 10);
  return 600 - v * 27; // higher slider value -> shorter interval -> faster
}

document.getElementById("surface-select").addEventListener("change", (e) => {
  surfaceKey = e.target.value;
  regenerateSurface();
});
document.getElementById("algo-select").addEventListener("change", (e) => {
  algoKey = e.target.value;
  changeAlgorithm();
});
document.getElementById("new-start").addEventListener("click", () => {
  lastStart = randomStart(currentSurface());
  resetTrajectory(lastStart);
});
document.getElementById("step").addEventListener("click", () => {
  if (status === "ready" || status === "paused") status = "running";
  if (status === "running") doStep();
});
document.getElementById("reset").addEventListener("click", () => resetTrajectory(lastStart));
document.getElementById("play-pause").addEventListener("click", () => {
  const btn = document.getElementById("play-pause");
  if (timer) {
    clearInterval(timer);
    timer = null;
    status = "paused";
    btn.textContent = "Start";
    render();
    return;
  }
  if (status === "converged" || status === "diverged") resetTrajectory(lastStart);
  status = "running";
  btn.textContent = "Pause";
  timer = setInterval(doStep, speedToInterval());
});
document.getElementById("speed-slider").addEventListener("input", () => {
  if (timer) {
    clearInterval(timer);
    timer = setInterval(doStep, speedToInterval());
  }
});

buildAlgoControls();
document.getElementById("formula-text").innerHTML = currentAlgo().formula;
regenerateSurface();
