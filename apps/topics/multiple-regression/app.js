// ---------- random numbers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- data generation ----------

const X1_MIN = 0, X1_MAX = 10;
const X2_MIN = 0, X2_MAX = 10;
const TRUE_B0 = 3, TRUE_B1 = 1.6, TRUE_B2 = -1.2;

let nextId = 0;

function generateData(n) {
  const sigma = parseFloat(document.getElementById("noise").value);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x1 = X1_MIN + Math.random() * (X1_MAX - X1_MIN);
    const x2 = X2_MIN + Math.random() * (X2_MAX - X2_MIN);
    const y = TRUE_B0 + TRUE_B1 * x1 + TRUE_B2 * x2 + randNormal() * sigma;
    pts.push({ id: nextId++, x1, x2, y });
  }
  return pts;
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

  const fitted = points.map(p => b0 + b1 * p.x1 + b2 * p.x2);
  const resid = points.map((p, i) => p.y - fitted[i]);
  const SSR = d3.sum(resid, r => r * r);
  const meanY = Sy / n;
  const SST = d3.sum(points, p => (p.y - meanY) ** 2);
  const R2 = SST === 0 ? 0 : 1 - SSR / SST;

  return { n, b0, b1, b2, R2 };
}

// ---------- 3D scene ----------

const width = 780, height = 520;
const center = { x: width / 2, y: height / 2 + 10 };
const cubeScale = 175;

let azimuth = -0.6, elevation = 0.35;
let yMid = 0, yHalf = 1;

const svg = d3.select("#main-panel").append("svg")
  .attr("width", width).attr("height", height)
  .attr("role", "img")
  .attr("aria-label", "Rotatable 3D scatter plot of two predictors and an outcome, with the OLS-fitted regression plane shown as a red wireframe mesh. Current fit statistics are reported in the text below the plot.");

const backgroundRect = svg.append("rect")
  .attr("x", 0).attr("y", 0).attr("width", width).attr("height", height)
  .attr("fill", "#fff")
  .style("pointer-events", "all");

const axisLayer = svg.append("g");
const planeLayer = svg.append("g");
const pointLayer = svg.append("g");

function toCube(x1, y, x2) {
  return {
    u: (x1 - (X1_MIN + X1_MAX) / 2) / ((X1_MAX - X1_MIN) / 2),
    v: (y - yMid) / yHalf,
    w: (x2 - (X2_MIN + X2_MAX) / 2) / ((X2_MAX - X2_MIN) / 2)
  };
}

function project(cube) {
  const { u, v, w } = cube;
  const cosA = Math.cos(azimuth), sinA = Math.sin(azimuth);
  const xr = u * cosA + w * sinA;
  const zr = -u * sinA + w * cosA;
  const cosE = Math.cos(elevation), sinE = Math.sin(elevation);
  const yr = v * cosE - zr * sinE;
  const zr2 = v * sinE + zr * cosE;
  return {
    x: center.x + xr * cubeScale,
    y: center.y - yr * cubeScale,
    depth: zr2
  };
}

function setYDomain(pts) {
  const yExt = d3.extent(pts, d => d.y);
  const pad = (yExt[1] - yExt[0]) * 0.15 || 1;
  yMid = (yExt[0] + yExt[1]) / 2;
  yHalf = (yExt[1] - yExt[0]) / 2 + pad;
}

function renderAxes() {
  const axisEnds = [
    { from: toCube(X1_MIN, yMid - yHalf, (X2_MIN + X2_MAX) / 2), to: toCube(X1_MAX, yMid - yHalf, (X2_MIN + X2_MAX) / 2), label: "x₁" },
    { from: toCube((X1_MIN + X1_MAX) / 2, yMid - yHalf, (X2_MIN + X2_MAX) / 2), to: toCube((X1_MIN + X1_MAX) / 2, yMid + yHalf, (X2_MIN + X2_MAX) / 2), label: "y" },
    { from: toCube((X1_MIN + X1_MAX) / 2, yMid - yHalf, X2_MIN), to: toCube((X1_MIN + X1_MAX) / 2, yMid - yHalf, X2_MAX), label: "x₂" }
  ];

  const lines = axisEnds.map(a => ({ p1: project(a.from), p2: project(a.to), label: a.label }));

  axisLayer.selectAll("line.axis3d").data(lines).join("line")
    .attr("class", "axis3d")
    .attr("x1", d => d.p1.x).attr("y1", d => d.p1.y)
    .attr("x2", d => d.p2.x).attr("y2", d => d.p2.y);

  axisLayer.selectAll("text.axis3d-label").data(lines).join("text")
    .attr("class", "axis3d-label")
    .attr("x", d => d.p2.x + 8).attr("y", d => d.p2.y)
    .text(d => d.label);
}

function renderPlane(fit) {
  if (!fit) {
    planeLayer.selectAll("line.plane-line").remove();
    return;
  }
  const steps = 6;
  const segs = [];
  for (let j = 0; j <= steps; j++) {
    const x2v = X2_MIN + (X2_MAX - X2_MIN) * j / steps;
    const yA = fit.b0 + fit.b1 * X1_MIN + fit.b2 * x2v;
    const yB = fit.b0 + fit.b1 * X1_MAX + fit.b2 * x2v;
    segs.push([toCube(X1_MIN, yA, x2v), toCube(X1_MAX, yB, x2v)]);
  }
  for (let i = 0; i <= steps; i++) {
    const x1v = X1_MIN + (X1_MAX - X1_MIN) * i / steps;
    const yA = fit.b0 + fit.b1 * x1v + fit.b2 * X2_MIN;
    const yB = fit.b0 + fit.b1 * x1v + fit.b2 * X2_MAX;
    segs.push([toCube(x1v, yA, X2_MIN), toCube(x1v, yB, X2_MAX)]);
  }

  const projected = segs.map(([a, b]) => ({ p1: project(a), p2: project(b) }));

  planeLayer.selectAll("line.plane-line").data(projected).join("line")
    .attr("class", "plane-line")
    .attr("x1", d => d.p1.x).attr("y1", d => d.p1.y)
    .attr("x2", d => d.p2.x).attr("y2", d => d.p2.y);
}

function renderPoints() {
  const withDepth = points.map(p => {
    const cube = toCube(p.x1, p.y, p.x2);
    const proj = project(cube);
    return { ...p, proj };
  });
  withDepth.sort((a, b) => a.proj.depth - b.proj.depth);

  const depthExtent = d3.extent(withDepth, d => d.proj.depth);
  const rScale = d3.scaleLinear().domain(depthExtent.length ? depthExtent : [-1, 1]).range([3, 7]);
  const opScale = d3.scaleLinear().domain(depthExtent.length ? depthExtent : [-1, 1]).range([0.45, 1]);

  pointLayer.selectAll("circle")
    .data(withDepth, d => d.id)
    .join("circle")
    .attr("class", "point3d")
    .attr("cx", d => d.proj.x)
    .attr("cy", d => d.proj.y)
    .attr("r", d => rScale(d.proj.depth))
    .attr("opacity", d => opScale(d.proj.depth))
    .order();
}

// ---------- state + wiring ----------

let points = [];

function render() {
  const fit = computeFit(points);
  renderAxes();
  renderPlane(fit);
  renderPoints();

  document.getElementById("n").textContent = points.length;
  document.getElementById("b0").textContent = fit ? fit.b0.toFixed(2) : "–";
  document.getElementById("b1").textContent = fit ? fit.b1.toFixed(2) : "–";
  document.getElementById("b2").textContent = fit ? fit.b2.toFixed(2) : "–";
  document.getElementById("r2").textContent = fit ? fit.R2.toFixed(3) : "–";
}

function regenerate() {
  points = generateData(60);
  setYDomain(points);
  render();
}

const rotateDrag = d3.drag().on("drag", (event) => {
  azimuth += event.dx * 0.01;
  elevation = Math.max(-1.5, Math.min(1.5, elevation - event.dy * 0.01));
  render();
});

svg.call(rotateDrag);

document.getElementById("regenerate").addEventListener("click", regenerate);
document.getElementById("noise").addEventListener("input", regenerate);
document.getElementById("reset-view").addEventListener("click", () => {
  azimuth = -0.6;
  elevation = 0.35;
  render();
});

regenerate();
