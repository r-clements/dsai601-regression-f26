// ---------- random numbers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// Points are shape-coded (not just color-coded) so class is legible without color vision:
// class 0 = circle, class 1 = triangle.
function classSymbolPath(cls, size) {
  return d3.symbol().type(cls === 0 ? d3.symbolCircle : d3.symbolTriangle).size(size)();
}

// ---------- data generation ----------

let nextId = 0;

function generateData(separation) {
  const perClass = 25;
  const pts = [];
  for (let i = 0; i < perClass; i++) {
    pts.push({ id: nextId++, cls: 0, x1: 3 + randNormal() * 1.3, x2: 3 + randNormal() * 1.3, jitter: (Math.random() - 0.5) * 0.06 });
  }
  for (let i = 0; i < perClass; i++) {
    pts.push({ id: nextId++, cls: 1, x1: 3 + separation + randNormal() * 1.3, x2: 3 + separation + randNormal() * 1.3, jitter: (Math.random() - 0.5) * 0.06 });
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

// ---------- logistic regression via IRLS ----------

function fitLogistic(points) {
  const n = points.length;
  const classes = new Set(points.map(p => p.cls));
  if (n < 4 || classes.size < 2) return null;

  let b0 = 0, b1 = 0, b2 = 0;
  const lambda = 1e-3; // tiny ridge penalty: keeps IRLS numerically stable under near-perfect separation

  for (let iter = 0; iter < 25; iter++) {
    let Sw = 0, Swx1 = 0, Swx2 = 0, Swx1x1 = 0, Swx2x2 = 0, Swx1x2 = 0, Swz = 0, Swx1z = 0, Swx2z = 0;
    for (const p of points) {
      const eta = b0 + b1 * p.x1 + b2 * p.x2;
      let pi = sigmoid(eta);
      pi = Math.min(Math.max(pi, 1e-6), 1 - 1e-6);
      const w = pi * (1 - pi);
      const z = eta + (p.cls - pi) / w;
      Sw += w; Swx1 += w * p.x1; Swx2 += w * p.x2;
      Swx1x1 += w * p.x1 * p.x1; Swx2x2 += w * p.x2 * p.x2; Swx1x2 += w * p.x1 * p.x2;
      Swz += w * z; Swx1z += w * p.x1 * z; Swx2z += w * p.x2 * z;
    }
    const A = [
      [Sw + lambda, Swx1, Swx2],
      [Swx1, Swx1x1 + lambda, Swx1x2],
      [Swx2, Swx1x2, Swx2x2 + lambda]
    ];
    const c = [Swz, Swx1z, Swx2z];
    const solved = solveLinearSystem3(A, c);
    if (!solved) break;
    [b0, b1, b2] = solved;
  }

  let correct = 0, logLossSum = 0;
  points.forEach(p => {
    const eta = b0 + b1 * p.x1 + b2 * p.x2;
    let pi = sigmoid(eta);
    pi = Math.min(Math.max(pi, 1e-9), 1 - 1e-9);
    if ((pi >= 0.5 ? 1 : 0) === p.cls) correct++;
    logLossSum += -(p.cls * Math.log(pi) + (1 - p.cls) * Math.log(1 - pi));
  });

  return {
    b0, b1, b2, n,
    accuracy: correct / n,
    logLoss: logLossSum / n
  };
}

// ---------- main scatter panel ----------

const mainWidth = 760, mainHeight = 440;
const mainMargin = { top: 20, right: 20, bottom: 45, left: 55 };

const mainSvg = d3.select("#main-panel").append("svg")
  .attr("width", mainWidth).attr("height", mainHeight)
  .attr("role", "img")
  .attr("aria-label", "Scatter plot of two classes: class 0 as blue circles, class 1 as orange triangles. A red decision boundary line and fainter probability contour lines are drawn across the plot. Current fit statistics are reported in the text below.");

const mainX = d3.scaleLinear().range([mainMargin.left, mainWidth - mainMargin.right]);
const mainY = d3.scaleLinear().range([mainHeight - mainMargin.bottom, mainMargin.top]);

const mainXAxisG = mainSvg.append("g").attr("transform", `translate(0,${mainHeight - mainMargin.bottom})`);
const mainYAxisG = mainSvg.append("g").attr("transform", `translate(${mainMargin.left},0)`);

const mainCaptureRect = mainSvg.append("rect")
  .attr("x", mainMargin.left).attr("y", mainMargin.top)
  .attr("width", mainWidth - mainMargin.left - mainMargin.right)
  .attr("height", mainHeight - mainMargin.top - mainMargin.bottom)
  .attr("fill", "transparent")
  .style("pointer-events", "all")
  .on("click", (event) => {
    const [px, py] = d3.pointer(event);
    const cls = parseInt(document.querySelector('input[name="addclass"]:checked').value, 10);
    points.push({ id: nextId++, cls, x1: mainX.invert(px), x2: mainY.invert(py), jitter: (Math.random() - 0.5) * 0.06 });
    update();
  });

const contourLayer = mainSvg.append("g");
const mainPointLayer = mainSvg.append("g");

const mainDrag = d3.drag()
  .subject((event, d) => ({ x: mainX(d.x1), y: mainY(d.x2) }))
  .on("drag", (event, d) => {
    d.x1 = mainX.invert(event.x);
    d.x2 = mainY.invert(event.y);
    update();
  });

function setMainDomain(pts) {
  const xExt = d3.extent(pts, d => d.x1);
  const yExt = d3.extent(pts, d => d.x2);
  const xPad = (xExt[1] - xExt[0]) * 0.25 || 1;
  const yPad = (yExt[1] - yExt[0]) * 0.25 || 1;
  mainX.domain([xExt[0] - xPad, xExt[1] + xPad]);
  mainY.domain([yExt[0] - yPad, yExt[1] + yPad]);
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

// Line where b0 + b1*x1 + b2*x2 = k, clipped to the visible domain by
// parametrizing over whichever axis has the larger coefficient (avoids
// dividing by a near-zero coefficient for near-vertical/horizontal boundaries).
function boundaryLine(fit, k, xDomain, yDomain) {
  if (Math.abs(fit.b2) >= Math.abs(fit.b1)) {
    const x1a = xDomain[0], x1b = xDomain[1];
    const x2a = (k - fit.b0 - fit.b1 * x1a) / fit.b2;
    const x2b = (k - fit.b0 - fit.b1 * x1b) / fit.b2;
    return [{ x1: x1a, x2: x2a }, { x1: x1b, x2: x2b }];
  } else {
    const x2a = yDomain[0], x2b = yDomain[1];
    const x1a = (k - fit.b0 - fit.b2 * x2a) / fit.b1;
    const x1b = (k - fit.b0 - fit.b2 * x2b) / fit.b1;
    return [{ x1: x1a, x2: x2a }, { x1: x1b, x2: x2b }];
  }
}

function renderMain(fit) {
  mainXAxisG.call(d3.axisBottom(mainX));
  mainYAxisG.call(d3.axisLeft(mainY));

  const xDomain = mainX.domain(), yDomain = mainY.domain();

  let contourSegs = [], boundarySeg = null;
  if (fit) {
    const contourKs = [-3, -1.5, 1.5, 3];
    contourSegs = contourKs.map(k => boundaryLine(fit, k, xDomain, yDomain));
    boundarySeg = boundaryLine(fit, 0, xDomain, yDomain);
  }

  contourLayer.selectAll("line.contour-line")
    .data(contourSegs)
    .join("line")
    .attr("class", "contour-line")
    .attr("x1", d => mainX(d[0].x1)).attr("y1", d => mainY(d[0].x2))
    .attr("x2", d => mainX(d[1].x1)).attr("y2", d => mainY(d[1].x2));

  contourLayer.selectAll("line.boundary-line")
    .data(boundarySeg ? [boundarySeg] : [])
    .join("line")
    .attr("class", "boundary-line")
    .attr("x1", d => mainX(d[0].x1)).attr("y1", d => mainY(d[0].x2))
    .attr("x2", d => mainX(d[1].x1)).attr("y2", d => mainY(d[1].x2));

  mainPointLayer.selectAll("path.pt")
    .data(points, d => d.id)
    .join("path")
    .attr("d", d => classSymbolPath(d.cls, 80))
    .call(mainDrag)
    .on("dblclick", (event, d) => {
      event.stopPropagation();
      points = points.filter(p => p.id !== d.id);
      update();
    })
    .attr("class", d => "pt class" + d.cls)
    .attr("transform", d => `translate(${mainX(d.x1)},${mainY(d.x2)})`);

  addAxisLabels(mainSvg, mainWidth, mainHeight, "x₁", "x₂");
}

// ---------- sigmoid panel ----------

const sigWidth = 760, sigHeight = 260;
const sigMargin = { top: 15, right: 20, bottom: 45, left: 55 };

const sigSvg = d3.select("#sigmoid-panel").append("svg")
  .attr("width", sigWidth).attr("height", sigHeight)
  .attr("role", "img")
  .attr("aria-label", "Sigmoid curve of predicted probability against the linear predictor, with each data point plotted at its own linear predictor value near y=0 (class 0, circles) or y=1 (class 1, triangles).");

const sigX = d3.scaleLinear().range([sigMargin.left, sigWidth - sigMargin.right]);
const sigY = d3.scaleLinear().domain([0, 1]).range([sigHeight - sigMargin.bottom, sigMargin.top]);

const sigXAxisG = sigSvg.append("g").attr("transform", `translate(0,${sigHeight - sigMargin.bottom})`);
const sigYAxisG = sigSvg.append("g").attr("transform", `translate(${sigMargin.left},0)`);

const sigDecorLayer = sigSvg.append("g");
const sigCurveLayer = sigSvg.append("g");
const sigPointLayer = sigSvg.append("g");

function renderSigmoid(fit) {
  if (!fit) {
    sigCurveLayer.selectAll("path").remove();
    sigPointLayer.selectAll("path").remove();
    return;
  }

  const zs = points.map(p => fit.b0 + fit.b1 * p.x1 + fit.b2 * p.x2);
  const zExt = d3.extent(zs);
  const zPad = (zExt[1] - zExt[0]) * 0.2 || 1;
  sigX.domain([zExt[0] - zPad, zExt[1] + zPad]);

  sigXAxisG.call(d3.axisBottom(sigX));
  sigYAxisG.call(d3.axisLeft(sigY).ticks(5));

  sigDecorLayer.selectAll("line.prob-ref").data([0.5]).join("line")
    .attr("class", "prob-ref")
    .attr("x1", sigX.range()[0]).attr("x2", sigX.range()[1])
    .attr("y1", sigY(0.5)).attr("y2", sigY(0.5));

  const curveSteps = 60;
  const [z0, z1] = sigX.domain();
  const curveData = d3.range(curveSteps + 1).map(i => {
    const z = z0 + (z1 - z0) * i / curveSteps;
    return { z, p: sigmoid(z) };
  });
  const line = d3.line().x(d => sigX(d.z)).y(d => sigY(d.p));
  sigCurveLayer.selectAll("path.sigmoid-curve").data([curveData]).join("path")
    .attr("class", "sigmoid-curve")
    .attr("d", line);

  const pointData = points.map((p, i) => ({ id: p.id, z: zs[i], y: p.cls + p.jitter, cls: p.cls }));
  sigPointLayer.selectAll("path.pt")
    .data(pointData, d => d.id)
    .join("path")
    .attr("d", d => classSymbolPath(d.cls, 50))
    .attr("class", d => "pt class" + d.cls)
    .attr("transform", d => `translate(${sigX(d.z)},${sigY(d.y)})`);

  addAxisLabels(sigSvg, sigWidth, sigHeight, "linear predictor (z = b₀ + b₁x₁ + b₂x₂)", "class / p̂");
}

// ---------- state + wiring ----------

let points = [];

function update() {
  const fit = fitLogistic(points);
  renderMain(fit);
  renderSigmoid(fit);

  document.getElementById("n").textContent = points.length;
  document.getElementById("b0").textContent = fit ? fit.b0.toFixed(2) : "–";
  document.getElementById("b1").textContent = fit ? fit.b1.toFixed(2) : "–";
  document.getElementById("b2").textContent = fit ? fit.b2.toFixed(2) : "–";
  document.getElementById("accuracy").textContent = fit ? (fit.accuracy * 100).toFixed(1) + "%" : "–";
  document.getElementById("logloss").textContent = fit ? fit.logLoss.toFixed(3) : "–";
}

function regenerate() {
  points = generateData(parseFloat(document.getElementById("separation").value));
  setMainDomain(points);
  update();
}

document.getElementById("regenerate").addEventListener("click", regenerate);
document.getElementById("separation").addEventListener("input", regenerate);

regenerate();
