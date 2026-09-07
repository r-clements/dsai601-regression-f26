// ---------- random number helpers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randSkewed() {
  // exponential(1) recentered to mean 0 -> right-skewed noise
  const u = Math.random();
  return -Math.log(1 - u) - 1;
}

function normalPDF(x, mu, sigma) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}

// Inverse standard normal CDF (Acklam's approximation)
function qnorm(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
  const plow = 0.02425;
  let q, r;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (p <= 1 - plow) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
}

// ---------- data generation ----------

let nextId = 0;

function generateData(n) {
  const heteroSeverity = parseFloat(document.getElementById("hetero-severity").value);
  const nonlinearSeverity = parseFloat(document.getElementById("nonlinear-severity").value);
  const skewSeverity = parseFloat(document.getElementById("skew-severity").value);

  const baseSigma = 1.3;
  const EX2_CENTERED = 8.3333; // E[(x-5)^2] for x ~ Uniform(0,10), keeps the curvature term from shifting the mean

  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 10;
    let trueY = 2 + 1.3 * x;
    trueY += nonlinearSeverity * ((x - 5) * (x - 5) - EX2_CENTERED);

    // Sigma fans out (or in) linearly across the x-range; severity=0 keeps it constant.
    const sigma = baseSigma * (1 + heteroSeverity * (x - 5) / 5);

    // Blend standard-normal and standardized-skewed noise, holding variance fixed at 1.
    // Weighting by sqrt(1-w)/sqrt(w) (rather than sqrt(1-w^2)/w) makes the resulting
    // skewness scale roughly as w^1.5 instead of w^3, so moderate slider values are
    // actually noticeable instead of being suppressed until close to the max.
    const w = skewSeverity;
    const noiseStd = Math.sqrt(1 - w) * randNormal() + Math.sqrt(w) * randSkewed();

    pts.push({ id: nextId++, x, y: trueY + noiseStd * sigma });
  }
  return pts;
}

// ---------- OLS + diagnostics ----------

function computeFit(points) {
  const n = points.length;
  const p = 2;
  if (n < 3) return null;

  const meanX = d3.mean(points, d => d.x);
  const meanY = d3.mean(points, d => d.y);
  const SSX = d3.sum(points, d => (d.x - meanX) ** 2);
  if (SSX === 0) return null;

  const slope = d3.sum(points, d => (d.x - meanX) * (d.y - meanY)) / SSX;
  const intercept = meanY - slope * meanX;
  const fitted = points.map(d => intercept + slope * d.x);
  const resid = points.map((d, i) => d.y - fitted[i]);
  const SSE = d3.sum(resid, r => r * r);
  const sigmaHat = Math.sqrt(SSE / (n - p));
  const leverage = points.map(d => 1 / n + (d.x - meanX) ** 2 / SSX);
  const stdResid = resid.map((r, i) => {
    const denom = sigmaHat * Math.sqrt(Math.max(1 - leverage[i], 1e-6));
    return denom === 0 ? 0 : r / denom;
  });
  const cooksD = stdResid.map((r, i) => (r * r / p) * (leverage[i] / Math.max(1 - leverage[i], 1e-6)));

  return { n, p, slope, intercept, fitted, resid, leverage, stdResid, cooksD, sigmaHat };
}

// ---------- main scatter panel ----------

const mainWidth = 900, mainHeight = 340;
const mainMargin = { top: 20, right: 20, bottom: 45, left: 55 };

const mainSvg = d3.select("#main-panel").append("svg")
  .attr("width", mainWidth).attr("height", mainHeight)
  .attr("role", "img")
  .attr("aria-label", "Scatter plot of the simulated data with an OLS fitted line and dashed residual segments; points flagged as highly influential are outlined and colored red. Current fit statistics are reported in the text below the plot.");

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
    points.push({ id: nextId++, x: mainX.invert(px), y: mainY.invert(py) });
    update();
  });

const mainResidualLayer = mainSvg.append("g");
const mainLineLayer = mainSvg.append("g");
const mainPointLayer = mainSvg.append("g");

const mainDrag = d3.drag()
  .subject((event, d) => ({ x: mainX(d.x), y: mainY(d.y) }))
  .on("drag", (event, d) => {
    d.x = mainX.invert(event.x);
    d.y = mainY.invert(event.y);
    update();
  });

function setMainDomain(pts) {
  const xExt = d3.extent(pts, d => d.x);
  const yExt = d3.extent(pts, d => d.y);
  const xPad = (xExt[1] - xExt[0]) * 0.25 || 1;
  const yPad = (yExt[1] - yExt[0]) * 0.4 || 1;
  mainX.domain([xExt[0] - xPad, xExt[1] + xPad]);
  mainY.domain([yExt[0] - yPad, yExt[1] + yPad]);
}

function renderMain(fit) {
  mainXAxisG.call(d3.axisBottom(mainX));
  mainYAxisG.call(d3.axisLeft(mainY));

  const domain = mainX.domain();
  const lineData = fit ? [
    { x: domain[0], y: fit.intercept + fit.slope * domain[0] },
    { x: domain[1], y: fit.intercept + fit.slope * domain[1] }
  ] : [];

  mainLineLayer.selectAll("line.fit-line")
    .data(fit ? [lineData] : [])
    .join("line")
    .attr("class", "fit-line")
    .attr("x1", d => mainX(d[0].x)).attr("y1", d => mainY(d[0].y))
    .attr("x2", d => mainX(d[1].x)).attr("y2", d => mainY(d[1].y));

  mainResidualLayer.selectAll("line.residual")
    .data(fit ? points : [], d => d.id)
    .join("line")
    .attr("class", "residual")
    .attr("x1", d => mainX(d.x)).attr("x2", d => mainX(d.x))
    .attr("y1", d => mainY(d.y)).attr("y2", d => mainY(d.fitted));

  mainPointLayer.selectAll("circle")
    .data(points, d => d.id)
    .join("circle")
    .attr("r", 5)
    .call(mainDrag)
    .on("dblclick", (event, d) => {
      event.stopPropagation();
      points = points.filter(p => p.id !== d.id);
      update();
    })
    .attr("class", d => "point" + (d.notable ? " notable" : ""))
    .attr("cx", d => mainX(d.x))
    .attr("cy", d => mainY(d.y));

  addAxisLabels(mainSvg, mainWidth, mainHeight, "x", "y");
}

// ---------- generic small-multiple panel ----------

function setupPanel(containerId, ariaLabel) {
  const width = 440, height = 260;
  const margin = { top: 15, right: 15, bottom: 40, left: 50 };
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
  panel.pointLayer.selectAll("circle")
    .data(data)
    .join("circle")
    .attr("class", d => "diag-point" + (d.notable ? " notable" : ""))
    .attr("r", 4)
    .attr("cx", d => panel.x(xAcc(d)))
    .attr("cy", d => panel.y(yAcc(d)));
}

const residFittedPanel = setupPanel("#resid-fitted", "Residuals versus fitted values, with a dashed reference line at zero. A curved or funnel-shaped pattern indicates nonlinearity or non-constant variance.");
const qqPanel = setupPanel("#qq-plot", "Normal quantile-quantile plot of standardized residuals against theoretical normal quantiles, with a dashed diagonal reference line. Points that stray from the diagonal indicate departure from normality.");
const residHistPanel = setupPanel("#resid-histogram", "Histogram of standardized residuals with the theoretical standard normal density overlaid in red. A shape that departs from this curve indicates non-normal errors.");
const residLeveragePanel = setupPanel("#resid-leverage", "Standardized residuals versus leverage, with dashed Cook's distance contours at 0.5 and 1. Points near or beyond the outer contour are highly influential.");

function renderResidFitted(points) {
  renderScatterPanel(residFittedPanel, points, d => d.fitted, d => d.resid);
  const [x0, x1] = residFittedPanel.x.domain();
  residFittedPanel.decorLayer.selectAll("line.ref-line").data([0]).join("line")
    .attr("class", "ref-line")
    .attr("x1", residFittedPanel.x(x0)).attr("x2", residFittedPanel.x(x1))
    .attr("y1", residFittedPanel.y(0)).attr("y2", residFittedPanel.y(0));
  addAxisLabels(residFittedPanel.svg, residFittedPanel.width, residFittedPanel.height, "Fitted values", "Residuals");
}

function renderQQ(points) {
  const n = points.length;
  const sorted = points.slice().sort((a, b) => a.stdResid - b.stdResid);
  const qqData = sorted.map((d, i) => ({
    tq: qnorm((i + 0.5) / n),
    sr: d.stdResid,
    notable: d.notable
  }));
  renderScatterPanel(qqPanel, qqData, d => d.tq, d => d.sr);
  const [x0, x1] = qqPanel.x.domain();
  qqPanel.decorLayer.selectAll("line.qq-diag").data([0]).join("line")
    .attr("class", "qq-diag")
    .attr("x1", qqPanel.x(x0)).attr("y1", qqPanel.y(x0))
    .attr("x2", qqPanel.x(x1)).attr("y2", qqPanel.y(x1));
  addAxisLabels(qqPanel.svg, qqPanel.width, qqPanel.height, "Theoretical quantiles", "Standardized residuals");
}

function renderResidHistogram(points) {
  const values = points.map(d => d.stdResid);
  const [dmin, dmax] = d3.extent(values);
  const pad = (dmax - dmin) * 0.1 || 1;
  const domain = [dmin - pad, dmax + pad];
  residHistPanel.x.domain(domain);

  const bins = d3.bin().domain(domain).thresholds(20)(values);
  const density = bins.map(b => ({ x0: b.x0, x1: b.x1, d: b.length / (values.length * (b.x1 - b.x0)) }));

  const curve = d3.range(80).map(i => {
    const x = domain[0] + (domain[1] - domain[0]) * i / 79;
    return { x, y: normalPDF(x, 0, 1) };
  });

  const maxY = Math.max(d3.max(density, d => d.d), d3.max(curve, d => d.y)) * 1.1;
  residHistPanel.y.domain([0, maxY]);

  residHistPanel.xAxisG.call(d3.axisBottom(residHistPanel.x).ticks(6));
  residHistPanel.yAxisG.call(d3.axisLeft(residHistPanel.y).ticks(5));

  residHistPanel.decorLayer.selectAll("rect.hist-bar").data(density).join("rect")
    .attr("class", "hist-bar")
    .attr("x", d => residHistPanel.x(d.x0) + 1)
    .attr("width", d => Math.max(0, residHistPanel.x(d.x1) - residHistPanel.x(d.x0) - 1))
    .attr("y", d => residHistPanel.y(d.d))
    .attr("height", d => residHistPanel.y(0) - residHistPanel.y(d.d));

  const lineGen = d3.line().x(d => residHistPanel.x(d.x)).y(d => residHistPanel.y(d.y));
  residHistPanel.pointLayer.selectAll("path.theory-curve").data([curve]).join("path")
    .attr("class", "theory-curve")
    .attr("d", lineGen);

  addAxisLabels(residHistPanel.svg, residHistPanel.width, residHistPanel.height, "Standardized residuals", "density");
}

function renderResidLeverage(points) {
  renderScatterPanel(residLeveragePanel, points, d => d.leverage, d => d.stdResid);

  const p = 2;
  const [hMin, hMax] = residLeveragePanel.x.domain();
  const hLo = Math.max(1e-4, hMin);
  const hHi = Math.min(0.999, hMax);
  const steps = 60;
  const contours = [0.5, 1].map(D => {
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const h = hLo + (hHi - hLo) * i / steps;
      pts.push({ h, r: Math.sqrt(D * p * (1 - h) / h) });
    }
    return pts;
  });
  const linePos = d3.line().x(d => residLeveragePanel.x(d.h)).y(d => residLeveragePanel.y(d.r));
  const lineNeg = d3.line().x(d => residLeveragePanel.x(d.h)).y(d => residLeveragePanel.y(-d.r));

  residLeveragePanel.decorLayer.selectAll("path.cooks-line").remove();
  contours.forEach(pts => {
    residLeveragePanel.decorLayer.append("path").datum(pts).attr("class", "cooks-line").attr("d", linePos);
    residLeveragePanel.decorLayer.append("path").datum(pts).attr("class", "cooks-line").attr("d", lineNeg);
  });

  residLeveragePanel.decorLayer.selectAll("line.ref-line").data([0]).join("line")
    .attr("class", "ref-line")
    .attr("x1", residLeveragePanel.x(hMin)).attr("x2", residLeveragePanel.x(hMax))
    .attr("y1", residLeveragePanel.y(0)).attr("y2", residLeveragePanel.y(0));

  addAxisLabels(residLeveragePanel.svg, residLeveragePanel.width, residLeveragePanel.height,
    "Leverage", "Standardized residuals");
}

// ---------- state + wiring ----------

let points = [];

function update() {
  const fit = computeFit(points);

  if (fit) {
    points.forEach((d, i) => {
      d.fitted = fit.fitted[i];
      d.resid = fit.resid[i];
      d.leverage = fit.leverage[i];
      d.stdResid = fit.stdResid[i];
      d.cooksD = fit.cooksD[i];
      d.notable = d.cooksD > 4 / fit.n;
    });
  } else {
    points.forEach(d => {
      d.fitted = d.resid = d.leverage = d.stdResid = d.cooksD = undefined;
      d.notable = false;
    });
  }

  renderMain(fit);

  document.querySelectorAll(".grid .cell").forEach(c => c.style.opacity = fit ? 1 : 0.35);
  if (fit) {
    renderResidFitted(points);
    renderQQ(points);
    renderResidHistogram(points);
    renderResidLeverage(points);
  }

  const flagged = fit ? points.filter(d => d.notable).length : 0;
  document.getElementById("n").textContent = points.length;
  document.getElementById("slope").textContent = fit ? fit.slope.toFixed(3) : "–";
  document.getElementById("intercept").textContent = fit ? fit.intercept.toFixed(3) : "–";
  document.getElementById("flagged").textContent = flagged;
}

function regenerate() {
  points = generateData(120);
  setMainDomain(points);
  update();
}

document.getElementById("regenerate").addEventListener("click", regenerate);

function wireSeveritySlider(sliderId, readoutId) {
  const slider = document.getElementById(sliderId);
  const readout = document.getElementById(readoutId);
  slider.addEventListener("input", () => {
    readout.textContent = parseFloat(slider.value).toFixed(2);
    regenerate();
  });
}
wireSeveritySlider("hetero-severity", "hetero-readout");
wireSeveritySlider("nonlinear-severity", "nonlinear-readout");
wireSeveritySlider("skew-severity", "skew-readout");

regenerate();
