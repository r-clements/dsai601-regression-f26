// ---------- random numbers + distribution helpers ----------

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
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

// Student's t quantile via the Cornish-Fisher expansion around the normal quantile.
// Accurate to a few decimal places for df >= ~4, which is all this app needs.
function qt(p, df) {
  const z = qnorm(p);
  const z2 = z * z, z3 = z2 * z, z5 = z3 * z2, z7 = z5 * z2, z9 = z7 * z2;
  const g1 = (z3 + z) / 4;
  const g2 = (5 * z5 + 16 * z3 + 3 * z) / 96;
  const g3 = (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / 384;
  const g4 = (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / 92160;
  return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4;
}

// ---------- true model + design ----------

const TRUE_B0 = 2, TRUE_B1 = 1.3;
const NUM_REPLICATES = 1000;
const CI_DISPLAY_COUNT = 100;

let xDesign = [];
let designStats = { n: 0, meanX: 0, Sxx: 0 };
let simResults = [];
let replicateIndex = 0;

// y-values for the first CI_DISPLAY_COUNT samples, retained so an individual
// sample can be re-displayed later ("show a missed example").
let displayedYs = [];
let lastLiveYs = null, lastLiveFit = null;
let inspectedIndex = null;
let missCycleIndex = -1;

function generateDesign(n) {
  // Randomize the generating range itself (not just the draws within a fixed
  // range) so that x-bar and the spread of x visibly differ between designs.
  const center = 3 + Math.random() * 4;
  const halfWidth = 2 + Math.random() * 3;
  xDesign = d3.range(n).map(() => center - halfWidth + Math.random() * 2 * halfWidth);
  const meanX = d3.mean(xDesign);
  const Sxx = d3.sum(xDesign, x => (x - meanX) ** 2);
  designStats = { n, meanX, Sxx };
}

function fitOLS(ys) {
  const n = xDesign.length;
  const meanY = d3.mean(ys);
  const b1 = d3.sum(xDesign, (x, i) => (x - designStats.meanX) * (ys[i] - meanY)) / designStats.Sxx;
  const b0 = meanY - b1 * designStats.meanX;
  const resid = ys.map((y, i) => y - (b0 + b1 * xDesign[i]));
  const SSR = d3.sum(resid, r => r * r);
  const s = Math.sqrt(SSR / Math.max(n - 2, 1));
  const seB1 = s / Math.sqrt(designStats.Sxx);
  const seB0 = s * Math.sqrt(1 / n + designStats.meanX ** 2 / designStats.Sxx);
  return { b0, b1, seB0, seB1 };
}

// ---------- generic panel setup ----------

function setupPanel(containerId, width, height, margin, ariaLabel) {
  const svg = d3.select(containerId).append("svg").attr("width", width).attr("height", height)
    .attr("role", "img")
    .attr("aria-label", ariaLabel);
  const x = d3.scaleLinear().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().range([height - margin.bottom, margin.top]);
  const xAxisG = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
  const yAxisG = svg.append("g").attr("transform", `translate(${margin.left},0)`);
  const layer = svg.append("g");
  return { svg, x, y, xAxisG, yAxisG, layer, width, height, margin };
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

// ---------- current sample panel ----------

const currentSampleWidth = 900, currentSampleHeight = 360;
const currentSampleMargin = { top: 15, right: 20, bottom: 40, left: 55 };

const currentSamplePanel = setupPanel("#current-sample-panel", currentSampleWidth, currentSampleHeight, currentSampleMargin,
  "The most recently drawn sample of points with its OLS-fitted line in red, against the fixed true line in gray.");
const currentX = currentSamplePanel.x, currentY = currentSamplePanel.y;
const currentXAxisG = currentSamplePanel.xAxisG, currentYAxisG = currentSamplePanel.yAxisG;
const currentTrueLineLayer = currentSamplePanel.svg.append("g");
const currentLineLayer = currentSamplePanel.svg.append("g");
const currentPointLayer = currentSamplePanel.svg.append("g");

function setCurrentSampleDomain() {
  const xExt = d3.extent(xDesign);
  const xPad = (xExt[1] - xExt[0]) * 0.1 || 1;
  currentX.domain([xExt[0] - xPad, xExt[1] + xPad]);

  const sigma = parseFloat(sigmaSlider.value);
  const domain = currentX.domain();
  const yAtMin = TRUE_B0 + TRUE_B1 * domain[0], yAtMax = TRUE_B0 + TRUE_B1 * domain[1];
  const yPad = 4 * sigma;
  currentY.domain([Math.min(yAtMin, yAtMax) - yPad, Math.max(yAtMin, yAtMax) + yPad]);

  currentXAxisG.call(d3.axisBottom(currentX));
  currentYAxisG.call(d3.axisLeft(currentY));
  addAxisLabels(currentSamplePanel.svg, currentSampleWidth, currentSampleHeight, "x", "y");
}

function renderTrueLine() {
  const domain = currentX.domain();
  const lineData = [
    { x: domain[0], y: TRUE_B0 + TRUE_B1 * domain[0] },
    { x: domain[1], y: TRUE_B0 + TRUE_B1 * domain[1] }
  ];
  currentTrueLineLayer.selectAll("line.true-ref-line").data([lineData]).join("line")
    .attr("class", "true-ref-line")
    .attr("x1", d => currentX(d[0].x)).attr("y1", d => currentY(d[0].y))
    .attr("x2", d => currentX(d[1].x)).attr("y2", d => currentY(d[1].y));
}

function clearCurrentSample() {
  currentPointLayer.selectAll("circle").remove();
  currentLineLayer.selectAll("line.fit-line").remove();
  document.getElementById("cur-est-b0").textContent = "–";
  document.getElementById("cur-est-b1").textContent = "–";
}

function renderCurrentSample(ys, fit, transitionMs) {
  const data = xDesign.map((x, i) => ({ x, y: ys[i], i }));

  currentPointLayer.selectAll("circle").data(data, d => d.i)
    .join(enter => enter.append("circle").attr("class", "point").attr("r", 4)
      .attr("cx", d => currentX(d.x)).attr("cy", d => currentY(d.y)))
    .transition().duration(transitionMs)
    .attr("cx", d => currentX(d.x))
    .attr("cy", d => currentY(d.y));

  const domain = currentX.domain();
  const lineData = [
    { x: domain[0], y: fit.b0 + fit.b1 * domain[0] },
    { x: domain[1], y: fit.b0 + fit.b1 * domain[1] }
  ];
  currentLineLayer.selectAll("line.fit-line").data([lineData])
    .join(enter => enter.append("line").attr("class", "fit-line")
      .attr("x1", d => currentX(d[0].x)).attr("y1", d => currentY(d[0].y))
      .attr("x2", d => currentX(d[1].x)).attr("y2", d => currentY(d[1].y)))
    .transition().duration(transitionMs)
    .attr("x1", d => currentX(d[0].x)).attr("y1", d => currentY(d[0].y))
    .attr("x2", d => currentX(d[1].x)).attr("y2", d => currentY(d[1].y));

  document.getElementById("cur-est-b0").textContent = fit.b0.toFixed(3);
  document.getElementById("cur-est-b1").textContent = fit.b1.toFixed(3);
}

// ---------- accumulated histogram panels ----------

const b0HistPanel = setupPanel("#b0-hist-panel", 440, 280, { top: 15, right: 15, bottom: 40, left: 50 },
  "Histogram of the estimated beta-0 accumulated so far, with the theoretical normal sampling distribution overlaid in red.");
const b1HistPanel = setupPanel("#b1-hist-panel", 440, 280, { top: 15, right: 15, bottom: 40, left: 50 },
  "Histogram of the estimated beta-1 accumulated so far, with the theoretical normal sampling distribution overlaid in red.");

function renderHistogram(panel, values, theoreticalMean, theoreticalSD, trueValue, xLabel) {
  if (values.length < 2) {
    panel.layer.selectAll("*").remove();
    return;
  }

  const [dmin, dmax] = d3.extent(values);
  const pad = (dmax - dmin) * 0.1 || 1;
  const domain = [dmin - pad, dmax + pad];
  panel.x.domain(domain);

  const bins = d3.bin().domain(domain).thresholds(30)(values);
  const density = bins.map(b => ({ x0: b.x0, x1: b.x1, d: b.length / (values.length * (b.x1 - b.x0)) }));

  const curve = d3.range(80).map(i => {
    const x = domain[0] + (domain[1] - domain[0]) * i / 79;
    return { x, y: normalPDF(x, theoreticalMean, theoreticalSD) };
  });

  const maxY = Math.max(d3.max(density, d => d.d), d3.max(curve, d => d.y)) * 1.1;
  panel.y.domain([0, maxY]);

  panel.xAxisG.call(d3.axisBottom(panel.x).ticks(6));
  panel.yAxisG.call(d3.axisLeft(panel.y).ticks(5));

  panel.layer.selectAll("rect.hist-bar").data(density).join("rect")
    .attr("class", "hist-bar")
    .attr("x", d => panel.x(d.x0) + 1)
    .attr("width", d => Math.max(0, panel.x(d.x1) - panel.x(d.x0) - 1))
    .attr("y", d => panel.y(d.d))
    .attr("height", d => panel.y(0) - panel.y(d.d));

  const lineGen = d3.line().x(d => panel.x(d.x)).y(d => panel.y(d.y));
  panel.layer.selectAll("path.theory-curve").data([curve]).join("path")
    .attr("class", "theory-curve")
    .attr("d", lineGen);

  panel.layer.selectAll("line.true-ref-line").data([trueValue]).join("line")
    .attr("class", "true-ref-line")
    .attr("x1", d => panel.x(d)).attr("x2", d => panel.x(d))
    .attr("y1", panel.y.range()[0]).attr("y2", panel.y.range()[1]);

  addAxisLabels(panel.svg, panel.width, panel.height, xLabel, "density");
}

function updateAccumulatedPanels(sigma) {
  const n = designStats.n;
  const theoryMeanB1 = TRUE_B1, theorySDB1 = sigma / Math.sqrt(designStats.Sxx);
  const theoryMeanB0 = TRUE_B0, theorySDB0 = sigma * Math.sqrt(1 / n + designStats.meanX ** 2 / designStats.Sxx);

  const b0Values = simResults.map(f => f.b0);
  const b1Values = simResults.map(f => f.b1);

  renderHistogram(b0HistPanel, b0Values, theoryMeanB0, theorySDB0, TRUE_B0, "β̂₀");
  renderHistogram(b1HistPanel, b1Values, theoryMeanB1, theorySDB1, TRUE_B1, "β̂₁");

  document.getElementById("mean-b0").textContent = b0Values.length >= 2 ? d3.mean(b0Values).toFixed(3) : "–";
  document.getElementById("theory-mean-b0").textContent = theoryMeanB0.toFixed(3);
  document.getElementById("sd-b0").textContent = b0Values.length >= 2 ? d3.deviation(b0Values).toFixed(3) : "–";
  document.getElementById("theory-sd-b0").textContent = theorySDB0.toFixed(3);

  document.getElementById("mean-b1").textContent = b1Values.length >= 2 ? d3.mean(b1Values).toFixed(3) : "–";
  document.getElementById("theory-mean-b1").textContent = theoryMeanB1.toFixed(3);
  document.getElementById("sd-b1").textContent = b1Values.length >= 2 ? d3.deviation(b1Values).toFixed(3) : "–";
  document.getElementById("theory-sd-b1").textContent = theorySDB1.toFixed(3);
}

// ---------- confidence interval panel ----------

const ciPanel = setupPanel("#ci-panel", 900, 420, { top: 15, right: 20, bottom: 40, left: 55 },
  "Forest plot of confidence intervals from samples 1 through 100 of the 1000 simulated samples, in the order they were drawn, for the selected coefficient. Each is colored by whether it captured the true value, with a reference line at the true value. The coverage rate reported below the panel is computed across all 1000 samples, not just the 100 shown here.");

function currentCoefKey() {
  return document.querySelector('input[name="ci-coef"]:checked').value;
}

function renderCIPanel(coefKey, tCrit) {
  const displayed = simResults.slice(0, CI_DISPLAY_COUNT);
  const trueValue = coefKey === "b0" ? TRUE_B0 : TRUE_B1;

  const intervals = displayed.map((fit, i) => {
    const est = fit[coefKey];
    const se = coefKey === "b0" ? fit.seB0 : fit.seB1;
    const lo = est - tCrit * se, hi = est + tCrit * se;
    return { i, est, lo, hi, covered: trueValue >= lo && trueValue <= hi };
  });

  const domain = [d3.min(intervals, d => d.lo), d3.max(intervals, d => d.hi)];
  const pad = (domain[1] - domain[0]) * 0.05 || 1;
  ciPanel.x.domain([domain[0] - pad, domain[1] + pad]);
  ciPanel.y.domain([CI_DISPLAY_COUNT + 1, 0]);

  ciPanel.xAxisG.call(d3.axisBottom(ciPanel.x));
  ciPanel.yAxisG.call(d3.axisLeft(ciPanel.y).ticks(5));

  ciPanel.layer.selectAll("line.true-ref-line").data([trueValue]).join("line")
    .attr("class", "true-ref-line")
    .attr("x1", d => ciPanel.x(d)).attr("x2", d => ciPanel.x(d))
    .attr("y1", ciPanel.y.range()[0]).attr("y2", ciPanel.y.range()[1]);

  ciPanel.layer.selectAll("line.ci-line").data(intervals, d => d.i).join("line")
    .attr("class", d => "ci-line " + (d.covered ? "covered" : "missed") + (d.i === inspectedIndex ? " inspected" : ""))
    .attr("x1", d => ciPanel.x(d.lo)).attr("x2", d => ciPanel.x(d.hi))
    .attr("y1", d => ciPanel.y(d.i + 1)).attr("y2", d => ciPanel.y(d.i + 1))
    .on("click", (event, d) => inspectSample(d.i, coefKey, tCrit));

  ciPanel.layer.selectAll("circle.ci-dot").data(intervals, d => d.i).join("circle")
    .attr("class", "ci-dot")
    .attr("r", 1.5)
    .attr("cx", d => ciPanel.x(d.est))
    .attr("cy", d => ciPanel.y(d.i + 1))
    .on("click", (event, d) => inspectSample(d.i, coefKey, tCrit));

  const coefLabel = coefKey === "b0" ? "β₀" : "β₁";
  addAxisLabels(ciPanel.svg, ciPanel.width, ciPanel.height, coefLabel + " interval", "sample # (1–100 of 1000)");

  const coverageCount = simResults.filter(fit => {
    const est = fit[coefKey];
    const se = coefKey === "b0" ? fit.seB0 : fit.seB1;
    return trueValue >= est - tCrit * se && trueValue <= est + tCrit * se;
  }).length;

  return coverageCount;
}

function renderCI() {
  const confidence = parseFloat(confidenceSelect.value);
  const alpha = 1 - confidence;
  const tCrit = qt(1 - alpha / 2, designStats.n - 2);
  const coefKey = currentCoefKey();
  const coverageCount = renderCIPanel(coefKey, tCrit);

  document.getElementById("coverage-rate").textContent =
    `${coverageCount} / ${simResults.length} (${(100 * coverageCount / simResults.length).toFixed(1)}%)`;
  document.getElementById("coverage-target").textContent = `${(confidence * 100).toFixed(0)}%`;
}

function showCI() {
  document.getElementById("ci-placeholder").hidden = true;
  document.getElementById("ci-content").hidden = false;
  document.getElementById("show-missed").disabled = false;
  document.getElementById("show-latest").disabled = false;
  renderCI();
}

function hideCI() {
  document.getElementById("ci-placeholder").hidden = false;
  document.getElementById("ci-content").hidden = true;
}

// ---------- inspecting individual samples from the CI panel ----------

function inspectSample(index, coefKey, tCrit) {
  inspectedIndex = index;
  const ys = displayedYs[index];
  const fit = simResults[index];
  renderCurrentSample(ys, fit, 0);

  const trueValue = coefKey === "b0" ? TRUE_B0 : TRUE_B1;
  const est = fit[coefKey];
  const se = coefKey === "b0" ? fit.seB0 : fit.seB1;
  const lo = est - tCrit * se, hi = est + tCrit * se;
  const covered = trueValue >= lo && trueValue <= hi;
  const coefLabel = coefKey === "b0" ? "β₀" : "β₁";

  document.getElementById("inspect-note").textContent =
    `Inspecting sample #${index + 1}: ${coefLabel} CI = [${lo.toFixed(3)}, ${hi.toFixed(3)}], ` +
    `true ${coefLabel} = ${trueValue.toFixed(2)} → ${covered ? "captured" : "MISSED"}.`;

  renderCI();
}

function showMissedExample() {
  const confidence = parseFloat(confidenceSelect.value);
  const tCrit = qt(1 - (1 - confidence) / 2, designStats.n - 2);
  const coefKey = currentCoefKey();
  const trueValue = coefKey === "b0" ? TRUE_B0 : TRUE_B1;

  const missed = [];
  simResults.slice(0, CI_DISPLAY_COUNT).forEach((fit, i) => {
    const est = fit[coefKey];
    const se = coefKey === "b0" ? fit.seB0 : fit.seB1;
    if (!(trueValue >= est - tCrit * se && trueValue <= est + tCrit * se)) missed.push(i);
  });

  if (missed.length === 0) {
    document.getElementById("inspect-note").textContent =
      "No misses among samples 1–100 at this confidence level — try a lower confidence level, or Restart for a fresh set of samples.";
    return;
  }

  missCycleIndex = (missCycleIndex + 1) % missed.length;
  inspectSample(missed[missCycleIndex], coefKey, tCrit);
}

function showLatestSample() {
  inspectedIndex = null;
  if (lastLiveYs) renderCurrentSample(lastLiveYs, lastLiveFit, 0);
  document.getElementById("inspect-note").textContent = "";
  renderCI();
}

// ---------- animation state + controls ----------

const nSlider = document.getElementById("n-slider");
const sigmaSlider = document.getElementById("sigma-slider");
const nReadout = document.getElementById("n-readout");
const sigmaReadout = document.getElementById("sigma-readout");
const confidenceSelect = document.getElementById("confidence-level");
const speedSlider = document.getElementById("speed-slider");
const playPauseBtn = document.getElementById("play-pause");
const stepBtn = document.getElementById("step");
const skipBtn = document.getElementById("skip-to-end");

let isPlaying = false;
let timerHandle = null;

function setControlsDisabled(disabled) {
  playPauseBtn.disabled = disabled;
  stepBtn.disabled = disabled;
  skipBtn.disabled = disabled;
}

function stepOnce() {
  if (replicateIndex >= NUM_REPLICATES) return;

  const sigma = parseFloat(sigmaSlider.value);
  const ys = xDesign.map(x => TRUE_B0 + TRUE_B1 * x + randNormal() * sigma);
  const fit = fitOLS(ys);
  simResults.push(fit);
  if (displayedYs.length < CI_DISPLAY_COUNT) displayedYs.push(ys);
  lastLiveYs = ys; lastLiveFit = fit;
  replicateIndex++;

  const transitionMs = Math.min(300, (1000 / speedsPerSecond()) * 0.7);
  renderCurrentSample(ys, fit, transitionMs);
  updateAccumulatedPanels(sigma);
  document.getElementById("rep-index").textContent = replicateIndex;

  if (replicateIndex >= NUM_REPLICATES) {
    pause();
    setControlsDisabled(true);
    showCI();
  }
}

function speedsPerSecond() {
  return parseInt(speedSlider.value, 10);
}

function scheduleNext() {
  if (!isPlaying) return;
  stepOnce();
  if (isPlaying && replicateIndex < NUM_REPLICATES) {
    timerHandle = setTimeout(scheduleNext, 1000 / speedsPerSecond());
  } else {
    isPlaying = false;
    playPauseBtn.textContent = "Play";
  }
}

function play() {
  if (replicateIndex >= NUM_REPLICATES) return;
  isPlaying = true;
  playPauseBtn.textContent = "Pause";
  scheduleNext();
}

function pause() {
  isPlaying = false;
  playPauseBtn.textContent = "Play";
  if (timerHandle) { clearTimeout(timerHandle); timerHandle = null; }
}

function stepManual() {
  pause();
  stepOnce();
}

function skipToEnd() {
  pause();
  const sigma = parseFloat(sigmaSlider.value);
  let lastYs = null;
  while (replicateIndex < NUM_REPLICATES) {
    const ys = xDesign.map(x => TRUE_B0 + TRUE_B1 * x + randNormal() * sigma);
    const fit = fitOLS(ys);
    simResults.push(fit);
    if (displayedYs.length < CI_DISPLAY_COUNT) displayedYs.push(ys);
    replicateIndex++;
    lastYs = ys;
    lastLiveFit = fit;
  }
  lastLiveYs = lastYs;
  if (lastYs) renderCurrentSample(lastYs, simResults[simResults.length - 1], 0);
  updateAccumulatedPanels(sigma);
  document.getElementById("rep-index").textContent = replicateIndex;
  setControlsDisabled(true);
  showCI();
}

function fullReset(regenerateDesign) {
  pause();
  setControlsDisabled(false);
  if (regenerateDesign) generateDesign(parseInt(nSlider.value, 10));
  simResults = [];
  replicateIndex = 0;
  displayedYs = [];
  lastLiveYs = null;
  lastLiveFit = null;
  inspectedIndex = null;
  missCycleIndex = -1;
  document.getElementById("inspect-note").textContent = "";
  document.getElementById("show-missed").disabled = true;
  document.getElementById("show-latest").disabled = true;
  setCurrentSampleDomain();
  renderTrueLine();
  clearCurrentSample();
  updateAccumulatedPanels(parseFloat(sigmaSlider.value));
  document.getElementById("rep-index").textContent = 0;
  hideCI();
}

// ---------- wiring ----------

document.getElementById("true-b0").textContent = TRUE_B0.toFixed(2);
document.getElementById("true-b1").textContent = TRUE_B1.toFixed(2);
document.getElementById("rep-total").textContent = NUM_REPLICATES;

nSlider.addEventListener("input", () => { nReadout.textContent = nSlider.value; });
nSlider.addEventListener("change", () => fullReset(true));

sigmaSlider.addEventListener("input", () => {
  const v = parseFloat(sigmaSlider.value).toFixed(1);
  sigmaReadout.textContent = v;
  document.getElementById("true-sigma").textContent = v;
});
sigmaSlider.addEventListener("change", () => fullReset(false));

function onCISettingsChanged() {
  missCycleIndex = -1;
  inspectedIndex = null;
  document.getElementById("inspect-note").textContent = "";
  if (replicateIndex >= NUM_REPLICATES) renderCI();
}
confidenceSelect.addEventListener("change", onCISettingsChanged);
document.querySelectorAll('input[name="ci-coef"]').forEach(el => el.addEventListener("change", onCISettingsChanged));

playPauseBtn.addEventListener("click", () => { isPlaying ? pause() : play(); });
stepBtn.addEventListener("click", stepManual);
skipBtn.addEventListener("click", skipToEnd);
document.getElementById("restart").addEventListener("click", () => fullReset(false));
document.getElementById("new-design").addEventListener("click", () => fullReset(true));
document.getElementById("show-missed").addEventListener("click", showMissedExample);
document.getElementById("show-latest").addEventListener("click", showLatestSample);

generateDesign(parseInt(nSlider.value, 10));
fullReset(false);
