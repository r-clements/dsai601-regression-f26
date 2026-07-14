const width = 700;
const height = 420;
const margin = { top: 20, right: 20, bottom: 40, left: 50 };

const svg = d3.select("#chart")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

const x = d3.scaleLinear().domain([0, 10]).range([margin.left, width - margin.right]);
const y = d3.scaleLinear().domain([0, 10]).range([height - margin.bottom, margin.top]);

svg.append("g")
  .attr("transform", `translate(0,${height - margin.bottom})`)
  .call(d3.axisBottom(x));

svg.append("g")
  .attr("transform", `translate(${margin.left},0)`)
  .call(d3.axisLeft(y));

// Transparent rect to capture clicks anywhere on the plot area
svg.append("rect")
  .attr("x", margin.left)
  .attr("y", margin.top)
  .attr("width", width - margin.left - margin.right)
  .attr("height", height - margin.top - margin.bottom)
  .attr("fill", "transparent")
  .on("click", (event) => {
    const [px, py] = d3.pointer(event);
    points.push({ x: x.invert(px), y: y.invert(py) });
    update();
  });

const residualLayer = svg.append("g");
const pointLayer = svg.append("g");
const lineLayer = svg.append("g");

let points = [];

function ols(pts) {
  const n = pts.length;
  const meanX = d3.mean(pts, d => d.x);
  const meanY = d3.mean(pts, d => d.y);
  const num = d3.sum(pts, d => (d.x - meanX) * (d.y - meanY));
  const den = d3.sum(pts, d => (d.x - meanX) ** 2);
  const slope = den === 0 ? 0 : num / den;
  const intercept = meanY - slope * meanX;
  const ssr = d3.sum(pts, d => (d.y - (slope * d.x + intercept)) ** 2);
  return { slope, intercept, ssr };
}

function update() {
  pointLayer.selectAll("circle")
    .data(points)
    .join("circle")
    .attr("class", "point")
    .attr("r", 5)
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y));

  document.getElementById("n").textContent = points.length;

  if (points.length < 2) {
    lineLayer.selectAll("line.fit-line").remove();
    residualLayer.selectAll("line.residual").remove();
    document.getElementById("slope").textContent = "–";
    document.getElementById("intercept").textContent = "–";
    document.getElementById("ssr").textContent = "–";
    return;
  }

  const { slope, intercept, ssr } = ols(points);
  const x0 = x.domain()[0], x1 = x.domain()[1];

  lineLayer.selectAll("line.fit-line")
    .data([{ x0, x1, y0: slope * x0 + intercept, y1: slope * x1 + intercept }])
    .join("line")
    .attr("class", "fit-line")
    .attr("x1", d => x(d.x0))
    .attr("y1", d => y(d.y0))
    .attr("x2", d => x(d.x1))
    .attr("y2", d => y(d.y1));

  residualLayer.selectAll("line.residual")
    .data(points)
    .join("line")
    .attr("class", "residual")
    .attr("x1", d => x(d.x))
    .attr("y1", d => y(d.y))
    .attr("x2", d => x(d.x))
    .attr("y2", d => y(slope * d.x + intercept));

  document.getElementById("slope").textContent = slope.toFixed(3);
  document.getElementById("intercept").textContent = intercept.toFixed(3);
  document.getElementById("ssr").textContent = ssr.toFixed(3);
}

document.getElementById("reset").addEventListener("click", () => {
  points = [];
  update();
});

update();
