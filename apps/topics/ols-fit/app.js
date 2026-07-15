const width = 700;
const height = 420;
const margin = { top: 20, right: 20, bottom: 40, left: 50 };

const svg = d3.select("#chart")
  .append("svg")
  .attr("width", width)
  .attr("height", height)
  .attr("role", "img")
  .attr("aria-label", "Scatter plot with an ordinary least squares fitted line and dashed residual segments; current fit statistics are reported in the text below the plot.");

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
  .style("pointer-events", "all")
  .on("click", (event) => {
    const [px, py] = d3.pointer(event);
    const id = nextId++;
    points.push({ id, x: x.invert(px), y: y.invert(py) });
    update();
    focusPointById(id);
  });

const residualLayer = svg.append("g");
const pointLayer = svg.append("g");
const lineLayer = svg.append("g");

let points = [];
let nextId = 0;

function focusPointById(id) {
  const node = pointLayer.selectAll("circle").filter(d => d.id === id).node();
  if (node) node.focus();
}

// Arrow keys nudge the focused point (Shift = larger step); Delete/Backspace removes it.
function handlePointKeydown(event, d) {
  const stepX = (x.domain()[1] - x.domain()[0]) * (event.shiftKey ? 0.1 : 0.02);
  const stepY = (y.domain()[1] - y.domain()[0]) * (event.shiftKey ? 0.1 : 0.02);
  switch (event.key) {
    case "ArrowLeft": event.preventDefault(); d.x -= stepX; update(); break;
    case "ArrowRight": event.preventDefault(); d.x += stepX; update(); break;
    case "ArrowUp": event.preventDefault(); d.y += stepY; update(); break;
    case "ArrowDown": event.preventDefault(); d.y -= stepY; update(); break;
    case "Delete":
    case "Backspace": {
      event.preventDefault();
      const idx = points.findIndex(p => p.id === d.id);
      points = points.filter(p => p.id !== d.id);
      update();
      const nodes = pointLayer.selectAll("circle").nodes();
      if (nodes.length > 0) {
        nodes[Math.min(idx, nodes.length - 1)].focus();
      } else {
        document.getElementById("add-point").focus();
      }
      break;
    }
  }
}

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
    .data(points, d => d.id)
    .join("circle")
    .attr("class", "point")
    .attr("r", 5)
    .attr("tabindex", 0)
    .attr("role", "button")
    .on("keydown", handlePointKeydown)
    .attr("aria-label", d => `Data point ${points.indexOf(d) + 1} of ${points.length} at x = ${d.x.toFixed(2)}, y = ${d.y.toFixed(2)}. Use arrow keys to move, Delete or Backspace to remove.`)
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

document.getElementById("add-point").addEventListener("click", () => {
  const id = nextId++;
  const cx = points.length ? d3.mean(points, p => p.x) : (x.domain()[0] + x.domain()[1]) / 2;
  const cy = points.length ? d3.mean(points, p => p.y) : (y.domain()[0] + y.domain()[1]) / 2;
  points.push({ id, x: cx, y: cy });
  update();
  focusPointById(id);
});

update();
