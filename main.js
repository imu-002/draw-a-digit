import { loadModel, predict, argmax } from "./network.js";
import { toMnist, SIZE } from "./preprocess.js";

const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const seen = document.getElementById("seen").getContext("2d");
const digitEl = document.getElementById("digit");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const barsEl = document.getElementById("bars");

let model = null;
let drawing = false;
let last = null;
let updateQueued = false;

// ---- one bar per digit ----

const rows = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML =
    `<span class="label">${d}</span>` +
    `<span class="track"><span class="fill"></span></span>` +
    `<span class="value">0%</span>`;
  barsEl.appendChild(row);
  rows.push(row);
}

// ---- drawing ----

function clearPad() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  showNothing();
}

// pointer position in canvas pixels (the canvas is 280 wide whatever
// size it is shown at)
function position(event) {
  const rect = pad.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * pad.width,
    y: ((event.clientY - rect.top) / rect.height) * pad.height,
  };
}

function stroke(from, to) {
  // MNIST strokes are about 2 pixels thick in a 20 pixel digit. A digit
  // drawn to fill most of this canvas shrinks about 10x, so 20 here.
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

pad.addEventListener("pointerdown", (event) => {
  drawing = true;
  last = position(event);
  stroke(last, last); // a tap leaves a dot
  pad.setPointerCapture(event.pointerId);
  queueUpdate();
});

pad.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const now = position(event);
  stroke(last, now);
  last = now;
  queueUpdate();
});

for (const name of ["pointerup", "pointercancel"]) {
  pad.addEventListener(name, () => {
    drawing = false;
    queueUpdate();
  });
}

document.getElementById("clear").addEventListener("click", clearPad);

// ---- reading the drawing ----

// Pointer events can fire far more often than the screen redraws, so run
// the network at most once per frame.
function queueUpdate() {
  if (updateQueued) return;
  updateQueued = true;
  requestAnimationFrame(() => {
    updateQueued = false;
    update();
  });
}

function update() {
  if (!model) return;

  // canvas pixels -> ink amounts: white paper is 0, black ink is 1
  const data = ctx.getImageData(0, 0, pad.width, pad.height).data;
  const ink = new Float32Array(pad.width * pad.height);
  for (let i = 0; i < ink.length; i++) ink[i] = 1 - data[i * 4] / 255;

  const input = toMnist(ink, pad.width, pad.height);
  if (!input) {
    showNothing();
    return;
  }

  const probs = predict(model, input);
  const best = argmax(probs);
  digitEl.textContent = best;
  confidenceEl.textContent = `${Math.round(probs[best] * 100)}% sure`;

  rows.forEach((row, d) => {
    row.classList.toggle("top", d === best);
    row.querySelector(".fill").style.width = `${probs[d] * 100}%`;
    row.querySelector(".value").textContent = `${Math.round(probs[d] * 100)}%`;
  });
  showInput(input);
}

function showNothing() {
  digitEl.textContent = "?";
  confidenceEl.textContent = "draw something";
  rows.forEach((row) => {
    row.classList.remove("top");
    row.querySelector(".fill").style.width = "0";
    row.querySelector(".value").textContent = "0%";
  });
  seen.fillStyle = "#fff";
  seen.fillRect(0, 0, SIZE, SIZE);
}

function showInput(input) {
  const image = seen.createImageData(SIZE, SIZE);
  for (let i = 0; i < input.length; i++) {
    const shade = Math.round(255 * (1 - input[i]));
    image.data[i * 4] = shade;
    image.data[i * 4 + 1] = shade;
    image.data[i * 4 + 2] = shade;
    image.data[i * 4 + 3] = 255;
  }
  seen.putImageData(image, 0, 0);
}

// ---- start ----

async function start() {
  try {
    const [meta, buffer] = await Promise.all([
      fetch("model.json").then((r) => r.json()),
      fetch("model.bin").then((r) => r.arrayBuffer()),
    ]);
    model = loadModel(meta, buffer);
    const weights = meta.layers.reduce((n, l) => n + l.inputs * l.outputs + l.outputs, 0);
    statusEl.textContent =
      `Network loaded: ${weights.toLocaleString()} weights, 98.40% accuracy on the MNIST test set.`;
    update();
  } catch (error) {
    statusEl.textContent =
      "Could not load the network. Open this page through a web server, not as a file.";
  }
}

clearPad();
start();
