// Run the JavaScript network over the whole MNIST test set and compare
// with what the Python version scored (98.40%, 160 mistakes).
//
//   node check_accuracy.js ../mnist-from-scratch/data
//
// It then repeats the test with every digit redrawn at a random size and
// position on a 280x280 canvas and passed through preprocess.js, which is
// what happens to a real drawing.

import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

import { loadModel, predict, argmax } from "./network.js";
import { toMnist, SIZE } from "./preprocess.js";

const dataDir = process.argv[2] || "../mnist-from-scratch/data";

function readGz(name) {
  return zlib.gunzipSync(fs.readFileSync(path.join(dataDir, name)));
}

function seeded(seed) {
  return () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
}

// Paste a 28x28 digit onto an empty canvas, enlarged `scale` times with
// its top left corner at (left, top).
function paste(image, canvasSize, scale, left, top) {
  const canvas = new Float32Array(canvasSize * canvasSize);
  for (let y = 0; y < SIZE * scale; y++) {
    for (let x = 0; x < SIZE * scale; x++) {
      const v = image[Math.floor(y / scale) * SIZE + Math.floor(x / scale)];
      canvas[(top + y) * canvasSize + left + x] = v;
    }
  }
  return canvas;
}

const meta = JSON.parse(fs.readFileSync("model.json", "utf8"));
const bin = fs.readFileSync("model.bin");
const model = loadModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

const images = readGz("t10k-images-idx3-ubyte.gz").subarray(16);
const labels = readGz("t10k-labels-idx1-ubyte.gz").subarray(8);
const count = labels.length;

const random = seeded(0);
const CANVAS = 280;
let plainCorrect = 0;
let redrawnCorrect = 0;
let rawCorrect = 0;

const start = performance.now();
for (let n = 0; n < count; n++) {
  const image = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) image[i] = images[n * SIZE * SIZE + i] / 255;

  // 1. the test image exactly as it is
  if (argmax(predict(model, image)) === labels[n]) plainCorrect++;

  // 2. enlarged 2 to 9 times and placed anywhere on the canvas
  const scale = 2 + Math.floor(random() * 8);
  const room = CANVAS - SIZE * scale;
  const left = Math.floor(random() * (room + 1));
  const top = Math.floor(random() * (room + 1));
  const canvas = paste(image, CANVAS, scale, left, top);

  if (argmax(predict(model, toMnist(canvas, CANVAS, CANVAS))) === labels[n]) redrawnCorrect++;

  // 3. the same canvas simply shrunk to 28x28, with no centring
  const naive = new Float32Array(SIZE * SIZE);
  const step = CANVAS / SIZE;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        for (let dx = 0; dx < step; dx++) sum += canvas[(y * step + dy) * CANVAS + x * step + dx];
      }
      naive[y * SIZE + x] = sum / (step * step);
    }
  }
  if (argmax(predict(model, naive)) === labels[n]) rawCorrect++;
}
const seconds = (performance.now() - start) / 1000;

const pct = (n) => `${((n / count) * 100).toFixed(2)}%`;
console.log(`${count} test images`);
console.log(`as they are:                         ${pct(plainCorrect)}  (${count - plainCorrect} wrong)`);
console.log(`random size and position, centred:   ${pct(redrawnCorrect)}  (${count - redrawnCorrect} wrong)`);
console.log(`random size and position, no centring: ${pct(rawCorrect)}  (${count - rawCorrect} wrong)`);
console.log(`took ${seconds.toFixed(1)}s`);
