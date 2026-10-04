// Run with: node --test

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import { loadModel, predict, argmax } from "./network.js";
import { toMnist, boundingBox, resize, centreOfMass, SIZE } from "./preprocess.js";

const meta = JSON.parse(fs.readFileSync("model.json", "utf8"));
const bin = fs.readFileSync("model.bin");
const model = loadModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const samples = JSON.parse(fs.readFileSync("fixtures/samples.json", "utf8"));

const pixels = (image) => Float32Array.from(image, (v) => v / 255);

// draw a filled rectangle of ink on an empty canvas
function canvasWith(width, height, rect) {
  const ink = new Float32Array(width * height);
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) ink[y * width + x] = 1;
  }
  return ink;
}

// ---- network ----

test("model file has the expected shape", () => {
  assert.deepEqual(model.layers.map((l) => [l.inputs, l.outputs]),
                   [[784, 128], [128, 64], [64, 10]]);
});

test("a model file of the wrong size is refused", () => {
  assert.throws(() => loadModel(meta, new ArrayBuffer(1000)));
});

test("JavaScript gives the same probabilities as Python", () => {
  // fixtures hold what network.py produced for the same 100 images
  samples.images.forEach((image, n) => {
    const probs = predict(model, pixels(image));
    probs.forEach((p, digit) => {
      assert.ok(Math.abs(p - samples.probabilities[n][digit]) < 1e-4,
                `image ${n}, digit ${digit}: ${p} vs ${samples.probabilities[n][digit]}`);
    });
  });
});

test("probabilities add up to 1", () => {
  const probs = predict(model, pixels(samples.images[0]));
  assert.equal(probs.length, 10);
  assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});

test("the 100 sample digits are all read correctly", () => {
  samples.images.forEach((image, n) => {
    assert.equal(argmax(predict(model, pixels(image))), samples.labels[n]);
  });
});

// ---- preprocessing ----

test("an empty canvas gives null", () => {
  assert.equal(toMnist(new Float32Array(100 * 100), 100, 100), null);
});

test("bounding box finds the ink", () => {
  const ink = canvasWith(100, 80, { x: 30, y: 10, w: 20, h: 50 });
  assert.deepEqual(boundingBox(ink, 100, 80), { x: 30, y: 10, w: 20, h: 50 });
});

test("the longer side of the drawing becomes 20 pixels", () => {
  // tall rectangle 40 wide, 200 high -> 4 wide, 20 high
  const out = toMnist(canvasWith(280, 280, { x: 10, y: 20, w: 40, h: 200 }), 280, 280);
  const box = boundingBox(out, SIZE, SIZE);
  assert.equal(box.h, 20);
  assert.equal(box.w, 4);
});

test("the drawing ends up centred wherever it was drawn", () => {
  for (const corner of [{ x: 0, y: 0 }, { x: 200, y: 0 }, { x: 0, y: 180 }, { x: 190, y: 170 }]) {
    const out = toMnist(canvasWith(280, 280, { ...corner, w: 60, h: 90 }), 280, 280);
    const [cx, cy] = centreOfMass(out, SIZE, SIZE);
    assert.ok(Math.abs(cx - 14) <= 0.5 && Math.abs(cy - 14) <= 0.5, `${cx}, ${cy}`);
  }
});

test("position on the canvas does not change the result", () => {
  const a = toMnist(canvasWith(280, 280, { x: 5, y: 5, w: 50, h: 100 }), 280, 280);
  const b = toMnist(canvasWith(280, 280, { x: 180, y: 150, w: 50, h: 100 }), 280, 280);
  assert.deepEqual(Array.from(a), Array.from(b));
});

test("resizing keeps the total amount of ink in proportion", () => {
  // half ink, half paper, shrunk 10x: average stays 0.5
  const ink = canvasWith(100, 100, { x: 0, y: 0, w: 50, h: 100 });
  const small = resize(ink, 100, { x: 0, y: 0, w: 100, h: 100 }, 10, 10);
  const mean = small.reduce((a, b) => a + b, 0) / small.length;
  assert.ok(Math.abs(mean - 0.5) < 1e-6);
});

test("output values stay between 0 and 1", () => {
  const out = toMnist(canvasWith(280, 280, { x: 33, y: 47, w: 71, h: 113 }), 280, 280);
  assert.equal(out.length, 784);
  assert.ok(out.every((v) => v >= 0 && v <= 1 + 1e-6));
});

test("a sample digit enlarged and moved to a corner is still read correctly", () => {
  // put each of the first 20 samples at 6x size in the bottom right corner
  let correct = 0;
  for (let n = 0; n < 20; n++) {
    const image = pixels(samples.images[n]);
    const canvas = new Float32Array(280 * 280);
    for (let y = 0; y < 168; y++) {
      for (let x = 0; x < 168; x++) {
        canvas[(112 + y) * 280 + 112 + x] = image[Math.floor(y / 6) * SIZE + Math.floor(x / 6)];
      }
    }
    if (argmax(predict(model, toMnist(canvas, 280, 280))) === samples.labels[n]) correct++;
  }
  assert.ok(correct >= 19, `${correct} of 20`);
});
