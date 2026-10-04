// Turn a drawing into the kind of image the network was trained on.
//
// MNIST digits were not just scaled to 28x28. Each one was fitted into a
// 20x20 box and then placed in the 28x28 image so that its centre of mass
// is in the middle. A network trained on that has never seen a digit in a
// corner, so a drawing has to go through the same steps or it will be
// misread.

export const SIZE = 28;
const BOX = 20;

// ink: Float32Array of width * height values, 0 = paper, 1 = full ink.
// Returns a Float32Array of 784 values, or null if nothing is drawn.
export function toMnist(ink, width, height) {
  const box = boundingBox(ink, width, height);
  if (!box) return null;

  // scale so the longer side of the drawing becomes 20 pixels
  const scale = BOX / Math.max(box.w, box.h);
  const w = Math.max(1, Math.round(box.w * scale));
  const h = Math.max(1, Math.round(box.h * scale));
  const small = resize(ink, width, box, w, h);

  // put the centre of mass at the centre of the 28x28 image
  const [cx, cy] = centreOfMass(small, w, h);
  let left = Math.round(SIZE / 2 - cx);
  let top = Math.round(SIZE / 2 - cy);
  left = Math.min(Math.max(left, 0), SIZE - w);
  top = Math.min(Math.max(top, 0), SIZE - h);

  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[(top + y) * SIZE + left + x] = small[y * w + x];
    }
  }
  return out;
}

export function boundingBox(ink, width, height, threshold = 0.1) {
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[y * width + x] > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// Shrink (or grow) the part of the image inside `box` to w x h.
// Each output pixel is the average of the source area it covers, with
// partly covered source pixels counted by how much of them is covered.
// Plain "pick the nearest pixel" would lose thin strokes when shrinking.
export function resize(ink, width, box, w, h) {
  const out = new Float32Array(w * h);
  const sx = box.w / w;
  const sy = box.h / h;

  for (let ty = 0; ty < h; ty++) {
    const y0 = ty * sy;
    const y1 = y0 + sy;
    for (let tx = 0; tx < w; tx++) {
      const x0 = tx * sx;
      const x1 = x0 + sx;

      let sum = 0;
      for (let y = Math.floor(y0); y < Math.ceil(y1); y++) {
        const coverY = Math.min(y + 1, y1) - Math.max(y, y0);
        for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
          const coverX = Math.min(x + 1, x1) - Math.max(x, x0);
          sum += ink[(box.y + y) * width + box.x + x] * coverX * coverY;
        }
      }
      out[ty * w + tx] = sum / (sx * sy);
    }
  }
  return out;
}

export function centreOfMass(image, w, h) {
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = image[y * w + x];
      total += v;
      sumX += v * (x + 0.5);
      sumY += v * (y + 0.5);
    }
  }
  return [sumX / total, sumY / total];
}
