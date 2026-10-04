// The forward pass of the network, in plain JavaScript.
// Same maths as network.py in mnist-from-scratch: Dense -> ReLU -> Dense
// -> ReLU -> Dense -> softmax. Only prediction, no training.

// meta is the parsed model.json, buffer is model.bin as an ArrayBuffer.
export function loadModel(meta, buffer) {
  const all = new Float32Array(buffer);
  const layers = [];
  let offset = 0;

  for (const { inputs, outputs } of meta.layers) {
    const weights = all.subarray(offset, offset + inputs * outputs);
    offset += inputs * outputs;
    const biases = all.subarray(offset, offset + outputs);
    offset += outputs;
    layers.push({ inputs, outputs, weights, biases });
  }
  if (offset !== all.length) {
    throw new Error("model.bin does not match model.json");
  }
  return { layers };
}

// input: 784 numbers between 0 and 1. Returns 10 probabilities.
export function predict(model, input) {
  let x = input;

  model.layers.forEach((layer, index) => {
    const out = new Float32Array(layer.outputs);
    for (let j = 0; j < layer.outputs; j++) {
      // weights for output j are stored in one run of `inputs` numbers
      let sum = layer.biases[j];
      const start = j * layer.inputs;
      for (let i = 0; i < layer.inputs; i++) {
        sum += x[i] * layer.weights[start + i];
      }
      const isLast = index === model.layers.length - 1;
      out[j] = isLast ? sum : Math.max(0, sum); // ReLU on hidden layers only
    }
    x = out;
  });

  return softmax(x);
}

function softmax(logits) {
  // subtract the largest value first so Math.exp cannot overflow
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
