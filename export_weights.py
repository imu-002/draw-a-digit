"""Convert the trained weights from mnist-from-scratch into files the
browser can load.

    python export_weights.py path/to/mnist-from-scratch

Writes:
    model.bin            all weights as 32-bit floats, one layer after another
    model.json           the shape of each layer, so the JS knows how to cut up model.bin
    fixtures/samples.json  100 test images with the Python network's own
                           outputs, used to check that the JS gives the same answers
"""

import json
import os
import sys

import numpy as np

repo = sys.argv[1] if len(sys.argv) > 1 else "../mnist-from-scratch"
sys.path.insert(0, repo)

import data  # noqa: E402  (from mnist-from-scratch)
from network import Network, softmax  # noqa: E402

weights = np.load(os.path.join(repo, "results", "weights.npz"))
n_layers = len(weights.files) // 2

layers = []
chunks = []
for i in range(n_layers):
    W, b = weights[f"W{i}"], weights[f"b{i}"]
    # NumPy stores W as (inputs, outputs). Write it as (outputs, inputs) so
    # that in JS the weights for one output neuron sit next to each other.
    chunks.append(W.T.astype("<f4").ravel())
    chunks.append(b.astype("<f4").ravel())
    layers.append({"inputs": int(W.shape[0]), "outputs": int(W.shape[1])})

np.concatenate(chunks).tofile("model.bin")
with open("model.json", "w") as f:
    json.dump({"layers": layers}, f)
print(f"model.bin: {os.path.getsize('model.bin'):,} bytes, layers {layers}")

# ---- fixtures ----
_, _, x_test, y_test = data.load(os.path.join(repo, "data"))
net = Network([layers[0]["inputs"]] + [l["outputs"] for l in layers])
net.load(os.path.join(repo, "results", "weights.npz"))

x = x_test[:100]
probs = softmax(net.forward(x))
os.makedirs("fixtures", exist_ok=True)
with open("fixtures/samples.json", "w") as f:
    json.dump({
        # pixels as 0-255 integers to keep the file small
        "images": np.round(x * 255).astype(int).tolist(),
        "labels": y_test[:100].tolist(),
        "probabilities": np.round(probs, 6).tolist(),
    }, f)
print(f"fixtures/samples.json: 100 images, "
      f"{(probs.argmax(axis=1) == y_test[:100]).sum()} classified correctly by Python")
