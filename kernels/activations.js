
// === Activation wrappers ===
const relu = (xArray, onGPU) => {
  return Math.max(0, xArray);
};

const sigmoid = (xArray, onGPU) => {
  return 1 / (1 + Math.exp(-xArray));
};

const tanh = (xArray) => {
    return Math.tanh(xArray);
};

const linear = (xArray) => {
    return xArray;
};

// Softmax stays CPU for stability
const softmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
};

// === Derivatives ===
const dreLu = x => x > 0 ? 1 : 0;
const dsigmoid = x => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 - s);
};
const dtanh = x => 1 - Math.pow(Math.tanh(x), 2);
const dlinear = () => 1;
const dsoftmax = () => 1;

module.exports = {
  relu,
  sigmoid,
  tanh,
  linear,
  softmax,
  derivatives: {
    relu: dreLu,
    sigmoid: dsigmoid,
    tanh: dtanh,
    linear: dlinear,
    softmax: dsoftmax
  },
};
