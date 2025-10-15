

/**
 * Compute forward dot product for one layer (GPU or CPU)
 */
const computeForward = (input, weights, biases) => {
    let output = [];
    for (let neuron = 0; neuron < biases.length; neuron++) {
        let sum = biases[neuron];
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * weights[i][neuron];
        }
        output.push(sum);
    }
    return output;
};



module.exports = {
    computeForward,
};
