const { getEmbeddings } = require('../../core/bindings');


/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer) => {
    const embeddingDim = current_layer.embeddingDim;

    const output = getEmbeddings(input, embeddingDim, pointer);

    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs on Embedding layer (feedforward)");
    
    return {
        outputs: output, 
        z_values: output,
        incrementor_value: 1
    };
}

module.exports = {
    feedforward,
}