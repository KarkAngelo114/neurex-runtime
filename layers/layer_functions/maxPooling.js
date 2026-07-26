const { MaxPool } = require("../../core/bindings");

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    const [inputh, inputw, inputd] = current_layer.inputShape;
    const [outputh, outputw, outputd] = current_layer.outputShape;
    const [poolHeight, poolWidth] = current_layer.poolSize;
    const strides = current_layer.strides;
                
    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides, outputTemplatePointer);

    current_layer.maxIndices = maxIndices;

    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

    return {
        outputs: output,
        z_values: output,
        incrementor_value:0
    }
}

module.exports = {
    feedforward,
}
