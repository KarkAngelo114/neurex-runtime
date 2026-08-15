const activation = require('../../core/bindings');
const { transConv } = require("../../core/bindings");
const { XavierInitialization, calculateTransposedTensorShape, getTransposedPaddingSizes } = require('../../utils/utils');


/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer) => {
    
    const inputShape = current_layer.inputShape; // [iH, iW, iD]
    const outputShape = current_layer.outputShape; // [oH, oW, oD]
    const weightShape = current_layer.weightShape; // [f, kh, kw, d]
    const strides = current_layer.strides;
    const filters = current_layer.filters;
    const activation_function = activation[current_layer.activation_function.name];

    const transConvOutput = transConv(input, inputShape, outputShape, strides, filters, weightShape, pointer);
    if (transConvOutput.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after trans conv Ops");

    const output = activation_function(transConvOutput);
    if (output.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after applying activation");

    return {
        outputs: output,
        z_values: transConvOutput,
        incrementor_value: 1
    }
}

module.exports = {
    feedforward,
}