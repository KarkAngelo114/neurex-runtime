const activation = require('../../core/bindings')
const { applyPadding, Convolve, Dilate_Input} = require("../../core/bindings");
const { calculateTensorShape, getPaddingSizes } = require("../../utils/utils");

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer) => {
    let [f, kh, kw, kd] = current_layer.weightShape;
    let [input_H, input_W, input_D] = current_layer.inputShape; 
    let padding = current_layer.padding;
    let strides = current_layer.strides;

    // 1. compute expected output tensor shape
    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, current_layer.strides, current_layer.padding);

    // 2. get padding sizes for each sides
    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

    // 3. apply padding
    const {data, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

    // 4. Perform the convolve operation using the shapes calculated in step 1
    const convolve_result = Convolve(data, current_layer.strides, [OutputHeight, OutputWidth], [f, kh, kw, kd], [shape[0], shape[1]], pointer);

    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

    // 5. activate each depth input using the given activation function
    const activation_function = activation[current_layer.activation_function.name];
    const outputs = activation_function(convolve_result);

    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

    return {
        outputs: outputs,
        z_values: convolve_result,
        incrementor_value: 1
    };
}

module.exports = {
    feedforward,
}
