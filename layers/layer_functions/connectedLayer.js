const { MatMul } = require("../../core/bindings");
const { XavierInitialization, ifOneHotEndcoded } = require("../../utils/utils");
const activation = require('../../core/bindings');

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    const [inputSize, outputSize] = current_layer.weightShape; // weight shape [input, output]
    const z_values = MatMul(input, inputSize, outputSize, pointer, outputTemplatePointer); // perform the MatMul() operation

    const activation_function = activation[current_layer.activation_function.name]; // activation function
    let outputs = activation_function(z_values); // use the activation function       
    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
    return {
        outputs, 
        z_values,
        incrementor_value: 1
    };
}


module.exports = {
    feedforward
}
