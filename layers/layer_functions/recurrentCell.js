const { recurrentMatMul } = require('../../core/bindings');
const activation = require('../../core/bindings');

/**
 * The feedforward logic of this layer
 * @param {Float32Array} inputSequence input sequence data 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (inputSequence, current_layer, pointer, outputTemplatePointer) => {

    const units = current_layer.units;
    // Assume inputSequence is flat: [units * sequence_length]
    const sequence_length = current_layer.maxSequenceLength || 1; 
    const feature_size = current_layer.weightShape[0]; // [feature_size/size, units]

    // 1. Initialize a clean hidden state vector for the start of THIS sequence sample
    let current_hidden = new Float32Array(units).fill(0); 

    // 2. Arrays to store history for Backpropagation Through Time (BPTT)
    const all_z_values = [];
    const all_hidden_states = []; 

    // 3. Loop through each time step sequentially
    for (let t = 0; t < sequence_length; t++) {
        
        // Extract x_t for the current time step
        let offset = t * feature_size;
        const sequence_data = inputSequence.subarray(offset, offset + feature_size);

        // Compute z_t = (x_t * W_x) + (current_hidden * W_h) + bias
        // Pass current_hidden explicitly so it doesn't leak between global samples
        const z_t = recurrentMatMul(
            sequence_data, 
            current_hidden, 
            [current_layer.weightShape[0], current_layer.weightShape[1]], 
            [current_layer.weightShape[2], current_layer.weightShape[3]], 
            pointer, 
            outputTemplatePointer
        );

        if (z_t.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs on Recurrent layer (feedforward)");
        
        // Update hidden state for the next step
        current_hidden = activation[current_layer.activation_function.name](z_t);

        // Record history for backprop
        all_z_values.push(z_t);
        all_hidden_states.push(new Float32Array(current_hidden));
    }

    // cache recurrent layer cell feedforward data
    current_layer.cache = {
        hidden_states: all_hidden_states,
        recurrentZs: all_z_values
    }

    let final_output;
    if (current_layer.return_sequence) {
        // Concatenate all hidden states into one big flat array if return_sequence is true
        final_output = concatenateFloat32Array(all_hidden_states);
    } else {
        // Just return the very last hidden state vector
        final_output = all_hidden_states[sequence_length - 1];
    }

    return {
        outputs: final_output, 
        z_values: all_z_values, // Pass the array of z_values back to Neurex
        incrementor_value: 1
    };
}

module.exports = {
    feedforward,
}