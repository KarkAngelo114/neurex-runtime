/**
  * this runtime version only needs feedforward related functions.
  * No need to use native bindings
  */

const functions = require('./float32Ops/ops.js');

/**
 *  "✅☑️"
 * @function getEmbeddings
 * @param {Array<Number>} tokenVector an array of token vector 
 * @param {Number} embeddingDim embedding dim value
 * @param {Number} pointer pointer value corresponding to the global parameter of weights and biases 
 * @param {Number} outputTemplatePointer pointer value correspondind to the output template tensor 
 * @returns {Float32Array} flattened embeddings
 */
const getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => functions.getEmbeddings(Array.from(tokenVector), embeddingDim, pointer, outputTemplatePointer);

/**
 * "✅☑️"
 * @function MatMul
 * @param {Float32Array} inputs - 1D float32array of input features
 * @param {Float32Array} weights - 1D float32array of weights
 * @param {Float32Array} biases - 1D float32array of biases
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @param {Number} pointer - a pointer that will be use to index the corresponding parameter from global params
 * @returns 1D array of output
 */
const MatMul = (inputs, inputSize, outputSize, pointer, outputTemplatePointer) => functions.MatMul(inputs, inputSize, outputSize, pointer, outputTemplatePointer);
/**
 * "✅☑️"
 * @function relu
 * @param {Float32Array} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => functions.Relu(input)

/**
 * "✅☑️"
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => functions.Sigmoid(input);

/**
 * "✅☑️"
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => functions.Tanh(input);

/**
 * "✅☑️"
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => functions.Softmax(input);

/**
 * "✅☑️"
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => functions.Linear(input); 

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Number} inputH 
 * @param {Number} inputW 
 * @param {Number} channels 
 * @param {Number} padTop 
 * @param {Number} padBottom 
 * @param {Number} padLeft 
 * @param {Number} padRight 
 * @returns padded tensor
 */
const applyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => functions.ApplyPadding(input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight);

 /**
  * "✅☑️"
  * @param {Float32Array} input Float32Array input 
  * @param {Number} strides Stride value 
  * @param {Numebr} outputH Expected output height
  * @param {Number} outputW Expected output width
  * @param {Numebr} num_filters number of filters
  * @param {Number} kernel_height kernel height
  * @param {Number} kernel_width kernel width
  * @param {Number} depth depth value
  * @param {Number} inputH current input height
  * @param {Number} inputW currnet input width
  * @param {Number} pointer pointer value to get the matching parameter from the global store
  * @param {Number} outputTemplatePointer pointer value to get the matching output template tensor from the global store 
  * @returns {Float32Array} Convolution result
  */
const Convolve = (input, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW, pointer, outputTemplatePointer) => functions.Convolve(input, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW, pointer, outputTemplatePointer);
/**
 * 
 * "✅☑️"
 * @function 
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after multiplying input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_mul = (flat_arr_1, flat_arr_2) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length}`);
    
    return functions.element_wise_mul(flat_arr_1, flat_arr_2);
}


/**
 * 
 * "✅☑️"
 * @function
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after subtracting input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_sub = (flat_arr_1, flat_arr_2) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length}`);
    return functions.element_wise_sub(new Float32Array(flat_arr_1), new Float32Array(flat_arr_2));
}

/**
 * "✅☑️"
 * @param {Foat32Array} arr1 a flat array input
 * @param {Foat32Array} arr2 a flat array input
 * @param {Foat32Array} arr3 a flat array input
 * @returns a flat array after performing `(arr1[i] - arr2[i]) * arr3[i]`
 * @throws {Error} - if any of the input array are not equal in length
 */
const scaleDiff = (arr1, arr2, arr3) => {
    if (arr1.length !== arr2.length || arr2.length !== arr3.length || arr1.length !== arr3.length) {
        throw new Error(`[ERROR]------- Error: All arrays must be equal in length. array1: ${arr1.length} | array2: ${arr2.length} | array3: ${arr3.length}`);
    }

    return functions.scaleDiff(new Float32Array(arr1), new Float32Array(arr2), new Float32Array(arr3));
}

/**
 * "✅☑️"
 * @function MaxPool
 * @param {Float32Array} input - current input passed down to this layer 
 * @param {Array<Number>} poolSize - pool size of the sliding window
 * @param {Array<Number>} inputShape - input shape of the current tensor
 * @param {Array<Number>} outputShape - output shape of the tensor
 * @param {Number} strides - determines how many pixels it will skipped
 */
const MaxPool = (input, poolSize, inputShape, outputShape, strides, outputTemplatePointer) => functions.MaxPooling(input, poolSize, inputShape, outputShape, strides, outputTemplatePointer);

const derivatives = {
    relu: (input) => {
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i] > 0 ? 1 : 0;
        }
        return output;
    },
    sigmoid: (input) => {
        const sig = functions.Sigmoid(input);
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = sig[i] * (1 - sig[i]);
        }
        return output;
    },
    tanh: (input) => {
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const t = Math.tanh(input[i]);
            output[i] = 1 - t * t;
        }
        return output;
    },
    softmax: (input) => {
        // Jacobian diagonal approximation (used in some loss+activation combos)
        const s = functions.Softmax(input);
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = s[i] * (1 - s[i]);
        }
        return output;
    },
    linear: (input) => {
        return new Float32Array(input.length).fill(1);
    }
};

module.exports = {
    getEmbeddings,
    MatMul,
    relu,
    sigmoid,
    tanh,
    softmax,
    linear,
    applyPadding,
    Convolve,
    element_wise_mul,
    element_wise_sub,
    MaxPool,
    scaleDiff,
    derivatives
}