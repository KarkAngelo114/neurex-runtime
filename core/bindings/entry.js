const { getGlobalParams } = require('../../params_init')

/**
  * this runtime version only needs feedforward related functions.
  * No need to use native bindings
  */

const functions = require('./float32Ops');

/**
 *  "✅☑️"
 * @function getEmbeddings
 * @param {Array<Number>} tokenVector an array of token vector 
 * @param {Number} embeddingDim embedding dim value
 * @param {Number} pointer pointer value corresponding to the global parameter of weights and biases 
 * @param {Number} outputTemplatePointer pointer value correspondind to the output template tensor 
 * @returns {Float32Array} flattened embeddings
 */
const getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => functions.getEmbeddings(
    Array.from(tokenVector), 
    embeddingDim, 
    getGlobalParams().globalWeights[pointer],
    outputTemplatePointer
);

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
const MatMul = (inputs, inputSize, outputSize, pointer, outputTemplatePointer) => functions.MatMul(
    inputs, 
    inputSize, 
    outputSize, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer], 
    outputTemplatePointer
);

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
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => functions.DReLu(input);

/**
 * "✅☑️"
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => functions.DSigmoid(input);

/**
 * "✅☑️"
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => functions.DTanh(input);

/**
 * "✅☑️"
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => functions.DSoftmax(input)

/**
 * "✅☑️"
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => functions.DLinear(input);

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
 * @param {Float32Array} input input to perform convolution
 * @param {Number} strides stride value
 * @param {Array<Number>} outputShape [oH, oW]
 * @param {Array<Number>} kernelShape [num_filters, Kh, Kw, channels]
 * @param {Array<Number>} inputShape [iH, iW] 
 * @param {Number} pointer pointer value to fetch corresponding parameters of the layer from the global store
 * @param {Number} outputTemplatePointer pointer value to fetch allocated tensor of the layer from the global store
 * @returns {Float32Array} convolution result
 */
const Convolve = (input, strides, outputShape, kernelShape, inputShape, pointer, outputTemplatePointer) => functions.Convolve(
    input, 
    strides, 
    outputShape, 
    kernelShape, 
    inputShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer], 
    outputTemplatePointer
);

/**
 * "✅☑️" dilate the input inserting 0s
 * @param {Float32Array} input 
 * @param {Array<Number>} shape_array 
 * @param {Number} strides 
 * @returns {Object} {data, dilatedHeight, dilatedWidth}
 */
const Dilate_Input = (input, shape_array, strides) => functions.DilateInput(input, shape_array, strides);

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

/**
 * "☑️"
 * @param {Float32Array} input input vector
 * @param {Float32Array} prevHiddenState hidden temporal state
 * @param {Array<Number>} inputWeightShape input weight shape
 * @param {Array<Number>} recurrentWeightShape recurrent weight shape
 * @param {Number} pointer value to reference the weights and biases 
 * @param {Number} outputTemplatePointer value to reference the output template pointer 
 * @returns 
 */
const recurrentMatMul = (input, prevHiddenState, inputWeightShape, recurrentWeightShape, pointer, outputTemplatePointer) => functions.recurrentMatMul(
    input, 
    prevHiddenState,
    inputWeightShape, 
    recurrentWeightShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer],
    outputTemplatePointer
);


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
    Dilate_Input,
    MaxPool,
    recurrentMatMul,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}