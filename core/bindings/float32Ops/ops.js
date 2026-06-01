/**
  * this runtime version only needs feedforward related functions
  */

const { getGlobalParams } = require("../../../params_init/globals.js");

const Relu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
    return output;
};

const Sigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = 1 / (1 + Math.exp(-output[i]));
    }
    return output;
};

const Tanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.tanh(output[i]);
    }
    return output;
};

const Softmax = (arr) => {
    const output = new Float32Array(arr);
    const maxVal = Math.max(...output);
    let sum = 0;

    for (let i = 0; i < output.length; i++) {
        output[i] = Math.exp(output[i] - maxVal);
        sum += output[i];
    }

    for (let i = 0; i < output.length; i++) {
        output[i] /= sum;
    }

    return output;
};

const Linear = (arr) => {
    return new Float32Array(arr);
};

const getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => {
    const {globalWeights, globalOutputTensorTemplate} = getGlobalParams();

    const lookup = globalWeights[pointer];
    const output = globalOutputTensorTemplate[outputTemplatePointer];

    // helper function
    const getRow = (tokenID) => {
        const start = tokenID * embeddingDim;

        return lookup.subarray(start, start + embeddingDim);
    }

    const sequence_length = tokenVector.length;

    for (let i = 0; i < sequence_length; i++) {
        const row = getRow(tokenVector[i]);

        output.set(row, i * embeddingDim);
    }

    return output;
}

const MatMul = (input, inputSize, outputSize, pointer, outputTemplatePointer) => {

    /**
     * since there's no weights and biases being passed to this function, we use the pointer to reference the parameters
     */

    const {globalWeights, globalBiases, globalOutputTensorTemplate} = getGlobalParams();
    
    const z_values = globalOutputTensorTemplate[outputTemplatePointer]; // use the output template pointer to get the corresponding pre-allocated output tensor

    
    // 1. Initialize with Biases (Faster than adding them in a separate loop later)
    z_values.set(globalBiases[pointer]);

    // 2. Perform Weighted Sum
    // We iterate through each input neuron
    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];
        
        // Calculate the starting offset for this specific input neuron's weights
        const offset = i * outputSize;

        // Multiply the input by every weight connecting to output neurons
        for (let j = 0; j < outputSize; j++) {
            z_values[j] += inputVal * globalWeights[pointer][offset + j];
        }
    }

    return z_values;
}

const ApplyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => {
    const newH = inputH + padTop + padBottom;
    const newW = inputW + padLeft + padRight;
    const output = new Float32Array(newH * newW * channels);

    for (let i = 0; i < inputH; i++) {
        for (let j = 0; j < inputW; j++) {
            for (let c = 0; c < channels; c++) {
                const oldIdx = (i * inputW + j) * channels + c;
                const newIdx = ((i + padTop) * newW + (j + padLeft)) * channels + c;
                output[newIdx] = input[oldIdx];
            }
        }
    }
    return {
        data: output,
        shape: [newH, newW, channels]
    };
};


const Convolve = ( input, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW, pointer, outputTemplatePointer ) => {

    const {globalWeights, globalBiases, globalOutputTensorTemplate} = getGlobalParams();

    const output = globalOutputTensorTemplate[outputTemplatePointer];

    for (let f = 0; f < num_filters; f++) {

        const bias = globalBiases[pointer][f];

        for (let y = 0; y < outputH; y++) {
            for (let x = 0; x < outputW; x++) {

                let sum = 0;

                for (let ky = 0; ky < kernel_height; ky++) {
                    for (let kx = 0; kx < kernel_width; kx++) {
                        for (let c = 0; c < depth; c++) {

                            const inY = y * strides + ky;
                            const inX = x * strides + kx;

                            if (inY < inputH && inX < inputW) {

                                const inputIndex = ((inY * inputW + inX) * depth + c);

                                const kernelIndex = (((f * kernel_height + ky) * kernel_width + kx) * depth + c);

                                sum += input[inputIndex] * globalWeights[pointer][kernelIndex];
                            }
                        }
                    }
                }

                const outIndex = ((y * outputW + x) * num_filters + f);

                output[outIndex] = sum + bias;
            }
        }
    }

    return output;
};


const MaxPooling = (arr, pool_size, inputShape, outputShape, strides, outputTemplatePointer) => {
    const {globalOutputTensorTemplate} = getGlobalParams();
    const [poolH, poolW] = pool_size;
    const [inputH, inputW, inputD] = inputShape;
    const [outputH, outputW, outputD] = outputShape;

    const output = globalOutputTensorTemplate[outputTemplatePointer];
    const maxIdexes = new Int32Array(outputH * outputW * outputD);

    for (let d = 0; d < inputD; d++) {
        for (let i = 0; i < outputH; i++) {
            for (let j = 0; j < outputW; j++) {
                let maxVal = -Infinity;
                let maxIdx = -1;
                // Define the window boundaries based on strides
                const startH = i * strides;
                const startW = j * strides;

                for (let ph = 0; ph < poolH; ph++) {
                    for (let pw = 0; pw < poolW; pw++) {
                        const currH = startH + ph;
                        const currW = startW + pw;

                        // Check bounds to handle cases where window might exceed input dimensions
                        if (currH < inputH && currW < inputW) {
                            // Calculate index in the flattened 1D array
                            const idx = (currH * inputW * inputD) + (currW * inputD) + d;
                            const val = arr[idx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            };
                        }
                    }
                }
                // Set the max value in the output array
                const outIdx = (i * outputW * outputD) + (j * outputD) + d;
                output[outIdx] = maxVal === -Infinity ? 0 : maxVal;
                maxIdexes[outIdx] = maxIdx;
            }
        }
    }
    return {
        output: output,
        maxIndices: maxIdexes
    };
}

const element_wise_mul = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < arr1.length; i++) {
        output[i] = arr1[i] * arr2[i];
    }

    return output;
}

const scaleDiff = (arr1, arr2, arr3) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = (arr1[i] - arr2[i]) * arr3[i];
    }

    return output;
}

const element_wise_sub = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = arr1[i] - arr2[i];
    }

    return output;
}


module.exports = {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    getEmbeddings,
    MatMul,
    ApplyPadding,
    Convolve,
    MaxPooling,
    element_wise_mul,
    scaleDiff,
    element_wise_sub
}