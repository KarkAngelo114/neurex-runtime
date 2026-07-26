const { getGlobalParams } = require("../../../params_init");

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

const DReLu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? 1 : 0;
    }
    return output;
};

const DSigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const s = 1 / (1 + Math.exp(-output[i]));
        output[i] = s * (1 - s);
    }
    return output;
};

const DTanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const t = Math.tanh(output[i]);
        output[i] = 1 - t * t;
    }
    return output;
};

const DSoftmax = (arr) => {

    return new Float32Array(arr.length).fill(1);
};

const DLinear = (arr) => {
    const output = new Float32Array(arr.length);
    output.fill(1);
    return output;
};

const getEmbeddings = (tokenVector, embeddingDim, lookup, outputTemplatePointer) => {
    const {globalOutputTensorTemplate} = getGlobalParams();

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

const MatMul = (input, inputSize, outputSize, weights, biases, outputTemplatePointer) => {
    const {globalOutputTensorTemplate} = getGlobalParams();
    
    const z_values = globalOutputTensorTemplate[outputTemplatePointer]; // use the output template pointer to get the corresponding pre-allocated output tensor

    // 1. Initialize with Biases (Faster than adding them in a separate loop later)
    z_values.set(biases);

    // 2. Perform Weighted Sum
    // We iterate through each input neuron
    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];
        
        // Calculate the starting offset for this specific input neuron's weights
        const offset = i * outputSize;

        // Multiply the input by every weight connecting to output neurons
        for (let j = 0; j < outputSize; j++) {
            z_values[j] += inputVal * weights[offset + j];
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

const Convolve = (input, strides, outputShape, kernelShape, inputShape, weights, biases) => {

    const [numFilters, kernelH, kernelW, depth] = kernelShape;
    const [inputH, inputW] = inputShape;
    const [outputH, outputW] = outputShape;

    const output = new Float32Array(outputH * outputW * numFilters);

    const kernelSize = kernelH * kernelW * depth;

    for (let y = 0; y < outputH; y++) {

        const baseY = y * strides;

        for (let x = 0; x < outputW; x++) {

            const baseX = x * strides;

            const outBase = (y * outputW + x) * numFilters;

            for (let f = 0; f < numFilters; f++) {

                let sum = biases[f];

                const filterOffset = f * kernelSize;

                for (let ky = 0; ky < kernelH; ky++) {

                    const inY = baseY + ky;

                    if (inY >= inputH) continue;

                    for (let kx = 0; kx < kernelW; kx++) {

                        const inX = baseX + kx;

                        if (inX >= inputW) continue;

                        const inputBase = (inY * inputW + inX) * depth;

                        const kernelBase = filterOffset + (ky * kernelW + kx) * depth;

                        let c = 0;

                        for (; c <= depth - 4; c += 4) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                            sum += input[inputBase + c + 1] * weights[kernelBase + c + 1];
                            sum += input[inputBase + c + 2] * weights[kernelBase + c + 2];
                            sum += input[inputBase + c + 3] * weights[kernelBase + c + 3];
                        }

                        for (; c < depth; c++) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                        }
                    }
                }

                output[outBase + f] = sum;
            }
        }
    }

    return output;
};

const DilateInput = (input, shape, stride) => {
    const [H, W, C] = shape;
    const dilatedH = (H - 1) * stride + 1;
    const dilatedW = (W - 1) * stride + 1;
    
    const dilatedSize = dilatedH * dilatedW * C;
    const dilated = new Float32Array(dilatedSize);

    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const srcIdx = (h * W + w) * C + c;
                const dilatedHIdx = h * stride;
                const dilatedWIdx = w * stride;
                const dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
                dilated[dstIdx] = input[srcIdx];
            }
        }
    }

    return {
        data: dilated,
        dilatedHeight: dilatedH,
        dilatedWidth: dilatedW
    };
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

const recurrentMatMul = (input, prevHiddenState,  inputWeightShape, recurrentWeightShape, weights, biases, outputTemplatePointer) => {
    const { globalOutputTensorTemplate } = getGlobalParams();
    // The weights were concatenated during initialization as:
    // [input_weights..., recurrent_weights...]
    const inputSize = inputWeightShape[0];
    const units = inputWeightShape[1];
    const range_input_weights = inputSize * units;
    const output = globalOutputTensorTemplate[outputTemplatePointer];

    const input_weights = weights.subarray(0, range_input_weights);
    const recurrent_weights = weights.subarray(range_input_weights, range_input_weights + recurrentWeightShape[0] * recurrentWeightShape[1]);

    for (let j = 0; j < units; j++) {
        let z = biases[j];

        for (let i = 0; i < inputSize; i++) {
            z += input[i] * input_weights[i * units + j];
        }

        for (let h = 0; h < units; h++) {
            z += prevHiddenState[h] * recurrent_weights[h * units + j];
        }

        output[j] = z;
    }

    return output;
}


module.exports = {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    DReLu,
    DSigmoid,
    DTanh,
    DSoftmax,
    DLinear,
    getEmbeddings,
    MatMul,
    ApplyPadding,
    Convolve,
    DilateInput,
    Convolve,
    MaxPooling,
    recurrentMatMul,
}