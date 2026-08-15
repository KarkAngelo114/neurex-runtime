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

const getEmbeddings = (tokenVector, embeddingDim, lookup) => {
    const output = new Float32Array(tokenVector.length * embeddingDim);

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

const MatMul = (input, inputSize, outputSize, weights, biases) => {

    const output = new Float32Array(outputSize);

    output.set(biases);

    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];

        const rowStart = i * outputSize;
        const rowEnd = rowStart + outputSize;
        const weightRow = weights.subarray(rowStart, rowEnd);

        for (let j = 0; j < outputSize; j++) {
            output[j] += inputVal * weightRow[j];
        }
    }

    return output;
};

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

const MaxPooling = (arr, pool_size, inputShape, outputShape, strides) => {
    const {globalOutputTensorTemplate} = getGlobalParams();
    const [poolH, poolW] = pool_size;
    const [inputH, inputW, inputD] = inputShape;
    const [outputH, outputW, outputD] = outputShape;

    const output = new Float32Array(outputH * outputW * outputD);
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

const recurrentMatMul = (input, prevHiddenState,  inputWeightShape, recurrentWeightShape, weights, biases) => {
    const { globalOutputTensorTemplate } = getGlobalParams();
    // The weights were concatenated during initialization as:
    // [input_weights..., recurrent_weights...]
    const inputSize = inputWeightShape[0];
    const units = inputWeightShape[1];
    const range_input_weights = inputSize * units;
    const output = new Float32Array(units);

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

const transConv = (input, inputShape, outputShape, strides, filters, weightShape, weights, biases) => {
    
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [f, kh, kw, d] = weightShape;

    const output = new Float32Array(oH * oW * oD);

    // Sanity checks
    if (d !== iD) {
        throw new Error(`TransConv: weight input depth (${d}) != input depth (${iD})`);
    }

    if (f !== oD) {
        throw new Error(`TransConv: number of filters (${f}) != output depth (${oD})`);
    }

    if (filters !== f) {
        throw new Error(`TransConv: filters (${filters}) != weightShape[0] (${f})`);
    }

    // Clear output first (just in case)
    output.fill(0);

    const padH = Math.max(0, (iH - 1) * strides + kh - oH);
    const padW = Math.max(0,(iW - 1) * strides + kw - oW);
    const padTop = Math.floor(padH / 2);
    const padLeft = Math.floor(padW / 2);

    // Flat index helpers.
    const inputIndex = (y, x, c) => (y * iW + x) * iD + c;
    const outputIndex = (y, x, c) => (y * oW + x) * f + c;
    const weightIndex = (filter, ky, kx, c) =>(((filter * kh) + ky) * kw + kx) * d + c;

    for (let iy = 0; iy < iH; iy++) {
        for (let ix = 0; ix < iW; ix++) {

            const inputBase = (iy * iW + ix) * iD;

            for (let ky = 0; ky < kh; ky++) {

                const oy = iy * strides + ky - padTop;

                // Kernel row falls outside output.
                if (oy < 0 || oy >= oH) continue;

                for (let kx = 0; kx < kw; kx++) {

                    const ox = ix * strides + kx - padLeft;

                    // Kernel column falls outside output.
                    if (ox < 0 || ox >= oW) continue;

                    const outputBase = (oy * oW + ox) * f;

                    /*
                     * For every output filter, accumulate the
                     * input channels multiplied by the kernel.
                     */
                    for (let filter = 0; filter < f; filter++) {

                        let sum = 0;

                        const weightBase = ((filter * kh + ky) * kw + kx) * d;

                        for (let c = 0; c < d; c++) {
                            sum += input[inputBase + c] * weights[weightBase + c];
                        }

                        output[outputBase + filter] += sum;
                    }
                }
            }
        }
    }

    /*
     * Bias is added ONCE per output element, after all
     * input/kernel contributions have been accumulated.
     */
    for (let y = 0; y < oH; y++) {
        for (let x = 0; x < oW; x++) {

            const outputBase = (y * oW + x) * f;

            for (let filter = 0; filter < f; filter++) {
                output[outputBase + filter] += biases[filter];
            }
        }
    }

    return output;
};


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
    transConv
}