

const calculateTensorShape = (inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding) => {
    // console.log(inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding);
    let oH, oW;
    if (padding === "same") {
        oH = Math.ceil(inputHeight / stride);
        oW = Math.ceil(inputWidth / stride);
    } else {
        oH = Math.floor((inputHeight - kernelHeight) / stride + 1);
        oW = Math.floor((inputWidth - kernelWidth) / stride + 1);
    }

    return {
        OutputHeight: oH,
        OutputWidth: oW,
        CalculatedTensorShape: oH * oW * depth
    };
};

/**
 * 
 * @param {Number} inputH - height of the input
 * @param {Number} inputW - width of the input 
 * @param {Number} kernelH - height of the kernel
 * @param {Number} kernelW - width of the kernel
 * @param {Number} stride - stride value
 * @param {String} padding - "same" or "valid"
 * @returns 
 */
const getPaddingSizes = (inputH, inputW, kernelH, kernelW, stride, padding) => {
    if (padding === "valid") {
        return { top: 0, bottom: 0, left: 0, right: 0 };
    }

    // Standard formula for total padding needed
    const outputH = Math.ceil(inputH / stride);
    const outputW = Math.ceil(inputW / stride);

    const padH = Math.max(0, (outputH - 1) * stride + kernelH - inputH);
    const padW = Math.max(0, (outputW - 1) * stride + kernelW - inputW);

    // Distribute padding to sides (asymmetric if necessary)
    return {
        top: Math.floor(padH / 2),
        bottom: padH - Math.floor(padH / 2),
        left: Math.floor(padW / 2),
        right: padW - Math.floor(padW / 2)
    };
}

/**
 * 
 * @param {Array<Float32Array>} chunks an array collection of float32 array 
 * @returns { Float32Array }
 */
const concatenateFloat32Array = (chunks) => {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Float32Array(totalLength);
    
    // 2. Copy each chunk into the target position
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    
    return result;
}

module.exports = {
    calculateTensorShape,
    getPaddingSizes,
    concatenateFloat32Array
}