
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases


const setGlobalParams = (weights, biases) => {
    globalWeights = weights;
    globalBiases = biases;
}

/** 
 * Use to get paramters from the global store. 
 * @returns {{globalWeights: Array<Float32Array>, globalBiases: Array<Float32Array>}}
*/
const getGlobalParams = () => {
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases,
    }
}

module.exports = {
    setGlobalParams,
    getGlobalParams
}