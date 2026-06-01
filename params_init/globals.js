
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases
let globalOutputTensorTemplate = []; // global array of output templates used in feedforward only so that no need to create new Flaot32Array each time a layer function is called and to return an output during feedforward. Applies only to layers


const setGlobalParams = (weights, biases, outputTemplates) => {
    globalWeights = weights;
    globalBiases = biases;
    globalOutputTensorTemplate = outputTemplates;
}

/** 
 * Use to get paramters from the global store. 
 * @returns {Object}
*/
const getGlobalParams = () => {
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases,
        globalOutputTensorTemplate: globalOutputTensorTemplate
    }
}

module.exports = {
    setGlobalParams,
    getGlobalParams
}