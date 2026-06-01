/**
 * Neurex follows a Plugin-style architecture where in modifications on the core engine (the core file) are minimal and the logic are exposed by these methods of the Layers class.
 * This allows the library to be extensible, flexible, and clean separation of concern without touching the core engine
 * Read here about Plugin-style architecture: https://medium.com/omarelgabrys-blog/plug-in-architecture-dec207291800
 */


const {
    getEmbeddings,
    MatMul, 
    applyPadding,
    Convolve,
    element_wise_mul,
    element_wise_sub,
    MaxPool,
    scaleDiff
} = require('../core/bindings/entry.js');

const {calculateTensorShape, getPaddingSizes} = require('../utils/utils.js');
const activation = require('../core/bindings/entry.js');


class Layers {

    /**
     * @method inputShape
     * @param {object} shapeConfig - specify the number of features
     * @returns {Object}
     * @example
     * model.sequentialBuild([
        layer.inputShape({features: 4}),
        layer.connectedLayer("relu", 5),
        layer.connectedLayer("softmax", 3);
     ]);

     the inputShape() method allows you to get the shape of your input.
     */
    inputShape(shapeConfig) {
        try {
            if (shapeConfig.features) {
                const features = shapeConfig.features;
                this.input_shape = null;
                return {
                    layer_name: "input_layer",
                    layer_size: features,
                    input_shape: null
                };
            } else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
                const { height, width, depth } = shapeConfig;

                return {
                    layer_name: "input_layer",
                    layer_size: height * width * depth,
                    input_shape: [height, width, depth]
                };
            } else {
                throw new Error(`[ERROR]------- Invalid input shape config`);
            }
        } catch (error) {
            console.error(error.message);
        }
    }

    /**
    * Creates an embedding layer for token encoding.
    *
    * @param {Number} vocabSize - The size of the vocabulary.
    * @param {Number} embeddingDim - The size of the dense vector used to represent each token.
    * @param {Number} maxSequenceLength - The length of the encoded token containing token IDs.
    * @returns {Object} - The embedding layer object configuration
    */
    embeddingLayer(vocabSize, embeddingDim, maxSequenceLength) {
        if (vocabSize <= 0 || embeddingDim <= 0 || maxSequenceLength <= 0) throw new Error(`VocabSize or embeddingDim should not be a negative number or 0. vocabSize: ${vocabSize} | embeddingDim: ${embeddingDim} | maxSequenceLength: ${maxSequenceLength}`);

        return {
            layer_name:"EmbeddingLayer",
            vocabSize: vocabSize,
            embeddingDim: embeddingDim,
            maxSequenceLength: maxSequenceLength,
            feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                const embeddingDim = current_layer.embeddingDim;

                const output = getEmbeddings(input, embeddingDim, pointer, outputTemplatePointer);
                return {
                    outputs: output, 
                    z_values: output,
                    incrementor_value: 1
                };
            }
        }
    }

    /**
     * @method connectedLayer
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     * @returns {Object}
     *
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     */
    connectedLayer(activation_function = 'relu', layer_size = 5) {
        try {

            if (!activation_function || !layer_size || layer_size <= 0) {
                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
            }

            let function_name = activation_function.toLowerCase();

            if (!activation[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"connected_layer", 
                "activation_function":activation[function_name], 
                "derivative_activation_function":activation.derivatives[function_name],
                "layer_size":layer_size,
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {

                    const [inputSize, outputSize] = current_layer.weightShape;
                    const z_values = MatMul(input, inputSize, outputSize, pointer, outputTemplatePointer);
                    const activation_function = activation[function_name];

                    let outputs = activation_function(z_values);
                    
                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
                },
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 
     * @method convolutionalLayer
     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function - the activation function to be use for this layer
     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
     * @throws {Error} - if any of the parameters are invalid.
     * @returns {Object}
     *
     * Allows you to add convolutional layers in your model architecture in sequential building.
     */
    convolutionalLayer(filters = 1, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 'same') {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);

            // check if the padding is same/valid, otherwise throw error
            let paddings = ["same", "valid"];
            if (!paddings.includes(padding.toLowerCase())) {
                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
            }

            // check if the activation function is valid
            const function_name = activation_function.toLowerCase();

            if (!activation[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"convolutionalLayer",
                "activation_function":activation[function_name],
                "derivative_activation_function":activation.derivatives[function_name],
                "kernel_size":kernel_size,
                "filters":filters,
                "padding":padding.toLowerCase(),
                "strides":strides,
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                    
                    let [f, kh, kw, kd] = current_layer.weightShape;
                    let [input_H, input_W, input_D] = current_layer.inputShape; 
                    let padding = current_layer.padding;
                    let strides = current_layer.strides;

                    // 1. compute expected output tensor shape
                    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, current_layer.strides, current_layer.padding);

                    // 2. get padding sizes for each sides
                    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

                    // 3. apply padding
                    const {data, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

                    // 4. Perform the convolve operation using the shapes calculated in step 1
                    const convolve_result = Convolve(data,current_layer.strides, OutputHeight, OutputWidth, f, kh, kw, kd, shape[0], shape[1], pointer, outputTemplatePointer);

                    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

                    // 5. activate each depth input using the given activation function
                    const activation_function = activation[function_name];

                    const outputs = activation_function(convolve_result);

                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

                    return {
                        outputs: outputs,
                        z_values: convolve_result,
                        incrementor_value: 1
                    };
                }
            }
        }
        catch (error) {
            console.error(error);
        }
    }

    /**
     * @method maxPooling
     * @param {Array<Number>} poolSize - determines the pool size window 
     * @param {Number} strides - It determines how much the pool window slides across the input tensor.
     * @param {String} padding - `same` or `valid`
     * @throws {Error} - if any of the values are 0s or negative for the pool size and strides or the padding is invalid
     *
     * `maxPooling` is use for downsampling operation that reduces the spatial dimensions of an input tensor by taking the maximum value over a defined sliding window
     */
    maxPooling(poolSize, strides = 1, padding = "same") {
        try {
            if (poolSize[0] <= 0 || poolSize[1] <= 0) {
                throw new Error(`[ERROR]------- pool size value cannot be 0 or a negative value`);
            }

            // check if the padding is same/valid, otherwise throw error
            let paddings = ["same", "valid"];
            if (!paddings.includes(padding.toLowerCase())) {
                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
            }

            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);

            return {
                "layer_name":"maxPooling",
                "poolSize": poolSize,
                "padding": padding,
                "strides":strides,
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                    const [inputh, inputw, inputd] = current_layer.inputShape;
                    const [outputh, outputw, outputd] = current_layer.outputShape;
                    const [poolHeight, poolWidth] = current_layer.poolSize;
                    const strides = current_layer.strides;
                
                    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides, outputTemplatePointer);

                    current_layer.maxIndices = maxIndices;

                    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

                    return {
                        outputs:output,
                        z_values: output,
                        incrementor_value:0
                    }
                },
            }
        }
        catch (error) {
            console.error(error);
        }
    }
}

module.exports = Layers;