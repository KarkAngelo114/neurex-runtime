
 /**
  * this runtime version only needs feedforward related functions
  */

// import necessary modules
const Layers = require('../layers');
const { setGlobalParams } = require('../params_init/globals.js');

class Runtime {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.num_layers = 0;
        this.input_size = 1;
        this.input_shape = [1, 1, 1];
        this.output_shape = [];
        this.currentShape = null;
        this.currentSize = null;
        this.accuracy = '';
        this.loss_function = '';
        this.output_size = 0;
        this.task = null;
        this.epoch_count = 0;
        this.batch_size = 0;
        this.depth = 0;
        this.filters = 1;
        this.layers = []; // layers (except input type layers) and their details will store here
        this.hasSequentiallyBuild = false;
        this.hasBuilt = false;

        // default configs
        this.optimizer = 'sgd';
        this.learning_rate = 0.001;

        this.isfailed = false;

        this.parametric_layers = [];
        this.miscellaneous = null;
    }

    /**
     * 
     * @param {String} model - path to your model
     */
    async loadSavedModel(modelData) {
        try {
            if (!modelData) {
                throw new Error('[ERROR]------- No JSON model provided.');
            }

            if (this.layers.length > 0) {
                console.warn(`[WARN]------- A model is already loaded in this instance. If you're going to load another model, you have to instantiate another class`);
                return;
            }

            // Assign properties
            this.miscellaneous = modelData.miscellaneous;
            this.task = modelData.task;
            this.loss_function = modelData.loss_function;
            this.epoch_count = modelData.epoch;
            this.batch_size = modelData.batch_size;
            this.optimizer = modelData.optimizer;
            this.learning_rate = modelData.learning_rate;
            this.input_size = modelData.input_size;
            this.output_size = modelData.output_size;
            this.num_layers = modelData.num_layers;
            this.weights = modelData.weights.map(w => new Float32Array(w));
            this.biases = modelData.biases.map(b => new Float32Array(b));
            this.input_shape = modelData.input_shape;
            const layerBuilder = new Layers();
            this.layers = modelData.layers.map(layerData => {
                let newLayer;
                if (layerData.layer_name === "connected_layer") {
                    // Recreate the connected layer with the correct activation and size
                    newLayer = layerBuilder.connectedLayer(layerData.layer_size, layerData.activation_function_name);
                    newLayer.weightShape = layerData.weightShape;
                    this.parametric_layers.push(layerData.layer_name);
                } else if (layerData.layer_name === "input_layer") {
                    // Recreate the input layer. Note: The input layer doesn't have methods, so this is just for consistency
                    newLayer = layerBuilder.inputShape({ features: layerData.layer_size });
                } else if (layerData.layer_name === "convolutionalLayer") {
                    // recreate Convolutional layer
                    newLayer = layerBuilder.convolutionalLayer(layerData.filters, layerData.strides, layerData.kernel_size, layerData.activation_function_name, layerData.padding);
                    newLayer.weightShape = layerData.weightShape;
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    const [H, W, D] = layerData.outputShape;
                    const totalSize = H * W * D;
                    this.parametric_layers.push(layerData.layer_name);
                } else if (layerData.layer_name === "maxPooling") {
                    newLayer = layerBuilder.maxPooling(layerData.poolSize, layerData.strides, layerData.padding);
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    const [H, W, D] = layerData.outputShape;
                    const totalSize = H * W * D;
                }
                else if (layerData.layer_name === "EmbeddingLayer") {
                    const vocabSize = layerData.vocabSize;
                    const embeddingDim = layerData.embeddingDim;
                    const sequence_length = layerData.maxSequenceLength;
                    const outputSize = sequence_length * embeddingDim;
                    newLayer = layerBuilder.embeddingLayer(vocabSize, embeddingDim, sequence_length);
                    newLayer.inputShape = [];
                    newLayer.outputShape = [1, 1, outputSize];
                    newLayer.weightShape = [vocabSize, embeddingDim];
                    newLayer.outputSize = outputSize;
                    this.parametric_layers.push(layerData.layer_name);
                }
                else if (layerData.layer_name === "recurrent_cell") {
                    const units = layerData.units;
                    const return_sequence = layerData.return_sequence || false;
                    const return_state = layerData.return_state || false;
                    newLayer = layerBuilder.recurrentCell(units, layerData.activation_function_name, return_sequence, return_state);
                    newLayer.weightShape = layerData.weightShape;
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    newLayer.maxSequenceLength = layerData.maxSequenceLength || 1;
                    this.parametric_layers.push(layerData.layer_name);
                }
                else {
                    throw new Error(`[ERROR] Unknown layer type '${layerData.layer_name}' found in model.`);
                }
                
                return newLayer;
            });

            console.log('[SUCCESS]------- Model loaded successfully');
        } catch (error) {
            console.error(error);
        }
    }

    get_task_type() {
        return this.task || "Task not specified";
    }

    /**
     * @method get_miscellaneous_data
     * @returns {Object} Saved miscellaneous data upon model saving
     */
    get_miscellaneous_data() {
        return this.miscellaneous;
    }

    /**
     *  @method predict()
        @param {Array} input - input data 
        @returns Array of predictions
        @throws Error when there's shape mismatch and no input data

     produces predictions based on the input data
    */
    async predict(input) {
        setGlobalParams(this.weights, this.biases, this.output_layers_templates);

        if (!this.weights || !this.biases || !this.output_layers_templates) {
            throw new Error("Parameters are missing");
        }

        try {
            if (!input) {
                throw new Error("\n[ERROR]-------No inputs")
            }

            for (let i = 0; i < input.length; i++) {
                if (input[i].length != (this.input_shape[0] * this.input_shape[1] * this.input_shape[2]) || input[i].length != this.input_size) {
                    this.isfailed = true;
                    console.log(`[ERROR]------- Input data must be the same shape set in the input layer\n- Use getTensorShape() or getInputSize()\n\nInput size/shape: ${input[i].length} || Expected: [${this.input_shape}] or ${this.input_size}\n`)
                    throw new Error(`Shape mismatch`);
                }

                input[i] = input[i] instanceof Float32Array ? input[i] : new Float32Array(input[i].flat(Infinity));
            }            

            let outputs = [];
            for (let sample_index = 0; sample_index < input.length; sample_index++) {
                let input_data = input[sample_index];

                const {predictions} = this.#Feedforward(input_data);
                outputs.push(predictions);
            }

            return outputs;
        }
        catch (error) {
            console.error(error);
        }
    }

    // ========= Private methods =======
    // forward propagation
    #Feedforward(input) {
        let current_input = input
        let all_layer_outputs = [input];
        let zs = [];
        
        let pointer = 0;
        for (let layer_index = 0; layer_index < this.num_layers; layer_index++) {
            const current_layer = this.layers[layer_index];

            const { outputs, z_values, incrementor_value } = current_layer.feedforward(current_input, current_layer, pointer);
            pointer+=incrementor_value;

            zs.push(z_values);
            current_input = outputs;
            all_layer_outputs.push(current_input);
        }

        return {
            predictions: current_input, 
            activations : all_layer_outputs,
            zs: zs
        };
    }
}

module.exports = Runtime;