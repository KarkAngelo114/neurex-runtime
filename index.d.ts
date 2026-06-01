
/**
 *
 * A ported, browser compatible library to run inference in the browser
 *
 * @module neurex-runtime
 */
declare module 'neurex-runtime' {
    /**
     * @class
     *
     * The `Runtime` class is the core of this library. Similar to `Neurex` core class in the `NeurexJS` library,
     * the `Runtime` class can load models and perform inference, but in different way.
     */
    export class Runtime {
        /**
         * @async
         * @method loadSavedModel is the method use for loading the model. But unlike on `NeurexJS` main library where you can load an `.nrx` file, this method accepts a parsed JSON. Therefore, your model must be in JSON format so that you use import the model and parsed it's contents.
         * @param modelData parsed JSON data
         * @throws {Error} if reading JSON data has a problem or in model configuration 
         */
        loadSavedModel(modelData: Object): void;

        /**
        * @async
        * @method predict produces predictions based on the input data
        * @param {Array<Array<Number>>} input input data 
        * @returns {Float32Array} Array of predictions
        * @throws Error when there's shape mismatch and no input data
        */
        predict(input: Number[][]): Float32Array;
    }
}