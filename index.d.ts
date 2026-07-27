
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
     * the `Runtime` class can also load models and perform inference
     */
    export class Runtime {
        
        /**
        * @method get_miscellaneous_data
        * @returns {any} Saved miscellaneous data upon model saving
        */
        get_miscellaneous_data(): any;

        /**
         * @async
         * @method loadSavedModel is the method use for loading the model. But unlike on `NeurexJS` main library where you can load an `.nrx` file, this method accepts a parsed JSON. Therefore, your model must be in JSON format so that you can import the model and parsed it's contents.
         * @param modelData parsed JSON data
         * @throws {Error} if reading JSON data has a problem or model configuration 
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

    /**
     * @function buildVocab - allows you to tokenized an entire corpus into tokens of words, symbols, numbers and removing duplicated words.
     * @param {String} corpus an array of sentences or large corpus
     * @returns {Array<String>} an array of tokenized words
     */
    export function buildVocab(corpus: String): Array<String>;

    /**
     * @function buildWord2Id - this function assign unique token IDs to tokenized words. These token IDs will be use to `Encode` input tokenized words.
     * @param {Array<String>} vocab tokenized words
     * @returns {Object} an object containing key value pairs. Each key (words) has corresponding value (token ID)
     */
    export function buildWord2Id(vocab: String[]): Object;

    /**
     * @function Encode - this function tokenize a sentence and assign token IDs returning an array of token IDs.
     * @param {String} sentence input sentence or prompt
     * @param {Object} buildWord2Id_output the output after calling `buildWord2Id()` function. This key-value object will be use to encode the input sentence and assign corresponding token IDs based on the words in `buildWord2Id_output`
     * @param {Number} max_length The length of the encoded token containing token IDs.
     * @returns {Array<Number>} an array of token IDs to be use for token embeddings in the embedding layer 
     */
    export function Encode(sentence: String, buildWord2Id_output: Object, max_length: Number): Array<Number>;
}