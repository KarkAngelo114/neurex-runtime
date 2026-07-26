// const Runtime = require('./core/core.js');
// const { buildVocab, buildWord2Id, Encode, tokenize } = require('./preprocessor/tokenizer.js')

// exports.Runtime = Runtime;

const Runtime = require('./core/core.js');
const { buildVocab, buildWord2Id, Encode, tokenize } = require('./preprocessor/tokenizer.js');


// under `NeurexRuntime` namespace, they can use the exported module like this:
// 
// const nrx = new NeurexRuntime.Runtime(); // this is the runtime class
// 
// const vocab = NeurexRuntime.buildVocab(extracted_text_data);
// const word2ID_output = NeurexRuntime.buildWord2Id(vocab);
// const encoded_data = NeurexRuntime.Encode("Hello World", word2ID_output, 20);
module.exports = {
  Runtime,
  buildVocab,
  buildWord2Id,
  Encode,
  tokenize
};