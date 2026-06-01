const resolve = require('@rollup/plugin-node-resolve');
const commonjs = require('@rollup/plugin-commonjs');
const { terser } = require('rollup-plugin-terser');

module.exports = [
    {
        input: 'index.js',
        output: {
            file: 'dist/neurex-runtime.umd.js',
            format: 'umd',
            name: 'NeurexRuntime',
            exports: 'named'
        },
        plugins: [resolve(), commonjs(), terser()]
    },

    {
        input: 'index.js',
        output: {
            file: 'dist/neurex-runtime.esm.js',
            format: 'esm',
            exports: 'named'
        },
        plugins: [resolve(), commonjs()]
    }
];