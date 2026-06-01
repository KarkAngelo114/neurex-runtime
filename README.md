# Neurex-Runtime

A browser compatible port of the main library [Neurex](https://github.com/KarkAngelo114/Neurex) to run models on your browsers for browser inferencing! Bring intelligence right on your browsers now! 🚀

## Installation
via CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/neurex-runtime/dist/neurex-runtime.umd.js"></script>
```

via NPM:
```bash
npm install neurex-runtime
```

## Usage
If you're working on vanilla projects, simply you can use the CDN in the script tag. The example below is how to load a model, and run an inference prediction of an XOR problem.

```html
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/neurex-runtime/dist/neurex-runtime.umd.js"></script>
        </head>

        <body>
            <input type="number" id = "num1"/>
            <input type="number" id = "num2"/>
            <button type="button" onclick="predict()">Predict</button>
            <p>Output is: <span id = "output"></span></p>
        </body>

        <script>

            let nrx;

            (async () => {
                const res = await fetch('./XOR.json'); // fetch the JSON file.
                const model = await res.json(); // parse the JSON response

                nrx = new NeurexRuntime.Runtime(); // initialzed the Runtime
                await nrx.loadSavedModel(model); // load the model
            })();

            async function predict() {
                const num1 = document.getElementById('num1').value;
                const num2 = document.getElementById('num2').value;
                const input = [parseInt(num1), parseInt(num2)];

                const pred = await nrx.predict([input]); // this function accepts matrix input [[0, 1],[1, 0],[1, 1], [0, 0]]

                document.getElementById('output').innerText = Array.from(pred[0]); // convert to JS array
            }
            
        </script>
    </html>
```

The library does not directly load an `.nrx` model because the `loadSavedModel()` accepts parsed JSON only. To use your trained model, you have to convert your `.nrx` model to JSON format by going to https://neurex-documentation.vercel.app/convert-to-json.
To get the contents of your model JSON file, you can use `fetch()` (or whatever you use for fetching) just like in the example above. After parsing, you can pass the parsed JSON data to the `loadSavedModel()` to reconstruct the model.


If you're working on JS frameworks (like ReactJS), you can directly import the `Runtime` class:

```Javascript
import {useEffect, useState, useRef} from 'react';
import { Runtime } from 'neurex-runtime';

function App() {
    const [isModelLoaded, setIsModelLoaded] = useState(false);

    let nrx = useRef(null); // we use useRef() to reference the class when the component is rendered

    useEffect(() => {
        const init = async () => {
            nrx.current = new Runtime(); // set reference of the initialize class
            const res = await fetch("/model.json"); // ensure that your model is inside the "public" folder
            const modelData = await res.json(); // parse the JSON response;

            // pass the parsed JSON data to loadSavedModel() to reconstruct the network
            await nrx.current.loadSavedModel(modelData); 
        }

        init();
    },[]);

    const handlePrediction = async () => {
        // do something... (e.g. preprocessing of your data like image pixels, preparing the data to be feed to the network, and the likes)

        const input = [0.0834, 0.4968, 0.4535]; // this can be an array of preprocessed image pixels flattened to 1D array, an input from another function (like sensor readings response) or from another external source to be used as inputs
        const pred = await nrx.current.predict([input]); // run the inference

        /*
        * Note: The outputs are float32array. You can convert it by using Array.from() if you have to. It is better to log first * what does the predict() outputs.
        */
    }

    return (
        <>
            {/* ...rest of the JSX code...*/}
        </>
    );
}

export default App;

```