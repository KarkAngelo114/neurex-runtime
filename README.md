# Neurex-Runtime

A browser port of the main library [Neurex](https://github.com/KarkAngelo114/Neurex) to run models on your browsers for browser inferencing! Bring intelligence right on your client's browsers now!

## Installation
CDN:
```html
<script src="https://cdn.jsdelivr.net/gh/KarkAngelo114/neurex-runtime@cdfd8ae/dist/neurex-runtime.umd.js" defer></script> 
```

via NPM:
```bash
npm install git+https://github.com/KarkAngelo114/neurex-runtime.git 
```

## Usage
If you're working on vanilla projects, simply you can use the CDN in the script tag. The example below is how to load a model, and run an inference prediction of an XOR.

```html
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/gh/KarkAngelo114/neurex-runtime@cdfd8ae/dist/neurex-runtime.umd.js"></script>
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