# neurex-runtime

A NodeJS library that allows you to run [neurex](https://www.npmjs.com/package/neurex) models for your NodeJS applications. 

## Installation
```bash
npm install neurex-runtime
```

## Usage
Import `Runtime` class and create an instance of it.

```Javascript

const { Runtime } = require('neurex-runtime');

const rn = new Runtime();

```

Load save models using the `loadSavedModel()` method:

```Javascript

const { Runtime } = require('neurex-runtime');

const rn = new Runtime();
rn.loadSavedModel('model.nrx');

```

To view your model, you use the `modelSummary()` method:

```Javascript

const { Runtime } = require('neurex-runtime');

const rn = new Runtime();
rn.loadSavedModel('model.nrx');
rn.modelSummary();


```

And run inference predictions using `predict()` method:

```Javascript

const { Runtime } = require('neurex-runtime');

const rn = new Runtime();
rn.loadSavedModel('model.nrx');
// rn.modelSummary();

const predictions = rn.predict(input);
console.log(predictions);
```

`neurex-runtime` also comes with built-in tools for loading dataset from CSV, normalizing inputs, labeling and evaluating model performance which is also present on the [neurex](https://www.npmjs.com/package/neurex) library.

Using `CsvDataHandler`:

import the class and create an instance of it:

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();

```

Load your CSV dataset `read.csv()`

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');

```

To view your loaded dataset, use `tabularize()`. This is very useful especially if you want to keep track what is the current structure of your dataset along the process.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
csv.tabularize();

```

If your working with numerical data, consider using `rowsToInt()` method. This is because when you loaded your CSV dataset, all cell elements on all rows are strings. You can view the changes by logging the first result of the process and after converting all rows to Numbers. Ensure that all rows contains elements that can be converted to numerical data, othewise, elements that cannot be converted to int (like words) will be represented as NaN.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
const formatted_dataset = csv.rowsToInt(dataset);

```

If you're going to feed your dataset to your model, consider normalizing your dataset first. Your dataset must not contain NaNs and all rows are already formatted to numbers. You can do this using `normalize()` method. As for now, the only available normalization method is `MinMax`, which normalize all values between 0 to 1.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
const formatted_dataset = csv.rowsToInt(dataset);
const normalized_dataset = csv.normalize('MinMax', formatted_dataset);

```

If you want to extract data under the column to be use as your labels or trainY, use `extractColumn()`. Note that this will alter the structure of your dataset.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
const formatted_dataset = csv.rowsToInt(dataset);
const normalized_dataset = csv.normalize('MinMax', formatted_dataset);
const labels = csv.extractColumn('Column_name', normalized_dataset);

```

You may have unwanted columns in your current dataset (or columns that contains NaN values) that cannot be used as features, considering dropping them using `removeColumns()` method. When removing columns, this will alter the structure of your dataset.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
const formatted_dataset = csv.rowsToInt(dataset);
const normalized_dataset = csv.normalize('MinMax', formatted_dataset);
const labels = csv.extractColumn('Column_name', normalized_dataset);
const train_dataset = csv.removeColumns(["column_name1","column_name2", /* other column names */], normalized_dataset);

```

If you need to get only specific number of features from a row, you can use `getRowElements()`. It will get the elements from the first element in the row up to the specified the range. Note that this will alter the structure of your dataset.

```Javascript

const { CsvDataHandler } = require('neurex-runtime');

const csv = new CsvDataHandler();
const dataset = csv.read.csv('awesome-dataset.csv');
const formatted_dataset = csv.rowsToInt(dataset);
const normalized_dataset = csv.normalize('MinMax', formatted_dataset);
const labels = csv.extractColumn('Column_name', normalized_dataset);
const train_dataset = csv.removeColumns(["column_name1","column_name2", /* other column names */], normalized_dataset);
const new_train_dataset = csv.getRowElements(5, train_dataset); // example: from 7 features, now returns 5 features only. This will get the values under the first column up to the N column.

```

Using `MinMaxScaler`:

The use of `MinMaxScaler` is to normalize and denormalize you data. It is being used by CsvDataHandler internally in the `normalize()` method. You can still do manual normalization on your dataset using the `transform()` and to denormalized, you would need to use `inverseTransform()`.

```Javascript

const { MinMaxScaler } = require('neurex-runtime');

const scaler = new MinMaxScaler();

```

To normalize and denormalize, you must use `fit()` first. Here, you can pass your X_train, Y_train, X_test or Y_test to get their minimum and maximum values. Then you can use `transform()` and `inverseTransform()`. The code below is how to use the MinMaxScaler.

```Javascript

const { MinMaxScaler } = require('neurex-runtime');

const scaler = new MinMaxScaler();


scaler.fit(X_train);
const normalized_Xtrain = scaler.transform(X_train);

/** though, not all cases you would need to normalize
* Y_train. 
* example: doing binary or multiclass-classification task
*/
scaler.fit(labels);
const normalize_Ytrain = scaler.transform(labels);

/**
* Works great if you're doing regression tasks
*/
scaler.fit(Y_test);
const denormalize = scaler.inverseTransform(predictions);
```

`OneHotEncoded`, `BinaryLabeling`, and `IntegerLabeling`

```Javascript

const {OneHotEncoded, BinaryLabeling, IntegerLabeling} = require('neurex-runtime');

const sample1 = OneHotEncoded(labels); // [ [1, 0, 0], [0, 1, 0], [0, 0, 1]]
const sample2 = BinaryLabeling(labels); // [[0],[0],[1],[1]]
const sample3 = IntegerLabeling(labels); // [[0],[1],[2]]

```


`RegressionMetrics` and `ClassificationMetrics` are use for evaluating your model's performance.

```Javascript

const {RegressionMetrics, ClassificationMetrics} = require('neurex-runtime');

RegressionMetrics(pred, acts);

ClassificationMetrics(pred, acts, 'binary', ["class1", "class2", /* add more classes if you trained your model for multi-class classification*/])

```

Note that when passing the third parameter in the `ClassificationMetrics` depends what type of classification task you trained your model ( binary or multiclass) and the type of label encoding you use if you use `OneHotEncoded()` then you will need to pass 'categorical' as the third parameter, otherwise if it's `IntegerLabeling()`, then you will need to pass 'sparse_categorical' as the third parameter.












