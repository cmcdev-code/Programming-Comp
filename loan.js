<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
// Load the pre-trained model (HDF5 format)
const model = await tf.loadLayersModel('testing.h5');


// Sample input data for the selected columns
const inputDataArray = [
  2,             // no_of_dependents
  0,    // education
  60000,         // income_annum
  300000,        // loan_amount
  24,            // loan_term
  750,           // cibil_score
  200000,        // residential_assets_value
  50000,         // commercial_assets_value
  10000,         // luxury_assets_value
  300000,        // bank_asset_value
];

// Convert the array to a TensorFlow.js Tensor
const inputDataTensor = tf.tensor(inputDataArray);

// Ensure that the shape of the input data matches the model's input shape
// You may need to reshape or preprocess the data accordingly
const reshapedInputData = inputDataTensor.reshape([1, inputDataArray.length]);

// Make a prediction using the model
const predictions = model.predict(reshapedInputData);
predictions.print();
