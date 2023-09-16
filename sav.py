import tensorflow as tf

# Load the HDF5 model
model = tf.keras.models.load_model('testing.h5')

# Convert to a SavedModel
tf.saved_model.save(model, "here")
