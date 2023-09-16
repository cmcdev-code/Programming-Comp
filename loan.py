import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')  # Replace 'loan_data.csv' with your dataset file path

data = data.dropna()

print(data.head())

# Select relevant columns for prediction
selected_columns = ["no_of_dependents", 'education', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',"residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value", 'loan_status']

data = data[selected_columns]

# Split the data into features and labels
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='swish', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(15, activation='swish'),
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Change activation to 'sigmoid'
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10000, batch_size=1000, validation_data=(X_test, y_test))

# Evaluate the model
y_prob = model.predict(X_test)  # This will give you probabilities between 0 and 1

model.save('best.h5')
# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
