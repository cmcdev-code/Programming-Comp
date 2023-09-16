import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import random
from sklearn.model_selection import RandomizedSearchCV

# Load the dataset
data = pd.read_csv('loan_train.csv')  # Replace 'loan_data.csv' with your dataset file path

data = data.dropna()

print(data.head())

# Select relevant columns for prediction
selected_columns = ["Education", 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term', 'Credit_History', 'Status']

data = data[selected_columns]

# Split the data into features and labels
X = data.drop('Status', axis=1)
y = data['Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the hyperparameter search space
param_dist = {
    'learning_rate': [random.uniform(0.01, 0.2) for _ in range(10)],
    'batch_size': [random.choice([32, 64, 128]) for _ in range(10)],
    # Add other hyperparameters to tune
}

# Build a function to create the model with specified hyperparameters
def create_model(learning_rate, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create Randomized Search object
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
random_search = RandomizedSearchCV(model, param_dist, cv=3, scoring='accuracy', n_iter=10)

# Fit the Randomized Search
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Train the final model with the best hyperparameters
best_model = create_model(**best_params)
best_model.fit(X_train, y_train, epochs=1000, batch_size=best_params['batch_size'], validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (best_model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
