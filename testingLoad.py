import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

load_model=tf.keras.models.load_model("testing.h5")

#["no_of_dependents", 'education', 
# 'income_annum', 'loan_amount', 'loan_term', 
# 'cibil_score',"residential_assets_value","commercial_assets_value",
# "luxury_assets_value","bank_asset_value", 
# 'loan_status']



testData = pd.DataFrame({
    "no_of_dependents": [1, 1, 0],
    "education": [1, 1, 0],
    'income_annum': [10000, 10000, 10000],
    'loan_amount': [10000000, 100000, 10000],
    "loan_term": [360, 360, 360],
    "cibil_score": [800, 200, 700],
    "residential_assets_value": [1000000, 1000, 1000000],
    "commercial_assets_value": [10000000, 10, 100000],
    "luxury_assets_value": [1000001, 10000, 10000],
    "bank_asset_value": [10000, 10000, 1000]
})

scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(testData)
predictions = load_model.predict(new_data_scaled)
print(predictions)