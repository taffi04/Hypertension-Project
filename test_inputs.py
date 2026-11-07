import pandas as pd
import pickle

print("1. Loading saved model pipeline...")
with open('hypertension_rf_model.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)

print("Model loaded successfully!\n")

# 2. Define custom patient test case profiles matching your training features
# Features: ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI', 'Family_History_Binary', 'Smoking_Status_Binary']
test_cases = pd.DataFrame([
    {'Age': 69, 'Salt_Intake': 8.0,  'Stress_Score': 9,  'Sleep_Duration': 6.4, 'BMI': 25.8, 'Family_History_Binary': 1, 'Smoking_Status_Binary': 0},
    {'Age': 32, 'Salt_Intake': 11.7, 'Stress_Score': 10, 'Sleep_Duration': 5.4, 'BMI': 23.4, 'Family_History_Binary': 0, 'Smoking_Status_Binary': 1},
    {'Age': 78, 'Salt_Intake': 9.5,  'Stress_Score': 3,  'Sleep_Duration': 7.1, 'BMI': 18.7, 'Family_History_Binary': 1, 'Smoking_Status_Binary': 0},
    {'Age': 38, 'Salt_Intake': 10.0, 'Stress_Score': 10, 'Sleep_Duration': 4.2, 'BMI': 22.1, 'Family_History_Binary': 0, 'Smoking_Status_Binary': 0},
    {'Age': 41, 'Salt_Intake': 9.8,  'Stress_Score': 1,  'Sleep_Duration': 5.8, 'BMI': 16.2, 'Family_History_Binary': 1, 'Smoking_Status_Binary': 1}
])

# 3. Make predictions and calculate probabilities using the saved pipeline
predictions = best_rf_model.predict(test_cases)
probabilities = best_rf_model.predict_proba(test_cases)[:, 1]

# 4. Display formatted results
print("--- Hypertension Risk Test Results ---")
for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
    risk_label = "High Risk (1)" if pred == 1 else "Low Risk (0)"
    print(f"Patient {idx + 1}: {risk_label} | Probability of Hypertension: {prob:.2%}")