import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("1. Loading dataset...")
df = pd.read_csv('hypertension_data.csv')

# Use your dataset's exact feature columns and correct target column ('Has_Hypertension_Binary')
X = df[['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI', 'Family_History_Binary', 'Smoking_Status_Binary']]
y = df['Has_Hypertension_Binary']

# Split data into training (80%) and testing (20%) sets with stratification for balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build a Pipeline with a Scaler and Random Forest Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid to improve model performance
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

print("2. Tuning and training the Random Forest model (GridSearchCV)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extract the best tuned model from grid search
best_rf_model = grid_search.best_estimator_
print(f"Best Hyperparameters Found: {grid_search.best_params_}\n")

# Evaluate the improved model on the test split
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Training & Tuning Complete! Test Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model pipeline using pickle
model_filename = 'hypertension_rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_rf_model, file)

print(f"Model successfully saved to {model_filename}!")