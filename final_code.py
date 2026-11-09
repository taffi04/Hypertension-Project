# Step 1: Import required libraries
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

#Displaying all rows and columns clearly.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

#Display floating point numbers nicely.

pd.set_option(
    'display.float_format',
    '{:.4f}'.format
)
# Step 2: Building a solid path to dataset.

BASE_DIR = Path(__file__).resolve().parent

csv_path = (
    BASE_DIR
    / "Dataset"
    / "hypertension_dataset.csv"
)

# Step 3: Code to load dataset and named it df.
df = pd.read_csv(csv_path)

print(
    "\nDataset loaded successfully."
)

print(
    "\nDataset shape:",
    df.shape
)
# Step 3.1: Displaying information about the dataset.
print("\nDataset Information:")
print(df.info())

print("\nFirst five rows of dataset:")
print(df.head())

print("\nSummary statistics:")
print(
    df.describe().T.to_string()
)

# Step 3.2: Checking for missing values inside the dataset.

print("\nMissing Values:")
print(df.isnull().sum())

#If there are missing numerical values, replace them with the mean.

df = df.fillna(
    df.mean(
        numeric_only=True
    )
)
# Step 4:Now to separate target variable (y)

y = df["Has_Hypertension"].map(
    {
        "Yes": 1,
        "No": 0
    }
)

# Step 5:Now to exclude the target and leakage columns

exclude = [
    "Has_Hypertension",
    "BP_History",
    "Medication"
]

exclude += [
    col for col in df.columns
    if col.startswith("BP_")
    or col.startswith("Medication_")
]

# Step 6:Time to define input features (X)

X = df.drop(
    columns=[
        col for col in exclude
        if col in df.columns
    ]
)

#Keeping only the columns with numbers to prevent string conversion errors.

X = X.select_dtypes(
    include=["number"]
)

# To display input and target features after seperating.

print("Input features:\n", X.head())

print(
    "\nTarget variable:\n",
    y.head()
)

print(
    "\nFeature columns:\n",
    X.columns.tolist()
)
# Step 6.1: Displaying class distribution.

print("\nClass Distribution:")

class_distribution = pd.DataFrame({

    "Class": [
        "No Hypertension",
        "Hypertension"
    ],

    "Count":
        y.value_counts()
         .sort_index()
         .values
})

print(
    class_distribution
    .to_string(index=False)
)

#Plotting class distribution.

plt.figure(figsize=(6,4))

sns.countplot(x=y)

plt.title(
    "Class Distribution"
)

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=300
)

plt.close()
# Step 6.2: Scaling the feature values.
#This prevents large values from dominating smaller values.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(
    X_scaled,
    columns=X.columns
)
# Step 7: #dividing the dataset into 2 parts:
#Training data is 80%, used to teach the model.
#Testing data is 20%, used to analyse how well the model will predict the data it has not been trined on.

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

#Now we display the input training and testing data.

print(
    "Training input data:",
    X_train.shape
)

print(
    "Testing input data:",
    X_test.shape
)

print(
    "Training target data:",
    y_train.shape
)

print(
    "Testing target data:",
    y_test.shape
)
# Step 8:Now to select the suitable machine learning algorithms.

models = {

    "Logistic Regression":

        make_pipeline(

            StandardScaler(),

            LogisticRegression(
                max_iter=1000
            )
        ),

    "Decision Tree":

        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "K-Nearest Neighbours":

        make_pipeline(

            StandardScaler(),

            KNeighborsClassifier(
                n_neighbors=5
            )
        ),

    "Support Vector Machine":

        make_pipeline(

            StandardScaler(),

            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
}
#Dictionary used to store model performances.

model_performance = []

#Used later for cross-validation export.

cross_validation_results = []

# Step 9: Training and evaluate each selected machine learning algorthims.

print(
    "\nThe machine learning algorithms selected:"
)

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    model_performance.append({

        "Model": name,

        "Accuracy": accuracy
    })

    print(
        f"\n== {name} ==="
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "\nConfusion Matrix:\n",
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    report_df = pd.DataFrame(

        classification_report(

            y_test,

            y_pred,

            output_dict=True

        )

    ).transpose()

    print("\nClassification Report")

    print(
        report_df.to_string(
            float_format="{:.4f}".format
        )
    )

    report_filename = (

        name.lower()
            .replace(" ", "_")
            .replace("-", "_")

        + "_classification_report.csv"
    )

    report_df.to_csv(
        report_filename
    )

    fig_height = max(
        4,
        len(report_df) * 0.6
    )

    fig, ax = plt.subplots(

        figsize=(10, fig_height)
    )

    ax.axis("off")

    table = ax.table(

        cellText=np.round(
            report_df.values,
            4
        ),

        rowLabels=report_df.index,

        colLabels=report_df.columns,

        cellLoc="center",

        loc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(9)

    table.scale(
        1.3,
        1.8
    )

    plt.title(
        f"{name} Classification Report"
    )

    plt.savefig(

        f"{name.lower().replace(' ','_')}_classification_report.png",

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()
    # Step 9.1: Comparing all machine learning models.

comparison_df = pd.DataFrame(
    model_performance
)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nMODEL COMPARISON")

print(
    comparison_df.to_string(
        index=False
    )
)

comparison_df.to_csv(
    "model_comparison_results.csv",
    index=False
)
plt.figure(figsize=(10,8))

for name, model in models.items():

    y_score = model.predict_proba(
        X_test
    )[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.2f})"
    )

plt.plot([0,1],[0,1],"k--")

plt.legend()

plt.savefig(
    "model_roc_curves.png",
    dpi=300
)

plt.close()
#Step 12: Cross-Validation (k-fold)

print(
    "\nCross-Validation Results:"
)

for name, model in models.items():

    scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="accuracy"
    )

    cross_validation_results.append({

        "Model": name,

        "Mean Accuracy":
            scores.mean(),

        "Standard Deviation":
            scores.std()
    })

    print(
        f"{name} : "
        f"{scores.mean():.4f}"
    )

cv_df = pd.DataFrame(
    cross_validation_results
)

cv_df.to_csv(
    "cross_validation_results.csv",
    index=False
)
#Step 13: Hyperparameter tuning with GridSearchCV

param_grid = {

    "n_estimators":
        [50,100,200],

    "max_depth":
        [None,5,10],

    "min_samples_split":
        [2,5]
}

grid_search = GridSearchCV(

    RandomForestClassifier(
        random_state=42
    ),

    param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = (
    grid_search.best_estimator_
)

print(
    grid_search.best_params_
)

# Step 13.1: Confusion Matrix for Best Model

y_pred_best = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    y_pred_best
)

print(
    "\nBEST MODEL CONFUSION MATRIX"
)

print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Hypertension",
        "Actual Hypertension"
    ],
    columns=[
        "Predicted No Hypertension",
        "Predicted Hypertension"
    ]
)

print(
    "\nFormatted Confusion Matrix"
)

print(
    cm_df.to_string()
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Hypertension",
        "Hypertension"
    ],
    yticklabels=[
        "No Hypertension",
        "Hypertension"
    ]
)

plt.title(
    "Confusion Matrix - Best Random Forest Model"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    "best_model_confusion_matrix.png",
    dpi=300
)

plt.close()

importances = (
    best_model.feature_importances_
)

feature_df = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        importances
})

feature_df = feature_df.sort_values(

    by="Importance",

    ascending=False
)

feature_df.to_csv(
    "feature_importance_results.csv",
    index=False
)

# Step 14: Saving the optimised model.

model_package = {

    "model": best_model,

    "scaler": scaler,

    "features": list(
        X.columns
    )
}

with open(

    "hypertension_rf_optimized.pkl",

    "wb"

) as file:

    pickle.dump(
        model_package,
        file
    )

print(
    "\nPROJECT COMPLETED SUCCESSFULLY"
)


# STEP 15: Testing the saved code with sample patients.


print("\nLoading the optimised Random Forest model...")

with open(
    "hypertension_rf_optimized.pkl",
    "rb"
) as file:
    model_package = pickle.load(file)

# Extract saved components
trained_model = model_package["model"]
saved_scaler = model_package["scaler"]
feature_names = model_package["features"]

print("Model loaded successfully!")


# Sample patient profiles.

test_cases = pd.DataFrame([
    {
        "Age": 65,
        "Salt_Intake": 9.2,
        "Stress_Score": 8,
        "Sleep_Duration": 6.0,
        "BMI": 28.5,
        "Family_History_Binary": 1,
        "Smoking_Status_Binary": 1,
        "Exercise_Level_Code": 1
    },
    {
        "Age": 30,
        "Salt_Intake": 5.0,
        "Stress_Score": 3,
        "Sleep_Duration": 7.5,
        "BMI": 22.0,
        "Family_History_Binary": 0,
        "Smoking_Status_Binary": 0,
        "Exercise_Level_Code": 3
    },
    {
        "Age": 58,
        "Salt_Intake": 10.5,
        "Stress_Score": 9,
        "Sleep_Duration": 5.2,
        "BMI": 31.0,
        "Family_History_Binary": 1,
        "Smoking_Status_Binary": 0,
        "Exercise_Level_Code": 0
    },
    {
        "Age": 42,
        "Salt_Intake": 6.8,
        "Stress_Score": 4,
        "Sleep_Duration": 7.0,
        "BMI": 23.5,
        "Family_History_Binary": 0,
        "Smoking_Status_Binary": 0,
        "Exercise_Level_Code": 2
    },
    {
        "Age": 72,
        "Salt_Intake": 11.0,
        "Stress_Score": 10,
        "Sleep_Duration": 4.8,
        "BMI": 29.2,
        "Family_History_Binary": 1,
        "Smoking_Status_Binary": 1,
        "Exercise_Level_Code": 0
    }
])

# Ensure correct feature order.

test_cases = test_cases.reindex(
    columns=feature_names,
    fill_value=0
)

# Scale test Data

test_cases_scaled = saved_scaler.transform(
    test_cases
)

# Making predictions

predictions = trained_model.predict(
    test_cases_scaled
)

probabilities = trained_model.predict_proba(
    test_cases_scaled
)[:, 1]

# Displaying results

results_data = []

print("HYPERTENSION RISK PREDICTIONS")

for idx, (pred, prob) in enumerate(
        zip(predictions, probabilities),
        start=1):

    status = (
        "High Risk (Hypertension)"
        if pred == 1
        else "Low Risk (Normal)"
    )

    probability = f"{prob * 100:.2f}%"

    print(
        f"Patient {idx}: "
        f"{status} | "
        f"Probability = {probability}"
    )

    results_data.append([
        f"Patient {idx}",
        status,
        probability
    ])

# Saving table results.

results_df = pd.DataFrame(
    results_data,
    columns=[
        "Patient Profile",
        "Prediction Status",
        "Risk Probability"
    ]
)

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.axis("off")
ax.axis("tight")

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

plt.title(
    "Hypertension Risk Predictions",
    fontsize=14,
    weight="bold",
    pad=20
)

plt.savefig(
    "patient_predictions_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPrediction results saved as "
    "'patient_predictions_result.png'"
)

print(
    "\nFULL PROJECT EXECUTED SUCCESSFULLY"
)