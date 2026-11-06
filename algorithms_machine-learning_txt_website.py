# Step 1: Import required libraries
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

# Step 2: Building a solid path to dataset.
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "Dataset" / "hypertension_dataset.csv"

# Step 3: Code to load dataset and named it df.
df = pd.read_csv(csv_path)

# Step 4:Now to separate target variable (y)
y = df["Has_Hypertension"].map({"Yes": 1, "No": 0})

# Step 5:Now to exclude the target and leakage columns
exclude = ["Has_Hypertension", "BP_History", "Medication"]
exclude += [col for col in df.columns if col.startswith("BP_") or col.startswith("Medication_")]

# Step 6:Time to define input features (X)
X = df.drop(columns=[col for col in exclude if col in df.columns])
#Keeping only the columns with numbers to prevent string conversion errors.
X = X.select_dtypes(include=['number'])

# To display input and target features after seperating.
print("Input features:\n", X.head())
print("\nTarget variable:\n", y.head())
print("\nFeature columns:\n", X.columns.tolist())

# Step 7: #dividing the dataset into 2 parts:
#Training data is 80%, used to teach the model.
#Testing data is 20%, used to analyse how well the model will predict the data it has not been trined on.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#Now we display the input training and testing data.
print("Training input data:", X_train.shape)
print("Testing input data:", X_test.shape)
print("Training target data:", y_train.shape)
print("Testing target data:", y_test.shape)

# Step 8:Now to select the suitable machine learning algorithms.
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbours" : make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42)
    )
}

# Step 9: Training and evaluate each selected machine learning algorthims.
print("\nThe machine learning algorithms selected:")
for name, model in models.items():

    #training the model. 
    model.fit(X_train, y_train)

    #This line is to predict the test data
    y_pred = model.predict(X_test)

    #This code evaluates performance. 
    print(f"\n== {name} ===")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

#Step 10 : Plotting the ROC curves for all the models.
plt.figure(figsize=(10,8))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else: 
        # SVM without prdeict_proba, use decision_function.
        y_score = model.decision_function(X_test)

    if 1 in y_test.values:
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label= f"{name} (AUC = {roc_auc:.2f})")
    else: 
        print(f"Skipping ROC curve for {name}: no positive samples in test set.")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Classifiers")
plt.legend(loc="lower right")
plt.show()

#Step 11: To show feature importance hich is Random Forest.
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x = importances[indices], y = X.columns[indices])
plt.title("Feature Importnace (Random Forest)")
plt.show()

#Step 12: Cross-Validation (k-fold)
print("\nCross-Validation Results:")
for name, model in models.items():
    scores = cross_val_score(models["Random Forest"], X, y, cv=5, scoring="accuracy")
    print(f"{name} : Mean Accuracy = {scores.mean():.3f}, Scores = {scores}")

#Step 13: Hyperparameter adjustment using GridSearchCV for Random Forest.
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv = 5,
    scoring = "accuracy"
)

grid_search.fit(X_train, y_train)
print("\n Best parameters for Random Forest:", grid_search.best_params_)
print("\n Best cross-validation accuracy:", grid_search.best_score_)
