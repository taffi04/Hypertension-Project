import pandas as pd
df = pd.read_csv("./Dataset/hypertension_dataset.csv")

df["Family_History_Binary"] = df["Family_History"].map({"No": 0, "Yes":1})
df["Smoking_Status_Binary"] = df["Smoking_Status"].map({"Non-Smoker": 0, "Smoker":1})
df["Has_Hypertension_Binary"] = df["Has_Hypertension"].map({"No": 0, "Yes":1})
df["Exercise_Level_Code"] = df["Exercise_Level"].map({"Low":0, "Moderate": 1, "High":2})

df["Medication_Filled"] = df["Medication"].fillna("None")

bp_dummies = pd.get_dummies(df["BP_History"], prefix="BP", dtype=int)
med_dummies = pd.get_dummies(df["Medication_Filled"], prefix="Medication", dtype=int)

df_prepared = pd.concat([df, bp_dummies, med_dummies], axis=1)

df_numeric_only = df_prepared.select_dtypes(include=["number"])

df_prepared = df_prepared.drop(columns=["Medication_Filled"])

df_numeric_only.to_csv("./Dataset/hypertension_dataset_numeric.csv", index=False)