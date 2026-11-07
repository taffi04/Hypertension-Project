from pathlib import Path
import pandas as pd

#Now need to get the directory in which the script is located.
BASE_DIR = Path(__file__).resolve().parent

#need to build a solid path to the dataset.
csv_path = BASE_DIR / "Dataset" / "hypertension_dataset.csv"

#code loads the CSV file into a DataFrame named df.
df = pd.read_csv(csv_path)

#Now to seperate the input and target variables.
#Target variable first.
y = df["Has_Hypertension"]

#Now to exclude the target and columns that could leak the diagnosis
exclude = [
    "Has_Hypertension",
    "BP_History",
    "Medication"
]

#Must also exclude any encoded blood-pressure history and medication coloumns if the exist.
exclude += [
    col for col in df.columns
    if col.startswith("BP_")
    or col.startswith("Medication_")
]

#Time for the imput variables.
x = df.drop(
    columns=[col for col in exclude if col in df.columns]
)

#Time to print the input features after seperating.
print("Input features:")
print(x.head())

#Now I display the target variables.
print("\nTarget variable:")
print(y.head())

#I will also display the  feature coloumns.
print("\nFeature columns:")
print(x.columns.tolist())

#importing this code to help train and test the data.
from sklearn.model_selection import train_test_split
#dividing the dataset into 2 parts:
#Training data is 80%, used to teach the model.
#Testing data is 20%, used to analyse how well the model will predict the data it has not been trined on.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

#Now we display the input training and testing data.
print("Training input data:", x_train.shape)
print("Testing input data:", x_test.shape)

#Also display the target training and testing data.
print("Training target data:", y_train.shape)
print("Testing target data:", y_test.shape)