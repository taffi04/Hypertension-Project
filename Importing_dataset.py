#in this part I will be importing, displaying.
#code below loads the pandas library for working with the data.
from pathlib import Path
import pandas as pd

#Now need to get the directory in which the script is located.
BASE_DIR = Path(__file__).resolve().parent

#need to build a solid path to the dataset.
csv_path = BASE_DIR / "Dataset" / "hypertension_dataset.csv"

#code loads the CSV file into a DataFrame named df.
df = pd.read_csv(csv_path)

print(df.head()) #displays the first fve rows.
print(df.shape)   #displays the number of rows and columns.
print(df.columns) #displays all the column names.