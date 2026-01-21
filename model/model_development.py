import pandas as pd
import os

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to train.csv (assumes it is in the parent folder of this script)
csv_path = os.path.join(script_dir, "..", "train.csv")

# Read the CSV
try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully!")
except FileNotFoundError:
    print(f"Error: 'train.csv' not found at {csv_path}")
    print("Please make sure the file exists in the project folder.")

# Continue with your model code...
# For example:
# X = df.drop("Survived", axis=1)
# y = df["Survived"]
