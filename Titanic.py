import pandas as pd

# Load the uploaded Titanic dataset
file_path = 'titanic.csv'
titanic_df = pd.read_csv(file_path)

# Display basic info and first few rows for initial exploration
titanic_df.info(), titanic_df.head()
