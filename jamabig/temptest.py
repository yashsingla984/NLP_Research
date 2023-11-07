import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file and 'Sheet1' with the sheet name
file_path = 'outputjama_big1.xlsx'
# sheet_name = 'Sheet1'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)
print(len(df))