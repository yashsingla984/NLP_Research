import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file and 'Sheet1' with the sheet name
file_path = 'gpt-4_RY.xlsx'
# sheet_name = 'Sheet1'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Replace 'Column_Name' with the actual name of the column containing "Yes" and "No" values
column_name = 'IsLLM_Correct'


# Count "Yes" and "No" values in the specified column
yes_count = df[column_name].eq('Yes').sum()
no_count = df[column_name].eq('No').sum()
empty_count = df[column_name].isna().sum()

print(f'Count of "Yes": {yes_count}')
print(f'Count of "No": {no_count}')
print(f'Count of "Empty": {empty_count}')
no_count=no_count+empty_count
print(f'Count of "Empty_and_No": {no_count}')

