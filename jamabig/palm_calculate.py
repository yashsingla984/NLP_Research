import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file and 'Sheet1' with the sheet name
file_path = 'output_palm_onlyalpha_big_temp2.xlsx'
# sheet_name = 'Sheet1'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Replace 'Column_Name' with the actual name of the column containing "Yes" and "No" values
column_name = 'IsPalmCorrect'

# Count "Yes" and "No" values in the specified column
yes_count = df[column_name].eq('Yes').sum()
no_count = df[column_name].eq('No').sum()

print(f'Count of "Yes": {yes_count}')
print(f'Count of "No": {no_count}')
