import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file and 'Sheet1' with the sheet name
file_path = 'llama-2-70b-chat_Y_5.xlsx'
# sheet_name = 'Sheet1'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Replace 'Column_Name' with the actual name of the column containing "Yes" and "No" values
column_name = 'IsLLM_Correct'
year_column = 'Year'

# Create a mask for the two separate periods
mask_2013_2018 = (df[year_column] >= 2013) & (df[year_column] <= 2018)
mask_2019_2023 = (df[year_column] >= 2019) & (df[year_column] <= 2023)

# Count "Yes" and "No" values in the specified column for each period
yes_count_2013_2018 = df.loc[mask_2013_2018, column_name].eq('Yes').sum()
no_count_2013_2018 = df.loc[mask_2013_2018, column_name].eq('No').sum()

yes_count_2019_2023 = df.loc[mask_2019_2023, column_name].eq('Yes').sum()
no_count_2019_2023 = df.loc[mask_2019_2023, column_name].eq('No').sum()

print(f'Count of "Yes" (2013-2018): {yes_count_2013_2018}')
print(f'Count of "No" (2013-2018): {no_count_2013_2018}')

print(f'Count of "Yes" (2019-2023): {yes_count_2019_2023}')
print(f'Count of "No" (2019-2023): {no_count_2019_2023}')
