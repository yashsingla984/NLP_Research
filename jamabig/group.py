import pandas as pd

# Load the data from the Excel file
file_path = 'output_chatgpt_onlyalpha_big_temp.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(file_path)

# Group the data by 'MedicalField' and count the 'IsChatGptCorrect' values
grouped = df.groupby('ModifiedMedicalField')['IsChatGptCorrect'].value_counts().unstack(fill_value=0)

grouped['Percentage_Yes'] = (grouped['Yes'] / (grouped['Yes'] + grouped['No'])) * 100
# Display the result
print(grouped)
