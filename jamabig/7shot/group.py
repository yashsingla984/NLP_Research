import pandas as pd

# Load the data from the Excel file
file_path = 'gpt-4_RY.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(file_path)

# Group the data by 'MedicalField' and count the 'IsChatGptCorrect' values
# grouped = df.groupby('ModifiedMedicalField')['IsChatGptCorrect'].value_counts().unstack(fill_value=0)

# print(len(grouped))
# grouped['Percentage_Yes'] = (grouped['Yes'] / (grouped['Yes'] + grouped['No'])) * 100


#df['IsLLM_Correct'].isna('Empty', inplace=True)

# Group the data by 'ModifiedMedicalField' and count the 'IsChatGptCorrect' values
grouped = df.groupby('MedicalField')['IsLLM_Correct'].value_counts().unstack(fill_value=0)

# Calculate the percentage of 'Yes' values
grouped['Percentage_Yes'] = (grouped['Yes'] / (grouped['Yes'] + grouped['No'] )) * 100
# Display the result
print(grouped)
