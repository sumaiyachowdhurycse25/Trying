import pandas as pd

# Input CSV file
input_file = 'domain_specific_chatbot_data.csv'  # Replace with your input file name

# Column based on which we want to separate rows
category_column = 'domain'  # Replace with your column name

# Read the CSV file
df = pd.read_csv(input_file, encoding='latin1')

# Get unique categories
categories = df[category_column].unique()

# Loop through each category and save a separate CSV
for category in categories:
    # Filter rows for the current category
    df_category = df[df[category_column] == category]
    
    # Create a filename for this category
    #output_file = f'{category}.csv'
    
    # Save to CSV
    #df_category.to_csv(output_file, index=False)

    #print(f'Saved {len(df_category)} rows to {output_file}')

import pandas as pd

# Load the CSV files
df1 = pd.read_csv('train_data_chatbot.csv', encoding='latin1')
df2 = pd.read_csv('validation_data_chatbot.csv', encoding='latin1')


merged = pd.concat([df1, df2], ignore_index=True)

# Save the result
merged.to_csv('chatbotfile.csv', index=False)

