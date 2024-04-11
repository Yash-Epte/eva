import pdfplumber
import pandas as pd

# Load PDF
with pdfplumber.open("iat.pdf") as pdf:
    # Extract all tables from the single page
    page = pdf.pages[0]
    tables = page.extract_tables()

    # Convert the extracted tables to a DataFrame
    data = []
    for table in tables:
        df = pd.DataFrame(table[1:])
        df = df.drop(df.index[0])
        #df = df.drop(df.index[1])  # Exclude the header row
        data.append(df)

    # Concatenate the DataFrames
    pdfconv = pd.concat(data, ignore_index=True)

    # Rename the headers
    new_column_names = ['Seat Number','Student Name','Question 1A','Question 1B','Question 1C','Question 2A', 'Question 2B','Question 3A','Question 3B','Total'] + list(pdfconv.columns[10:])
    pdfconv.columns = new_column_names

# Save the DataFrame to a CSV file
pdfconv.to_csv("iat.csv", index=False, header=True)