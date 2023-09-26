import os
import sqlite3

import pandas as pd

# Open connection
connector = sqlite3.connect("processed_credit_risk.db")

# Get cursor to run SQL queries
cursor = connector.cursor()

# Read dataset from csv
file_path = os.path.abspath('.')
df = pd.read_csv(file_path + '/data/processed_credit_risk_data.csv')

# Insert data into tables
df.to_sql(
    name='processed_credit_risk',
    con=connector,
    if_exists='replace',
    index=True,
    index_label='id',
    dtype={'id': 'INTEGER PRIMARY KEY'}
    )

# Commit data
connector.commit()

# Close connection
connector.close()
