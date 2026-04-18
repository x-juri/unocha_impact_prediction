import pandas as pd
import os

cirv_path = 'initial_data/cirv/CIRV_DataTable_2022-I.xlsx'
df_cirv = pd.read_excel(cirv_path, header=None)
print("--- CIRV First 10 rows (raw) ---")
print(df_cirv.head(10))
