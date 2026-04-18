import pandas as pd
import os

cirv_path = 'initial_data/cirv/CIRV_DataTable_2022-I.xlsx'
df_cirv = pd.read_excel(cirv_path, header=6)
print("--- CIRV Columns (from header index 6) ---")
print(df_cirv.columns.tolist())
print("\n--- CIRV Sample ---")
print(df_cirv.head(3))
