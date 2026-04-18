import pandas as pd
import os
import csv

hdx_path = 'initial_data/hdx_cerf_allocations.csv'

try:
    print("--- Reading HDX with multi-line support ---")
    df_hdx = pd.read_csv(hdx_path, quotechar='"', skipinitialspace=True)
    print("Success!")
    print(df_hdx.columns)
    print(df_hdx.head())
except Exception as e:
    print(f"Failed with standard read: {e}")
    # Try reading with different settings
    try:
        df_hdx = pd.read_csv(hdx_path, on_bad_lines='skip')
        print("Success skip bad lines!")
        print(f"Rows: {len(df_hdx)}")
    except Exception as e2:
        print(f"Failed again: {e2}")

# Let's inspect CIRV 2022-I columns specifically
cirv_path = 'initial_data/cirv/CIRV_DataTable_2022-I.xlsx'
if os.path.exists(cirv_path):
    df_cirv = pd.read_excel(cirv_path)
    print("\n--- CIRV Columns ---")
    print(df_cirv.columns.tolist())
    print("\n--- CIRV Sample ---")
    print(df_cirv.head(2))
