import pandas as pd
import os

hdx_path = 'initial_data/hdx_cerf_allocations.csv'
cirv_dir = 'initial_data/cirv'

# 1. Inspect HDX
print("--- HDX Data ---")
df_hdx = pd.read_csv(hdx_path)
df_hdx_recent = df_hdx[df_hdx['year'] >= 2022]
print(f"Total rows after 2022: {len(df_hdx_recent)}")
print(df_hdx_recent[['year', 'dateUSGSignature', 'windowFullName', 'countryCode']].head(10))
print("\nUnique windowFullName values:")
print(df_hdx_recent['windowFullName'].unique())

# 2. Inspect CIRV
print("\n--- CIRV Data (2022-I) ---")
cirv_22_i = pd.read_excel(os.path.join(cirv_dir, 'CIRV_DataTable_2022-I.xlsx'))
print(f"Columns in 2022-I: {cirv_22_i.columns.tolist()}")
# Look for something like iso3 or country code
potential_iso3 = [c for c in cirv_22_i.columns if 'iso' in c.lower() or 'code' in c.lower()]
print(f"Potential ISO3 columns: {potential_iso3}")
print(cirv_22_i.head(5))

print("\n--- CIRV Data (2022-II) ---")
cirv_22_ii = pd.read_excel(os.path.join(cirv_dir, 'CIRV_DataTable_2022-II.xlsx'))
print(f"Columns in 2022-II: {cirv_22_ii.columns.tolist()}")

# 3. Check 2025
print("\n--- CIRV Data (2025-I) ---")
cirv_25_i = pd.read_excel(os.path.join(cirv_dir, 'CIRV_DataTable_2025-I.xlsx'))
print(f"Columns in 2025-I: {cirv_25_i.columns.tolist()}")

# 4. Check that weird CSV for 2025-II
print("\n--- CIRV Data (2025-II CSV) ---")
cirv_25_ii = pd.read_csv(os.path.join(cirv_dir, 'CERF UFE 2025-II_CIRV Data table (October 2025)(CERF UFE 2025-II All Data).csv'))
print(f"Columns in 2025-II: {cirv_25_ii.columns.tolist()}")
