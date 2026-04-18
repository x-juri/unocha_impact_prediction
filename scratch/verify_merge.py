import pandas as pd

PATH = 'initial_data/final_merged_cerf_cirv_2022_2025.csv'
df = pd.read_csv(PATH)

print(f"Total rows: {len(df)}")
print("\n--- Rows by Year ---")
print(df['year'].value_counts().sort_index())

print("\n--- Check 2025 Data ---")
df_25 = df[df['year'] == 2025]
print(f"2025 records: {len(df_25)}")
print(f"2025 records with CIRV Score: {df_25['CIRV_Score'].notna().sum()}")
print(f"2025 records with HDX Funding: {df_25['totalAmountApproved'].notna().sum()}")

print("\n--- Sample Join Result (2025) ---")
print(df_25[df_25['CIRV_Score'].notna() & df_25['totalAmountApproved'].notna()].head(10)[['countryName', 'Round', 'CIRV_Score', 'totalAmountApproved']])

# Check for duplicate projects
dupes = df['projectID'].duplicated().sum() - df['projectID'].isna().sum()
print(f"\nDuplicate Project IDs (excluding NaNs): {dupes}")

# Check outer join coverage
only_hdx = df[df['ISO3'].isna()]
only_cirv = df[df['projectID'].isna()]
print(f"\nRecords with ONLY HDX data (no CIRV match): {len(only_hdx)}")
print(f"Records with ONLY CIRV data (no HDX match): {len(only_cirv)}")
