import pandas as pd

PATH = 'initial_data/final_merged_cerf_cirv_2022_2025.csv'
df = pd.read_csv(PATH)

# We want only Underfunded Emergencies for the comparison with PDFs
ufe_df = df[df['windowFullName'] == 'Underfunded Emergencies'].copy()

# Fix year as int
ufe_df['year'] = ufe_df['year'].astype(int)

# Aggregate
agg = ufe_df.groupby(['year', 'Round'])['totalAmountApproved'].sum().reset_index()
agg['total_million'] = (agg['totalAmountApproved'] / 1_000_000).round(2)

print("--- Aggregated UFE Totals from CSV ---")
print(agg[['year', 'Round', 'total_million']])

# PDF Values (Manually extracted/verified from script output)
pdf_values = {
    (2022, 'I'): 150.0,
    (2022, 'II'): 100.0,
    (2023, 'I'): 146.0,
    (2023, 'II'): 125.0,
    (2024, 'II'): 100.0,
    (2025, 'I'): 100.0
}

print("\n--- Comparison ---")
for (y, r), pdf_val in pdf_values.items():
    csv_row = agg[(agg['year'] == y) & (agg['Round'] == r)]
    if not csv_row.empty:
        csv_val = csv_row.iloc[0]['total_million']
        diff = csv_val - pdf_val
        status = "MATCH" if abs(diff) < 0.1 else f"MISMATCH (Diff: {diff:.2f}M)"
        print(f"{y}-{r}: PDF={pdf_val}M, CSV={csv_val}M -> {status}")
    else:
        print(f"{y}-{r}: PDF={pdf_val}M, CSV=MISSING")

# Check 2024-I presence in CSV
c24_i = agg[(agg['year'] == 2024) & (agg['Round'] == 'I')]
if not c24_i.empty:
    print(f"\n2024-I (Found in CSV but no PDF): {c24_i.iloc[0]['total_million']}M")
else:
    print("\n2024-I: Missing from CSV as well.")
