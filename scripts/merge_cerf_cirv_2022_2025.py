import pandas as pd
import os
import glob
from datetime import datetime

# Paths
HDX_PATH = 'initial_data/hdx_cerf_allocations.csv'
CIRV_DIR = 'initial_data/cirv/'
OUTPUT_PATH = 'initial_data/final_merged_cerf_cirv_2022_2025.csv'

def load_hdx():
    print("Loading HDX data...")
    # Use quotechar and on_bad_lines for robust parsing
    df = pd.read_csv(HDX_PATH, quotechar='"', on_bad_lines='warn')
    
    # Filter for 2022 onwards
    df = df[df['year'] >= 2022].copy()
    
    # Convert dates
    df['dateUSGSignature'] = pd.to_datetime(df['dateUSGSignature'])
    
    # Official Country-Round mapping from PDF reports
    UFE_MAPPING = {
        2022: {
            'I': ['AGO', 'TCD', 'COD', 'HTI', 'HND', 'KEN', 'LBN', 'MDG', 'MMR', 'NER', 'SDN', 'SYR'],
            'II': ['BFA', 'DZA', 'BGD', 'CMR', 'MLI', 'MOZ', 'NGA', 'SSD', 'UGA', 'VEN', 'YEM']
        },
        2023: {
            'I': ['TCD', 'COL', 'COD', 'ERI', 'ETH', 'HND', 'KEN', 'LBN', 'MDG', 'PAK', 'SDN', 'SYR'],
            'II': ['AFG', 'BGD', 'BFA', 'CAF', 'CMR', 'HTI', 'MWI', 'MLI', 'MOZ', 'MMR', 'UGA', 'VEN', 'YEM', 'PSE']
        },
        2024: {
            'I': ['TCD', 'COD', 'ERI', 'ETH', 'MDG', 'NER', 'SDN', 'SYR', 'SSD'],
            'II': ['BFA', 'BDI', 'CMR', 'ETH', 'HTI', 'MWI', 'MLI', 'MOZ', 'MMR', 'YEM']
        },
        2025: {
            'I': ['AFG', 'CAF', 'TCD', 'HND', 'MRT', 'NER', 'SOM', 'SDN', 'VEN', 'ZMB']
        }
    }

    import re
    def get_alloc_year(code):
        if not isinstance(code, str): return None
        match = re.search(r'(\d{2})-(UF|RR)', code)
        if match: return 2000 + int(match.group(1))
        match = re.search(r'-(\d{2})-(UF|RR)-', code)
        if match: return 2000 + int(match.group(1))
        return None

    df['alloc_year'] = df['projectCode'].apply(get_alloc_year)
    df['alloc_year'] = df['alloc_year'].fillna(df['year']).astype(int)

    def assign_round(row):
        year = row['alloc_year']
        country = row['countryCode']
        is_ufe = row['windowFullName'] == 'Underfunded Emergencies'
        
        if is_ufe:
            if year in UFE_MAPPING:
                if country in UFE_MAPPING[year].get('I', []):
                    return 'I'
                if country in UFE_MAPPING[year].get('II', []):
                    return 'II'
                return 'None' # Strict mapping for UFE
            return 'None'
        
        # Fallback for Rapid Response: date heuristic
        month = row['dateUSGSignature'].month
        if month <= 6:
            return 'I'
        else:
            return 'II'
            
    df['Round'] = df.apply(assign_round, axis=1)
    
    df['year'] = df['alloc_year']
    df.drop(columns=['alloc_year'], inplace=True)
    
    # Project year might differ from signature year
    # But usually, they align for our purposes of matching with the CIRV year.
    # We will use the 'year' column from HDX as the primary year for matching.
    
    print(f"Loaded {len(df)} HDX records.")
    return df

def load_cirv_excel(file_path, year, round_label):
    print(f"Loading CIRV Excel: {file_path}")
    # The "All Data" sheet name varies slightly, so we look for it
    xl = pd.ExcelFile(file_path)
    sheet_name = [s for s in xl.sheet_names if 'All Data' in s][0]
    
    # Header is typically on row 2 (index 2)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    
    # Column mapping based on research
    # 0: Country, 1: ISO3, 22: CIRV Index
    cols = df.columns.tolist()
    
    # Find columns by name patterns to be more robust
    iso_col = [c for c in cols if 'ISO3' in str(c) or 'ISO Code' in str(c)][0]
    cirv_col = [c for c in cols if 'CIRV - CERF Index' in str(c)][0]
    inform_col = [c for c in cols if 'I. INFORM Risk' in str(c) or 'INFORM score' in str(c)]
    gsci_col = [c for c in cols if 'Global Crisis Severity Index' in str(c) or 'GSCI' in str(c)]
    
    # Inclusion of Adjusted CIRV if available
    adj_cirv_col = [c for c in cols if 'CIRV (Adjusted)' in str(c)]

    selected_data = df[[iso_col, cirv_col] + inform_col + gsci_col + adj_cirv_col].copy()
    
    # Standardize names
    rename_map = {
        iso_col: 'ISO3',
        cirv_col: 'CIRV_Score',
    }
    if inform_col: rename_map[inform_col[0]] = 'INFORM_Risk'
    if gsci_col: rename_map[gsci_col[0]] = 'GCSI_Score'
    if adj_cirv_col: rename_map[adj_cirv_col[0]] = 'CIRV_Adjusted'
    
    selected_data.rename(columns=rename_map, inplace=True)
    selected_data['Year'] = year
    selected_data['Round'] = round_label
    
    return selected_data

def load_cirv_csv_2025_ii(file_path):
    print(f"Loading CIRV 2025-II CSV: {file_path}")
    # Load first few rows to find where headers are
    raw_df = pd.read_csv(file_path, sep=';', header=None, nrows=5)
    
    # We'll load the data starting from row 3 (data starts after the multi-line header)
    df = pd.read_csv(file_path, sep=';', header=None, skiprows=3)
    
    # Based on the inspection of Row 2 and Row 1:
    # Col 1: ISO3
    # Col 3: INFORM Score
    # Col 5: GCSI
    # Col 22: CIRV (without Inform)
    # Col 23: CIRV (Incl Inform)
    # Col 24: CIRV (Adjusted)
    
    selected_data = pd.DataFrame()
    selected_data['ISO3'] = df[1]
    selected_data['INFORM_Risk'] = df[3]
    selected_data['GCSI_Score'] = df[5]
    selected_data['CIRV_Score'] = df[22]
    selected_data['CIRV_Adjusted'] = df[24]
    
    # Clean up strings
    for col in selected_data.columns:
        if col != 'ISO3':
            if selected_data[col].dtype == 'object':
                selected_data[col] = selected_data[col].str.replace(',', '.').str.strip()
                # Handle possible non-numeric 'TEST' or similar
                selected_data[col] = pd.to_numeric(selected_data[col], errors='coerce')
        else:
            selected_data[col] = selected_data[col].str.strip()
            
    selected_data['Year'] = 2025
    selected_data['Round'] = 'II'
    
    # Drop rows without ISO3 (like 'TEST' or headers)
    selected_data = selected_data[selected_data['ISO3'].notna() & (selected_data['ISO3'] != 'TEST')].copy()
    
    return selected_data

def main():
    hdx_df = load_hdx()
    
    cirv_frames = []
    
    # 2022-2024 Rounds
    for year in range(2022, 2025):
        for rnd in ['I', 'II']:
            pattern = os.path.join(CIRV_DIR, f"CIRV_DataTable_{year}-{rnd}.xlsx")
            files = glob.glob(pattern)
            if files:
                cirv_frames.append(load_cirv_excel(files[0], year, rnd))
            else:
                print(f"Warning: Missing CIRV file for {year}-{rnd}")

    # 2025-I
    pattern_25_i = os.path.join(CIRV_DIR, "CIRV_DataTable_2025-I.xlsx")
    files_25_i = glob.glob(pattern_25_i)
    if files_25_i:
        cirv_frames.append(load_cirv_excel(files_25_i[0], 2025, 'I'))
        
    # 2025-II (CSV)
    pattern_25_ii = os.path.join(CIRV_DIR, "*2025-II*.csv")
    files_25_ii = glob.glob(pattern_25_ii)
    if files_25_ii:
        cirv_frames.append(load_cirv_csv_2025_ii(files_25_ii[0]))

    all_cirv = pd.concat(cirv_frames, ignore_index=True)
    print(f"Aggregated {len(all_cirv)} CIRV records.")

    # Outer join
    # HDX keys: countryCode, year, Round
    # CIRV keys: ISO3, Year, Round
    
    print("Merging datasets...")
    merged_df = pd.merge(
        hdx_df,
        all_cirv,
        left_on=['countryCode', 'year', 'Round'],
        right_on=['ISO3', 'Year', 'Round'],
        how='outer',
        suffixes=('', '_cirv')
    )
    
    # Fill Year if missing from join
    merged_df['year'] = merged_df['year'].fillna(merged_df['Year'])
    
    # Drop redundant Year column but keep ISO3 for verification if needed
    merged_df.drop(columns=['Year'], inplace=True)
    
    print(f"Final merged dataframe has {len(merged_df)} rows.")
    
    # Save
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
