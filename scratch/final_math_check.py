import pandas as pd
import re

df = pd.read_csv('initial_data/hdx_cerf_allocations.csv')
ufe = df[df['windowFullName'] == 'Underfunded Emergencies'].copy()

# Extract allocation year from project code
def get_alloc_year(code):
    # Match various formats: 25-UF, CERF-SDN-25-UF, 2022-UF
    match = re.search(r'(\d{2})-(UF|RR)', code)
    if match:
        return 2000 + int(match.group(1))
    match = re.search(r'-(\d{2})-(UF|RR)-', code)
    if match:
        return 2000 + int(match.group(1))
    return None

ufe['alloc_year'] = ufe['projectCode'].apply(get_alloc_year)

# Mapping from calibration plan
mapping = {
    2022: {
        'I': ['AGO', 'TCD', 'COD', 'HTI', 'HND', 'KEN', 'LBN', 'MDG', 'MMR', 'NER', 'SDN', 'SYR'],
        'II': ['DZA', 'BGD', 'CMR', 'MLI', 'MOZ', 'NGA', 'SSD', 'UGA', 'VEN', 'YEM']
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

def assign_round(row):
    ay = row['alloc_year']
    iso = row['countryCode']
    if ay in mapping:
        if iso in mapping[ay].get('I', []): return 'I'
        if iso in mapping[ay].get('II', []): return 'II'
    return 'None'

ufe['assigned_round'] = ufe.apply(assign_round, axis=1)

# Group by alloc_year and assigned_round
summary = ufe.groupby(['alloc_year', 'assigned_round'])['totalAmountApproved'].sum() / 1e6
print(summary)
