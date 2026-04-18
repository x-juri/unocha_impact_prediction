import urllib.parse
import requests
import pandas as pd
import os

links = [
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202020-I%20Funding%20%26%20Vulnerability%20Analysis_0.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202020-II_CIRV%20and%20Funding.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202021-I_CIRV_0.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202022-II_CIRV.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202022-I_CIRV%20and%20Funding.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202023-II_CIRV%20May_2023.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202023-I_CIRV_0.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202024-II_CIRV%202024_Funding_July_2024.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202024-I_CIRV%20January%202024.xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202025-II_CIRV%20Data%20table%20(October%202025).xlsx",
    "https://cerf.un.org/sites/default/files/resources/CERF%20UFE%202025-I_CIRV_Data%20table%20(Dec%202024).xlsx"
]

os.makedirs('initial_data/cirv', exist_ok=True)

for link in links:
    file_name = urllib.parse.unquote(link.split('/')[-1])
    print(f"Downloading {file_name}...")
    try:
        response = requests.get(link)
        response.raise_for_status()
        xlsx_path = os.path.join('initial_data/cirv', file_name)
        with open(xlsx_path, 'wb') as f:
            f.write(response.content)
            
        csv_name = file_name.replace('.xlsx', '.csv')
        csv_path = os.path.join('initial_data/cirv', csv_name)
        
        # Read the excel file; CIRV usually has data on 'CIRV' or 'Data' sheet, but read the first one by default if not specified
        df = pd.read_excel(xlsx_path, sheet_name=0)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_name}")
        
    except Exception as e:
        print(f"Failed to process {link}. Error: {e}")
