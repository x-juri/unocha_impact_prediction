import pandas as pd
import os

data_dir = 'initial_data/cirv'
for f in os.listdir(data_dir):
    if f.endswith('.xlsx'):
        xlsx_path = os.path.join(data_dir, f)
        csv_path = os.path.join(data_dir, f.replace('.xlsx', '.csv'))
        
        xl = pd.ExcelFile(xlsx_path)
        all_data_sheet = None
        for sheet_name in xl.sheet_names:
            if 'All Data' in sheet_name:
                all_data_sheet = sheet_name
                break
        
        if all_data_sheet:
            print(f"Extracting {all_data_sheet} from {f}...")
            df = pd.read_excel(xlsx_path, sheet_name=all_data_sheet, header=None) # Read without header to find the real header
            
            # Find the actual header row. The header usually contains 'Country' or 'ISO3'
            header_row = -1
            for idx, row in df.iterrows():
                row_str = ' '.join([str(x).lower() for x in row.values])
                # Often "iso" or "country" is in the header
                if 'iso' in row_str or ('country' in row_str and 'cirv' in row_str):
                    header_row = idx
                    break
            
            if header_row != -1:
                df = pd.read_excel(xlsx_path, sheet_name=all_data_sheet, header=header_row)
            else:
                df = pd.read_excel(xlsx_path, sheet_name=all_data_sheet)
                
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")
        else:
            print(f"Could not find 'All Data' sheet in {f}")
