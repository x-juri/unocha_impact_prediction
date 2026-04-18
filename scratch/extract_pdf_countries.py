import pdfplumber
import os
import pandas as pd

pdf_dir = 'initial_data/allocations'
round_countries = {}

for filename in sorted(os.listdir(pdf_dir)):
    if filename.endswith('.pdf'):
        match = filename.replace('.pdf', '').split('_')[-1]
        path = os.path.join(pdf_dir, filename)
        countries = []
        with pdfplumber.open(path) as pdf:
            print(f"--- Extracting from {filename} ---")
            for page in pdf.pages:
                tables = page.find_tables()
                for table in tables:
                    rows = table.extract()
                    if rows:
                        # Print some rows to see the structure
                        for row in rows:
                            if row and len(row) > 0 and row[0]:
                                # Clean country name
                                country = row[0].strip().replace('\n', ' ')
                                if country and country.lower() != 'country' and len(country) < 50:
                                    countries.append(country)
            
            # Print unique detected countries
            unique_countries = sorted(list(set(countries)))
            print(f"Detected {len(unique_countries)} countries: {unique_countries}")
            round_countries[match] = unique_countries

print("\nFinal Mapping Summary:")
for k, v in round_countries.items():
    print(f"{k}: {len(v)} countries")
