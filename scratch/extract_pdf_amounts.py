import pdfplumber
import os
import re

pdf_dir = 'initial_data/allocations'
results = {}

for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        match = re.search(r'(\d{4})-(I+)', filename)
        if match:
            year = match.group(1)
            rnd = match.group(2)
            path = os.path.join(pdf_dir, filename)
            
            with pdfplumber.open(path) as pdf:
                # Usually the total is on the first page
                text = pdf.pages[0].extract_text()
                # Look for patterns like "$150 million" or "$150M"
                amounts = re.findall(r'\$\d+(?:\.\d+)?\s*(?:million|M)', text, re.IGNORECASE)
                results[f"{year}-{rnd}"] = amounts
                # Also look for table with "Total"
                print(f"--- {filename} ---")
                print(text[:500])
                print(f"Detected amounts: {amounts}\n")

print("\nSummary of PDF Amounts:")
for k, v in results.items():
    print(f"{k}: {v}")
