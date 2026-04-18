import pdfplumber
import os
import re

pdf_dir = 'initial_data/allocations'

def get_countries(filename):
    path = os.path.join(pdf_dir, filename)
    with pdfplumber.open(path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages])
        # Look for the "Tier" tables or lists of countries
        # Often there is a list behind bold bullets or similar
        print(f"--- {filename} ---")
        
        # Look for phrases like "final selection of countries for ... totalling $X million"
        # and see if the countries are listed right after.
        # Alternatively, extract words that look like countries or specific markers.
        
        # Let's just look at the first page summary which often lists them.
        first_page = pdf.pages[0].extract_text()
        print(first_page)

for f in sorted(os.listdir(pdf_dir)):
    if f.endswith('.pdf'):
        get_countries(f)
