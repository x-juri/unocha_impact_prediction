import pdfplumber
import os

with pdfplumber.open('initial_data/allocations/Allocation_2025-I.pdf') as pdf:
    # Print the table on page 1 or 2
    for i in range(len(pdf.pages)):
        print(f"--- Page {i+1} ---")
        text = pdf.pages[i].extract_text()
        print(text)
        table = pdf.pages[i].extract_table()
        if table:
            for row in table:
                print(row)
