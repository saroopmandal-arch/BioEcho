"""Extract EXACT raw source lines for cell 6 of NB1 to see formatting."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\saroo\Downloads\BioEcho\training\files\1 voice.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][6]
src_list = cell['source']
if isinstance(src_list, list):
    for li, line in enumerate(src_list):
        if 'RAVDESS' in line or 'ravdess' in line or 'zenodo' in line:
            # Print this line and surrounding 10 lines
            start = max(0, li - 2)
            end = min(len(src_list), li + 20)
            print(f'Found at source line {li}. Context [{start}:{end}]:')
            for j in range(start, end):
                print(f'  [{j}] {repr(src_list[j])}')
            print()
            break
