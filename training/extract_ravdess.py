"""Extract full RAVDESS download code from NB1 and show line numbers."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\saroo\Downloads\BioEcho\training\files\1 voice.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    if 'zenodo' in src.lower() or 'RAVDESS' in src:
        if c['cell_type'] == 'code':
            print(f'=== CELL {i} id={c.get("id","")} type={c["cell_type"]} ===')
            lines = src.split('\n')
            for li, line in enumerate(lines):
                if 'ravdess' in line.lower() or 'zenodo' in line.lower() or 'RAVDESS' in line:
                    print(f'  LINE {li}: {line}')
            # Print 5 lines before and after each RAVDESS reference
            for li, line in enumerate(lines):
                if 'ravdess' in line.lower() or 'zenodo' in line.lower():
                    start = max(0, li - 3)
                    end = min(len(lines), li + 15)
                    print(f'\n  --- Context around line {li} ---')
                    for j in range(start, end):
                        marker = '>>>' if j == li else '   '
                        print(f'  {marker} {j}: {lines[j]}')
                    break
            print(f'  Total lines: {len(lines)}')
            print('=== END ===\n')
