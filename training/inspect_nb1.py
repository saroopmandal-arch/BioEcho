"""Show cell structure of notebook 1."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\saroo\Downloads\BioEcho\training\files\1 voice.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    first = src[:100].replace('\n', ' | ')
    print(f'Cell {i} [{c["cell_type"]}] id={c.get("id","")}')
    print(f'  {first}...')
    if 'RAVDESS' in src or 'ravdess' in src.lower() or 'zenodo' in src.lower():
        print('  >>> HAS RAVDESS/ZENODO <<<')
    print()
