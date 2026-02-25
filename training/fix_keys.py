"""Fix remaining API key hardcoding in notebooks 5 and 6."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

DIR = os.path.join(os.path.dirname(__file__), 'files')

for fname in ['5_face_disease.ipynb', '6_fusion.ipynb']:
    path = os.path.join(DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        if 'KGAT_2a969618d36d94f56d0989908ec94774' in src and 'os.environ' not in src:
            # Find and replace the json.dump line with creds
            old = "json.dump({'username':'saroopmandal','key':'KGAT_2a969618d36d94f56d0989908ec94774'}, open(cp,'w'))"
            new = (
                "# On Kaggle, credentials are auto-injected. No hardcoded key needed.\n"
                "import os\n"
                "json.dump({'username': os.environ.get('KAGGLE_USERNAME', 'saroopmandal'),\n"
                "           'key': os.environ.get('KAGGLE_KEY', '')}, open(cp, 'w'))"
            )
            if old in src:
                src = src.replace(old, new)
            else:
                # Try with different quote/space patterns
                import re
                src = re.sub(
                    r"json\.dump\(\{['\"]username['\"]:\s*['\"]saroopmandal['\"],\s*['\"]key['\"]:\s*['\"]KGAT_[^'\"]+['\"]\},\s*open\(cp,\s*['\"]w['\"]\)\)",
                    "# On Kaggle, credentials are auto-injected.\nimport os\njson.dump({'username': os.environ.get('KAGGLE_USERNAME', 'saroopmandal'),\n           'key': os.environ.get('KAGGLE_KEY', '')}, open(cp, 'w'))",
                    src
                )

            if isinstance(cell['source'], list):
                lines = src.split('\n')
                result = []
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        result.append(line + '\n')
                    else:
                        result.append(line)
                cell['source'] = result
            else:
                cell['source'] = src
            print(f'Fixed API key in {fname}')

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Saved {fname}')

print('Done')
