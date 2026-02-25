"""Quick verification of all 6 BioEcho notebooks."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

DIR = os.path.join(os.path.dirname(__file__), 'files')
files = [
    '1 voice.ipynb', '2_face.ipynb', '3_gaze.ipynb',
    '4_typing.ipynb', '5_face_disease.ipynb', '6_fusion.ipynb'
]

all_ok = True
for fname in files:
    path = os.path.join(DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    issues = []
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

    for ci, cell in enumerate(code_cells):
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Check BF16
        if 'bfloat16' in src:
            issues.append(f'  BF16 still in code cell {ci}')

        # Check nn.RMSNorm (should be _RMSNorm in NB4)
        if 'nn.RMSNorm(' in src and 'try:' not in src and '_RMSNorm = nn.RMSNorm' not in src:
            issues.append(f'  nn.RMSNorm() still in cell {ci}')

        # Check hardcoded API key
        if 'KGAT_2a969618d36d94f56d0989908ec94774' in src and 'os.environ' not in src:
            issues.append(f'  Hardcoded API key in cell {ci}')

        # Syntax check
        try:
            compile(src, f'{fname}:cell_{ci}', 'exec')
        except SyntaxError as e:
            issues.append(f'  SyntaxError cell {ci}: {e}')

    # Check NB1 specific: save_checkpoint defined
    if '1 voice' in fname:
        all_src = '\n'.join(
            ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in code_cells
        )
        if 'def save_checkpoint' not in all_src:
            issues.append('  Missing save_checkpoint definition')
        if 'CKPT_DIR' not in all_src:
            issues.append('  Missing CKPT_DIR definition')
        # Check duplicate cell removed
        cell_ids = [c.get('id', '') for c in nb['cells']]
        if 'cell-datasets' in cell_ids:
            issues.append('  Duplicate cell-datasets not removed')

    # Check NB4 specific: _RMSNorm used
    if '4_typing' in fname:
        all_src = '\n'.join(
            ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in code_cells
        )
        if 'self.final_norm=nn.RMSNorm' in all_src:
            issues.append('  nn.RMSNorm not replaced with _RMSNorm')

    if issues:
        print(f'FAIL {fname}:')
        for iss in issues:
            print(iss)
        all_ok = False
    else:
        print(f'OK   {fname} ({len(code_cells)} code cells, all syntax valid)')

# Check cu118 vs cu121
for fname in files:
    path = os.path.join(DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'cu118' in content:
        print(f'WARN {fname}: still has cu118 (should be cu121)')

print()
if all_ok:
    print('ALL 6 NOTEBOOKS PASSED VERIFICATION')
else:
    print('SOME NOTEBOOKS HAVE ISSUES')
