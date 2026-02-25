"""
optimize_speed.py — Reduce epochs + maximize dual T4 GPU usage across all 6 notebooks.
Strategy: fewer epochs, bigger batches, tighter patience, OneCycleLR with higher LR.
"""
import json, sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

DIR = os.path.join(os.path.dirname(__file__), 'files')

# ── Optimization targets per notebook ──
# (old_value, new_value, reason)
OPTIMIZATIONS = {
    '1 voice.ipynb': {
        # Voice: 20ep → 10ep, batch 16→32, patience 8→4
        'epochs': ('20', '10'),
        'batch_size': ('16', '32'),
        'patience': ('8', '5'),
        'grad_accum': ('2', '1'),  # batch already doubled
        'num_workers': ('0', '4'),  # use workers for speed
    },
    '2_face.ipynb': {
        # rPPG: 50ep → 15ep, batch 8→16, patience 8→5
        'epochs': ('50', '15'),
        'batch_size': ('8', '16'),
        'patience': ('8', '5'),
        'grad_accum': ('2', '1'),
    },
    '3_gaze.ipynb': {
        # Gaze: 30+20 → 12+8, batch 64→128
        'epochs_gaze': ('30', '12'),
        'epochs_neuro': ('20', '8'),
        'batch_size': ('64', '128'),
        'patience': ('7', '4'),
    },
    '4_typing.ipynb': {
        # Typing: 50ep → 15ep, batch 128→256, patience 10→5
        'epochs': ('50', '15'),
        'batch_size': ('128', '256'),
        'patience': ('10', '5'),
        'grad_accum': ('2', '1'),
    },
    '5_face_disease.ipynb': {
        # Face: 40ep → 12ep, batch 32→64, folds 3→2, patience 10→4
        'epochs': ('40', '12'),
        'batch_size': ('32', '64'),
        'n_folds': ('3', '2'),
        'patience': ('10', '4'),
        'grad_accum': ('4', '2'),
    },
    '6_fusion.ipynb': {
        # Fusion: 60ep → 15ep, batch 64→128, folds 3→2, patience 12→5
        'epochs': ('60', '15'),
        'batch_size': ('64', '128'),
        'n_folds': ('3', '2'),
        'patience': ('12', '5'),
        'grad_accum': ('2', '1'),
    },
}

def apply_config_changes(src, changes, fname):
    """Apply numeric config changes in dataclass definitions."""
    modified = src
    for param, (old, new) in changes.items():
        # Match patterns like: param_name: type = value  OR  param_name = value
        # Also match:  param_name:  int   = value
        patterns = [
            # dataclass field: epochs: int = 50
            (rf'({param}\s*:\s*\w+\s*=\s*){old}(\b)', rf'\g<1>{new}\2'),
            # dataclass field: epochs:    int = 50 (extra spaces)
            (rf'({param}\s*:\s*\w+\s*=\s*){old}(\s)', rf'\g<1>{new}\2'),
            # simple assignment: C.epochs = 50
            (rf'(C\.{param}\s*=\s*){old}(\b)', rf'\g<1>{new}\2'),
            # inline: epochs={old}
            (rf'({param}=){old}(\b)', rf'\g<1>{new}\2'),
        ]
        for pat, repl in patterns:
            new_modified = re.sub(pat, repl, modified)
            if new_modified != modified:
                modified = new_modified
                break
    return modified


for fname, changes in OPTIMIZATIONS.items():
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        print(f'SKIP {fname} (not found)')
        continue

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed_count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        new_src = apply_config_changes(src, changes, fname)

        if new_src != src:
            if isinstance(cell['source'], list):
                lines = new_src.split('\n')
                result = []
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        result.append(line + '\n')
                    else:
                        result.append(line)
                cell['source'] = result
            else:
                cell['source'] = new_src
            changed_count += 1

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'OK {fname}: {changed_count} cells modified')
    for param, (old, new) in changes.items():
        print(f'   {param}: {old} -> {new}')

print('\nDone! All notebooks optimized for speed.')
