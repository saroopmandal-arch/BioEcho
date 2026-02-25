"""Fix NB1 RAVDESS: replace Zenodo download with Kaggle input path. Handles any format."""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\Users\saroo\Downloads\BioEcho\training\files\1 voice.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    # Get source as single string regardless of format
    if isinstance(cell['source'], list):
        src = ''.join(cell['source'])
    else:
        src = cell['source']

    if 'zenodo' not in src:
        continue

    print(f'Cell {i}: Found zenodo reference')

    # Show the lines with RAVDESS / zenodo
    lines = src.split('\n')
    rav_start = None
    rav_end = None
    for li, line in enumerate(lines):
        if '# ── 1. RAVDESS' in line or '# -- 1. RAVDESS' in line or '1. RAVDESS' in line:
            rav_start = li
        if rav_start is not None and 'unlink' in line and 'rav' in line:
            rav_end = li
            break
        # Also catch end by next section marker
        if rav_start is not None and li > rav_start + 2 and line.startswith('# ── 2'):
            rav_end = li - 1
            break

    if rav_start is not None:
        actual_end = rav_end if rav_end else rav_start + 12
        print(f'  RAVDESS block: lines {rav_start} to {actual_end}')
        for j in range(rav_start, min(actual_end + 1, len(lines))):
            print(f'  [{j}] {lines[j]}')
    else:
        # Just look for zenodo line
        for li, line in enumerate(lines):
            if 'zenodo' in line:
                print(f'  zenodo at line {li}: {line}')

    # Now do the replacement: replace RAVDESS download section
    new_ravdess = [
        "# ── 1. RAVDESS (Kaggle input — skip download)",
        "USE_RAVDESS = False",
        "for _rav_path in [Path('/kaggle/input/ravdess-emotional-speech-audio'),",
        "                  Path('/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24'),",
        "                  Path('/kaggle/input/ravdess'),",
        "                  Path('/kaggle/input/ravdess-emotional-song-audio')]:",
        "    if _rav_path.exists():",
        "        rav_dir = DATA / 'ravdess'",
        "        if not rav_dir.exists(): rav_dir.symlink_to(_rav_path)",
        "        USE_RAVDESS = True",
        "        console.print(f'[green]\\u2705 RAVDESS found: {_rav_path}[/]')",
        "        break",
        "if not USE_RAVDESS:",
        "    console.print('[yellow]RAVDESS not found in /kaggle/input/ — add it as a dataset[/]')",
    ]

    if rav_start is not None and rav_end is not None:
        # Replace the lines
        new_lines = lines[:rav_start] + new_ravdess + lines[rav_end + 1:]
        new_src = '\n'.join(new_lines)
    else:
        # Regex fallback: find the zenodo download block
        # Pattern: from "RAVDESS" section to unlink or next section
        new_src = re.sub(
            r'# [─\-]+ 1\. RAVDESS.*?(?:rav_zip\.unlink\([^)]*\)|(?=# [─\-]+ 2\.))',
            '\n'.join(new_ravdess),
            src,
            flags=re.DOTALL
        )

    # Save back
    if isinstance(cell['source'], list):
        result_lines = new_src.split('\n')
        result = []
        for j, line in enumerate(result_lines):
            if j < len(result_lines) - 1:
                result.append(line + '\n')
            else:
                result.append(line)
        cell['source'] = result
    else:
        cell['source'] = new_src

    # Verify
    final_src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    has_zenodo = 'zenodo' in final_src
    has_kaggle_input = '/kaggle/input/ravdess' in final_src
    print(f'  After fix: zenodo={has_zenodo}, kaggle_input={has_kaggle_input}')
    break

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
