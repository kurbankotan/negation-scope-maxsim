"""Build cacsa2_project.zip, the package archive that the Colab notebooks read from Google Drive.

The notebooks NegPairs_v3_build_colab.ipynb, NegPairs_v3_main_colab.ipynb and NegPairs_v3_followup_colab.ipynb (and
the archived notebooks of the earlier manifests) expect `cacsa2_project.zip` with a top folder `cacsa2_project/` that
holds the package, the tests, the notebooks, the manifests and NegPairs v1 under `data/negpairs_v1/`.
NegPairs_public_systems_colab.ipynb and NegPairs_finetune_li_colab.ipynb embed the package and need no archive.

usage: python make_package_zip.py [OUT.zip]     (default: cacsa2_project.zip next to this script)
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'cacsa2_project.zip'
TOP = sorted([p.name for p in ROOT.glob('*.ipynb')] + [p.name for p in ROOT.glob('make_*notebook*.py')]
             + ['README.md', 'CACSA_manifest_v2_0.md', 'CACSA_manifest_v3_0.md', 'negpairs_v3_audit.md', 'np_v3_main_audit.md'])

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('cacsa2_project/', '')
    for name in TOP:
        z.write(ROOT / name, f'cacsa2_project/{name}')
    for sub in ('cacsa2', 'tests'):
        z.writestr(f'cacsa2_project/{sub}/', '')
        for p in sorted((ROOT / sub).iterdir()):
            if p.is_file() and p.suffix in ('.py', '.json'):
                z.write(p, f'cacsa2_project/{sub}/{p.name}')
    z.writestr('cacsa2_project/data/', '')
    z.writestr('cacsa2_project/data/negpairs_v1/', '')
    for p in sorted((ROOT / 'data/negpairs_v1').iterdir()):
        if p.is_file():
            z.write(p, f'cacsa2_project/data/negpairs_v1/{p.name}')
print(OUT, OUT.stat().st_size // 1024, 'kB')
