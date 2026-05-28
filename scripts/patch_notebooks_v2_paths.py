#!/usr/bin/env python3
"""Parchea notebooks para usar PROJECT_ROOT (v2) + DATA_ROOT (v1 en Drive)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

SETUP_CELL = '''import sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_ROOT = Path('/content/drive/MyDrive/d10sformer-v2')
    DATA_ROOT = Path('/content/drive/MyDrive/d10sformer')
else:
    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    DATA_ROOT = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from paths import ensure_paths, print_paths

paths = ensure_paths(project_root=PROJECT_ROOT, data_root=DATA_ROOT)
print_paths(paths)

# Alias legacy usados en notebooks v1
ROOT = paths.project_root
DATA_PROCESSED = paths.data_processed
CORPUS_DIR = paths.corpus_dir
VOCAB_PATH = paths.vocab_path
CKPT_DIR = paths.checkpoints
CHECKPOINTS_V1 = paths.checkpoints_v1
DATA_RAW = paths.data_raw
DATA_INTERIM = paths.data_interim
'''

IMPORTS_PATCH = [
    ("ROOT = Path(PROJECT_ROOT)", "# paths: ROOT ya definido en setup"),
    ("sys.path.insert(0, str(ROOT / 'src'))", "# paths: sys.path ya configurado"),
    ("DATA_PROCESSED = ROOT / 'data' / 'processed'", "DATA_PROCESSED = paths.data_processed"),
    ("CORPUS_DIR = DATA_PROCESSED / 'corpus'", "CORPUS_DIR = paths.corpus_dir"),
    ("VOCAB_PATH = DATA_PROCESSED / 'vocab.json'", "VOCAB_PATH = paths.vocab_path"),
    ("CKPT_DIR = ROOT / 'checkpoints'", "CKPT_DIR = paths.checkpoints"),
    (
        "PRETRAIN_CKPT = CKPT_DIR / 'pretrain_5ep' / 'best.pt'",
        "PRETRAIN_CKPT = paths.checkpoints_v1 / 'pretrain_5ep' / 'best.pt'",
    ),
    ("DATA_RAW = ROOT / 'data' / 'raw'", "DATA_RAW = paths.data_raw"),
    ("DATA_INTERIM = ROOT / 'data' / 'interim'", "DATA_INTERIM = paths.data_interim"),
]


def patch_notebook(path: Path) -> bool:
    if path.name.startswith("99_"):
        return False
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "drive.mount" in src and "PROJECT_ROOT" in src and "d10sformer-v2" not in src:
            cell["source"] = [SETUP_CELL]
            changed = True
            continue
        new_src = src
        for old, new in IMPORTS_PATCH:
            if old in new_src:
                new_src = new_src.replace(old, new)
        if new_src != src:
            cell["source"] = [new_src]
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> int:
    updated = []
    for p in sorted(NOTEBOOKS.glob("*.ipynb")):
        if patch_notebook(p):
            updated.append(p.name)
    print(f"Actualizados ({len(updated)}):")
    for n in updated:
        print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
