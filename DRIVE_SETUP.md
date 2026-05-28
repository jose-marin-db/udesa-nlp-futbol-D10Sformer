# Configuración Google Drive — D10Sformer v2

## Carpetas

| Rol | Nombre en MyDrive | Folder ID |
|-----|-------------------|-----------|
| **Datos (v1)** | `d10sformer` | `1fLNr0QUdJsFqtxPSzz6hsDaKGgi5Bx4F` |
| **Código v2** | `d10sformer-v2` | `1Xz1rbw8t8jF_6J5Ez-_vUb7MuPG69w-O` |

> Si compartiste otra URL (`1dnikOS2QcY7cmuLZ4qoUkfNckQkwLJEw`), renombrala a **`d10sformer-v2`** o mové su contenido a la carpeta anterior para que Colab resuelva las rutas automáticamente.

## Inventario de datos (carpeta original)

Requerido para v2:

| Artefacto | Ruta |
|-----------|------|
| martj42 CSVs | `data/raw/international_results/*.csv` |
| StatsBomb | `data/raw/statsbomb/open-data/data/` |
| ELO interim | `data/interim/*.parquet` |
| Vocabulario | `data/processed/vocab.json` |
| Corpus ML | `data/processed/corpus/*.pkl` |
| Pretrain ckpt | `checkpoints/pretrain_5ep/best.pt` (en v1 o re-entrenar en v2) |

## Colab — setup mínimo

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
from pathlib import Path
sys.path.insert(0, '/content/drive/MyDrive/d10sformer-v2/src')
from paths import ensure_paths, print_paths

paths = ensure_paths(require_vocab=True, require_corpus=True)
print_paths(paths)
```

## Subir código actualizado a Drive

Corré en Colab (GPU no necesaria): **`notebooks/99_upload_project_to_drive.ipynb`**

O desde tu máquina, si tenés `gcloud auth application-default login`:

```bash
python scripts/push_v2_to_drive.py
```
