# Datos compartidos (v1)

En **v2** los datos pesados viven en la carpeta Drive original **`d10sformer`**, no en `d10sformer-v2`.

| Carpeta | Ubicación en Drive |
|---------|-------------------|
| Raw (martj42, StatsBomb) | `MyDrive/d10sformer/data/raw/` |
| Interim (ELO, parquet) | `MyDrive/d10sformer/data/interim/` |
| Processed (vocab, corpus .pkl) | `MyDrive/d10sformer/data/processed/` |

Los notebooks v2 usan `src/paths.py` con `DATA_ROOT` → `d10sformer` y `PROJECT_ROOT` → `d10sformer-v2`.

Podés **borrar** `d10sformer-v2/data/` en Drive si fue una copia duplicada (ahorra espacio).
