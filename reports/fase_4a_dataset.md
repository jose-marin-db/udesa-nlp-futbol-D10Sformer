# Fase 4a — Dataset, Corpus Builder y Collator MLM

## Resumen

Construimos toda la infraestructura de datos que alimenta al training loop.
La salida de esta sub-fase son **4 pickles** en `data/processed/corpus/`:
`pretrain.pkl`, `finetune_train.pkl`, `val.pkl`, `test.pkl`.

## Archivos creados

```
src/data/
├── dataset.py                  — MatchDataset (PyTorch Dataset) + MatchSample
├── collator.py                 — MLMCollator (BERT-style 15% / 80-10-10) + CollatedBatch
└── match_corpus_builder.py     — builders DataFrame → MatchDocument

tests/
├── test_dataset.py             — 5 tests
└── test_collator.py            — 9 tests
```

## Decisiones de diseño

### 1. Dataset in-memory (no `IterableDataset`)
**Por qué:** el corpus combinado son ~52K partidos. Caben en RAM sin problema
(~200MB serializado). Esto simplifica el shuffling (`DataLoader(shuffle=True)`
funciona out-of-the-box) y elimina I/O del bottleneck. Sería distinto si el
corpus tuviera 10M de samples.

### 2. Masking aplicado **on-the-fly** en el collator, no precomputado
**Por qué:** queremos que el modelo vea **distintas máscaras en cada época**,
no la misma máscara repetida 10 veces. Esto es exactamente lo que hace BERT
(Devlin et al., 2018, §3.1) y la práctica estándar.

Costo: el masking se hace en CPU por cada batch. Para batch=32 y len≈50, son
~16K bernoullis: irrelevante comparado con el forward/backward en GPU.

### 3. Tokens **no enmascarables**
La lista incluye todos los `SPECIAL_TOKENS` ([CLS], [SEP], [PAD], [MASK],
[UNK]) y cualquier token que empiece con `[` (estructurales: `[LINEUP_A]`,
`[FEATURES_START]`, etc.). Enmascarar estos es ruido — el modelo aprendería
trivialmente a predecir el token estructural a partir de su posición.

### 4. `mlm_labels` con sentinel **-100**
Convención estándar de PyTorch: `nn.CrossEntropyLoss(ignore_index=-100)`
descarta esas posiciones del cómputo de loss y del gradiente. Mucho más
limpio que un masking aparte.

### 5. **Result/score labels también usan -100** cuando faltan
Los partidos sin target (raros, pero existen — entradas incompletas) se
incluyen en pre-training (donde la loss principal es MLM) y se ignoran en
las heads de result/score. Permite usar el corpus completo para MLM
mientras solo se entrena el head clasificador donde hay etiqueta.

### 6. Splits temporales en el notebook, **no aleatorios**
- `pretrain` = clubes (StatsBomb < 2023-07) + selecciones (martj42 < 2023-07).
- `finetune_train` = solo selecciones < 2023-07.
- `val` = selecciones [2023-07, 2024-07).
- `test` = selecciones ≥ 2024-07 — incluye los amistosos del 2026 pre-Mundial.

## Estadísticas esperadas (a confirmar en Colab)

| Dataset | Docs aproximados |
|---|---|
| `docs_int` (martj42, ≥2014) | ~30K |
| `docs_sb` (StatsBomb con lineups) | ~3.4K |
| `pretrain` (intl_train + sb_train) | ~32K |
| `finetune_train` (solo intl_train) | ~28K |
| `val` (intl, jul 2023 → jul 2024) | ~2K |
| `test` (intl, ≥ jul 2024) | ~2-3K |

## Lo que falta antes de Fase 4b

1. Vic corre `notebooks/04a_dataset_collator.ipynb` en Colab y reporta:
   - Tamaños finales de cada split
   - Distribución de longitudes (mediana, p90, p99)
   - Masking ratio empírico (debe estar cerca de 0.15)
   - Tests verdes (5 dataset + 9 collator = 14 nuevos)
   - **Confirmación visual** del demo de masking (sección 6 del notebook)

2. Una vez confirmado, paso a **Fase 4b**: training loop con AdamW + cosine
   warmup, mixed precision, logging local JSON, checkpoint guardado cada N
   pasos.

## Riesgos identificados

1. **Longitudes muy cortas para StatsBomb sin eventos:** si la mayoría de
   docs son <50 tokens, el batch puede ser ineficiente. Vamos a ver el p90 y
   decidir si conviene `max_seq_length=128` en lugar de 512 para acelerar.

2. **Eventos StatsBomb deshabilitados por defecto:** los eventos rich
   (goles + cards + subs) los dejé fuera de Fase 4a porque cargar 3.5K
   archivos de events.json adicionales lleva ~30 min y agrega complejidad.
   Si Fase 5 muestra que el modelo se beneficia de eventos, los habilitamos
   ahí (basta con activar la rama en `build_statsbomb_document`).
