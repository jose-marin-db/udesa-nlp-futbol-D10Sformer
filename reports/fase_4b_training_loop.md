# Fase 4b — Training Loop

## Resumen

Implementamos el `Trainer` step-based con AMP, AdamW con weight decay
selectivo, scheduler cosine con warmup, logging local en JSONL y
checkpointing resumible. El smoke test (200 pasos sobre un subset de 1000
docs) corre en pocos minutos en T4 y nos permite validar que el modelo
aprende antes de invertir tiempo en el pre-training completo.

## Archivos creados

```
src/training/
├── __init__.py
├── scheduler.py             — get_warmup_cosine_schedule (linear warmup → cosine to 0)
├── training_metrics.py      — mlm_loss, mlm_accuracy, perplexity
└── trainer.py               — Trainer + TrainerConfig + LossSpec

tests/
├── test_scheduler.py        — 5 tests (warmup, decay, min_lr_ratio, validación)
├── test_training_metrics.py — 5 tests (ignore_index, accuracy, perplexity)
└── test_trainer.py          — 5 smoke tests (steps, log, loss decreases, no NaN, ckpt roundtrip)

notebooks/
└── 04b_training_loop.ipynb  — smoke test + curvas + checkpoint roundtrip

reports/
└── fase_4b_training_loop.md — este archivo
```

## Decisiones de diseño

### 1. Step-based, no epoch-based
**Por qué:** las épocas no son comparables entre experimentos cuando varía
el batch size. El estándar moderno (BERT, GPT, ViT) cuenta **pasos** (=
updates del optimizer). Schedulers, checkpoints y evals se anclan a
`step`. Esto también nos permite mezclar corpus heterogéneos (clubes +
selecciones) sin definir una "época" artificial.

### 2. AdamW con weight decay selectivo
Por convención BERT/HuggingFace, los biases, `LayerNorm.weight` y
`LayerNorm.bias` NO reciben weight decay (su rol es de escala/centrado, no
de feature extractor). El método `_build_optimizer` separa los parámetros
en dos grupos:

```
decay group:    Linear weights, Embedding weights  →  wd = config.weight_decay
no_decay group: biases, LN params                   →  wd = 0
```

Loshchilov & Hutter (2017) muestran que esta separación mejora la
generalización en Transformers.

### 3. Cosine schedule con warmup lineal
- **Warmup lineal** los primeros `warmup_ratio × max_steps` pasos (10%
  por defecto) evita que el LR inicial alto destruya los pesos
  iniciales aleatorios. Es estándar desde el paper original de
  Transformer (Vaswani et al., 2017).
- **Cosine decay** suaviza la convergencia hacia el final del
  entrenamiento (Loshchilov & Hutter, 2017 — *SGDR*).

### 4. Mixed Precision (AMP)
Usamos `torch.amp.autocast` + `GradScaler`. En T4 esto **duplica el
throughput** sin pérdida de precisión final (Micikevicius et al., 2018).
El `GradScaler` previene el underflow de gradientes pequeños que ocurre
en FP16. Lo desactivamos automáticamente en CPU.

### 5. Gradient clipping (norma 1.0)
Estándar BERT. Previene gradient explosions, especialmente al inicio del
entrenamiento cuando el LR aún es alto y los pesos no están alineados.

### 6. LossSpec — composable
El mismo Trainer sirve para:
- **Pre-training:** `LossSpec(use_mlm=True)` — solo MLM, λ=1.
- **Fine-tuning:** `LossSpec(use_mlm=True, use_result=True, use_score=True,
  lambda_mlm=0.2, lambda_result=1.0, lambda_score=0.3)` —
  MLM auxiliar + result principal + score secundario.

Esto evita tener dos archivos de training duplicados.

### 7. Logging local en JSONL
Cada N pasos (`log_every`) escribimos una línea JSON al archivo
`metrics.jsonl`. El notebook 04b lee ese archivo y plotea curvas. Si
en el futuro queremos wandb, basta con agregar una línea de
`wandb.log(metrics)` en `Trainer._log`.

### 8. Checkpoints resumibles
`save_checkpoint` guarda model + optimizer + scheduler + scaler + step,
de modo que `load_checkpoint` permite reanudar exactamente donde se
quedó. Crítico cuando Colab se desconecta.

## Configuración elegida para el smoke

| Hyper-parámetro | Valor | Razón |
|---|---|---|
| `max_seq_length` | **80** | p99 de Fase 4a fue 64; margen de 16 |
| `batch_size` | 64 | 6× más grande que d=256, max_len=512 permitiría |
| `lr` | 5e-4 | Estándar BERT-small |
| `weight_decay` | 0.01 | Estándar |
| `warmup_ratio` | 0.1 | 20 pasos sobre 200 totales |
| `max_steps` | **200** | smoke test rápido (~2-3 min en T4) |
| `mlm_probability` | 0.15 | BERT |
| `subset` | 1000 docs (de 11.852) | smoke test |

## Qué esperar del smoke test (criterios de aceptación)

| Métrica | Valor esperado | Acción si falla |
|---|---|---|
| Loss inicial (paso 10) | ~6-8 (cerca de ln(V)=8.42) | Si <5: bug de masking |
| Loss final (paso 200) | <6 (al menos -1.0 vs inicial) | Si no baja: lr, gradient flow |
| Perplexity final | <400 | Sigue siendo alta porque V=4521 |
| MLM accuracy final | >0.10 | Random baseline = 1/V = 0.0002 |
| Steps por segundo | 5-15 en T4 | Si <2: revisar AMP/batch/num_workers |
| Tests (15 nuevos) | 15/15 verdes | Si falla `test_loss_decreases_on_memorizable_data`: bug grave |
| Checkpoint roundtrip | parámetros idénticos | Si difiere: bug en state_dict |

## Riesgos identificados

1. **DataLoader workers en Colab.** `num_workers > 0` a veces falla en
   Colab por restricciones de fork. Si pasa, dejamos `num_workers=0`
   (la diferencia es marginal con corpus en RAM).

2. **NaN en AMP con gradientes muy pequeños al inicio.** Mitigamos con
   `GradScaler` (es justamente para esto) y warmup lineal. Si vemos
   NaN, bajamos a `mixed_precision=False`.

3. **Out-of-memory en T4 con batch=64.** Improbable con d=256 y len=80
   (~512 MB de activations). Si ocurre, bajamos a batch=32 + AMP.

4. **Smoke insuficiente para detectar problemas de pre-training real.**
   El smoke es 200 pasos sobre 1000 docs. El pre-training real será
   ~1000-3000 pasos sobre 12K docs. Algunos bugs (e.g., overfitting
   precoz) solo se manifiestan a mayor escala. Pero la **ausencia de
   reducción de loss en el smoke** es una señal **suficiente** de bug.

## Cómo proceder

1. Vic corre `04b_training_loop.ipynb` en Colab y reporta:
   - Las 4 curvas (loss, perplexity, accuracy, lr)
   - Diagnóstico cuantitativo (reducción % de loss)
   - 15 tests verdes
   - Checkpoint roundtrip OK

2. Si todo bien, paso a **Fase 4c: pre-training real**:
   - Corpus completo (11.852 docs)
   - 3 épocas (~550 pasos a batch=64)
   - Logging completo en JSONL
   - Checkpoint del mejor por val_loss
   - **Estimación T4: 30-50 minutos**

3. Después, **Fase 4d: fine-tuning** sobre `finetune_train.pkl` (8.658
   docs, solo selecciones, con todas las heads activas:
   `LossSpec(use_mlm=True, use_result=True, use_score=True)`).
