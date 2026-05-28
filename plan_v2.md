# D10Sformer v2.0 — Plan de implementación

**Fecha:** Mayo 2026  
**Objetivo:** Fine-tune centrado en marcador (score), vocabulario y datos ampliados, inferencia productiva.

---

## Estado v1.x (baseline)

| Componente | Valor |
|------------|--------|
| Pretrain | MLM, ~11.8k partidos (intl + StatsBomb) |
| Finetune | W/D/L (λ=1.0) + score 36 clases (λ=0.3) + MLM (λ=0.2) |
| Finetune corpus | ~8.6k selecciones (≥2014) |
| Vocabulario | ~4.521 tokens |
| Inferencia prod. | Solo `result_logits` → W/D/L |

**Hallazgo:** LogReg/XGB superan al Transformer en log-loss; el score head se entrena pero no se usa en simulación.

---

## Google Drive

| Rol | Carpeta MyDrive | Folder ID |
|-----|-----------------|-----------|
| Datos + corpus v1 | `d10sformer` | `1fLNr0QUdJsFqtxPSzz6hsDaKGgi5Bx4F` |
| Código v2 | `d10sformer-v2` | `1Xz1rbw8t8jF_6J5Ez-_vUb7MuPG69w-O` |

Ver `DRIVE_SETUP.md`. Sincronizar código: `notebooks/99_upload_project_to_drive.ipynb`.

---

## Fase 1 — Productizar score (hecho en repo local)

- [x] `LabelMappedCollator` en `src/data/collator.py`
- [x] `LossSpec.finetune_score_primary()` en `src/training/trainer.py`
- [x] `src/inference/predictions.py` — agregar score → W/D/L
- [x] `D10SformerPredictor` con modo `score_derived`
- [x] Tests unitarios
- [x] Notebook `04f_finetune_score_primary.ipynb`
- [ ] Entrenar checkpoint `finetune_score_15ep` (Colab, notebook 04f)
- [ ] Correr `99_upload_project_to_drive.ipynb` en Colab para subir `src/` actualizado

### Uso entrenamiento score-first

```python
from data.collator import MLMCollator, LabelMappedCollator
from training import LossSpec, Trainer, TrainerConfig

collator = LabelMappedCollator(MLMCollator(vocab, mlm_probability=0.15, seed=42))
loss_spec = LossSpec.finetune_score_primary()  # λ_score=1.0, sin result head
```

### Uso inferencia

```python
predictor = D10SformerPredictor(..., inference_mode="score_derived")
probs = predictor.predict("Argentina", "France")  # [p_home, p_draw, p_away]
score_probs = predictor.predict_score(...)        # (36,) distribución conjunta
```

---

## Fase 2 — Escalar datos y vocabulario

- [ ] Rebuild vocab con `min_match_year=2000` (o 1990) en pretrain
- [ ] Bajar `k_player_threshold` (10 → 5) para más `PLAYER_*`
- [ ] Activar eventos StatsBomb en `match_corpus_builder`
- [ ] Corpus pretrain ampliado → re-pretrain obligatorio
- [ ] Finetune por tiers: Tier A (StatsBomb+lineups), Tier B (martj42 metadata)
- [ ] Actualizar `configs/base_config.yaml` con presets v2

### Fuentes adicionales a evaluar

| Fuente | Qué aporta | Esfuerzo |
|--------|------------|----------|
| martj42 completo | +40k partidos históricos | Bajo |
| StatsBomb más ligas | Lineups, eventos | Bajo (ya clonado) |
| FBref / Understat | xG, ratings | Medio (scraping/API) |
| FIFA rankings | Feature tabular | Bajo |

---

## Fase 3 — Evaluación y simulación

- [ ] Métricas de score: RPS, log-loss por clase de marcador
- [ ] Comparar W/D/L derivado vs LogReg en val/test
- [ ] Monte Carlo con muestreo de marcador (`06c` o extender `06b`)
- [ ] Calibración temperature scaling sobre probs derivadas

---

## Decisiones de diseño v2

### Score: 36 clases vs Poisson marginal

| Enfoque | Pros | Contras |
|---------|------|---------|
| **36/49 clases (actual)** | Ya implementado, distribución conjunta | Clases raras, clamp 5+ |
| **Poisson H × Poisson A** | Mejor con pocos datos, goles altos | Nueva cabeza + pérdida |
| **MLM en `SCORE_*`** | Coherente con BERT | Rediseño collator |

**Recomendación v2.0:** score-first con 36 clases; v2.1 evaluar Poisson si RPS no mejora.

### W/D/L derivado

`P(home_win) = Σ_{h>a} P(SCORE_h_a)` — implementado en `src/inference/predictions.py`.

---

## Checklist antes del Mundial 2026

1. Checkpoint score-first entrenado y subido a Drive
2. Playground con toggle LogReg / D10Sformer / score_derived
3. Documentar en paper sección “v2 score-first”
4. Re-simular WC2026 con ambos predictores y comparar bracket
