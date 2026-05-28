# Plan de Implementación — Temporal Relational Transformer para Predicción de Fútbol

**Proyecto:** MIA305 (UdeSA, 2026) — Proyecto Final
**Autor:** Vic
**Fecha de inicio:** 16 de mayo de 2026
**Deadline crítico:** 11 de junio de 2026 (inicio Mundial 2026)
**Objetivo híbrido:** Entregable académico riguroso + Sistema funcional de predicción para el Mundial 2026

---

## Filosofía del Plan

Este plan asume el principio de **"baseline first, transformer second"**. La razón es doble:

1. **Académica:** sin un baseline serio, no podés afirmar que el Transformer "agrega valor". Toda contribución de NLP moderna se valida contra un baseline más simple (Jurafsky & Martin enfatizan esto repetidamente).
2. **Práctica:** si por cualquier razón el Transformer no llega a tiempo o no converge bien, los baselines tabulares (XGBoost) ya son suficientes para generar predicciones reales del Mundial 2026. **Nunca quedamos sin nada que mostrar.**

El plan está dividido en **8 fases secuenciales** + verificación final, con ~26 días de calendario hasta el inicio del Mundial. Cada fase produce artefactos concretos y reproducibles.

---

## Stack Técnico

| Componente | Elección | Justificación |
|---|---|---|
| Lenguaje | Python 3.11+ | Estándar de facto en ML/NLP |
| Framework DL | PyTorch 2.x | Mencionado en el doc del proyecto; mayor flexibilidad investigativa que TF |
| Entorno principal | Google Colab Pro (T4/L4) | Confirmado por el usuario |
| Versionado | Git + GitHub | Imprescindible para reproducibilidad académica |
| Tracking experimental | Weights & Biases (opcional, free tier) | Loguea hiperparámetros, métricas, checkpoints |
| Datos primarios | StatsBomb Open Data | Eventos detallados, gratuito, incluye Mundiales 2018 y 2022 |
| Datos secundarios | Kaggle "International Football Results" | Para baselines y features de ELO |
| Modelos tabulares | scikit-learn, XGBoost, LightGBM | Baselines estándar |
| Testing | pytest | Para tokenizer y data pipeline |

---

## Estructura de Carpetas del Proyecto

```
Pampa-NLP/
├── data/
│   ├── raw/              # StatsBomb JSONs originales (no commiteado)
│   ├── interim/          # Datos parseados a CSVs/Parquet intermedio
│   └── processed/        # Tokens listos para el modelo
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_baselines.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_transformer_training.ipynb
│   └── 04_tournament_simulation.ipynb
├── src/
│   ├── data/
│   │   ├── statsbomb_loader.py
│   │   ├── feature_engineering.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── embeddings.py
│   │   ├── transformer.py
│   │   └── heads.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── masking.py
│   │   └── train.py
│   ├── eval/
│   │   ├── metrics.py        # Brier, ECE, log-loss
│   │   └── calibration.py
│   └── simulation/
│       └── tournament.py
├── tests/
│   └── test_tokenizer.py
├── configs/
│   └── base_config.yaml
├── reports/
│   └── final_paper.md
├── requirements.txt
└── README.md
```

---

# Fase 0 — Setup y Adquisición de Datos

**Duración estimada:** 1–2 días
**Días calendario:** 16–17 de mayo

## Objetivos

Tener un entorno reproducible, datos en disco, y un análisis exploratorio que valide la viabilidad antes de invertir esfuerzo en arquitectura.

## Entregables

1. Repositorio GitHub con la estructura de carpetas descrita
2. `requirements.txt` congelado con versiones exactas
3. Notebook `00_eda.ipynb` con análisis exploratorio
4. StatsBomb Open Data descargada en `data/raw/`

## Pasos concretos

**0.1.** Crear repo Git, configurar `.gitignore` (excluir `data/raw/`, checkpoints, `.ipynb_checkpoints`).

**0.2.** Crear notebook de Colab base que monte Google Drive para persistencia de datos y checkpoints. Esto es crítico — sin esto, cada vez que se desconecta Colab perdés todo.

**0.3.** Clonar StatsBomb Open Data:

```bash
git clone https://github.com/statsbomb/open-data.git
```

Los datos relevantes están en `open-data/data/`. La estructura clave:
- `competitions.json` → lista de torneos disponibles
- `matches/{comp_id}/{season_id}.json` → metadatos de cada partido
- `lineups/{match_id}.json` → alineaciones
- `events/{match_id}.json` → eventos detallados (~3500 eventos por partido)

**0.4.** EDA mínimo en `00_eda.ipynb`:
- ¿Cuántos partidos hay disponibles? (Esperamos miles entre Mundiales, Champions, ligas top)
- Distribución de torneos, fechas, equipos
- Distribución de resultados (W/D/L) — verificar balance de clases
- Análisis del vocabulario potencial: # jugadores únicos, # equipos únicos, # tipos de eventos
- Validar suposición clave: **¿hay solapamiento de jugadores entre competiciones?** Esto determina si los embeddings de jugadores van a generalizar.

## Riesgos conocidos

- **StatsBomb cubre principalmente ligas top + Champions + algunos Mundiales.** Para el Mundial 2026, vamos a tener que combinar con datos de Kaggle para selecciones que no estén bien cubiertas.
- **Tamaño de los JSONs:** los eventos de un partido pueden pesar 5–10 MB. Para 1000+ partidos, esto se vuelve pesado en RAM. Vamos a necesitar streaming o parquet.

## Verificación de comprensión (Fase 0)

- ¿Por qué hacemos el split temporal (entrenamos en partidos viejos, validamos en partidos nuevos) en vez del split aleatorio típico?
- ¿Qué tipo de "data leakage" podría meterse si no tenemos cuidado al construir features de "forma reciente"?

---

# Fase 1 — Baselines Tabulares

**Duración estimada:** 3 días
**Días calendario:** 18–20 de mayo

## Objetivos

Tener un sistema completo, evaluable y funcionalmente predictivo del Mundial **antes de tocar Transformers**. Esto es el seguro contra falla del modelo complejo.

## Por qué esto importa académicamente

Los papers de NLP serios (Jurafsky & Martin, Eisenstein) siempre reportan baselines fuertes. Decir "mi Transformer logra 65% de accuracy" no significa nada si XGBoost logra 64%. La métrica relevante es **delta sobre baseline**, no accuracy absoluta.

## Features para baselines

Estas features replican lo que un analista deportivo usaría. La idea es darle al baseline todas las chances:

| Feature | Cálculo |
|---|---|
| ELO rating de cada equipo | Sistema ELO clásico actualizado partido a partido |
| Diferencia de ELO | `ELO_A - ELO_B` |
| Forma reciente (últimos 5) | % puntos ganados en últimos 5 partidos |
| Goles a favor promedio (últimos 10) | Media de goles convertidos |
| Goles en contra promedio (últimos 10) | Media de goles recibidos |
| Días de descanso | Días desde último partido |
| Localía | `home / away / neutral` (Mundial = mayormente neutral) |
| Importancia del partido | Fase de grupos / eliminatoria / final |
| Head-to-head histórico | % de victorias en últimos 5 enfrentamientos |
| Confederación | UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC |

## Modelos

1. **Regresión Logística Multinomial** — baseline interpretable, sirve como cota inferior
2. **XGBoost Classifier** — workhorse del ML tabular, casi siempre fuerte
3. **LightGBM** — alternativa más rápida, vale para comparar
4. **Promedio de Poisson** — modelo deportivo clásico (Maher 1982): modela goles como Poisson independiente. Útil como sanity check de dominio.

## Métricas (CRÍTICO)

Acá viene una decisión metodológica importante. **NO usar accuracy como métrica principal.** Las métricas correctas para predicción probabilística son:

- **Log-Loss / Cross-Entropy:** $\mathcal{L} = -\frac{1}{N}\sum_i \log P(y_i \mid x_i)$ — penaliza fuerte la sobreconfianza incorrecta
- **Brier Score:** $\frac{1}{N}\sum_i \|p_i - y_i\|^2$ — promedio cuadrático del error de probabilidad
- **Expected Calibration Error (ECE):** mide qué tan bien la probabilidad reportada matchea la frecuencia empírica
- **Reliability Diagram:** visualización de la calibración

Accuracy se reporta solo como referencia secundaria.

## Validación temporal

Esto es CRÍTICO y donde muchos proyectos fallan:

```
Train:     partidos hasta 31-Dic-2022
Validation: partidos 2023
Test:      partidos 2024–2025
```

NUNCA hacer random split en datos temporales. Eso te leakea información del futuro al pasado y rompe la validez del modelo.

## Entregables Fase 1

- `src/data/feature_engineering.py` — feature extraction pipeline
- `src/models/baselines.py` — implementación de los 4 modelos
- `src/eval/metrics.py` — Brier, ECE, log-loss, reliability diagrams
- Notebook `01_baselines.ipynb` con resultados comparativos
- **Tabla final de baselines** con métricas en validation y test

---

# Fase 2 — Vocabulario y Tokenización

**Duración estimada:** 2–3 días
**Días calendario:** 21–23 de mayo

## Objetivos

Diseñar el "lenguaje del fútbol" que el Transformer va a procesar. Esta fase **determina el techo de performance del modelo** — si la tokenización es pobre, ninguna arquitectura lo arregla.

## Diseño del vocabulario

Categorías de tokens:

**Tokens especiales:**
```
[CLS], [SEP], [PAD], [MASK], [UNK]
[MATCH_START], [MATCH_END]
[LINEUP_A], [LINEUP_B], [BENCH_A], [BENCH_B]
[CONTEXT_START], [CONTEXT_END]
[PREDICT_RESULT]
```

**Tokens de entidades:**
```
PLAYER_<id>     # ~5000 jugadores top
TEAM_<id>       # ~200 selecciones + clubes
COACH_<id>      # ~500 entrenadores
REFEREE_<id>    # opcional
```

**Tokens de eventos:**
```
GOAL, OWN_GOAL, PENALTY_GOAL, PENALTY_MISS
YELLOW_CARD, RED_CARD, SECOND_YELLOW
SUBSTITUTION, INJURY
SHOT_ON, SHOT_OFF, SAVE
ASSIST, KEY_PASS
CORNER, FREE_KICK
```

**Tokens de contexto:**
```
TOURNAMENT_<id>       # WORLD_CUP_2026, UCL_2024_25, etc.
STAGE_GROUP, STAGE_R16, STAGE_QF, STAGE_SF, STAGE_FINAL
HOME, AWAY, NEUTRAL
MIN_<bucket>          # Bucketizado: MIN_0_15, MIN_16_30, etc.
```

**Tokens numéricos bucketizados:**
```
FORM_LOW, FORM_MID, FORM_HIGH
XG_LOW, XG_MID, XG_HIGH
GOALS_T_<bucket>      # goles en torneo actual
```

Tamaño total estimado: **8.000–15.000 tokens**.

## Decisión técnica importante: rolling features como tokens vs. como side-input

Hay dos formas de pasarle al modelo las features rolling (ej. "Julián metió 4 goles en torneo actual"):

**Opción A (tokens):** bucketizamos y agregamos como tokens `GOALS_T_4`, `FORM_HIGH`, etc. Pierde precisión pero mantiene la arquitectura pura Transformer.

**Opción B (concat embeddings):** features numéricas crudas → MLP → vector → se suma al embedding del jugador. Mantiene precisión pero complica el modelo.

**Recomendación:** empezar con A (tokens bucketizados). Es más simple, interpretable, y permite que la atención decida qué pesar. Si la performance es pobre, migrar a B.

## Conversión partido → secuencia de tokens

Función crítica: `match_to_token_sequence(match_obj) -> List[int]`

Estructura típica de la secuencia:

```
[CLS]
TOURNAMENT_WC_2026 STAGE_GROUP HOME
TEAM_A_ARG TEAM_B_MEX
[LINEUP_A] PLAYER_001 PLAYER_002 ... PLAYER_011
[BENCH_A]  PLAYER_012 PLAYER_013 ... PLAYER_023
[LINEUP_B] PLAYER_201 ...
[CONTEXT_START]
  RECENT_MATCH_1: ... eventos clave ...
  RECENT_MATCH_2: ...
[CONTEXT_END]
[FEATURES]
  PLAYER_MESSI FORM_HIGH GOALS_T_2
  PLAYER_JULIAN FORM_HIGH GOALS_T_4
  ...
[SEP]
[PREDICT_RESULT]
```

## Entregables Fase 2

- `src/data/tokenizer.py` con clases `FootballVocab` y `FootballTokenizer`
- Vocabulario serializado (`data/processed/vocab.json`)
- Función inversa para debugging (`decode`)
- Test suite en `tests/test_tokenizer.py`
- Notebook `02_tokenization.ipynb` con ejemplos visuales

---

# Fase 3 — Arquitectura del Transformer

**Duración estimada:** 3–4 días
**Días calendario:** 24–27 de mayo

## Objetivos

Implementar la arquitectura completa en PyTorch. La arquitectura es **Transformer Encoder-only** (BERT-like), no Decoder.

## Justificación arquitectónica

Encoder porque queremos comprensión y clasificación, no generación. La salida es una distribución sobre `{WIN_A, DRAW, WIN_B}`, no una secuencia generada autoregresivamente. Esto sigue la lógica de BERT (Devlin et al., 2018) más que GPT.

## Configuración tentativa (Modelo Base)

| Hiperparámetro | Valor inicial |
|---|---|
| Vocabulario | ~10,000 tokens |
| Hidden size $d$ | 256 |
| FFN intermediate size | 1024 (4×d) |
| Capas | 6 |
| Attention heads | 8 |
| Dimensión por head $d_k$ | 32 ($d/h$) |
| Secuencia máxima | 512 tokens |
| Dropout | 0.1 |
| Parámetros totales | ~15M |

Si Colab Pro tolera, escalamos a 8 capas / 512 hidden = ~50M params para el run final.

## Embeddings compuestos

Esto es donde implementamos el insight central del proyecto:

```python
embedding(token, position, segment, time_context) =
    E_token[token] +
    E_position[position] +
    E_segment[segment] +     # lineup A, lineup B, context, etc.
    E_temporal[time_bucket]  # discretización temporal
```

Tres embeddings sumados, igual que BERT (token + position + segment), más uno extra (temporal) específico de nuestro dominio.

## Mecanismo de atención: la "fontanería"

Para que quede claro lo que implementamos, recordando la clase 4:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Con $d_k = 32$, el escalado por $\sqrt{32} \approx 5.66$ evita que el producto $QK^\top$ tome valores muy grandes que saturen el softmax y desvanezcan el gradiente (esto es el punto central del paper "Attention is All You Need", sección 3.2.1).

Multi-head: usamos $h = 8$ cabezas en paralelo, cada una con su propia proyección de $Q, K, V$. La idea es que cada cabeza pueda especializarse en un tipo de relación (sinergia ofensiva, matchup defensivo, contexto temporal, etc.).

## Cabezas de salida (multi-task)

```python
output_logits = TransformerEncoder(input_tokens)  # [batch, seq, d]
cls_repr = output_logits[:, 0, :]                 # [batch, d] — repr del [CLS]

# Cabeza 1: clasificación de resultado
result_logits = Linear(d, 3)(cls_repr)            # W/D/L

# Cabeza 2: regresión de goles
goals_pred = Linear(d, 2)(cls_repr)               # (goals_A, goals_B)

# Cabeza 3: MLM auxiliar (regularización)
mlm_logits = Linear(d, vocab_size)(output_logits) # [batch, seq, vocab]
```

## Función de pérdida combinada

$$\mathcal{L} = \mathcal{L}_{CE}^{result} + \lambda_1 \mathcal{L}_{MSE}^{goals} + \lambda_2 \mathcal{L}_{CE}^{MLM}$$

Valores iniciales: $\lambda_1 = 0.3$, $\lambda_2 = 0.2$. Estos hiperparámetros se ajustan en la fase de tuning.

## Entregables Fase 3

- `src/models/embeddings.py` — embeddings compuestos
- `src/models/transformer.py` — Encoder con multi-head attention
- `src/models/heads.py` — 3 cabezas multi-task
- Test que verifica que un forward pass produce shapes correctas
- Conteo total de parámetros documentado

---

# Fase 4 — Pipeline de Entrenamiento

**Duración estimada:** 4–5 días
**Días calendario:** 28 de mayo – 1 de junio

## Objetivos

Entrenar el modelo. Acá es donde se gasta el tiempo de GPU y donde más cosas pueden salir mal.

## Componentes

**4.1. Dataset y DataLoader (`src/training/dataset.py`)**

Clase `FootballMatchDataset(torch.utils.data.Dataset)` que:
- Carga partidos preprocesados
- Aplica tokenización
- Genera el target (W/D/L + score)
- Aplica masking estocástico (ver 4.2)

**4.2. Masking estocástico de features (`src/training/masking.py`)**

Acá implementamos el insight clave de que el modelo debe degradar elegantemente con info parcial:

```python
def stochastic_mask(tokens, p_mask_uniform=(0.0, 0.8)):
    p = uniform(*p_mask_uniform)  # cada batch, p distinto
    for field in [LINEUP_A, LINEUP_B, BENCH_A, BENCH_B, CONTEXT, FEATURES]:
        if random() < 0.3:  # 30% probability de enmascarar el campo entero
            mask_field(tokens, field, with=[MASK])
    return tokens
```

La intuición: durante entrenamiento, el modelo a veces ve TODA la info, a veces solo equipos, a veces solo lineup sin contexto, etc. Esto fuerza robustez en inferencia.

**4.3. Training loop (`src/training/train.py`)**

- Optimizer: AdamW con weight_decay=0.01
- Learning rate: 5e-4 con warmup lineal (10% steps) + cosine decay
- Mixed precision (autocast + GradScaler) para que entre cómodo en T4
- Gradient clipping a norm=1.0
- Batch size: empezar en 32, ajustar según VRAM
- Epochs: empezar con 20, early stopping en val log-loss

**4.4. Split temporal**

```python
train_data: partidos < 2023-07-01
val_data:   partidos 2023-07-01 a 2024-07-01
test_data:  partidos > 2024-07-01
```

El Mundial 2026 es **out-of-distribution** estricto — predecir un partido nunca visto, en una competencia que el modelo ve solo en patrones similares (Mundiales 2018/2022). Este es el régimen difícil pero realista.

## Riesgos conocidos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Overfitting (modelo memoriza partidos) | MLM auxiliar, dropout, masking estocástico, weight decay |
| Colab desconecta a la 4ta hora | Checkpoints cada N steps a Google Drive |
| Loss explota (NaN) | Gradient clipping, mixed precision con cuidado, monitoreo continuo |
| Métricas no mejoran sobre baseline | Esto es un resultado válido — reportar honestamente, analizar por qué |

## Entregables Fase 4

- `src/training/train.py` ejecutable end-to-end
- Notebook `03_transformer_training.ipynb` con resultados
- Checkpoints versionados en Google Drive
- Curvas de entrenamiento (loss train/val, métricas)
- Run de W&B (opcional pero recomendado para reproducibilidad)

---

# Fase 5 — Evaluación y Calibración

**Duración estimada:** 2–3 días
**Días calendario:** 2–4 de junio

## Objetivos

Responder con honestidad académica: **¿el Transformer es mejor que el baseline?** Y si lo es, ¿en qué condiciones?

## Análisis a producir

**5.1. Tabla comparativa de métricas**

| Modelo | Log-Loss | Brier | ECE | Accuracy (ref) |
|---|---|---|---|---|
| Logistic Regression | ? | ? | ? | ? |
| XGBoost | ? | ? | ? | ? |
| LightGBM | ? | ? | ? | ? |
| Transformer (ours) | ? | ? | ? | ? |

**5.2. Análisis de calibración**

Reliability diagrams comparativos. Si el modelo dice 70% de probabilidad de victoria, ¿en qué fracción de esos casos efectivamente gana?

Si la calibración es mala pero el ranking es bueno, aplicar **Platt scaling** o **temperature scaling** post-hoc. Recomiendo temperature scaling porque es más simple y funciona bien.

**5.3. Análisis estratificado**

¿Dónde el modelo gana y pierde frente al baseline?

- Por confederación (UEFA vs CONMEBOL vs CAF...)
- Por fase del torneo (grupos vs. eliminatoria)
- Por diferencia de ELO (partidos parejos vs. desbalanceados)
- Por cantidad de info disponible (full vs. solo equipos)

Este análisis es lo que distingue un proyecto de posgrado de un proyecto de TP.

**5.4. Ablation studies**

¿Qué componente del modelo contribuye más?

- Sin MLM auxiliar
- Sin masking estocástico
- Sin embeddings temporales
- Sin contexto reciente

Cada uno entrenado en condiciones idénticas, comparado en validation.

## Entregables Fase 5

- Notebook con tabla comparativa, reliability diagrams, análisis estratificado
- Modelo calibrado guardado para usar en simulación
- Resumen de hallazgos para incluir en el reporte final

---

# Fase 6 — Simulación Monte Carlo del Mundial 2026

**Duración estimada:** 2 días
**Días calendario:** 5–6 de junio

## Objetivos

Generar predicciones del Mundial 2026 completo: probabilidad de campeón, de cada selección de pasar de fase, brackets más probables.

## Estructura del Mundial 2026

48 selecciones, formato nuevo:
- Fase de grupos: 12 grupos de 4
- Pasan los 2 primeros + 8 mejores terceros = 32 a R16
- Eliminatoria estándar desde R16

Esto es nuevo respecto a 2018/2022, **importante a programar bien**.

## Algoritmo

```python
def simulate_world_cup(model, bracket_2026, n_sims=10000, info_t):
    results = []
    for _ in range(n_sims):
        groups = simulate_groups(model, bracket_2026, info_t)
        r16 = compute_qualified(groups)
        winners_r16 = [sample(model.predict(a, b, info_t)) for a, b in r16_pairings(r16)]
        # ... propagación hasta final
        champion = winners_final
        results.append(champion)
    return Counter(results)
```

**CRÍTICO:** `sample(probs)`, no `argmax(probs)`. Discutimos por qué — argmax colapsa la incertidumbre y subestima outcomes sorpresa.

## Análisis de sensibilidad

Variar inputs y ver cómo cambian las predicciones:
- Sacar a un jugador clave (lesión de Messi, Mbappé, etc.)
- Cambiar formaciones
- Comparar predicciones con y sin contexto reciente

Esto demuestra que el modelo **responde** a la información que le metés, no que predice ciegamente.

## Entregables Fase 6

- `src/simulation/tournament.py`
- Notebook `04_tournament_simulation.ipynb`
- **Output principal:** tabla de probabilidades por selección de:
  - Ganar el Mundial
  - Llegar a la final
  - Llegar a SF
  - Pasar de grupos
- Comparación con casas de apuestas (sanity check)
- Análisis de sensibilidad

---

# Fase 7 — Pipeline Live de Actualización

**Duración estimada:** 2 días + ejecución durante todo el Mundial
**Días calendario:** 7–10 de junio (build), luego live durante Mundial

## Objetivos

Sistema que permita re-predecir cada partido del Mundial conforme se va sabiendo más información (alineaciones confirmadas, lesiones, etc.).

## Componentes

**7.1. Script de actualización diaria**

Cada día del Mundial:
1. Cargar resultados de partidos previos
2. Actualizar features rolling (forma, goles_T, etc.)
3. Re-predecir partidos restantes
4. Logear probabilidades en histórico

**7.2. Tracking de probabilidades**

Para cada partido, graficar la evolución de $P(\text{result})$ desde 1 mes antes hasta el kickoff. Esperamos ver:
- Entropía alta lejos del partido
- Entropía decreciente conforme se acerca
- Cambios bruscos en eventos clave (lesión confirmada, lineup oficial)

Esto valida empíricamente la hipótesis del Mecanismo B de la conversación previa.

**7.3. Output presentable**

Una página simple (markdown o HTML) que se actualiza con:
- Predicción de cada partido del próximo round
- Estado actual de probabilidades de campeón
- Cambios desde la última actualización

## Entregables Fase 7

- Script `src/simulation/live_update.py`
- Documento que se actualiza día a día con predicciones
- Al final del Mundial: análisis retrospectivo de calibración real

---

# Fase Final — Verificación, Tests, Documento Académico

**Duración estimada:** durante todo el proyecto, finalizar la última semana

## Componentes

**F.1. Tests unitarios**

- Tokenizer: round-trip (encode → decode == original)
- DataLoader: shapes correctas, masking funcional
- Modelo: forward pass produce salidas en rangos válidos (probabilidades suman 1)

**F.2. Reproducibilidad**

- `requirements.txt` con versiones exactas
- Seeds fijas en toda generación aleatoria
- README con instrucciones paso a paso
- Notebook que reproduce los resultados principales en < 1 hora

**F.3. Documento académico final (`reports/final_paper.md`)**

Estructura:
1. **Introducción** — motivación, hipótesis central, contribución
2. **Trabajo relacionado** — RAG, BERT, NLP para deportes (poco), modelos clásicos de predicción deportiva
3. **Metodología** — vocabulario, arquitectura, masking, entrenamiento
4. **Experimentos** — baselines, ablations, calibración
5. **Resultados** — tabla principal, análisis estratificado
6. **Discusión** — limitaciones, qué aprendimos, qué haríamos diferente
7. **Conclusión y trabajo futuro** — GNN, multimodal, etc.

Citas obligatorias: Devlin et al. (BERT), Vaswani et al. (Transformers), Lewis et al. (RAG), Jurafsky & Martin, Eisenstein.

---

# Cronograma Resumido

| Fase | Días | Fechas | Output crítico |
|---|---|---|---|
| 0. Setup + EDA | 2 | 16–17 may | Datos + entorno listos |
| 1. Baselines | 3 | 18–20 may | Sistema funcional para Mundial (seguro) |
| 2. Tokenización | 3 | 21–23 may | Vocabulario + tokenizer testeado |
| 3. Arquitectura | 4 | 24–27 may | Modelo en PyTorch, forward pass OK |
| 4. Entrenamiento | 5 | 28 may – 1 jun | Modelo entrenado con checkpoints |
| 5. Eval + calibración | 3 | 2–4 jun | Comparación rigurosa vs baselines |
| 6. Simulación Mundial | 2 | 5–6 jun | Predicción Mundial 2026 v1.0 |
| 7. Pipeline live | 2 | 7–10 jun | Sistema de updates |
| Mundial empieza | — | 11 jun | Live predictions activas |
| Documento final | continuo | hasta fin | Entregable académico |

**Total:** 24 días hábiles, llegamos justo al kickoff con margen de 1 día.

---

# Decisiones a tomar antes de empezar Fase 0

1. **¿Repositorio GitHub público o privado?** Privado mientras desarrollamos, decidir al final.
2. **¿Usamos W&B para tracking experimental?** Recomiendo sí, free tier suficiente.
3. **¿Idioma del código y comentarios?** Recomiendo inglés en código, español en documentación académica.
4. **¿Empezamos por StatsBomb (Fase 0) o ya quieren ver el código de algún componente específico?**

---

*Plan creado: 16 de mayo de 2026. Sujeto a revisión semanal.*
