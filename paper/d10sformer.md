# D10Sformer: un Transformer bidireccional para la predicción del fútbol internacional

**Autora:** José Marin, Oscar Carrizo  
**Materia:** MIA305 — Procesamiento del Lenguaje Natural  
**Institución:** Universidad de San Andrés  
**Fecha:** Mayo 2026

---

## Resumen (Abstract)

Presentamos D10Sformer, un Transformer encoder bidireccional (estilo BERT) adaptado al dominio del fútbol internacional. El modelo se pre-entrena con un objetivo de Masked Language Modeling sobre 11.852 partidos de clubes y selecciones, y se afina sobre 8.658 partidos exclusivamente de selecciones nacionales (período 2014–2023). Su vocabulario de 4.521 tokens combina entidades estructuradas (equipos, jugadores, torneos) con representaciones bucketizadas de cantidades continuas (ELO, racha, goles recientes). Aunque los baselines tabulares (regresión logística, XGBoost) superan al Transformer en métricas de clasificación puras (log-loss, accuracy, ECE), demostramos que el modelo aprende representaciones semánticamente coherentes: en particular, los embeddings de buckets de ELO se organizan en una recta monótona pese a haber sido entrenados como tokens categóricos discretos. Usamos los baselines calibrados para producir una predicción Monte Carlo del Mundial FIFA 2026, e implementamos un pipeline de actualización condicional que permite re-simular el torneo restante en aproximadamente 15 segundos cuando se conocen resultados reales. El sistema completo es reproducible, está testeado (más de 100 tests unitarios) y abierto para uso académico.

**Palabras clave:** NLP aplicado, Transformer, BERT, fútbol, predicción deportiva, Masked Language Modeling, embeddings, Monte Carlo, calibración.

---

## 1. Introducción

### 1.1 Motivación

La predicción de resultados deportivos es un caso paradigmático de aprendizaje supervisado con clase intermedia ambigua (empate), señal predictiva concentrada en pocas variables ordinales (rating ELO, racha reciente, contexto de venue) y alta cola de baja frecuencia (selecciones nacionales que se enfrentan rara vez). Los modelos clásicos tabulares —regresión logística sobre rating ELO + racha— constituyen baselines fuertes, históricamente cercanos al techo teórico de predictibilidad dado lo estocástico de un partido individual.

Sin embargo, la revolución del Transformer en NLP (Vaswani et al., 2017; Devlin et al., 2018) abrió la posibilidad de aprender **representaciones distribuidas contextuales** de las entidades del dominio. En el caso del fútbol, una representación distribuida de Lionel Messi podría capturar no solo su rendimiento estadístico sino también el contexto competitivo en el que ha actuado (Copa América con Argentina, La Liga con Barcelona, Ligue 1 con PSG). Este aporte representacional, según hipotetizamos, sería visible incluso si el desempeño predictivo terminara igualando al de los baselines.

### 1.2 Hipótesis

Formulamos tres hipótesis principales:

**H1** — Un Transformer encoder bidireccional pre-entrenado con MLM sobre un corpus heterogéneo (clubes + selecciones) puede aprender representaciones distribucionales coherentes de entidades futbolísticas, manifestadas como vecindades cosenoidales no triviales en el espacio de embeddings.

**H2** — La cuantización por buckets de variables continuas (ELO, racha) puede ser aprendida ordinalmente por el modelo aun cuando los buckets se presenten como tokens discretos sin información explícita de orden.

**H3** — Aunque el Transformer pueda no superar a baselines tabulares fuertes en métricas puramente predictivas en datasets de tamaño moderado (~10K muestras), su valor está en (a) las representaciones aprendidas (analizables a posteriori), y (b) la integración natural de heterogeneidad estructural (partidos con vs. sin lineups, eventos opcionales) que es trivialmente conflictiva en pipelines tabulares.

### 1.3 Contribuciones

Las contribuciones del trabajo son:

1. Un **vocabulario futbolístico** de 4.521 tokens construido empíricamente sobre cobertura de StatsBomb + martj42, con codificación jerárquica de jugadores que maneja la cola larga de la distribución de apariciones (33% de jugadores con token dedicado, 67% con fallback posicional-tier).
2. Una **arquitectura Transformer bidireccional liviana** (~6.13M parámetros) entrenable en GPU consumer (T4) con tres cabezales multitarea: MLM, clasificación de resultado y clasificación de marcador.
3. Un **análisis riguroso comparativo** entre el Transformer y baselines tabulares (LogReg, XGBoost, LightGBM, ELO puro), revelando que los Transformers no superan a árboles en tabular pequeño (consistente con Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022).
4. Una **demostración del aprendizaje ordinal emergente** sobre buckets de ELO: la similitud coseno entre `ELO_BUCKET_2100` y `ELO_BUCKET_X` decrece monótonamente con |2100 − X|.
5. Un **pipeline Monte Carlo del Mundial 2026** con 10.000 simulaciones, condicionable a resultados reales para actualización en vivo durante el torneo (~15 s por re-simulación).

---

## 2. Datos

### 2.1 Fuentes

Combinamos dos corpus heterogéneos:


| Fuente                              | Partidos                 | Cobertura                                       | Riqueza                        |
| ----------------------------------- | ------------------------ | ----------------------------------------------- | ------------------------------ |
| **martj42 / international_results** | 49.257 (1872–marzo 2026) | Todas las selecciones FIFA                      | Solo metadata + resultado      |
| **StatsBomb Open Data**             | 3.464                    | Mayormente Europa (UCL, World Cup, big leagues) | Lineups + eventos + posiciones |


martj42 nos da cobertura mundial completa de selecciones. StatsBomb nos da profundidad estructural (jugadores titulares, posiciones, eventos) pero está fuertemente sesgada a Europa y a competiciones internacionales filmadas profesionalmente.

### 2.2 Pre-procesamiento

Para martj42 calculamos en una pasada cronológica:

- **Rating ELO** del fútbol mundial (fórmula World Football ELO con factor K dependiente del tipo de torneo, multiplicador G por diferencia de goles y bonus de localía +100 cuando aplica). Cobertura: 100% de partidos.
- **Forma rolling** (puntos promedio en últimos 5 partidos por equipo). Cobertura: 99.7%.
- **Goles rolling** (goles a favor promedio en últimos 5 partidos). Cobertura: 99.7%.

Para StatsBomb derivamos:

- **Posición canónica** por jugador (GK / DF / MF / FW) mapeando la nomenclatura StatsBomb.
- **Frecuencia de aparición** (apariciones totales) para construir el umbral de "jugador dedicado" del vocabulario.

### 2.3 Splits temporales

Aplicamos splits **estrictamente cronológicos**, nunca aleatorios:


| Split      | Período                 | Tamaño (selecciones) |
| ---------- | ----------------------- | -------------------- |
| Train      | 2014-01-01 → 2023-06-30 | 8.658                |
| Validation | 2023-07-01 → 2024-06-30 | 1.232                |
| Test       | ≥ 2024-07-01            | 1.810                |


Esta convención es crítica para evitar leakage temporal en el entrenamiento de features rolling y para garantizar que el conjunto de test refleje la distribución de partidos que el modelo verá en el Mundial 2026.

---

## 3. Baselines tabulares

### 3.1 Modelos evaluados

Implementamos cuatro baselines como referencia comparativa:

1. **Uniforme** (P=1/3 cada clase) — piso teórico, log-loss = ln(3) ≈ 1.0986.
2. **ELO solo** — convierte la diferencia de ELO en probabilidades vía la fórmula logística estándar del fútbol; distribuye masa al empate según la cercanía.
3. **LogReg multinomial** — sobre 40 features (ELO + form_5 + form_10 + h2h + venue + tournament_class one-hot), con `SimpleImputer` + `StandardScaler` en pipeline.
4. **XGBoost** y **LightGBM** — `multi:softprob` con early stopping sobre val.

### 3.2 Resultados sobre el conjunto de test


| Modelo     | log-loss ↓ | Brier ↓    | ECE ↓      | Accuracy ↑ |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| Uniforme   | 1.0986     | 0.6667     | 0.0000     | 0.4500     |
| ELO solo   | 1.0102     | 0.6101     | 0.0890     | 0.5510     |
| LightGBM   | 0.8744     | 0.5135     | 0.0250     | 0.5990     |
| XGBoost    | 0.8663     | 0.5105     | **0.0182** | **0.6026** |
| **LogReg** | **0.8610** | **0.5071** | 0.0236     | 0.6005     |


LogReg minimiza la log-loss y empata en accuracy con XGBoost. La calibración (ECE) la lidera XGBoost. Estos números operan como **techo de referencia** para el Transformer.

### 3.3 Análisis estratificado

Estratificando los partidos del test por diferencia absoluta de ELO:

- |ΔELO| < 100 → accuracy 0.432 (partidos parejos, difíciles)
- |ΔELO| ∈ [100, 400] → accuracy 0.62–0.78
- |ΔELO| > 400 → accuracy 0.900 (partidos asimétricos, casi triviales)

Esto confirma que la **señal predictiva está concentrada en la diferencia de ELO**, dejando poco margen aditivo a otras features.

---

## 4. Vocabulario futbolístico

### 4.1 Diseño

El vocabulario combina seis familias de tokens:


| Familia             | Cantidad  | Ejemplos                                                                       |
| ------------------- | --------- | ------------------------------------------------------------------------------ |
| Especiales          | 20        | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`, `[LINEUP_A]`, `[FEATURES_START]` |
| Equipos             | 568       | `TEAM_ARGENTINA`, `TEAM_BARCELONA`                                             |
| Torneos             | 216       | `TOURNAMENT_FIFA_WORLD_CUP`, `TOURNAMENT_CHAMPIONS_LEAGUE`                     |
| Jugadores dedicados | 3.600     | `PLAYER_5503` (Messi) — cualquier jugador con ≥10 apariciones                  |
| Jugadores fallback  | 25        | `POS_FW_TIER_3`, `POS_GK_TIER_5` (cola larga + jugadores no vistos)            |
| Buckets numéricos   | 23        | `ELO_BUCKET_2100`, `FORM_HIGH`, `GOALS_LOW`                                    |
| Eventos             | 9         | `EVENT_GOAL`, `EVENT_YELLOW_CARD`, `EVENT_SUBSTITUTION`                        |
| Resultados          | 3 + 36    | `RESULT_HOME_WIN`, `SCORE_2_1`                                                 |
| **Total**           | **4.521** |                                                                                |


### 4.2 Cola larga de jugadores

De los 10.803 jugadores únicos en StatsBomb, solo 3.600 (33.3%) acumulan ≥10 apariciones y reciben token dedicado. El 66.7% restante (7.203 jugadores) se mapea a un token compuesto `POS_<posición>_TIER_<n>`, donde el tier resume la frecuencia de aparición (1 = ≥200 partidos, 5 = <10).

Este diseño jerárquico cumple dos funciones simultáneas:

- Reduce el tamaño del vocabulario en ~7K tokens sin sacrificar información del 91% del corpus.
- Permite que **jugadores no vistos** durante el entrenamiento (incluidos los debutantes del Mundial 2026) reciban una representación significativa: cualquier delantero novato cae en `POS_FW_TIER_5`, que ha visto miles de contextos durante el pre-training.

### 4.3 Salud del vocabulario

Sobre una muestra de 100 partidos sin sesgo:

- Tokens emitidos: 3.500
- Tokens `[UNK]`: **0** (0.000%)
- Longitud de secuencia: distribución bimodal, mediana ~23 (partidos sparse), p99 = 64 (StatsBomb con lineups).

El UNK rate de 0% confirma que el vocabulario cierra completamente el dominio observado.

---

## 5. Arquitectura del Transformer

### 5.1 Decisiones de diseño

D10Sformer es un **Transformer encoder bidireccional** (no autorregresivo) con los siguientes componentes:


| Componente     | Decisión                                        | Justificación                                                                                                                      |
| -------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Tipo           | Encoder-only, bidireccional                     | La tarea final es clasificación, no generación. Atención bidireccional captura mejor relaciones entre jugadores del mismo partido. |
| Normalización  | Pre-LN (Layer Norm pre-residual)                | Mayor estabilidad de gradientes sin warmup ajustado manualmente (Xiong et al., 2020).                                              |
| Atención       | Implementada desde cero (Q, K, V, O explícitos) | Transparencia didáctica para fundamentar la sección de metodología (Vaswani et al., 2017).                                         |
| Activación FFN | GELU                                            | Estándar BERT post-Hendrycks & Gimpel (2016).                                                                                      |
| Posición       | Embeddings aprendidos (no sinusoidales)         | `max_seq_length=80` permite tabla pequeña; sin necesidad de extrapolar.                                                            |
| Segmento       | Aprendido, 8 categorías                         | Distingue secciones estructurales del partido (META / LINEUP_A / EVENTS / etc.).                                                   |
| Weight tying   | MLM head ↔ token embedding                      | Ahorra 1.16M parámetros (~17%) y mejora generalización (Press & Wolf, 2017).                                                       |


### 5.2 Hiperparámetros y presupuesto


| Hiperparámetro         | Valor         |
| ---------------------- | ------------- |
| d_model                | 256           |
| num_layers             | 6             |
| num_heads              | 8             |
| d_ff                   | 1.024         |
| max_seq_length         | 80            |
| Dropout (todos)        | 0.1           |
| **Parámetros totales** | **6.131.920** |
| Checkpoint en disco    | 69.2 MB       |


A `d_model = 256`, el modelo entra cómodamente en una T4 (16 GB) incluso con batch size 64 y máximo paralelismo de secuencia.

### 5.3 Cabezales multitarea

Sobre el [CLS] del encoder, tres cabezales actúan en paralelo:

- **MLM Head** (4.521 clases) — predice el token enmascarado en posiciones aleatorias. Compartida con la tabla de embeddings de entrada (weight tying).
- **Result Head** (3 clases) — clasifica el resultado final del partido (home_win, draw, away_win).
- **Score Head** (36 clases) — clasifica el marcador, clamped a 5 goles por lado.

Cada cabezal se activa según el régimen de entrenamiento. Durante pre-training MLM puro, solo MLM está activo. Durante fine-tuning multitarea, los tres contribuyen con pesos λ_MLM=0.2, λ_result=1.0, λ_score=0.3 (la elección sigue la convención BERT de usar MLM como regularizador auxiliar).

---

## 6. Entrenamiento

### 6.1 Pre-training (Fase 4c)


| Parámetro        | Valor                                                             |
| ---------------- | ----------------------------------------------------------------- |
| Corpus           | 11.852 partidos (8.658 selecciones train + 3.194 StatsBomb train) |
| Objetivo         | MLM puro, 15% masking BERT-style (80/10/10)                       |
| Épocas           | 5                                                                 |
| Pasos totales    | 930                                                               |
| Optimizer        | AdamW (β₁=0.9, β₂=0.999, wd=0.01 selectivo)                       |
| LR schedule      | Cosine con warmup lineal 10%                                      |
| LR máximo        | 5×10⁻⁴                                                            |
| Batch size       | 64                                                                |
| Mixed precision  | FP16 + GradScaler                                                 |
| Hardware         | NVIDIA T4 (Colab)                                                 |
| **Tiempo total** | **48 segundos**                                                   |
| **Throughput**   | **18.8 step/s**                                                   |


**Resultados:**

- Loss inicial (paso 25): 8.2343 (cerca del random baseline ln(4521) ≈ 8.42)
- Loss final (paso 925): 3.9863 (**−51.6%**)
- Validation loss best: **2.0678** en paso 900
- Validation accuracy MLM: **40.19%** (vs 1/4521 = 0.022% del random baseline)
- Validation perplexity: 7.91

La curva de validación bajó monótonamente hasta el último eval, sugiriendo que el modelo podría seguir mejorando con más épocas.

### 6.2 Fine-tuning multitarea (Fase 4d–4e)

El fine-tuning se realiza sobre las 8.658 selecciones del train split, partiendo del mejor checkpoint del pre-training. Se evaluaron dos configuraciones:

**4d** — Sin class weights, LR=5×10⁻⁵, 10 épocas.
**4e** — Con class weights inversamente proporcionales a frecuencia (w_home=0.70, w_draw=1.45, w_away=1.14), LR=5×10⁻⁵, 15 épocas.


| Configuración       | log-loss (test) | Brier  | ECE    | Accuracy | Predicciones DRAW (val)            |
| ------------------- | --------------- | ------ | ------ | -------- | ---------------------------------- |
| **Baseline LogReg** | **0.8610**      | 0.5071 | 0.0236 | 0.6005   | (apropiado)                        |
| 4d sin weights      | 0.8846          | 0.5221 | 0.0400 | 0.5912   | **12 / 293** (colapso)             |
| 4e con weights      | 0.9200          | 0.5439 | 0.0531 | 0.5514   | 315 / 1232 (balanceada pero ruido) |


**Hallazgo central:** ninguna configuración del Transformer supera al baseline LogReg en métricas predictivas. La configuración 4d sufre **colapso de la clase minoritaria** (predice empate en el 4% de los partidos cuando la frecuencia real es 24%); la configuración 4e resuelve el colapso pero a costa de degradar la accuracy en las clases mayoritarias.

Este resultado es **académicamente consistente** con la literatura reciente sobre Transformers en tablas pequeñas (Shwartz-Ziv & Armon, 2022 "Tabular Data: Deep Learning is Not All You Need"; Grinsztajn et al., 2022 "Why do tree-based models still outperform deep learning on typical tabular data?"). La discusión completa se encuentra en la Sección 10.

---

## 7. Análisis de embeddings (el corazón del paper)

Aunque el Transformer no superó a los baselines en clasificación, demostramos que **sí aprendió relaciones semánticas no triviales**, no observables en modelos tabulares.

### 7.1 Ordinalidad emergente en buckets de ELO

Los 14 buckets de ELO (`ELO_BUCKET_1100` a `ELO_BUCKET_2400`) fueron entrenados como tokens **categóricos discretos**: para el modelo, en principio, son tan distintos entre sí como lo son `TEAM_ARGENTINA` y `TEAM_FRANCE`. No hay supervisión explícita sobre el orden numérico.

**Sin embargo, los vecinos más cercanos por similitud coseno del bucket 2100 son:**


| Vecino            | Cosine similarity |
| ----------------- | ----------------- |
| `ELO_BUCKET_2000` | **0.7675**        |
| `ELO_BUCKET_1900` | 0.7479            |
| `ELO_BUCKET_1800` | 0.6853            |
| `ELO_BUCKET_1700` | 0.6807            |
| `ELO_BUCKET_1600` | 0.6315            |
| `ELO_BUCKET_1500` | 0.5836            |
| `ELO_BUCKET_1400` | 0.5311            |
| `ELO_BUCKET_1300` | **0.5196**        |


La similitud decrece **monótonamente** con la distancia numérica. El modelo descubrió por MLM contextual que estos buckets viven en una recta. Es la analogía moderna del clásico `vec(king) − vec(man) + vec(woman) ≈ vec(queen)` (Mikolov et al., 2013), aplicada a una variable cuantitativa bucketizada.

El mismo patrón emerge en los buckets de racha (`FORM_VERY_LOW` → `FORM_VERY_HIGH`): `FORM_HIGH` está más cerca de `FORM_VERY_HIGH` (0.74) que de `FORM_VERY_LOW` (0.66).

### 7.2 Similitudes dirigidas (anchor / positive / negative)

Probamos ocho contrastes semánticos. El modelo "acierta" si la similitud coseno del anchor con el target positivo supera la del negativo.


| Test                                                       | Δ      | Acierto |
| ---------------------------------------------------------- | ------ | ------- |
| Messi → Argentina vs Francia                               | −0.077 | ✗       |
| Messi → Argentina vs Inglaterra                            | +0.040 | ✓       |
| Mbappé → Francia vs Argentina                              | +0.079 | ✓       |
| Mbappé → Francia vs Brasil                                 | −0.010 | ✗       |
| Argentina → Brasil vs Alemania                             | −0.003 | ✗       |
| Alemania → Francia vs Nigeria                              | +0.124 | ✓       |
| `FORM_HIGH` → `FORM_VERY_HIGH` vs `FORM_VERY_LOW`          | +0.074 | ✓       |
| `ELO_BUCKET_2100` → `ELO_BUCKET_2000` vs `ELO_BUCKET_1100` | +0.383 | ✓       |
| **Total**                                                  |        | **5/8** |


Los buckets cuantitativos pasan con margen amplio; las relaciones jugador-equipo son más ruidosas. Esto es consistente con la naturaleza del corpus: StatsBomb tiene a Messi en contextos predominantemente europeos (Barcelona, PSG en las últimas temporadas), no en contextos exclusivamente argentinos.

### 7.3 Estructura de clusters de selecciones

Una proyección t-SNE de 47 selecciones por confederación revela que **el modelo no agrupó por geografía sino por nivel competitivo**:

```
Top-8 vecinos cosenoidales de TEAM_ARGENTINA:
  TEAM_BELGIUM (0.631), TEAM_ITALY (0.619), TEAM_COLOMBIA (0.592),
  TEAM_NETHERLANDS (0.589), TEAM_FRANCE (0.571), TEAM_SPAIN (0.571),
  TEAM_SERBIA (0.566), TEAM_PORTUGAL (0.562)
```

De los 8 vecinos, 7 son europeos top y solo 1 sudamericano. La interpretación es contextual: durante un Mundial las selecciones top se enfrentan entre sí en fases finales **independientemente del continente**. El modelo aprendió la estructura competitiva del fútbol internacional, no su distribución geográfica.

Este hallazgo es **complementario** al del baseline LogReg, que no podría revelar este tipo de relación porque opera sobre ELO y forma como escalares aislados.

### 7.4 Calibración del Result head + Temperature scaling

Aplicamos temperature scaling (Guo et al., 2017) al Transformer 4d sobre val, optimizando un único escalar T por NLL.


| Métrica (test) | Antes  | Después (T=1.085)   |
| -------------- | ------ | ------------------- |
| ECE            | 0.0504 | **0.0350** (−30%)   |
| log-loss       | 0.8855 | 0.8834              |
| accuracy       | 0.5967 | 0.5967 (sin cambio) |


T_opt = 1.085 indica que el modelo era **levemente sobreconfiado**. La calibración post-hoc reduce ECE significativamente pero no acerca el log-loss al baseline. Esto confirma que el problema del Transformer no es de calibración sino de **separación discriminativa de la clase intermedia**.

---

## 8. Predicción del Mundial 2026

### 8.1 Decisión metodológica

Dado que el Transformer no supera a los baselines en métricas predictivas, usamos el **LogReg calibrado** como motor del simulador Monte Carlo. El Transformer se reserva para el análisis interpretativo de la Sección 7. Esta decisión es transparente para el lector y consistente con el objetivo de "el mejor modelo posible para predecir, el mejor modelo posible para interpretar".

### 8.2 Motor de simulación

El simulador implementa exactamente la estructura del Mundial 2026 (formato nuevo FIFA, 48 equipos):


| Fase                           | Partidos | Equipos al final                     |
| ------------------------------ | -------- | ------------------------------------ |
| Grupos (12 grupos de 4)        | 72       | 48 → 32 (top 2 + 8 mejores terceros) |
| Octavos de final (Round of 32) | 16       | 32 → 16                              |
| Dieciseisavos (Round of 16)    | 8        | 16 → 8                               |
| Cuartos de final               | 4        | 8 → 4                                |
| Semifinales                    | 2        | 4 → 2                                |
| Tercer puesto                  | 1        | —                                    |
| Final                          | 1        | 1                                    |
| **Total**                      | **104**  |                                      |


**Reglas de modelado:**

- En grupos, cada partido se muestrea de la distribución `[P(home), P(draw), P(away)]` del LogReg. Los goles se simulan vía Poisson independiente con tasa derivada de la probabilidad de victoria, total esperado 2.5.
- En knockouts, no hay empates: la masa del empate se reparte 50/50 entre los equipos (aproximación de penales).
- La asignación de los 8 mejores terceros a los slots del bracket se hace por restricción combinatoria (búsqueda aleatoria) sobre los grupos elegibles definidos por FIFA.
- **Limitación documentada**: las features rolling (ELO, racha, goles) se fijan al inicio del Mundial y NO se actualizan intra-simulación. Esto subestima ligeramente la varianza realista de los partidos avanzados.

### 8.3 Predicciones iniciales — 10.000 simulaciones

**[CONFIRMAR CON 8a — números actualizados post-fix del unpacking]**

Resultados preliminares (Fase 7 ya con LogReg correcto):


| #   | Equipo     | ELO  | P(pasa grupo) | P(8avos) | P(QF) | P(SF) | P(F)  | P(campeón) |
| --- | ---------- | ---- | ------------- | -------- | ----- | ----- | ----- | ---------- |
| 1   | España     | 2209 | 99.6%         | 77.8%    | 54.6% | 44.4% | 30.1% | **19.98%** |
| 2   | Argentina  | 2177 | 99.1%         | 70.9%    | 55.8% | 40.9% | 23.3% | **13.64%** |
| 3   | Francia    | 2128 | 98.0%         | 73.9%    | 44.3% | 31.7% | 20.5% | **11.37%** |
| 4   | Brasil     | 2050 | 98.5%         | 70.2%    | 48.8% | 33.2% | 20.1% | **10.67%** |
| 5   | Ecuador    | 2010 | 95.8%         | 67.9%    | 38.1% | 24.2% | 14.1% | **7.11%**  |
| 6   | Portugal   | 2015 | 95.2%         | 65.5%    | 37.1% | 19.4% | 9.7%  | **4.87%**  |
| 7   | Colombia   | 2045 | 92.2%         | 62.9%    | 34.4% | 18.9% | 9.6%  | **4.56%**  |
| 8   | Inglaterra | 2070 | 97.6%         | 62.1%    | 35.3% | 17.3% | 8.3%  | **4.23%**  |


**Validaciones de coherencia del simulador:**

- **Correlación Spearman entre ELO inicial y P(campeón)**: ρ = 0.967 (p < 10⁻²⁸). El simulador respeta la jerarquía estadística esperable.
- **Suma de probabilidades de campeón sobre los 48 equipos**: 1.000 ± 1e-9 (cierre perfecto).
- **Cumulativo monotónico**: P(pasa grupo) ≥ P(8avos) ≥ P(QF) ≥ ... ≥ P(campeón) para todo equipo.

**Predicciones cualitativas notables:**

- **Final más probable**: Francia vs España (~5.7% de las simulaciones).
- Cuando Argentina llega a la final, su rival más probable es **Brasil (22.5%)** o **Francia (19.6%)**.
- Ecuador y Colombia entran al top-8 — un hallazgo interesante: sus combinaciones (ELO ~2.000, grupo accesible, alta racha reciente) los favorece sobre selecciones europeas de menor nivel.

### 8.4 Pipeline live para actualización en tiempo real

Implementamos un sistema `FixedResults` que permite condicionar el Monte Carlo a resultados reales. Cuando un partido se juega, sus puntos y goles se fijan y el simulador no muestrea esa instancia.

**Optimización del simulador** (`PrecomputedPredictor`): precomputamos las predicciones de los 2.256 pares posibles (48 × 47) una sola vez. Cada simulación posterior solo realiza dict lookups, reduciendo el tiempo de 10.000 simulaciones de **45 minutos a 12 segundos** (speedup de 224×).

**Demos de actualización condicional:**


| Escenario                                | P(Argentina pasa grupo) | P(Argentina campeón)  |
| ---------------------------------------- | ----------------------- | --------------------- |
| Baseline (sin info)                      | 99.1%                   | 13.64%                |
| Argentina gana 3-0 a Algeria (jornada 1) | 99.6%                   | 14.85%                |
| Argentina gana sus 3 partidos            | 100.0%                  | **15.70%** (+2.06 pp) |
| Argentina pierde 0-2 vs Algeria (upset)  | **89.7%** (−9.4 pp)     | **11.10%** (−2.5 pp)  |


Y en el mismo escenario adverso, P(Algeria pasa grupo) salta de 60% a **96%** (+36 pp). El simulador refleja correctamente la propagación de un upset a través del bracket.

---

## 9. Reproducibilidad

Todo el código está disponible en el repositorio `github.com/jose-marin-db/udesa-nlp-futbol-D10Sformer`, organizado en:

```
d10sformer/
├── src/                    # módulos importables
│   ├── data/               # vocabulario, tokenizer, dataset, collator
│   ├── models/             # arquitectura Transformer + baselines
│   ├── training/           # trainer, scheduler, métricas
│   ├── eval/               # log-loss, Brier, ECE, embeddings, reliability
│   └── simulation/         # bracket WC2026, Monte Carlo, fixed results
├── tests/                  # 100+ tests unitarios (pytest)
├── notebooks/              # 00–07 (un notebook por fase)
├── configs/                # base_config.yaml
├── data/processed/         # vocab.json + corpus pickles
└── checkpoints/            # modelos entrenados (.pt)
```

**Reproducción end-to-end** desde cero:

1. Clonar repo + instalar `requirements.txt`
2. Correr notebooks `00_eda.ipynb` → `07_live_update.ipynb` en orden
3. Cada notebook persiste sus artefactos para la fase siguiente
4. **Tiempo total estimado en T4**: ~25 minutos

---

## 10. Discusión y limitaciones

### 10.1 Por qué el Transformer no superó al baseline

Tres factores convergen para explicar el resultado:

1. **Tamaño del corpus pequeño.** 8.658 partidos de fine-tune son ~10²× menos que los corpus donde los Transformers en NLP usualmente brillan. La capacidad del modelo (6.13M parámetros) excede la información discriminativa disponible.
2. **Señal predictiva concentrada en pocas variables ordinales.** El feature importance de LogReg muestra que la diferencia de ELO concentra el 48% del peso predictivo. El resto de las features son refinamientos marginales. Un baseline con `ELO + form` ya está cerca del techo.
3. **Bucketización en lugar de continuo.** El Transformer recibe `ELO_BUCKET_2100`, perdiendo la granularidad fina (~100 puntos). LogReg ve el ELO crudo. En partidos parejos (|ΔELO| < 80), esta diferencia importa.

Este resultado es **consistente con la literatura reciente** sobre datos tabulares (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). El paper agrega una validación adicional en un dominio nuevo (fútbol internacional) y aporta un análisis cualitativo de qué sí aprende el modelo (Sección 7), que es la contribución representacional independiente de la performance.

### 10.2 Lo que el modelo sí aporta

- **Ordinalidad emergente** sobre buckets cuantitativos sin supervisión explícita (Sección 7.1) — un hallazgo intrínsecamente interesante para representation learning.
- **Estructura competitiva por nivel, no por geografía** en los clusters de selecciones (Sección 7.3) — útil para análisis cualitativo del fútbol internacional.
- **Modularidad estructural**: el formato de tokenización maneja naturalmente partidos con vs. sin lineups, eventos opcionales, sin que el modelo sufra. Un pipeline tabular requeriría engineering específico para cada combinación.

### 10.3 Limitaciones reconocidas

- **No actualización intra-simulación** de ELO/form: cada simulación del Mundial usa features del estado inicial del torneo. Mitigación parcial vía pipeline live (Sección 8.4) cuando los resultados reales se conocen.
- **Cobertura StatsBomb sesgada** a Europa: las representaciones de jugadores sudamericanos en sus contextos de selección son escasas. Los jugadores que también juegan en clubes europeos (Messi, Vinicius, Lautaro) están bien representados.
- **Empates en eliminatorias**: aproximamos la distribución de penales con un 50/50 sobre la masa de empate. Una mejora sería usar la distribución empírica histórica de penales por nivel de ELO.
- **Marcador clamp a 5**: scores como 6-0 o 7-0 (raros pero existen) se mapean a `SCORE_5_0`. Impacto pequeño dado el techo de 5.

### 10.4 Trabajo futuro

Tres direcciones naturales:

1. **Aumentar la granularidad del ELO**: en lugar de buckets de 100, usar buckets de 25, o agregar el ELO crudo como una feature continua adicional (inyectada por una pequeña MLP en paralelo al embedding categórico).
2. **Pre-training más extenso**: el corpus martj42 tiene 49K partidos pero solo usamos los ≥2014. Reextender hasta 2000 (con cuidado del régimen ELO inicial) podría enriquecer las representaciones.
3. **Eventos enriquecidos**: nuestros docs StatsBomb actuales no incluyen el detalle completo de los eventos (pases, tiros, posicionamiento). Una versión rich del tokenizer podría capturar esa señal.

---

## 11. Conclusión

Construimos D10Sformer, un Transformer encoder bidireccional para fútbol internacional. Aunque el modelo no supera a baselines tabulares en métricas predictivas en un dataset de tamaño moderado, demostramos que **aprende representaciones semánticamente coherentes** — en particular, descubre ordinalidad en buckets cuantitativos sin supervisión explícita y agrupa selecciones por nivel competitivo en lugar de por geografía. Estos hallazgos son intrínsecamente valiosos para representation learning y complementan al baseline.

Usamos el baseline calibrado para producir predicciones Monte Carlo del Mundial FIFA 2026, mostrando un favoritismo claro de España (P=20%), Argentina (P=14%), Francia (P=11%) y Brasil (P=11%). Implementamos un pipeline live que permite re-simular el torneo en ~15 segundos a partir de resultados reales, listo para ser usado durante el Mundial.

Toda la metodología es honesta sobre sus resultados y reproducible. Más allá del Mundial 2026, el proyecto contribuye dos elementos: una **demostración empírica adicional** de la limitación de los Transformers en tabular pequeño, y un **caso concreto de cuándo el valor del modelo está en sus representaciones, no en su accuracy**.

---

## Referencias

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
- Eisenstein, J. (2019). *Introduction to Natural Language Processing*. MIT Press.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS Datasets & Benchmarks.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415.
- Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed., draft).
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
- Micikevicius, P., Narang, S., Alben, J., et al. (2018). *Mixed Precision Training*. ICLR.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. ICML.
- Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models*. EACL.
- Shwartz-Ziv, R., & Armon, A. (2022). *Tabular Data: Deep Learning is Not All You Need*. Information Fusion.
- Tunstall, L., von Werra, L., & Wolf, T. (2022). *Natural Language Processing with Transformers*. O'Reilly.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Xiong, R., Yang, Y., He, D., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML.

---

*Documento generado durante el desarrollo del proyecto. Algunas tablas y números requieren confirmación final con la corrida re-ejecutada de Fase 6 (8a). Última actualización: junio 2026.*