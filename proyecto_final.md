````markdown
# CONTEXTO DEL PROYECTO — Temporal Relational Transformer para Predicción de Fútbol

## Overview del Proyecto

Estamos diseñando un sistema de IA/ML capaz de predecir resultados de partidos de fútbol utilizando:
- datos históricos estructurados,
- evolución temporal de jugadores,
- embeddings relacionales contextuales,
- secuencias de eventos futbolísticos.

Este NO es un proyecto clásico de NLP sobre lenguaje natural.

La idea central es modelar el fútbol como:
- un problema secuencial temporal,
- un sistema relacional entre entidades,
- y un problema de embeddings contextuales dinámicos.

La arquitectura probablemente combinará:
- Transformers Encoder,
- embeddings dinámicos,
- modelado temporal,
- potencialmente Graph Neural Networks en futuras versiones.

El objetivo inicial es construir un MVP capaz de superar modelos estadísticos tradicionales de predicción deportiva.

---

# Hipótesis Central

El resultado de un partido depende más de:
- relaciones contextuales entre jugadores,
- evolución temporal individual,
- sinergias tácticas,
- estado de forma,
- composición del plantel,
- historial reciente,

que del nombre histórico del equipo.

Ejemplos:
- Messi 2012 ≠ Messi 2025
- Argentina 2014 ≠ Argentina 2022

Por lo tanto:
- los embeddings deben ser temporales y contextuales,
- NO embeddings estáticos.

---

# Objetivo Conceptual Principal

Queremos que el modelo aprenda:

```text
estado(jugador, tiempo, contexto)
````

en lugar de:

```text
identidad_estatica(jugador)
```

---

# Dirección Técnica Inicial

## Tipo de Datos

NO vamos a utilizar texto natural.

Vamos a trabajar con:

* eventos estructurados,
* secuencias tokenizadas,
* representación semántica de partidos.

Ejemplo:

```text
MATCH
DATE: 2025-06-10

TEAM_A: Argentina
TEAM_B: Brasil

LINEUP_A:
Messi
Julian_Alvarez
DePaul

LINEUP_B:
Vinicius
Rodrygo

EVENT:
MIN_23 GOAL Argentina Messi ASSIST Julian_Alvarez

EVENT:
MIN_51 YELLOW Brazil Casemiro

RESULT:
2-1
```

---

# Filosofía Arquitectónica Importante

El “lenguaje” del fútbol NO es lenguaje humano.

El lenguaje del sistema está compuesto por:

* jugadores,
* equipos,
* eventos,
* contexto temporal,
* relaciones tácticas.

Por eso:

* NO queremos simplemente hacer fine tuning de BERT,
* preferimos entrenar un Transformer propio sobre secuencias futbolísticas.

---

# Enfoque de Modelado

## Fase 1 — Dataset Estructurado

Construir un pipeline de preprocessing que transforme datasets de fútbol en secuencias tokenizadas.

Fuentes potenciales:

* StatsBomb Open Data
* Wyscout Public Dataset
* Kaggle Football Datasets

Datos mínimos necesarios:

* alineaciones,
* goles,
* asistencias,
* tarjetas,
* sustituciones,
* fechas,
* torneo,
* local/visitante,
* resultado.

Datos opcionales futuros:

* xG,
* tracking de jugadores,
* posiciones tácticas,
* métricas físicas.

---

# Fase 2 — Vocabulario Futbolístico

Diseñar un vocabulario propio.

Ejemplos de tokens:

```text
PLAYER_MESSI
TEAM_ARGENTINA
GOAL
YELLOW_CARD
HOME
AWAY
MIN_23
```

Tamaño estimado:

* 5k–20k tokens.

---

# Fase 3 — Embeddings

Necesitamos embeddings para:

* jugadores,
* equipos,
* eventos,
* torneos,
* estados temporales.

Posible composición conceptual:

```text
E_final = E_player + E_time + E_form + E_context
```

Importante:

* la identidad del jugador debe mantenerse,
* el contexto temporal modifica su influencia.

---

# Modelado Temporal

El tiempo es CRÍTICO.

El sistema debe aprender:

* envejecimiento de jugadores,
* evolución táctica,
* estado de forma,
* recencia,
* evolución del equipo.

Posibles mecanismos:

* temporal decay,
* rolling windows,
* temporal embeddings,
* secuencias cronológicas.

Ejemplos de features temporales:

* edad,
* partidos últimos 30 días,
* goles últimos 10 partidos,
* asistencias recientes,
* minutos recientes,
* fase de carrera.

NO queremos usar fechas crudas directamente.

Preferimos:

* features temporales procesadas/contextuales.

---

# Diseño del Transformer

Recomendación inicial:

* arquitectura Transformer Encoder,
* NO GPT autoregresivo.

Configuración tentativa:

* 4–8 capas,
* hidden size 256–512,
* 4–8 attention heads.

Tamaño estimado:

* 20M–80M parámetros.

Framework recomendado:

* PyTorch
* HuggingFace opcional.

---

# Objetivos Iniciales

## Pretraining

Posibles tareas:

* masked token prediction,
* predicción de próximo evento,
* predicción de resultado.

## Fine Tuning

Tareas finales:

* victoria/empate/derrota,
* score exacto,
* xG,
* simulación de torneos.

---

# Objetivo Relacional Temporal

El mecanismo de atención debe aprender:

* sinergias,
* rivalidades,
* matchups tácticos,
* impacto contextual.

Ejemplos:

* Messi + Julián Álvarez
* Vinicius vs Molina
* Otamendi + Romero

---

# Insight Principal

NO queremos:

* embeddings estáticos de jugadores.

Queremos:

* embeddings dinámicos contextuales.

Conceptualmente:

```text
Messi + contexto(2025)
```

debe comportarse distinto a:

```text
Messi + contexto(2012)
```

manteniendo la identidad base del jugador.

---

# Alcance Recomendado para MVP

## MVP v1

Debe incluir:

* eventos estructurados,
* features temporales,
* Transformer Encoder,
* clasificación de resultados.

NO incluir inicialmente:

* Graph Neural Networks,
* multimodal,
* tracking espacial,
* LLMs,
* NLP de texto natural.

Mantener el sistema simple inicialmente.

---

# Baselines

Antes de evaluar Transformers:
implementar modelos tradicionales.

Ejemplos:

* XGBoost
* LightGBM
* Logistic Regression

Features:

* ELO,
* forma reciente,
* goles,
* localía,
* ratings.

El Transformer debe superar esos baselines.

---

# Restricciones Computacionales

Entorno objetivo:

* Google Colab Pro,
* GPUs T4/L4.

Factibilidad:

* completamente entrenable en Colab.

Tiempo estimado:

* horas a pocos días.

El cuello de botella principal será:

* ingeniería del dataset,
* tokenización,
* representación.

NO la GPU.

---

# Roadmap Futuro

Versiones futuras podrían incorporar:

* Temporal Graph Neural Networks,
* Graph Transformers,
* grafos de interacción entre jugadores,
* modelado táctico,
* embeddings espaciales,
* sistemas multimodales.

Pero eso queda FUERA del alcance inicial.

---

# Qué Necesitamos Diseñar

Necesitamos ayuda para:

* arquitectura general,
* modelado de datos,
* estrategia de tokenización,
* estructura de inputs,
* diseño de embeddings,
* objetivos de entrenamiento,
* métricas,
* preprocessing,
* roadmap experimental,
* comparación contra baselines,
* implementación eficiente.

El objetivo es construir un sistema realista, escalable y orientado a research serio.

```
```
