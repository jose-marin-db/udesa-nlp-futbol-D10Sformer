# Speech de Presentación — D10SFormer
**Materia:** MIA305 · Universidad de San Andrés  
**Presentadores:** Oscar Carrizo & José Marin  
**Tiempo estimado:** ~12–15 minutos

---

## SLIDE 1 — Título

*(Al abrir la presentación)*

Buenas, somos Oscar y José. El proyecto que les vamos a presentar hoy se llama **D10SFormer**: un Transformer bidireccional para el modelado y predicción del fútbol internacional.

El nombre no es casualidad — D10S es como se le dice a Maradona en Argentina. Y Transformer porque, bueno, si estamos en MIA305, ya saben de qué se trata. La idea de fondo es exactamente esa: aplicar la arquitectura de los grandes modelos de lenguaje al fútbol. No como metáfora, sino literalmente.

---

## SLIDE 2 — Hipótesis Central

Todo el proyecto descansa sobre una analogía que queremos que se lleven de esta presentación:

> **El fútbol tiene la misma estructura que el lenguaje natural.**

¿Qué quiere decir esto? Cuando BERT procesa la palabra *"banco"* en la frase *"Messi fue al banco"*, produce un embedding muy distinto al que produce para *"banco"* en *"Messi bancó la parada"*. La misma palabra, contextos radicalmente distintos → representaciones distintas. Eso es la semántica distribucional contextual en acción.

Ahora traslademos eso al fútbol. Messi en el Barcelona de 2012 —con Xavi, Iniesta, Guardiola— es un Messi completamente diferente al de Inter Miami en 2026. Mismo jugador, distinto contexto táctico, distintos compañeros, distinta presión competitiva. La hipótesis de D10SFormer es que **ese contexto importa más que la identidad histórica del equipo**, y que un Transformer puede aprenderlo automáticamente desde los datos.

Dicho de otra forma: el resultado de un partido depende de las relaciones contextuales y temporales entre jugadores, no de quiénes son en abstracto.

---

## SLIDE 3 — Motivación y Contribuciones

*(Slide "Quién pudiera ser el Pulpo Paul")*

¿Qué asumimos concretamente?

Que los **eventos y jugadores dentro de un partido funcionan como palabras en un contexto táctico**. Si yo tengo la secuencia de formaciones, ratings ELO, lineups y resultados de 18.000 partidos internacionales, esa secuencia es mi corpus. Los jugadores son tokens. Los partidos son oraciones. Los torneos son documentos.

Las contribuciones principales son cuatro:

Primero, construimos un **corpus híbrido** combinando StatsBomb Open Data e international_results de martj42 — más de 18.000 partidos de fútbol internacional con distintos niveles de granularidad de datos.

Segundo, el modelo aprende de manera **autodidacta relaciones ordinales entre variables continuas** como el ELO, sin que nosotros le digamos explícitamente que el ELO 2200 es "mejor" que el 1900.

Tercero, usamos **embeddings compartidos por posición** para equipos o jugadores de baja frecuencia — una solución pragmática para el problema del vocabulario escaso.

Y cuarto, demostramos que el fútbol puede modelarse como un **problema de NLP secuencial y contextual** de principio a fin.

---

## SLIDE 4 — Datos ("¿Cuál es la formación?")

Antes de hablar de arquitectura, la pregunta obvia es: ¿con qué datos trabajamos?

*(Explicar el dataset: StatsBomb Open Data + international_results. Mencionar el desafío de la asimetría — algunos partidos tienen lineups completos, otros solo el resultado final, lo cual el modelo tiene que manejar elegantemente.)*

---

## SLIDE 5 — Fuentes

Las dos fuentes principales son **StatsBomb Open Data**, que provee datos granulares de partidos seleccionados, y el repositorio **international_results de martj42** en GitHub, que cubre décadas de resultados del fútbol internacional. La combinación de ambas nos da riqueza semántica donde está disponible y cobertura histórica amplia donde no.

---

## SLIDE 6 — Arquitectura D10SFormer

Pasemos al corazón técnico. La arquitectura tiene **tres pilares de diseño** optimizados para este dominio:

**Primero: Encoder-only Bidireccional — estilo BERT.**

Esto es una decisión de diseño fundamental. ¿Por qué bidireccional y no un decoder autorregressivo tipo GPT? Porque cuando predecimos sobre un partido, tenemos acceso al contexto *completo* de ese partido: ambos equipos, el torneo, la instancia. No estamos generando texto token a token; estamos clasificando. La atención bidireccional es mandatoria para este caso de uso.

**Segundo: Pre-Layer Normalization (Pre-LN).**

Esta es una mejora sobre la arquitectura Transformer original. En el paper "Attention is All You Need", la normalización va *después* de cada sub-capa (Post-LN). La investigación posterior mostró que Pre-LN — normalizar *antes* — estabiliza significativamente el entrenamiento, especialmente durante el pre-entrenamiento con Masked Language Modeling. Evita la inestabilidad de gradientes que puede aparecer en redes profundas.

**Tercero: Weight Tying en el MLM Head.**

Durante el pre-entrenamiento, la cabeza de predicción de tokens enmascarados comparte los pesos con la matriz de embeddings original. Esto reduce el número de parámetros y actúa como regularizador implícito — la idea es que la representación que el modelo aprende para un jugador en el espacio de embeddings debe ser consistente con la que usa para predecirlo cuando está enmascarado.

---

## SLIDE 7 — Entrenamiento en Dos Etapas

El entrenamiento sigue el paradigma clásico de los modelos de lenguaje modernos: **pre-entrenamiento seguido de fine-tuning**.

En la **primera etapa**, pre-entrenamos con Masked Language Modeling sobre el corpus de partidos. Enmascaramos tokens — jugadores, ratings, formaciones — y le pedimos al modelo que los prediga desde el contexto. Así aprende las relaciones tácticas y contextuales sin supervisión explícita de resultados.

En la **segunda etapa**, hacemos fine-tuning supervisado sobre la tarea de predicción de resultados: dado el contexto previo al partido, ¿cuál es la probabilidad de que gane el local, empaten, o gane el visitante?

Esta separación es clave: el pre-entrenamiento le da al modelo una comprensión profunda del "idioma del fútbol". El fine-tuning lo especializa en la tarea concreta.

---

## SLIDE 8 — Comparativa de Modelos ("La Paradoja Tabular")

Acá viene algo importante que queremos ser honestos sobre. Cuando comparamos D10SFormer contra los baselines clásicos — regresión logística, árboles de decisión — encontramos lo que llamamos **la paradoja tabular**:

En conjuntos de datos deportivos estructurados de tamaño moderado, la regresión logística y los árboles de decisión son competidores **excelentes** en precisión pura.

Esto no es una sorpresa si conocen la literatura: los métodos tabulares tradicionales suelen brillar en datasets de decenas de miles de filas con features bien estructuradas. El verdadero valor añadido del Transformer, por lo tanto, **no radica en superar estas métricas en bruto**, sino en algo cualitativamente diferente: la riqueza de sus representaciones semánticas distribuidas.

El modelo aprende embeddings de jugadores, equipos, y contextos tácticos que pueden ser reutilizados, analizados, y transferidos. Eso la regresión logística no lo puede hacer.

---

## SLIDE 9 — Ordinalidad Emergente

Un resultado que nos pareció particularmente elegante: el modelo aprende **ordinalidad emergente** de variables continuas como el ELO.

Sin decirle explícitamente que ELO 2209 > ELO 2050, el modelo aprende en el espacio de embeddings que hay un orden geométrico coherente. Los equipos de mayor rating quedan más cerca entre sí en el espacio vectorial, y sus representaciones capturan esa jerarquía de manera automática. Es exactamente lo que pasa en Word2Vec cuando emergen relaciones semánticas como rey - hombre + mujer ≈ reina. Acá emerge la estructura ordinal del fútbol competitivo.

*(Mencionar los valores de cosine similarity del gráfico: 0.7675, 0.7471, 0.6315, 0.5196)*

---

## SLIDE 10 — Transición a Resultados

Bien, ya tienen la teoría. Ahora lo que todos vinieron a ver.

*(Pausa dramática.)*

---

## SLIDE 11 — Resultados Baseline (LogReg)

El baseline de regresión logística da como favorita a **España** con una probabilidad de campeonato del ~21%, seguida de Argentina (~14%) y Francia (~10%).

Los resultados son coherentes con el ELO: España tiene el rating más alto (2209) y el modelo logístico, básicamente, le da mucho peso a esa variable.

---

## SLIDE 12 & 13 — Resultados D10SFormer

El modelo D10SFormer produce una distribución diferente. **Argentina aparece como favorita** con un 10.64% de probabilidad de ganar el torneo, seguida de cerca por Francia (10.36%) y Colombia (10.17%).

Noten que la distribución es más **plana y democrática** que la del logístico. El Transformer no le da tanta ventaja a España como el ELO sugeriría, porque aprendió que el contexto táctico y la dinámica de los partidos recientes importan más que el rating histórico puro.

España baja al 5.10% — una corrección significativa respecto al 21% del baseline.

---

## SLIDE 14 — "Nosotros elegimos creer"

*(Slide con Argentina como campeón proyectado)*

Nosotros elegimos creer. 

---

## SLIDE 15 — Validaciones del Simulador

Para validar que el simulador es internamente consistente, chequeamos dos propiedades:

**Primero**, que la suma de probabilidades de ser campeón sobre todos los equipos sea exactamente 1.0 — lo cumple con error del orden de 1e-9 en ambos modelos.

**Segundo**, monotonicidad: la probabilidad de avanzar de fase debe ser siempre mayor que la de la siguiente fase. P(pasar grupos) > P(octavos) > P(cuartos), etc. Ambos modelos la respetan.

La final más frecuente en 2.000 simulaciones con el baseline es Francia vs. España (~5.7%). Con D10SFormer, es Argentina vs. Francia (~2.9%). La distribución más plana del Transformer se traduce en finales más variadas.

---

## SLIDE 16 — Demo

*(Acá pasamos a la demo en vivo.)*

---

## SLIDE 17 — Conclusiones

Para cerrar:

**D10SFormer demuestra que es totalmente viable modelar el deporte de alta competencia mediante secuencias complejas utilizando la semántica de NLP.**

Más allá de la precisión predictiva absoluta, el verdadero aporte del proyecto tiene dos dimensiones:

**Flexibilidad de tokenización**: El modelo maneja naturalmente secuencias asimétricas — partidos con lineups completos, partidos solo con resultado, eventos parciales. No necesita inputs perfectamente estructurados porque el mecanismo de atención aprende a ignorar lo que no está y a valorizar lo que sí está.

**Descubrimiento automático de estructura**: El modelo encuentra el orden geométrico de variables continuas como el ELO sin intervenciones estructuradas previas. Eso es representación distribuida en acción — algo que los modelos tabulares clásicos simplemente no pueden ofrecer.

La paradoja tabular no es una derrota. Es una señal de que estamos operando en el régimen correcto para explorar lo que los Transformers realmente aportan: **semántica, no solo precisión**.

Gracias.

---

## NOTAS PARA LA PRESENTACIÓN

- **Tiempo estimado por sección:**
  - Introducción + hipótesis: 2–3 min
  - Datos + arquitectura: 4–5 min
  - Resultados + validación: 3–4 min
  - Demo: variable
  - Conclusiones: 1–2 min

- **Puntos donde anticipar preguntas del jurado:**
  - "¿Por qué BERT y no GPT?" → Respuesta: tarea de clasificación con contexto completo disponible, la bidireccionalidad es la elección correcta.
  - "¿Por qué no supera en accuracy a LogReg?" → La paradoja tabular: el valor está en las representaciones, no en la métrica puntual.
  - "¿Cómo manejan el desbalance de clases?" → Mencionar si aplica en el fine-tuning.
  - "¿Qué significa Weight Tying concretamente?" → Los pesos W_embedding ∈ R^(V×d) son los mismos que los del linear output layer de la cabeza MLM, reduciendo parámetros y mejorando coherencia.
