# D10Sformer Lab (Streamlit)

Demo enfocada: secuencias de tokens, mapas de embeddings y simulador del Mundial 2026.

## Ejecutar

```bash
pip install -r playground/requirements.txt
streamlit run playground/app_lab.py
```

## Estructura

```
playground/
├── app_lab.py              # entrada
├── assets/lab.css
├── components/ui_lab.py
├── tabs/
│   ├── token_studio.py
│   ├── embedding_space.py
│   └── bracket_simulator.py
├── services/
│   ├── nlp_explorer.py
│   ├── predictors.py
│   ├── tournament_journey.py
│   ├── bracket_view.py
│   └── i18n.py
├── data/                   # JSON precomputados
└── artifacts/              # opcional: logreg, features, modelo
```

## Artefactos opcionales (`playground/artifacts/`)

| Archivo | Uso |
|---------|-----|
| `logreg_wc2026.pkl` | Predictor del simulador (si falta → ELO) |
| `team_features_wc2026.json` | Features de los 48 equipos |
| `vocab.json` + `d10sformer_finetune.pt` | Embeddings en vivo |

Actualizar predicciones MC desde el notebook 06: exportar CSV → `data/wc2026_predictions.json`.
