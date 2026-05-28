"""Predictor loading for the Streamlit playground.

Orden de prioridad (de mejor a peor):

    1. D10Sformer fine-tuneado (checkpoints/finetune_weighted_15ep/best.pt)
       — el modelo central del proyecto.
    2. LogReg calibrado (artifacts/logreg_wc2026.pkl)
       — baseline tabular.
    3. ELO baseline (ELOBaseline)
       — último fallback puro analítico.

Para forzar uno específico, setear la variable de entorno:

    D10SFORMER_ENGINE=transformer | logreg | elo
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_baselines_path = SRC / "models" / "baselines.py"
_spec = importlib.util.spec_from_file_location("playground_baselines", _baselines_path)
_baselines = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_baselines)
ELOBaseline = _baselines.ELOBaseline

PLAYGROUND = Path(__file__).resolve().parents[1]
DATA = PLAYGROUND / "data"
ARTIFACTS = PLAYGROUND / "artifacts"

# Rutas para el D10Sformer (ajustables vía env vars)
DEFAULT_D10S_CKPT = ROOT / "checkpoints" / "finetune_weighted_15ep" / "best.pt"
DEFAULT_D10S_VOCAB = ROOT / "data" / "processed" / "vocab.json"

DEFAULT_FEATURES = {"elo": 1500.0, "form_pts": 1.0, "recent_goals": 1.0}


def load_team_features() -> dict[str, dict]:
    for path in (ARTIFACTS / "team_features_wc2026.json", DATA / "team_features.json"):
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: {kk: float(vv) if kk != "team" else vv for kk, vv in v.items()} for k, v in raw.items()}
    return {}


def build_feature_vector(
    team_a: str,
    team_b: str,
    team_features: dict[str, dict],
    venue: str = "neutral",
) -> pd.DataFrame:
    fa = team_features.get(team_a, DEFAULT_FEATURES)
    fb = team_features.get(team_b, DEFAULT_FEATURES)
    elo_diff = fa["elo"] - fb["elo"]
    return pd.DataFrame(
        [
            {
                "tournament_class": "world_cup_final",
                "neutral": 1 if venue == "neutral" else 0,
                "home_elo": fa["elo"],
                "away_elo": fb["elo"],
                "elo_diff": elo_diff,
                "expected_home_win_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
                "home_rest_days": 7,
                "away_rest_days": 7,
                "home_form5_pts": fa.get("form_pts", 1.0),
                "home_form5_gf": fa.get("recent_goals", 1.0),
                "home_form5_ga": 1.0,
                "home_form5_gd": 0.0,
                "home_form5_n": 5,
                "away_form5_pts": fb.get("form_pts", 1.0),
                "away_form5_gf": fb.get("recent_goals", 1.0),
                "away_form5_ga": 1.0,
                "away_form5_gd": 0.0,
                "away_form5_n": 5,
                "home_form10_pts": fa.get("form_pts", 1.0),
                "home_form10_gf": fa.get("recent_goals", 1.0),
                "home_form10_ga": 1.0,
                "home_form10_gd": 0.0,
                "home_form10_n": 10,
                "away_form10_pts": fb.get("form_pts", 1.0),
                "away_form10_gf": fb.get("recent_goals", 1.0),
                "away_form10_ga": 1.0,
                "away_form10_gd": 0.0,
                "away_form10_n": 10,
                "h2h_n_matches": 0,
                "h2h_home_wins": 0,
                "h2h_draws": 0,
                "h2h_away_wins": 0,
                "h2h_avg_gd_for_home": 0.0,
            }
        ]
    )


# ============================================================================
# Loaders por engine
# ============================================================================

def _load_logreg_model():
    path = ARTIFACTS / "logreg_wc2026.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _try_load_d10sformer(team_features: dict[str, dict]):
    """Intenta cargar el D10Sformer fine-tuneado.

    Devuelve (predict_fn, engine_label) si tuvo éxito, o (None, motivo) si no.
    """
    ckpt = Path(os.getenv("D10SFORMER_CKPT", str(DEFAULT_D10S_CKPT)))
    vocab = Path(os.getenv("D10SFORMER_VOCAB", str(DEFAULT_D10S_VOCAB)))

    if not ckpt.exists():
        return None, f"checkpoint no encontrado: {ckpt}"
    if not vocab.exists():
        return None, f"vocab no encontrado: {vocab}"

    try:
        # Import perezoso — evita cargar torch al inicio si no hace falta
        from .d10sformer_predictor import D10SformerPredictor
    except Exception as exc:  # pragma: no cover
        return None, f"import D10SformerPredictor falló: {exc!r}"

    try:
        predictor = D10SformerPredictor(
            ckpt_path=ckpt,
            vocab_path=vocab,
            team_features=team_features,
            device=os.getenv("D10SFORMER_DEVICE", "cpu"),
        )
    except Exception as exc:  # pragma: no cover
        return None, f"instanciación falló: {exc!r}"

    label = f"D10Sformer · {ckpt.parent.name}"
    return predictor, label


# ============================================================================
# Factory principal
# ============================================================================

def make_predictor(team_features: dict[str, dict]) -> tuple[Callable[..., np.ndarray], str]:
    """Devuelve (predict_fn, engine_label) según prioridad y env var.

    Forzar:
        D10SFORMER_ENGINE=transformer  → solo Transformer; falla si no hay ckpt
        D10SFORMER_ENGINE=logreg       → solo LogReg
        D10SFORMER_ENGINE=elo          → solo ELO
        (unset / 'auto')               → prioridad: transformer → logreg → ELO
    """
    forced = (os.getenv("D10SFORMER_ENGINE") or "auto").lower()

    # 1) Transformer
    if forced in ("auto", "transformer", "d10sformer"):
        predictor, info = _try_load_d10sformer(team_features)
        if predictor is not None:
            def predict(team_a: str, team_b: str, venue: str = "neutral") -> np.ndarray:
                return predictor.predict(team_a, team_b, venue=venue)
            return predict, info
        if forced in ("transformer", "d10sformer"):
            # El usuario lo forzó pero no se pudo cargar — abortar con error explícito
            raise RuntimeError(
                f"D10SFORMER_ENGINE=transformer pero no se pudo cargar: {info}"
            )
        # auto → seguir al siguiente engine
        print(f"[predictors] D10Sformer no disponible ({info}). Cayendo a LogReg.")

    # 2) LogReg
    if forced in ("auto", "logreg"):
        logreg = _load_logreg_model()
        if logreg is not None:
            def predict(team_a: str, team_b: str, venue: str = "neutral") -> np.ndarray:
                x = build_feature_vector(team_a, team_b, team_features, venue=venue)
                aligned = x.copy()
                for col in logreg.feature_names_:
                    if col not in aligned.columns:
                        aligned[col] = 0.0
                return logreg.predict_proba(aligned[logreg.feature_names_].values)[0]
            return predict, "LogReg calibrado"
        if forced == "logreg":
            raise RuntimeError("D10SFORMER_ENGINE=logreg pero no se encontró el .pkl")

    # 3) ELO baseline
    elo = ELOBaseline()

    def predict(team_a: str, team_b: str, venue: str = "neutral") -> np.ndarray:
        x = build_feature_vector(team_a, team_b, team_features, venue=venue)
        return elo.predict_proba(x)[0]

    return predict, "ELO baseline (fallback)"
