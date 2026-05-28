"""
D10Sformer Lab — focused demo: token sequences, embedding space, full WC bracket.

Run from repo root:
  streamlit run playground/app_lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PLAYGROUND = Path(__file__).resolve().parent

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(PLAYGROUND))

from components.ui_lab import inject_lab_css, page_hero  # noqa: E402
from services.predictors import load_team_features, make_predictor  # noqa: E402
from simulation.bracket import WC2026_GROUPS_RAW, to_english  # noqa: E402
from simulation.simulator import PrecomputedPredictor  # noqa: E402
from tabs.token_studio import render_token_studio  # noqa: E402
from tabs.embedding_space import render_embedding_space  # noqa: E402
from tabs.bracket_simulator import render_bracket_simulator  # noqa: E402


LAB_SECTIONS = [
    ("tokens", "🧬 Secuencias", "Cómo se arman los tokens de entrenamiento"),
    ("embeddings", "🗺️ Vocabulario", "Mapas 2D/3D y relaciones del embedding"),
    ("mundial", "🏆 Mundial", "Simulador con grupos y llaves completas"),
]


def _load_predictors():
    team_features = load_team_features()
    predictor, engine = make_predictor(team_features)
    teams_en = [to_english(t) for t in sorted({t for g in WC2026_GROUPS_RAW.values() for t in g})]
    precomputed = PrecomputedPredictor(predictor, teams_en, venue="neutral")
    return precomputed, predictor, engine


@st.cache_resource
def _predictors():
    return _load_predictors()


def render_sidebar() -> str:
    st.sidebar.markdown("### D10Sformer Lab")
    st.sidebar.caption("Secuencias → embeddings → simulador Mundial 2026.")
    labels = [s[1] for s in LAB_SECTIONS]
    choice = st.radio(
        "Sección",
        options=[s[0] for s in LAB_SECTIONS],
        format_func=lambda k: next(l for key, l, _ in LAB_SECTIONS if key == k),
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Atajos**")
    st.sidebar.code("streamlit run playground/app_lab.py", language="bash")
    _precomputed, _live, engine = _predictors()
    st.sidebar.metric("Predictor activo", engine)
    st.sidebar.caption(
        "Forzá engine con `D10SFORMER_ENGINE=transformer|logreg|elo`."
    )
    return choice


def main() -> None:
    st.set_page_config(
        page_title="D10Sformer Lab",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_lab_css(PLAYGROUND / "assets" / "lab.css")

    page_hero(
        "Laboratorio interactivo",
        "Secuencias de tokens · espacio de embeddings · Mundial 2026 con llaves",
    )

    section = render_sidebar()
    precomputed, live, _engine = _predictors()

    if section == "tokens":
        render_token_studio()
    elif section == "embeddings":
        render_embedding_space()
    else:
        render_bracket_simulator(live, precomputed, default_n_sims=2000)


if __name__ == "__main__":
    main()
