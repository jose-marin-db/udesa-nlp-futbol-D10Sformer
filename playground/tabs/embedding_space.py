"""Embedding space explorer — 2D/3D maps and vocabulary relations."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from components.ui_lab import section_intro
from services.nlp_explorer import (
    compute_embedding_projection,
    get_team_neighbors,
    load_json,
    neighbor_edges_for_highlight,
    projection_to_dataframe,
    projection_token_sets,
    try_load_live_model,
)


@st.cache_resource(show_spinner="Cargando modelo…")
def _bootstrap():
    return try_load_live_model()


@st.cache_data(show_spinner="Calculando proyección…")
def _cached_projection(mode_tag, token_group, method, dims, perplexity, _v=3):
    model, vocab, mode = _bootstrap()
    result = compute_embedding_projection(model, vocab, token_group, method, dims, perplexity)
    return projection_to_dataframe(result), result, mode


def _hover_for(df: pd.DataFrame, dims: int) -> dict:
    hover = {"confederation": True, "x": False, "y": False, "token": False}
    if "equipo_asociado" in df.columns:
        hover["equipo_asociado"] = True
    if "elo" in df.columns and df["elo"].notna().any():
        hover["elo"] = ":.0f"
    if dims == 3 and "z" in df.columns:
        hover["z"] = False
    return hover


def _add_lines(fig, df, edges, dims):
    by_label = df.set_index("label")
    for src, tgt, sim in edges:
        if src not in by_label.index or tgt not in by_label.index:
            continue
        xs = by_label.loc[[src, tgt], "x"]
        ys = by_label.loc[[src, tgt], "y"]
        line = dict(color="rgba(116,172,223,0.55)", width=1 + sim)
        if dims == 3:
            zs = by_label.loc[[src, tgt], "z"]
            fig.add_trace(
                go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line, hoverinfo="skip", showlegend=False)
            )
        else:
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", line=line, hoverinfo="skip", showlegend=False)
            )


def render_embedding_space() -> None:
    section_intro(
        "02 · Embeddings",
        "Relaciones aprendidas en el vocabulario",
        "Proyectamos vectores a 2D/3D para ver clusters: selecciones, buckets ELO, jugadores. "
        "La distancia ≈ similitud cosenoidal.",
        "Probá «Final 2022: equipos + jugadores» en 3D y resaltá Argentina.",
    )

    model, vocab, mode = _bootstrap()
    disc = load_json("discoveries.json")
    scores = disc["monotonicity_scores"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mono. ELO", f"{scores['ELO buckets']:.3f}")
    c2.metric("Mono. FORM", f"{scores['FORM buckets']:.3f}")
    c3.metric("Modo", mode)
    c4.metric("Vocab", "4.521")

    sub_map, sub_neigh, sub_elo = st.tabs(["🗺️ Mapa 2D/3D", "🔗 Vecinos", "📊 Heatmap ELO"])

    with sub_map:
        groups = list(projection_token_sets().keys())
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            token_group = st.selectbox("Conjunto", groups, index=groups.index("Final 2022: equipos + jugadores") if "Final 2022: equipos + jugadores" in groups else 0)
        with c2:
            method = st.selectbox("Método", ["pca", "tsne", "mds"], format_func=str.upper)
        with c3:
            dims = st.radio("Dim", [2, 3], horizontal=True)
        with c4:
            perp = st.slider("Perplexity", 5.0, 40.0, 15.0, 1.0)

        df, result, _ = _cached_projection(mode, token_group, method, dims, perp)
        st.caption(result.source)
        if result.explained_variance:
            st.info(f"PCA: PC1 {result.explained_variance[0]:.0%}, PC2 {result.explained_variance[1]:.0%}")

        color_map = {
            "CONMEBOL": "#6A994E", "UEFA": "#74ACDF", "CONCACAF": "#F6B504",
            "AFC": "#E63946", "CAF": "#9B5DE5", "OFC": "#F4A261",
            "Jugador": "#2D6A4F", "ELO bucket": "#BBBBBB", "FORM bucket": "#888888", "otro": "#CCCCCC",
        }
        hover = _hover_for(df, dims)

        if dims == 3:
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="confederation", text="label",
                                hover_name="label", hover_data=hover, color_discrete_map=color_map, height=650)
            fig.update_traces(marker=dict(size=6))
        else:
            fig = px.scatter(df, x="x", y="y", color="confederation", text="label",
                             hover_name="label", hover_data=hover, color_discrete_map=color_map, height=580)
            fig.update_traces(marker=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

        team_df = df[df["token"].str.startswith("TEAM_")]
        opts = ["(ninguno)"] + sorted(team_df["label"].tolist())
        hi = st.selectbox("Resaltar vecinos", opts, index=opts.index("ARGENTINA") if "ARGENTINA" in opts else 0)
        if hi != "(ninguno)":
            row = df[df["label"] == hi].iloc[0]
            edges = neighbor_edges_for_highlight(row["token"], df, model, vocab, k=5)
            _add_lines(fig, df, edges, dims)
            st.plotly_chart(fig, use_container_width=True)

    with sub_neigh:
        q = st.selectbox("Equipo", ["TEAM_ARGENTINA", "TEAM_FRANCE", "TEAM_BRAZIL", "TEAM_SPAIN"])
        neighbors = get_team_neighbors(q, model, vocab, k=10)
        if neighbors:
            dfn = pd.DataFrame(neighbors, columns=["token", "sim"])
            dfn["equipo"] = dfn["token"].str.replace("TEAM_", "").str.replace("_", " ")
            fig = px.bar(dfn.sort_values("sim"), x="sim", y="equipo", orientation="h", color="sim",
                         color_continuous_scale=["#E8F4FC", "#74ACDF"])
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        for item in disc["embedding_findings"]:
            st.markdown(f"**{item['title']}** — {item['detail']}")

    with sub_elo:
        elo_data = load_json("elo_cosine_matrix.json")
        labels = elo_data["labels"]
        fig_h = go.Figure(data=go.Heatmap(z=elo_data["matrix"], x=labels, y=labels, colorscale="RdYlBu_r", zmin=-0.2, zmax=1))
        fig_h.update_layout(title="Similitud entre buckets ELO (sin orden explícito en training)", height=480)
        st.plotly_chart(fig_h, use_container_width=True)
