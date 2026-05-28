"""Full World Cup 2026 bracket simulator with groups + knockout tree."""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui_lab import metric_strip, section_intro, step_pills
from services.bracket_view import (
    build_group_standings_rows,
    build_knockout_bracket,
    build_pyramid_bracket_html,
    build_round_columns,
    champion_banner,
)
from services.i18n import to_spanish
from services.tournament_journey import (
    STAGE_LABELS,
    STAGE_ORDER,
    JourneyState,
    journey_to_fixed_results,
    run_full_tournament,
    run_group_stage,
    run_knockout_stage,
)
from simulation.bracket import GROUP_NAMES, WC2026_GROUPS
from simulation.simulator import monte_carlo


ROUND_DISPLAY = [
    ("round_of_32", "32avos"),
    ("round_of_16", "Octavos"),
    ("quarterfinals", "Cuartos"),
    ("semifinals", "Semis"),
    ("third_place", "3.er puesto"),
    ("final", "Final"),
]


def _init() -> None:
    if "lab_journey" not in st.session_state:
        st.session_state.lab_journey = None
    if "lab_seed" not in st.session_state:
        st.session_state.lab_seed = 42


def _reset(seed: int) -> None:
    st.session_state.lab_journey = JourneyState(seed=seed)
    st.session_state.lab_seed = seed


def _active_stage_index(state: JourneyState | None) -> int:
    if state is None:
        return -1
    if state.is_finished():
        return len(STAGE_ORDER)
    idx = STAGE_ORDER.index(state.next_stage) if state.next_stage else len(STAGE_ORDER)
    return max(0, idx - 1) if state.completed_stages else -1


def _match_played(row: dict) -> bool:
    res = row.get("resultado", "—")
    win = row.get("ganador", "—")
    return res not in ("—", "", None) and win not in ("—", "", None)


def _render_match_card_ui(row: dict) -> None:
    """Native Streamlit card — avoids broken nested HTML in st.markdown columns."""
    played = _match_played(row)
    score = row.get("resultado", "—")
    winner = row.get("ganador", "—")
    with st.container(border=True):
        st.markdown(f"**{row['local']}** vs **{row['visitante']}**")
        if played:
            st.markdown(f"### {score}")
            st.caption(f"Ganador: **{winner}**")
        else:
            st.caption("Por jugar · equipos según llave FIFA")


def _render_bracket_pyramid(state: JourneyState, cols_data: dict) -> None:
    """Horizontal pyramid: many matches on the left → final on the right."""
    st.caption(
        "Izquierda = dieciseisavos · derecha = final (vértice). "
        "Las filas alinean cada cruce con el partido que lo alimenta."
    )
    st.html(build_pyramid_bracket_html(cols_data))


def _render_bracket_tree(state: JourneyState) -> None:
    cols_data = build_round_columns(state)
    if not cols_data.get("round_of_32"):
        st.warning("Simulá al menos la fase de grupos para ver las llaves.")
        return

    st.markdown("##### Árbol eliminatorio")
    view = st.radio(
        "Vista",
        ["Pirámide (llave)", "Por fase (lista)"],
        horizontal=True,
        label_visibility="collapsed",
        key="bracket_view_mode",
    )

    if view.startswith("Pirámide"):
        _render_bracket_pyramid(state, cols_data)
    else:
        phase_tabs = st.tabs([label for _, label in ROUND_DISPLAY])
        for tab, (key, label) in zip(phase_tabs, ROUND_DISPLAY):
            matches = cols_data.get(key, [])
            with tab:
                if not matches:
                    st.caption("Sin partidos en esta fase.")
                    continue
                n_cols = 2 if len(matches) > 2 else 1
                grid = st.columns(n_cols)
                for i, m in enumerate(matches):
                    with grid[i % n_cols]:
                        _render_match_card_ui(m)


def _render_groups_grid(state: JourneyState) -> None:
    if "groups" not in state.completed_stages:
        st.info("Avanzá la fase de grupos para ver tablas.")
        return

    st.markdown("##### 12 grupos · top 2 + 8 mejores terceros")
    with st.expander("¿Qué significa «Mejor 3.º»?", expanded=False):
        st.markdown(
            """
            En el **Mundial 2026 (48 equipos)** clasifican los **2 primeros** de cada grupo (24)
            más los **8 mejores terceros** entre los 12 grupos (por puntos, diferencia y goles).

            No es un wildcard al azar: solo los 8 terceros con mejor récord siguen;
            los otros 4 quedan eliminados aunque hayan sido 3.º en su grupo.
            """
        )
    rows = build_group_standings_rows(state)
    df = pd.DataFrame(rows)
    group_letters = sorted(df["Grupo"].unique())
    grid_cols = st.columns(4)
    for i, letter in enumerate(group_letters):
        sub = df[df["Grupo"] == letter].copy()
        sub["Equipo"] = sub["Equipo"]
        with grid_cols[i % 4]:
            st.markdown(f"**Grupo {letter}**")
            st.dataframe(
                sub[["Pos", "Equipo", "Pts", "DG", "Clasifica"]],
                hide_index=True,
                height=175,
            )


def render_bracket_simulator(predictor, precomputed, default_n_sims: int) -> None:
    _init()
    section_intro(
        "03 · Mundial 2026",
        "Simulador completo con llaves",
        "Recorré el torneo fase a fase o en un clic. Grupos de 4, 32avos con mejores terceros, "
        "y árbol hasta la final. Cada partido se muestrea (no es determinístico).",
        "Usá «Simular todo» para ver el bracket completo; luego MC condicional para P(campeón).",
    )

    state: JourneyState | None = st.session_state.lab_journey
    labels = [STAGE_LABELS[s].split("(")[0].strip()[:12] for s in STAGE_ORDER]
    step_pills(labels, _active_stage_index(state))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seed = st.number_input("Semilla", 0, 99999, st.session_state.lab_seed, key="lab_sim_seed")
    with c2:
        if st.button("🔄 Nuevo torneo", use_container_width=True, key="lab_new"):
            _reset(int(seed))
            st.rerun()
    with c3:
        fast = st.button("⚡ Torneo completo", use_container_width=True, key="lab_fast")
    with c4:
        mc_n = st.number_input("MC", 100, 5000, min(1000, default_n_sims), step=100, key="lab_mc_n")

    if fast:
        with st.spinner("Simulando 104 partidos…"):
            st.session_state.lab_journey = run_full_tournament(predictor, seed=int(seed))
        st.rerun()

    state = st.session_state.lab_journey
    if state is None:
        st.markdown(
            """
            <div class="lab-tip">
              <strong>Empezá acá:</strong> elegí una semilla y pulsá <em>Nuevo torneo</em>.
              Luego avanzá fase a fase o saltá con <em>Torneo completo</em>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _preview_groups()
        return

    next_stage = state.next_stage
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶️ Siguiente fase", type="primary", disabled=state.is_finished(), key="lab_next"):
            with st.spinner(STAGE_LABELS.get(next_stage, "")):
                if next_stage == "groups":
                    run_group_stage(state, predictor)
                else:
                    run_knockout_stage(state, predictor, next_stage)
            st.rerun()
    with b2:
        if st.button("🇦🇷 Solo Grupo J", disabled="groups" in state.completed_stages, key="lab_gj"):
            run_group_stage(state, predictor, groups=["Group_J"])
            st.rerun()
    with b3:
        if st.button("📊 MC condicional", disabled=not state.completed_stages, key="lab_mc"):
            _run_conditional_mc(state, precomputed, int(seed), int(mc_n))

    banner = champion_banner(state)
    if banner:
        st.balloons()
        with st.container(border=True):
            st.markdown("#### Campeón simulado")
            st.markdown(f"## 🏆 {banner['campeon']}")
            st.caption(f"Subcampeón: {banner['subcampeon']} · 3.er puesto: {banner['tercero']}")

    n_played = len(state.match_log)
    n_ko = len([m for m in state.match_log if m.stage != "groups"])
    metric_strip(
        [
            ("Partidos", str(n_played)),
            ("Eliminatoria", str(n_ko)),
            ("Fases listas", str(len(state.completed_stages))),
            ("Siguiente", (STAGE_LABELS.get(next_stage, "Fin")[:18] if next_stage else "Fin")),
        ]
    )

    tab_groups, tab_bracket, tab_log, tab_probs = st.tabs(
        ["📋 Grupos", "🏆 Llaves", "📜 Cronología", "🎲 Probabilidades"]
    )

    with tab_groups:
        _render_groups_grid(state)

    with tab_bracket:
        _render_bracket_tree(state)
        with st.expander("Tabla detallada eliminatoria"):
            kb = build_knockout_bracket(state)
            if kb:
                st.dataframe(
                    pd.DataFrame(kb)[["fase", "local", "visitante", "resultado", "ganador"]],
                    hide_index=True,
                    height=420,
                )

    with tab_log:
        _render_match_log(state)

    with tab_probs:
        _render_precomputed_probs(precomputed)
        if state.completed_stages:
            st.caption("Pulsá «MC condicional» en la barra superior para actualizar con tu trayectoria.")


def _preview_groups() -> None:
    st.markdown("##### Vista previa · plantilla de grupos")
    preview = []
    for grp in GROUP_NAMES[:6]:
        teams = WC2026_GROUPS.get(grp, [])
        letter = grp.replace("Group_", "")
        for t in teams:
            preview.append({"Grupo": letter, "Equipo": to_spanish(t)})
    st.dataframe(pd.DataFrame(preview), hide_index=True, height=220)


def _render_match_log(state: JourneyState) -> None:
    rows = []
    for m in state.match_log:
        rows.append(
            {
                "Fase": STAGE_LABELS.get(m.stage, m.stage),
                "Partido": f"{to_spanish(m.home)} vs {to_spanish(m.away)}",
                "Marcador": f"{m.home_score}–{m.away_score}",
                "Ganador": to_spanish(m.winner) if m.winner != "draw" else "Empate",
                "Grupo": m.group or "",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=480)
    else:
        st.caption("Sin partidos aún.")


def _run_conditional_mc(state: JourneyState, precomputed, seed: int, n: int) -> None:
    fr = journey_to_fixed_results(state)
    with st.spinner(f"{n:,} simulaciones…"):
        t0 = time.time()
        agg = monte_carlo(precomputed, n_iters=n, seed=seed, progress=False, fixed_results=fr)
        elapsed = time.time() - t0
    df = agg.to_dataframe()
    df["Equipo"] = df["team"].map(to_spanish)
    st.success(f"Listo en {elapsed:.1f}s · trayectoria fijada")
    top = df.sort_values("P_champion", ascending=False).head(12)
    fig = px.bar(
        top,
        x="P_champion",
        y="Equipo",
        orientation="h",
        color="P_champion",
        color_continuous_scale=["#E8F4FC", "#6A994E"],
        title=f"P(campeón | estado actual) · n={n:,}",
    )
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    arg = df[df["team"] == "Argentina"]
    if not arg.empty:
        st.metric("🇦🇷 Argentina P(campeón)", f"{arg.iloc[0]['P_champion'] * 100:.2f}%")


def _render_precomputed_probs(precomputed) -> None:
    from services.nlp_explorer import load_json

    preds = load_json("wc2026_predictions.json")
    if not preds:
        st.caption("Cargá wc2026_predictions.json para ver prior global.")
        return
    df = pd.DataFrame(preds).sort_values("P_champion", ascending=False).head(15)
    df["Equipo"] = df["team"].map(to_spanish)
    fig = px.bar(
        df,
        x="P_champion",
        y="Equipo",
        orientation="h",
        color="P_champion",
        color_continuous_scale=["#E8F4FC", "#74ACDF"],
        title="Prior Monte Carlo 10k (LogReg, sin condicionar)",
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
