"""Token sequence studio — how training sequences are built."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from components.ui_lab import metric_strip, section_intro
from services.nlp_explorer import (
    SEGMENT_COLORS,
    SEGMENT_DETAILS,
    SEGMENT_NAMES,
    build_custom_token_sequence,
    example_tokens_by_segment,
    format_token_label,
    load_json,
    vocab_counts_by_segment,
)


def _render_token_chips(tokens: list[dict], mask_highlight: bool = True) -> None:
    chips = []
    for t in tokens:
        seg = t.get("segment_name", SEGMENT_NAMES.get(t.get("segment", 0), "?"))
        color = SEGMENT_COLORS.get(seg, "#ccc")
        if mask_highlight and t.get("token") == "[MASK]":
            color = SEGMENT_COLORS["mask"]
        fg = "white" if seg == "mask" else "#1a1a2e"
        label = format_token_label(t["token"])
        chips.append(
            f'<span class="token-chip" style="background:{color};color:{fg}">{label}</span>'
        )
    st.markdown(
        f'<div class="token-flow">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _segment_legend() -> None:
    cols = st.columns(4)
    items = list(SEGMENT_COLORS.items())
    for i, (name, color) in enumerate(items):
        if name == "mask":
            continue
        cols[i % 4].markdown(
            f'<span class="token-chip" style="background:{color};color:#1a1a2e">'
            f'{name}</span> <span style="color:#666;font-size:0.8rem">{name}</span>',
            unsafe_allow_html=True,
        )


def render_token_studio() -> None:
    section_intro(
        "01 · Representación",
        "Cómo armamos la secuencia de entrenamiento",
        "Cada partido internacional se convierte en una secuencia ordenada de tokens BERT. "
        "El modelo aprende leyendo segmentos: contexto del partido, features tabulares, "
        "alineaciones y (opcional) eventos.",
        "Usá el constructor para ver cómo un cambio en ELO o forma altera los buckets del input.",
    )

    stats = load_json("vocab_stats.json")
    metric_strip(
        [
            ("Tokens vocab", f"{stats['total']:,}"),
            ("Max seq len", str(stats.get("max_seq_length", 80))),
            ("Jugadores dedicados", f"{stats['players_dedicated']:,}"),
            ("Segmentos", "8"),
        ]
    )

    st.markdown("#### Pipeline token → modelo")
    # Usamos graphviz porque Streamlit lo renderiza nativo (st.graphviz_chart).
    # Mermaid en st.markdown queda como texto plano.
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            bgcolor="transparent";
            node [
                shape=box, style="rounded,filled",
                fontname="Helvetica", fontsize=12,
                fillcolor="#F8FAFC", color="#475569", fontcolor="#0F172A",
                margin="0.2,0.12"
            ];
            edge [color="#94A3B8", penwidth=1.4, arrowsize=0.7];

            M [label="Partido crudo",           fillcolor="#E0E7FF", color="#4338CA"];
            T [label="Tokenizador",             fillcolor="#FEF3C7", color="#B45309"];
            S [label="Secuencia\\nsegmentada",  fillcolor="#FDE68A", color="#B45309"];
            P [label="Pre-train\\nMLM",         fillcolor="#D1FAE5", color="#047857"];
            F [label="Fine-tune\\nresultado",   fillcolor="#FECACA", color="#B91C1C"];

            M -> T -> S;
            S -> P;
            S -> F;
        }
        """,
        use_container_width=True,
    )

    sub_a, sub_b, sub_c = st.tabs(["🛠️ Constructor", "📋 Ejemplo real (ARG–FRA)", "📖 Segmentos"])

    with sub_a:
        st.markdown("##### Armar un partido a mano")
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Local", ["Argentina", "France", "Brazil", "Spain"], key="lab_home")
            home_elo = st.slider("ELO local", 1500, 2300, 2177, 50, key="lab_elo_h")
            home_form = st.slider("Forma local", 0.0, 3.0, 2.4, 0.1, key="lab_form_h")
        with c2:
            away = st.selectbox("Visitante", ["France", "Argentina", "Germany", "England"], key="lab_away")
            away_elo = st.slider("ELO visitante", 1500, 2300, 2128, 50, key="lab_elo_a")
            away_form = st.slider("Forma visitante", 0.0, 3.0, 2.6, 0.1, key="lab_form_a")

        mask_bench = st.toggle("Simular MLM: enmascarar banco visitante", value=True)
        tokens = build_custom_token_sequence(
            home, away, home_elo, away_elo, home_form, away_form, mask_bench=mask_bench
        )
        st.caption("Cada chip = un token. El orden importa: el Transformer lee izquierda → derecha.")
        _render_token_chips(tokens)
        st.dataframe(pd.DataFrame(tokens)[["idx", "token", "segment_name"]], hide_index=True, height=280)

    with sub_b:
        ex = load_json("token_sequence_example.json")
        st.info(ex["description"])
        real_tokens = ex["tokens"]
        _render_token_chips(
            [{"token": t["token"], "segment_name": SEGMENT_NAMES.get(t["segment"], "?")} for t in real_tokens],
            mask_highlight=False,
        )
        st.write(f"**Target entrenamiento:** `{ex['target_result']}` · `{ex['target_score']}`")
        st.dataframe(pd.DataFrame(real_tokens), hide_index=True, height=320)

    with sub_c:
        st.markdown("##### Los 8 segmentos de la secuencia")
        _segment_legend()

        # Selector interactivo
        labels = [
            f"{seg}  ·  {info['name']}  —  {info['title']}"
            for seg, info in SEGMENT_DETAILS.items()
        ]
        choice = st.radio(
            "Elegí un segmento para inspeccionarlo",
            options=list(SEGMENT_DETAILS.keys()),
            format_func=lambda s: labels[s],
            horizontal=False,
            key="lab_segment_pick",
        )
        info = SEGMENT_DETAILS[choice]
        color = SEGMENT_COLORS.get(info["name"], "#CCC")

        # Header con color del segmento
        st.markdown(
            f'<div style="margin:1rem 0 0.5rem 0; padding:0.6rem 0.9rem; '
            f'background:{color}; border-radius:8px; color:#1a1a2e">'
            f'<strong>Segmento {choice} · {info["name"]}</strong> — {info["title"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(info["description"])

        col_ex, col_vocab = st.columns([3, 2])

        # ---- Tokens reales del ejemplo ARG-FRA ----
        by_seg = example_tokens_by_segment()
        sample = by_seg.get(choice, [])
        with col_ex:
            st.markdown("**Tokens reales en el ejemplo ARG–FRA**")
            if sample:
                _render_token_chips(
                    [{"token": t["token"], "segment_name": info["name"]} for t in sample],
                    mask_highlight=True,
                )
                df_sample = pd.DataFrame(
                    [{"idx en secuencia": t["idx"], "token raw": t["token"], "label": t["label"]} for t in sample]
                )
                st.dataframe(df_sample, hide_index=True, height=min(35 * (len(df_sample) + 1) + 3, 300))
            else:
                st.info(
                    "Este segmento no aparece en el ejemplo. "
                    "Para `events` (6), el ejemplo no incluyó eventos; en inferencia futura suele ir enmascarado."
                )

        # ---- Conteo del vocab por prefijo ----
        with col_vocab:
            st.markdown("**Cobertura del vocabulario**")
            counts = vocab_counts_by_segment().get(choice, {})

            spec_rows = [
                {"tipo": tok, "n en vocab": 1, "categoría": "especial"}
                for tok in info["specials"]
            ]
            prefix_rows = [
                {"tipo": f"{prefix}*", "n en vocab": n, "categoría": "prefijo"}
                for prefix, n in counts.items()
            ]
            df_cov = pd.DataFrame(spec_rows + prefix_rows)
            if df_cov.empty:
                st.caption("Solo tokens especiales en este segmento.")
            else:
                total = df_cov["n en vocab"].sum()
                st.metric("Total tokens posibles", f"{total:,}")
                st.dataframe(df_cov, hide_index=True, height=min(35 * (len(df_cov) + 1) + 3, 260))

        st.success(
            "**Para entrenar:** el collator puede enmascarar tokens al azar (MLM) o predecir "
            "`[PREDICT_RESULT]` al final (fine-tune). Los buckets ELO/FORM son discretos pero "
            "el embedding aprende orden ordinal."
        )
