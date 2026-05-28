"""Bracket and group visualization helpers for the lab simulator."""

from __future__ import annotations

import html as html_lib
from typing import Any, Optional

from services.i18n import to_spanish
from services.tournament_journey import JourneyState, STAGE_LABELS, KNOCKOUT_BY_STAGE
from simulation.bracket import (
    FINAL,
    GROUP_NAMES,
    THIRD_PLACE,
    round_of_match,
)
from simulation.simulator import resolve_slot


ROUND_LABELS = {
    "round_of_32": "Dieciseisavos",
    "round_of_16": "Octavos",
    "quarterfinals": "Cuartos",
    "semifinals": "Semifinales",
    "third_place": "3.er puesto",
    "final": "Final",
}


def _slot_label(slot: str, ctx: dict[str, str]) -> str:
    if slot.startswith("winner_match_") or slot.startswith("loser_match_"):
        mid = int(slot.split("_")[-1])
        return "TBD"
    try:
        return to_spanish(resolve_slot(slot, ctx))
    except KeyError:
        return slot.replace("_", " ")


def _match_from_log(state: JourneyState, match_id: int) -> dict[str, Any] | None:
    for m in state.match_log:
        if m.match_id == match_id:
            return {
                "home": to_spanish(m.home),
                "away": to_spanish(m.away),
                "score": f"{m.home_score}–{m.away_score}",
                "winner": to_spanish(m.winner),
            }
    return None


def _clasifica_label(state: JourneyState, pos: int, team: str) -> str:
    """Human-readable knockout status after groups."""
    if pos <= 2:
        return "Directo (top 2)"
    if pos == 4:
        return "Eliminado"
    # Third place: only 8 of 12 advance as "mejores terceros" (FIFA 2026 format).
    picked = set(state.third_assignment.values()) if state.third_assignment else set()
    if team in picked:
        return "Mejor 3.º (8 cupos)"
    return "3.º — no clasifica"


def build_group_standings_rows(state: JourneyState) -> list[dict[str, Any]]:
    rows = []
    for grp in GROUP_NAMES:
        standings = state.result.group_standings.get(grp, [])
        for pos, s in enumerate(standings, start=1):
            rows.append(
                {
                    "Grupo": grp.replace("Group_", ""),
                    "Pos": pos,
                    "Equipo": to_spanish(s.team),
                    "Pts": s.points,
                    "GF": s.goals_for,
                    "GC": s.goals_against,
                    "DG": s.goal_diff,
                    "Clasifica": _clasifica_label(state, pos, s.team),
                }
            )
    return rows


def _pyramid_slot(
    match_index: int,
    round_index: int,
    n_matches_in_round: int,
    total_rows: int,
) -> tuple[int, int]:
    """1-based grid row start and span for a horizontal knockout pyramid."""
    if n_matches_in_round <= 1:
        return 1, total_rows
    start = (2**round_index) * (2 * match_index + 1)
    span = 2 ** (round_index + 2)
    return start, span


def _match_node_html(row: dict) -> str:
    played = row.get("resultado") not in ("—", "", None) and row.get("ganador") not in ("—", "", None)
    score = html_lib.escape(str(row.get("resultado", "—")))
    winner = html_lib.escape(str(row.get("ganador", "—")))
    home = html_lib.escape(str(row["local"]))
    away = html_lib.escape(str(row["visitante"]))
    cls = "bracket-node played" if played else "bracket-node"
    body = (
        f'<div class="bn-teams">{home} <span class="bn-vs">vs</span> {away}</div>'
        f'<div class="bn-score">{score}</div>'
        f'<div class="bn-winner">→ {winner}</div>'
    ) if played else (
        f'<div class="bn-teams">{home} <span class="bn-vs">vs</span> {away}</div>'
        f'<div class="bn-pending">Por jugar</div>'
    )
    return f'<div class="{cls}">{body}</div>'


def build_pyramid_bracket_html(cols_data: dict[str, list[dict[str, Any]]]) -> str:
    """Single HTML document for horizontal pyramid bracket (Streamlit st.html)."""
    r32 = cols_data.get("round_of_32", [])
    n_first = max(len(r32), 1)
    total_rows = 2 * n_first

    pyramid_rounds = [
        ("round_of_32", "32avos", 0),
        ("round_of_16", "Octavos", 1),
        ("quarterfinals", "Cuartos", 2),
        ("semifinals", "Semis", 3),
    ]

    parts = [
        '<div class="bracket-pyramid-scroll">',
        '<div class="bracket-pyramid">',
    ]

    for key, label, r_idx in pyramid_rounds:
        matches = cols_data.get(key, [])
        parts.append(
            f'<div class="bracket-round" style="--rows:{total_rows}">'
            f'<div class="bracket-round-title">{html_lib.escape(label)}</div>'
            f'<div class="bracket-round-grid" style="grid-template-rows: repeat({total_rows}, minmax(20px, auto));">'
        )
        for i, m in enumerate(matches):
            start, span = _pyramid_slot(i, r_idx, len(matches), total_rows)
            node = _match_node_html(m)
            parts.append(
                f'<div class="bracket-slot" style="grid-row: {start} / span {span};">{node}</div>'
            )
        parts.append("</div></div>")

    # Final column: apex = final, 3rd place below center
    parts.append(f'<div class="bracket-round bracket-round--apex" style="--rows:{total_rows}">')
    parts.append('<div class="bracket-round-title">Cierre</div>')
    parts.append(
        f'<div class="bracket-round-grid" style="grid-template-rows: repeat({total_rows}, minmax(20px, auto));">'
    )
    final_m = cols_data.get("final", [])
    third_m = cols_data.get("third_place", [])
    if final_m:
        start, span = _pyramid_slot(0, 4, 1, total_rows)
        parts.append(
            f'<div class="bracket-slot bracket-slot--final" style="grid-row: {start} / span {span};">'
            f'<div class="bracket-round-title" style="margin:0 0 4px">Final</div>'
            f"{_match_node_html(final_m[0])}</div>"
        )
    if third_m:
        # Lower half of tree
        row_start = total_rows // 2 + total_rows // 8
        parts.append(
            f'<div class="bracket-slot bracket-slot--third" style="grid-row: {row_start} / span {total_rows // 4};">'
            f'<div class="bracket-round-title" style="margin:0 0 4px">3.er puesto</div>'
            f"{_match_node_html(third_m[0])}</div>"
        )
    parts.append("</div></div></div></div>")

    return "".join(parts)


def build_knockout_bracket(state: JourneyState) -> list[dict[str, Any]]:
    """One row per knockout match with resolved teams when available."""
    if "groups" not in state.completed_stages:
        return []

    ctx = state.ctx
    rows = []
    for stage_key, matches in KNOCKOUT_BY_STAGE.items():
        for match in matches:
            mid = match.match_id
            played = _match_from_log(state, mid)
            try:
                home = to_spanish(resolve_slot(match.slot_a, ctx)) if not played else played["home"]
                away = to_spanish(resolve_slot(match.slot_b, ctx)) if not played else played["away"]
            except KeyError:
                home = _slot_label(match.slot_a, ctx)
                away = _slot_label(match.slot_b, ctx)

            winner = state.result.knockout_winners.get(mid)
            rows.append(
                {
                    "match_id": mid,
                    "fase": ROUND_LABELS.get(round_of_match(mid), stage_key),
                    "local": home,
                    "visitante": away,
                    "resultado": played["score"] if played else ("—" if winner is None else "jugado"),
                    "ganador": to_spanish(winner) if winner else "—",
                    "stage_key": stage_key,
                }
            )
    return rows


def build_round_columns(state: JourneyState) -> dict[str, list[dict[str, Any]]]:
    """Group knockout matches by round for column layout."""
    all_rows = build_knockout_bracket(state)
    cols: dict[str, list[dict[str, Any]]] = {k: [] for k in ROUND_LABELS}
    for row in all_rows:
        key = row["stage_key"]
        if key == "finals":
            mid = row["match_id"]
            if mid == THIRD_PLACE.match_id:
                cols["third_place"].append(row)
            elif mid == FINAL.match_id:
                cols["final"].append(row)
        else:
            cols.setdefault(key, []).append(row)
    return cols


def champion_banner(state: JourneyState) -> Optional[dict[str, str]]:
    r = state.result
    if not r.champion:
        return None
    return {
        "campeon": to_spanish(r.champion),
        "subcampeon": to_spanish(r.runner_up) if r.runner_up else "—",
        "tercero": to_spanish(r.third_place) if r.third_place else "—",
    }
