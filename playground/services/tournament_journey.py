"""Step-by-step WC 2026 simulation for the interactive playground."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from simulation.bracket import (
    FINAL,
    QUARTERFINALS,
    ROUND_OF_16,
    ROUND_OF_32,
    SEMIFINALS,
    THIRD_PLACE,
    WC2026_GROUPS,
    GROUP_NAMES,
)
from simulation.simulator import (
    FixedResults,
    GroupStanding,
    Predictor,
    TournamentResult,
    assign_thirds_to_slots,
    resolve_slot,
    sample_goals,
    sample_knockout_winner,
    sample_match_result,
    select_best_thirds,
)


STAGE_ORDER = [
    "groups",
    "round_of_32",
    "round_of_16",
    "quarterfinals",
    "semifinals",
    "finals",
]

STAGE_LABELS = {
    "groups": "Fase de grupos",
    "round_of_32": "Dieciseisavos de final (32 equipos)",
    "round_of_16": "Octavos de final",
    "quarterfinals": "Cuartos de final",
    "semifinals": "Semifinales",
    "finals": "Tercer puesto y final",
}

KNOCKOUT_BY_STAGE = {
    "round_of_32": ROUND_OF_32,
    "round_of_16": ROUND_OF_16,
    "quarterfinals": QUARTERFINALS,
    "semifinals": SEMIFINALS,
    "finals": [THIRD_PLACE, FINAL],
}


@dataclass
class MatchRecord:
    stage: str
    group: Optional[str]
    match_id: Optional[int]
    home: str
    away: str
    home_score: int
    away_score: int
    winner: str


@dataclass
class JourneyState:
    """Mutable tournament path for one Monte Carlo trajectory."""

    seed: int = 42
    completed_stages: list[str] = field(default_factory=list)
    result: TournamentResult = field(default_factory=TournamentResult)
    ctx: dict[str, str] = field(default_factory=dict)
    match_log: list[MatchRecord] = field(default_factory=list)
    third_assignment: dict[str, str] = field(default_factory=dict)

    @property
    def next_stage(self) -> Optional[str]:
        for stage in STAGE_ORDER:
            if stage not in self.completed_stages:
                return stage
        return None

    def is_finished(self) -> bool:
        return self.next_stage is None


def _record_group_matches(
    state: JourneyState,
    group_name: str,
    teams: list[str],
    predictor: Predictor,
    rng: np.random.Generator,
    fixed_results: Optional[FixedResults] = None,
) -> list[GroupStanding]:
    """Simulate one group and append every match to the journey log."""
    standings = {t: GroupStanding(team=t) for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            fixed = fixed_results.get_group_match(group_name, a, b) if fixed_results else None
            if fixed is not None:
                if fixed["home"] == a:
                    ga_h, ga_a = fixed["home_score"], fixed["away_score"]
                else:
                    ga_h, ga_a = fixed["away_score"], fixed["home_score"]
                if ga_h > ga_a:
                    outcome = "home_win"
                elif ga_h < ga_a:
                    outcome = "away_win"
                else:
                    outcome = "draw"
            else:
                probs = predictor(a, b, venue="neutral")
                p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
                outcome = sample_match_result(p_h, p_d, p_a, rng)
                ga_h, ga_a = sample_goals(p_h, p_d, p_a, rng=rng)
                if outcome == "home_win" and ga_h <= ga_a:
                    ga_h = ga_a + 1
                elif outcome == "away_win" and ga_a <= ga_h:
                    ga_a = ga_h + 1
                elif outcome == "draw" and ga_h != ga_a:
                    ga_h = ga_a = max(ga_h, ga_a)

            winner = a if ga_h > ga_a else (b if ga_a > ga_h else "draw")
            state.match_log.append(
                MatchRecord(
                    stage="groups",
                    group=group_name.replace("Group_", ""),
                    match_id=None,
                    home=a,
                    away=b,
                    home_score=ga_h,
                    away_score=ga_a,
                    winner=winner,
                )
            )

            standings[a].goals_for += ga_h
            standings[a].goals_against += ga_a
            standings[b].goals_for += ga_a
            standings[b].goals_against += ga_h
            if outcome == "home_win":
                standings[a].points += 3
            elif outcome == "away_win":
                standings[b].points += 3
            else:
                standings[a].points += 1
                standings[b].points += 1

    return sorted(standings.values(), key=lambda s: s.sort_key)


def _build_ctx_from_groups(state: JourneyState) -> None:
    ctx: dict[str, str] = {}
    for grp, standings in state.result.group_standings.items():
        ctx[f"1st_{grp}"] = standings[0].team
        ctx[f"2nd_{grp}"] = standings[1].team
        for s in standings[3:]:
            state.result.progressions[s.team] = "group"
        for s in standings[:2]:
            state.result.progressions[s.team] = "round_of_32"

    best_thirds = select_best_thirds(state.result.group_standings, n=8)
    third_assignment = assign_thirds_to_slots(best_thirds, ROUND_OF_32, np.random.default_rng(state.seed))
    ctx.update(third_assignment)
    state.third_assignment = third_assignment

    picked = set(third_assignment.values())
    for grp, standings in state.result.group_standings.items():
        if len(standings) >= 3 and standings[2].team not in picked:
            state.result.progressions[standings[2].team] = "group"
    for team in picked:
        state.result.progressions[team] = "round_of_32"

    state.ctx = ctx


def run_group_stage(
    state: JourneyState,
    predictor: Predictor,
    groups: Optional[list[str]] = None,
    fixed_results: Optional[FixedResults] = None,
) -> JourneyState:
    """Simulate all groups (or a subset) and prepare knockout context."""
    if "groups" in state.completed_stages:
        return state

    rng = np.random.default_rng(state.seed + len(state.completed_stages))
    target_groups = groups or GROUP_NAMES

    for grp in target_groups:
        if grp not in WC2026_GROUPS:
            continue
        teams = WC2026_GROUPS[grp]
        if groups is not None and grp in state.result.group_standings:
            continue
        standings = _record_group_matches(state, grp, teams, predictor, rng, fixed_results)
        state.result.group_standings[grp] = standings

    if groups is None:
        _build_ctx_from_groups(state)
        state.completed_stages.append("groups")
    return state


def _simulate_knockout_matches(
    state: JourneyState,
    predictor: Predictor,
    matches: list,
    stage_key: str,
    rng: np.random.Generator,
    fixed_results: Optional[FixedResults] = None,
) -> None:
    for match in matches:
        team_a = resolve_slot(match.slot_a, state.ctx)
        team_b = resolve_slot(match.slot_b, state.ctx)

        fixed_winner = fixed_results.get_knockout_winner(match.match_id) if fixed_results else None
        if fixed_winner is not None and fixed_winner in (team_a, team_b):
            winner = fixed_winner
            loser = team_b if winner == team_a else team_a
            ga_h, ga_a = (2, 1) if winner == team_a else (1, 2)
        else:
            probs = predictor(team_a, team_b, venue="neutral")
            p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
            outcome = sample_knockout_winner(p_h, p_d, p_a, rng)
            winner, loser = (team_a, team_b) if outcome == "home_win" else (team_b, team_a)
            ga_h, ga_a = sample_goals(p_h, p_d, p_a, rng=rng)
            if winner == team_a and ga_h <= ga_a:
                ga_h = ga_a + 1
            elif winner == team_b and ga_a <= ga_h:
                ga_a = ga_h + 1

        state.result.knockout_winners[match.match_id] = winner
        state.result.knockout_losers[match.match_id] = loser
        state.ctx[f"winner_match_{match.match_id}"] = winner
        state.ctx[f"loser_match_{match.match_id}"] = loser

        state.match_log.append(
            MatchRecord(
                stage=stage_key,
                group=None,
                match_id=match.match_id,
                home=team_a,
                away=team_b,
                home_score=ga_h,
                away_score=ga_a,
                winner=winner,
            )
        )

        round_name = "final" if match.match_id == FINAL.match_id else stage_key
        for t in (team_a, team_b):
            current = state.result.progressions.get(t, "group")
            if t == winner:
                state.result.progressions[t] = round_name
            elif current == "group":
                state.result.progressions[t] = round_name

    if stage_key == "finals":
        state.result.champion = state.result.knockout_winners[FINAL.match_id]
        state.result.runner_up = state.result.knockout_losers[FINAL.match_id]
        state.result.third_place = state.result.knockout_winners[THIRD_PLACE.match_id]
        state.result.progressions[state.result.champion] = "champion"
        state.result.progressions[state.result.runner_up] = "final"


def run_knockout_stage(
    state: JourneyState,
    predictor: Predictor,
    stage_key: str,
    fixed_results: Optional[FixedResults] = None,
) -> JourneyState:
    if stage_key in state.completed_stages:
        return state
    if "groups" not in state.completed_stages:
        raise RuntimeError("Primero simulá la fase de grupos")

    rng = np.random.default_rng(state.seed + len(state.completed_stages) + 7)
    matches = KNOCKOUT_BY_STAGE[stage_key]
    _simulate_knockout_matches(state, predictor, matches, stage_key, rng, fixed_results)
    state.completed_stages.append(stage_key)
    return state


def run_full_tournament(
    predictor: Predictor,
    seed: int = 42,
    fixed_results: Optional[FixedResults] = None,
) -> JourneyState:
    state = JourneyState(seed=seed)
    run_group_stage(state, predictor, fixed_results=fixed_results)
    for stage in STAGE_ORDER[1:]:
        run_knockout_stage(state, predictor, stage, fixed_results=fixed_results)
    return state


def journey_to_fixed_results(state: JourneyState) -> FixedResults:
    """Convert the current path into FixedResults for conditional Monte Carlo."""
    fr = FixedResults()
    for m in state.match_log:
        if m.stage == "groups" and m.group is not None:
            fr.add_group_match(
                f"Group_{m.group}",
                m.home,
                m.away,
                m.home_score,
                m.away_score,
            )
        elif m.match_id is not None:
            fr.add_knockout_result(m.match_id, m.winner)
    return fr


def matches_for_stage(state: JourneyState, stage_key: str) -> list[MatchRecord]:
    return [m for m in state.match_log if m.stage == stage_key]


def standings_table(state: JourneyState, group_key: str = "Group_J") -> list[dict[str, Any]]:
    standings = state.result.group_standings.get(group_key, [])
    return [
        {
            "Equipo": s.team,
            "Pts": s.points,
            "GF": s.goals_for,
            "GC": s.goals_against,
            "DG": s.goal_diff,
        }
        for s in standings
    ]
