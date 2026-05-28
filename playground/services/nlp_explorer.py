"""NLP / vocabulary / embedding exploration for the playground."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PLAYGROUND = Path(__file__).resolve().parents[1]
DATA = PLAYGROUND / "data"
ARTIFACTS = PLAYGROUND / "artifacts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.vocabulary import (  # noqa: E402
    bucketize_elo,
    bucketize_form,
    bucketize_goals,
    team_token,
)

SEGMENT_NAMES = {
    0: "meta",
    1: "features",
    2: "lineup_a",
    3: "bench_a",
    4: "lineup_b",
    5: "bench_b",
    6: "events",
    7: "sep",
}

SEGMENT_COLORS = {
    "meta": "#74ACDF",
    "features": "#F6B504",
    "lineup_a": "#6A994E",
    "bench_a": "#A7C957",
    "lineup_b": "#E63946",
    "bench_b": "#F4A261",
    "events": "#9B5DE5",
    "sep": "#CCCCCC",
    "mask": "#1A1A2E",
}


def load_json(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def try_load_live_model() -> tuple[Any | None, Any | None, str]:
    """Load vocab + finetuned checkpoint if exported to artifacts/."""
    vocab_path = ARTIFACTS / "vocab.json"
    ckpt_path = ARTIFACTS / "d10sformer_finetune.pt"
    if not vocab_path.exists() or not ckpt_path.exists():
        return None, None, "datos precomputados (exportá vocab.json + checkpoint para modo live)"

    try:
        import torch
        from data.vocabulary import FootballVocab
        from models.d10sformer import D10Sformer, D10SformerConfig
        from eval.embedding_analysis import top_k_neighbours
    except ImportError:
        return None, None, "datos precomputados (instalá torch para modo live)"

    vocab = FootballVocab.load(vocab_path)
    cfg = D10SformerConfig(
        vocab_size=len(vocab),
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        pad_token_id=vocab.encode("[PAD]"),
        tie_mlm_weights=True,
    )
    model = D10Sformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model, vocab, "modelo en vivo"


def get_team_neighbors(query_token: str, model=None, vocab=None, k: int = 8) -> list[tuple[str, float]]:
    static = load_json("team_neighbors.json")
    if query_token in static:
        return [(t, s) for t, s in static[query_token][:k]]

    if model is not None and vocab is not None:
        from eval.embedding_analysis import top_k_neighbours

        restrict = [t for t in vocab.token_to_id if t.startswith("TEAM_")]
        return top_k_neighbours(model, vocab, query_token, k=k, restrict_to=restrict)

    return []


def build_custom_token_sequence(
    home: str,
    away: str,
    home_elo: float,
    away_elo: float,
    home_form: float = 2.4,
    away_form: float = 2.6,
    home_goals: float = 2.2,
    away_goals: float = 2.8,
    mask_bench: bool = False,
) -> list[dict]:
    """Build a simplified token sequence using vocabulary helpers (no full tokenizer)."""
    tokens: list[dict] = []
    seg = 0

    def add(token: str, segment: int) -> None:
        tokens.append(
            {
                "idx": len(tokens),
                "segment": segment,
                "token": token,
                "segment_name": SEGMENT_NAMES.get(segment, "?"),
            }
        )

    add("[CLS]", seg)
    add("TOURNAMENT_FIFA_WORLD_CUP", seg)
    add("STAGE_FINAL", seg)
    add("VENUE_NEUTRAL", seg)
    add(team_token(home), seg)
    add(team_token(away), seg)

    seg = 1
    add("[FEATURES_START]", seg)
    add(bucketize_elo(home_elo), seg)
    add(bucketize_elo(away_elo), seg)
    add(bucketize_form(home_form), seg)
    add(bucketize_form(away_form), seg)
    add(bucketize_goals(home_goals), seg)
    add(bucketize_goals(away_goals), seg)
    add("[FEATURES_END]", seg)

    seg = 2
    add("[LINEUP_A]", seg)
    for i in range(3):
        add(f"PLAYER_{1000 + i}", seg)

    seg = 3
    add("[BENCH_A]", seg)
    add("[MASK]" if mask_bench else "POS_MF_TIER_3", seg)

    seg = 4
    add("[LINEUP_B]", seg)
    for i in range(3):
        add(f"PLAYER_{2000 + i}", seg)

    add("[SEP]", 7)
    add("[PREDICT_RESULT]", 0)
    return tokens


def elo_similarity_profile(bucket: int) -> dict[str, float]:
    data = load_json("elo_cosine_matrix.json")
    labels = data["labels"]
    matrix = data["matrix"]
    if bucket not in labels:
        bucket = min(labels, key=lambda x: abs(x - bucket))
    idx = labels.index(bucket)
    row = matrix[idx]
    return {str(labels[j]): float(row[j]) for j in range(len(labels))}


@dataclass
class _VocabCache:
    player_names: dict[str, str]  # player_id_str -> human-readable name


_VOCAB_CACHE: _VocabCache | None = None


def _load_vocab_cache() -> _VocabCache:
    """Lee data/processed/vocab.json una sola vez y arma el mapa id → nombre.

    Si el vocab no existe en el repo (caso raro), devuelve un cache vacío.
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE

    vocab_path = ROOT / "data" / "processed" / "vocab.json"
    names: dict[str, str] = {}
    if vocab_path.exists():
        try:
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            for pid, info in payload.get("player_info", {}).items():
                name = info.get("name") or ""
                if name:
                    # Acortar nombres largos para que entren en el chip.
                    short = _short_player_name(name)
                    names[str(pid)] = short
        except Exception:
            names = {}
    _VOCAB_CACHE = _VocabCache(player_names=names)
    return _VOCAB_CACHE


_SURNAME_PARTICLES = {
    "di", "de", "da", "do", "du", "del", "della",
    "van", "von", "der", "den",
    "le", "la", "el", "al",
    "mc", "o'",
}


def _take_with_particle(parts: list[str], start_idx: int) -> str:
    """Si parts[start_idx] es una partícula ('Di', 'De', 'Van'…), concatena el siguiente.

    Convierte casos como ['Di', 'María'] en 'Di María' — apellido compuesto.
    """
    if start_idx >= len(parts):
        return parts[-1]
    if parts[start_idx].lower() in _SURNAME_PARTICLES and start_idx + 1 < len(parts):
        return f"{parts[start_idx]} {parts[start_idx + 1]}"
    return parts[start_idx]


def _short_player_name(full: str) -> str:
    """Acorta un nombre largo a 'I. Apellido' priorizando el apellido conocido.

    Reglas:
        4+ partes  → inicial + apellido paterno (parts[2], patrón hispano /
                     portugués: Nombre + Segundo + ApellidoPaterno + ApellidoMaterno).
        2-3 partes → inicial + último apellido.
        1 parte    → devolver tal cual.

    Maneja apellidos con partícula (Di María, De Paul, Van Persie, etc.):
    si la posición elegida cae sobre una partícula, se concatena la siguiente palabra.

    Filtra sufijos tipo 'Jr.' / 'Junior' antes de contar.

    Casos cubiertos:
        'Lionel Andrés Messi Cuccittini'      → 'L. Messi'
        'Ángel Fabián Di María Hernández'     → 'Á. Di María'
        'Rodrigo Javier De Paul'              → 'R. De Paul'
        'Antoine Griezmann'                   → 'A. Griezmann'
        'Lautaro Javier Martínez'             → 'L. Martínez'

    Caso conocido sin solución general:
        'Kylian Mbappé Lottin' → 'K. Lottin' (Mbappé es apellido paterno medio,
        pero no hay forma de distinguirlo de un segundo nombre sin metadata).
    """
    parts = [p for p in full.split() if p and p.lower() not in {"jr.", "jr", "junior"}]
    if not parts:
        return full
    first_initial = parts[0][0].upper() + "."
    if len(parts) >= 4:
        return f"{first_initial} {_take_with_particle(parts, 2)}"
    if len(parts) >= 2:
        return f"{first_initial} {_take_with_particle(parts, len(parts) - 1)}"
    return parts[0]


def format_token_label(token: str) -> str:
    """Etiqueta corta y human-readable para mostrar en chips / tablas / plots."""
    if token.startswith("PLAYER_"):
        pid = token.replace("PLAYER_", "")
        name = _load_vocab_cache().player_names.get(pid)
        if name:
            return name
        # Fallback: si por algún motivo no está en el vocab, mantenemos P#id.
        return f"P#{pid}"
    return (
        token.replace("TEAM_", "")
        .replace("ELO_BUCKET_", "ELO ")
        .replace("_", " ")
    )


TEAM_CONFEDERATIONS: dict[str, str] = {
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Spain": "UEFA",
    "France": "UEFA",
    "England": "UEFA",
    "Germany": "UEFA",
    "Netherlands": "UEFA",
    "Portugal": "UEFA",
    "Switzerland": "UEFA",
    "Belgium": "UEFA",
    "Norway": "UEFA",
    "Croatia": "UEFA",
    "Austria": "UEFA",
    "Turkey": "UEFA",
    "Sweden": "UEFA",
    "Czech Republic": "UEFA",
    "Scotland": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Mexico": "CONCACAF",
    "United States": "CONCACAF",
    "Canada": "CONCACAF",
    "Panama": "CONCACAF",
    "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF",
    "Morocco": "CAF",
    "Senegal": "CAF",
    "Algeria": "CAF",
    "Ghana": "CAF",
    "Ivory Coast": "CAF",
    "Egypt": "CAF",
    "Tunisia": "CAF",
    "South Africa": "CAF",
    "DR Congo": "CAF",
    "Cape Verde": "CAF",
    "Japan": "AFC",
    "South Korea": "AFC",
    "Iran": "AFC",
    "Saudi Arabia": "AFC",
    "Qatar": "AFC",
    "Iraq": "AFC",
    "Jordan": "AFC",
    "Uzbekistan": "AFC",
    "Australia": "AFC",
    "New Zealand": "OFC",
}


@dataclass
class ProjectionResult:
    tokens: list[str]
    coords: np.ndarray
    method: str
    source: str
    explained_variance: tuple[float, float] | None = None


def token_team_name(token: str) -> str | None:
    if not token.startswith("TEAM_"):
        return None
    # Tokens are uppercase (TEAM_ARGENTINA); metadata uses human-readable names.
    return token.replace("TEAM_", "").replace("_", " ").title()


def confederation_for_token(token: str) -> str:
    name = token_team_name(token)
    if name is None:
        if token.startswith("PLAYER_"):
            return "Jugador"
        if token.startswith("ELO_BUCKET_"):
            return "ELO bucket"
        if token.startswith("FORM_"):
            return "FORM bucket"
        return "otro"
    return TEAM_CONFEDERATIONS.get(name, "otro")


def demo_player_tokens() -> dict[str, list[str]]:
    """Player tokens from the Argentina-France sequence used in the token demo."""
    example = load_json("token_sequence_example.json")
    argentina = [
        row["token"]
        for row in example["tokens"]
        if row["segment"] == 2 and row["token"].startswith("PLAYER_")
    ]
    france = [
        row["token"]
        for row in example["tokens"]
        if row["segment"] == 4 and row["token"].startswith("PLAYER_")
    ]
    return {
        "Argentina": argentina,
        "France": france,
        "Todos": argentina + france,
    }


def demo_player_team_map() -> dict[str, str]:
    players = demo_player_tokens()
    return {
        **{token: "TEAM_ARGENTINA" for token in players["Argentina"]},
        **{token: "TEAM_FRANCE" for token in players["France"]},
    }


# ---------------------------------------------------------------------------
# Segment introspection — used by the "📖 Segmentos" tab
# ---------------------------------------------------------------------------

# Descripción canónica de los 8 segmentos del MatchTokenizer.
SEGMENT_DETAILS: dict[int, dict] = {
    0: {
        "name": "meta",
        "title": "Meta del partido",
        "description": (
            "Encabezado de la secuencia: marcadores `[CLS]` y `[PREDICT_RESULT]`, "
            "torneo, etapa, venue y los dos equipos. El `[CLS]` es la posición "
            "desde la que el head de clasificación lee la predicción final."
        ),
        "vocab_prefixes": ["TOURNAMENT_", "STAGE_", "VENUE_", "TEAM_"],
        "specials": ["[CLS]", "[PREDICT_RESULT]"],
    },
    1: {
        "name": "features",
        "title": "Features tabulares bucketizadas",
        "description": (
            "ELO, forma reciente (puntos por partido) y goles recientes — "
            "todos discretizados en buckets. El vocabulario aprende relaciones "
            "ordinales entre buckets aunque nunca recibe esa señal explícita."
        ),
        "vocab_prefixes": ["ELO_BUCKET_", "FORM_", "GOALS_"],
        "specials": ["[FEATURES_START]", "[FEATURES_END]"],
    },
    2: {
        "name": "lineup_a",
        "title": "Alineación titular · equipo A",
        "description": (
            "Los 11 titulares del equipo local (o equipo A en venue neutral). "
            "Cada jugador → token dedicado `PLAYER_<id>` si juega ≥10 partidos, "
            "o fallback positional `POS_<pos>_TIER_<t>` si es long-tail."
        ),
        "vocab_prefixes": ["PLAYER_", "POS_"],
        "specials": ["[LINEUP_A]"],
    },
    3: {
        "name": "bench_a",
        "title": "Suplentes · equipo A",
        "description": (
            "Banco del equipo A. Misma codificación que la titular. "
            "Durante el entrenamiento, este segmento puede enmascararse "
            "estocásticamente para forzar al modelo a ser robusto a info parcial."
        ),
        "vocab_prefixes": ["PLAYER_", "POS_"],
        "specials": ["[BENCH_A]", "[MASK]"],
    },
    4: {
        "name": "lineup_b",
        "title": "Alineación titular · equipo B",
        "description": "Idéntico al segmento 2, para el equipo visitante.",
        "vocab_prefixes": ["PLAYER_", "POS_"],
        "specials": ["[LINEUP_B]"],
    },
    5: {
        "name": "bench_b",
        "title": "Suplentes · equipo B",
        "description": "Idéntico al segmento 3, para el equipo visitante.",
        "vocab_prefixes": ["PLAYER_", "POS_"],
        "specials": ["[BENCH_B]", "[MASK]"],
    },
    6: {
        "name": "events",
        "title": "Eventos del partido",
        "description": (
            "Subset filtrado (goles, tarjetas, sustituciones). Cada evento se "
            "tokeniza como [TEAM_<X>_TURN] + bucket de minuto + tipo + jugador. "
            "En inferencia (partidos futuros) este segmento típicamente va con [MASK]."
        ),
        "vocab_prefixes": ["EVENT_", "MIN_"],
        "specials": [
            "[EVENTS_START]", "[EVENTS_END]",
            "[TEAM_A_TURN]", "[TEAM_B_TURN]",
        ],
    },
    7: {
        "name": "sep",
        "title": "Separador",
        "description": (
            "Marca el cierre de la secuencia. Solo contiene `[SEP]`. Existe por "
            "compatibilidad con el paradigma BERT/sentence-pair, aunque acá no "
            "hay frase B."
        ),
        "vocab_prefixes": [],
        "specials": ["[SEP]"],
    },
}


def example_tokens_by_segment() -> dict[int, list[dict]]:
    """Tokens del ejemplo demo (ARG-FRA) agrupados por segmento.

    Devuelve {segment_id: [{"token": str, "label": str, "idx": int}, ...]}
    """
    example = load_json("token_sequence_example.json")
    by_seg: dict[int, list[dict]] = {seg: [] for seg in SEGMENT_DETAILS}
    for row in example["tokens"]:
        seg = int(row["segment"])
        if seg not in by_seg:
            by_seg[seg] = []
        by_seg[seg].append(
            {
                "idx": row["idx"],
                "token": row["token"],
                "label": format_token_label(row["token"]),
            }
        )
    return by_seg


def vocab_counts_by_segment() -> dict[int, dict[str, int]]:
    """Cuántos tokens del vocab caen en cada categoría de cada segmento.

    Devuelve {segment_id: {prefix: count, ...}}.
    """
    vocab_path = ROOT / "data" / "processed" / "vocab.json"
    if not vocab_path.exists():
        return {seg: {} for seg in SEGMENT_DETAILS}

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    all_tokens = list(payload.get("token_to_id", {}).keys())

    out: dict[int, dict[str, int]] = {}
    for seg, info in SEGMENT_DETAILS.items():
        counts = {}
        for prefix in info["vocab_prefixes"]:
            counts[prefix] = sum(1 for t in all_tokens if t.startswith(prefix))
        out[seg] = counts
    return out


def projection_token_sets() -> dict[str, list[str]]:
    """Token groups available for embedding maps."""
    features = load_json("team_features.json")
    teams = [team_token(name) for name in features.keys()]
    demo_players = demo_player_tokens()
    final_teams = ["TEAM_ARGENTINA", "TEAM_FRANCE"]
    elo = [f"ELO_BUCKET_{b}" for b in range(1500, 2301, 100)]
    form = [
        "FORM_VERY_LOW",
        "FORM_LOW",
        "FORM_MEDIUM_LOW",
        "FORM_MEDIUM",
        "FORM_MEDIUM_HIGH",
        "FORM_HIGH",
        "FORM_VERY_HIGH",
    ]
    return {
        "Selecciones WC 2026": teams,
        "Final 2022: equipos + jugadores": final_teams + demo_players["Todos"],
        "Final 2022: jugadores": demo_players["Todos"],
        "Argentina: equipo + jugadores": ["TEAM_ARGENTINA"] + demo_players["Argentina"],
        "Francia: equipo + jugadores": ["TEAM_FRANCE"] + demo_players["France"],
        "Selecciones + buckets ELO": teams + elo,
        "Solo buckets ELO": elo,
        "Solo buckets FORM": form,
    }


def _pca_nd(model, vocab, tokens: list[str], dims: int) -> tuple[list[str], np.ndarray, tuple[float, float] | None]:
    from eval.embedding_analysis import get_embeddings_matrix

    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    x = mat.cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    n = min(dims, vt.shape[0])
    coords = x @ vt[:n].T
    total_var = float((s**2).sum())
    ev = None
    if total_var > 0 and n >= 2:
        ev = (float(s[0] ** 2) / total_var, float(s[1] ** 2) / total_var)
    return resolved, coords, ev


def _tsne_nd(model, vocab, tokens: list[str], dims: int, perplexity: float) -> tuple[list[str], np.ndarray, None]:
    from sklearn.manifold import TSNE
    from eval.embedding_analysis import get_embeddings_matrix

    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    x = mat.cpu().numpy()
    n = x.shape[0]
    if n < 4:
        raise ValueError("t-SNE necesita al menos 4 puntos")
    perp = min(perplexity, max(5.0, n / 4 - 1))
    tsne = TSNE(
        n_components=dims,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(x)
    return resolved, coords, None


def _mds_from_similarity(tokens: list[str], sim: np.ndarray, dims: int) -> np.ndarray:
    from sklearn.manifold import MDS

    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=dims, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    return mds.fit_transform(dist)


def _fallback_similarity_matrix(tokens: list[str]) -> np.ndarray:
    """Approximate cosine layout when no checkpoint is available."""
    features = load_json("team_features.json")
    neighbors = load_json("team_neighbors.json")
    player_team = demo_player_team_map()
    n = len(tokens)
    sim = np.eye(n, dtype=float)
    for i, ti in enumerate(tokens):
        for j in range(i + 1, n):
            tj = tokens[j]
            value = None
            for src, pairs in neighbors.items():
                if src == ti:
                    for nb, s in pairs:
                        if nb == tj:
                            value = float(s)
                            break
                elif src == tj:
                    for nb, s in pairs:
                        if nb == ti:
                            value = float(s)
                            break
                if value is not None:
                    break
            if value is None:
                ni, nj = token_team_name(ti), token_team_name(tj)
                if ti.startswith("PLAYER_") or tj.startswith("PLAYER_"):
                    team_i = player_team.get(ti, ti if ti.startswith("TEAM_") else None)
                    team_j = player_team.get(tj, tj if tj.startswith("TEAM_") else None)
                    if team_i is not None and team_i == team_j:
                        value = 0.72
                    elif team_i is not None and team_j is not None:
                        value = 0.36
                    else:
                        value = 0.22
                elif ni and nj and ni in features and nj in features:
                    de = abs(features[ni]["elo"] - features[nj]["elo"])
                    value = max(0.12, 1.0 - de / 900.0)
                elif ti.startswith("ELO_BUCKET_") and tj.startswith("ELO_BUCKET_"):
                    bi = int(ti.split("_")[-1])
                    bj = int(tj.split("_")[-1])
                    value = max(0.05, 1.0 - abs(bi - bj) / 800.0)
                else:
                    value = 0.25
            sim[i, j] = sim[j, i] = value
    return sim


def _cosine_similarity_matrix(model, vocab, tokens: list[str]) -> tuple[list[str], np.ndarray]:
    from eval.embedding_analysis import get_embeddings_matrix

    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    mat_n = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    sim = (mat_n @ mat_n.T).cpu().numpy()
    return resolved, sim


def compute_embedding_projection(
    model,
    vocab,
    token_group: str,
    method: Literal["pca", "tsne", "mds"],
    dims: Literal[2, 3],
    perplexity: float = 15.0,
) -> ProjectionResult:
    tokens = projection_token_sets()[token_group]
    has_model = model is not None and vocab is not None

    if has_model and method == "pca":
        resolved, coords, ev = _pca_nd(model, vocab, tokens, dims)
        return ProjectionResult(resolved, coords, "pca", "embeddings reales (checkpoint)", ev)
    if has_model and method == "tsne":
        resolved, coords, _ = _tsne_nd(model, vocab, tokens, dims, perplexity)
        return ProjectionResult(resolved, coords, "tsne", "embeddings reales (checkpoint)", None)
    if has_model and method == "mds":
        resolved, sim = _cosine_similarity_matrix(model, vocab, tokens)
        coords = _mds_from_similarity(resolved, sim, dims)
        return ProjectionResult(resolved, coords, "mds", "MDS sobre cosenos reales (checkpoint)", None)

    sim = _fallback_similarity_matrix(tokens)
    coords = _mds_from_similarity(tokens, sim, dims)
    label = "MDS sobre similitudes aproximadas (ELO + vecinos precomputados)"
    if not has_model and method in {"pca", "tsne"}:
        label += f". Exportá vocab.json + d10sformer_finetune.pt para {method.upper()} real."
    return ProjectionResult(tokens, coords, "mds", label, None)


def projection_to_dataframe(result: ProjectionResult) -> pd.DataFrame:
    rows = []
    features = load_json("team_features.json")
    for i, token in enumerate(result.tokens):
        name = format_token_label(token)
        conf = confederation_for_token(token)
        team_name = token_team_name(token)
        player_team = demo_player_team_map().get(token)
        elo = features.get(team_name, {}).get("elo") if team_name else None
        row = {
            "token": token,
            "label": name,
            "confederation": conf,
            "equipo_asociado": format_token_label(player_team) if player_team else "",
            "elo": elo,
            "x": float(result.coords[i, 0]),
            "y": float(result.coords[i, 1]),
        }
        if result.coords.shape[1] > 2:
            row["z"] = float(result.coords[i, 2])
        rows.append(row)
    return pd.DataFrame(rows)


def neighbor_edges_for_highlight(
    token: str,
    df: pd.DataFrame,
    model=None,
    vocab=None,
    k: int = 4,
) -> list[tuple[str, str, float]]:
    """Return (source_label, target_label, sim) for drawing proximity lines."""
    neighbors = get_team_neighbors(token, model, vocab, k=k)
    by_token = df.set_index("token")
    if token not in by_token.index:
        return []
    src = by_token.loc[token, "label"]
    edges = []
    for nb, sim in neighbors:
        if nb not in by_token.index:
            continue
        edges.append((src, by_token.loc[nb, "label"], sim))
    return edges
