"""
Embedding analysis utilities for D10Sformer.

Extracts the static token embeddings from a trained model and provides
classical analyses (cosine similarity, vector analogies, k-NN, t-SNE/PCA
projection) that demonstrate whether the model learned semantically
meaningful representations of teams, players, and bucketed features.

References:
    Mikolov et al. (2013) "Distributed Representations of Words and Phrases"
        — vector analogies (King - Man + Woman ≈ Queen).
    van der Maaten & Hinton (2008) "Visualizing Data using t-SNE".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_token_embedding(model, vocab, token: str) -> torch.Tensor:
    """Returns the d_model-dim embedding vector of `token` from the model.

    Falls back to [UNK] if the token is not present. The returned tensor
    is detached (no grad) and on the same device as the model.
    """
    tid = vocab.encode(token)
    with torch.no_grad():
        emb = model.embeddings.token_embedding.weight[tid].detach().clone()
    return emb


def get_embeddings_matrix(model, vocab, tokens: Iterable[str]) -> tuple[list[str], torch.Tensor]:
    """Stack embeddings of multiple tokens into a (N, d_model) tensor.

    Returns:
        (resolved_tokens, embeddings) — `resolved_tokens` is the input list
        with unknown tokens dropped.
    """
    resolved = []
    vectors = []
    for t in tokens:
        if vocab.has(t):
            resolved.append(t)
            vectors.append(get_token_embedding(model, vocab, t))
    if not vectors:
        raise ValueError("None of the requested tokens are in the vocabulary.")
    return resolved, torch.stack(vectors, dim=0)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors (or matched batched tensors)."""
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def top_k_neighbours(
    model,
    vocab,
    query: str | torch.Tensor,
    k: int = 10,
    restrict_to: Optional[list[str]] = None,
    exclude_self: bool = True,
) -> list[tuple[str, float]]:
    """Returns the top-k tokens by cosine similarity to `query`.

    Args:
        query: token string OR a 1-D embedding vector.
        k: number of neighbours.
        restrict_to: optional list of candidate tokens. If None, search the
            full vocab.
        exclude_self: if True, drop the query itself from results (when query
            is a string).

    Returns:
        List of (token, cosine_similarity) sorted descending.
    """
    if isinstance(query, str):
        q_emb = get_token_embedding(model, vocab, query)
        self_token = query
    else:
        q_emb = query
        self_token = None

    if restrict_to is None:
        all_tokens = list(vocab.token_to_id.keys())
    else:
        all_tokens = [t for t in restrict_to if vocab.has(t)]

    resolved, mat = get_embeddings_matrix(model, vocab, all_tokens)
    # Normalised cosine
    q_norm = q_emb / (q_emb.norm() + 1e-12)
    mat_norm = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    sims = (mat_norm @ q_norm).cpu().numpy()

    order = np.argsort(-sims)
    out = []
    for idx in order:
        tok = resolved[idx]
        if exclude_self and tok == self_token:
            continue
        out.append((tok, float(sims[idx])))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Vector analogies — "A is to B as C is to ___"
# ---------------------------------------------------------------------------

def analogy_query(
    model, vocab,
    a: str, b: str, c: str,
    k: int = 5,
    restrict_to: Optional[list[str]] = None,
    normalize: bool = False,
) -> list[tuple[str, float]]:
    """Mikolov-style additive analogy: query = vec(b) - vec(a) + vec(c).

    Returns the top-k tokens nearest to that query vector (excluding the
    inputs themselves).

    Args:
        normalize: if True, L2-normalise each vector before the arithmetic.
            Empirically reduces magnitude bias in Transformer embeddings.

    Example:
        analogy_query(model, vocab, 'TEAM_ARGENTINA', 'PLAYER_MESSI',
                      'TEAM_FRANCE') → ['PLAYER_MBAPPE', ...]
    """
    a_emb = get_token_embedding(model, vocab, a)
    b_emb = get_token_embedding(model, vocab, b)
    c_emb = get_token_embedding(model, vocab, c)
    if normalize:
        a_emb = a_emb / (a_emb.norm() + 1e-12)
        b_emb = b_emb / (b_emb.norm() + 1e-12)
        c_emb = c_emb / (c_emb.norm() + 1e-12)
    query = b_emb - a_emb + c_emb

    raw = top_k_neighbours(model, vocab, query, k=k + 3, restrict_to=restrict_to,
                            exclude_self=False)
    excluded = {a, b, c}
    filtered = [(t, s) for t, s in raw if t not in excluded]
    return filtered[:k]


def analogy_query_3cosmul(
    model, vocab,
    a: str, b: str, c: str,
    k: int = 5,
    restrict_to: Optional[list[str]] = None,
    epsilon: float = 1e-3,
) -> list[tuple[str, float]]:
    """Levy & Goldberg (2014) 3CosMul analogy. Multiplicative variant.

    Instead of arg max_d cos(d, b - a + c), it computes:

        arg max_d   [cos(d, b) * cos(d, c)] / [cos(d, a) + epsilon]

    where cosine similarities are first shifted to be non-negative
    (s' = (s + 1) / 2) to avoid sign issues with the division.

    Empirically this method outperforms additive 3CosAdd on the standard
    Word2Vec analogy benchmarks by ~5pp because no single term can dominate.
    """
    import torch.nn.functional as F

    a_emb = get_token_embedding(model, vocab, a)
    b_emb = get_token_embedding(model, vocab, b)
    c_emb = get_token_embedding(model, vocab, c)

    if restrict_to is None:
        candidate_tokens = list(vocab.token_to_id.keys())
    else:
        candidate_tokens = [t for t in restrict_to if vocab.has(t)]
    resolved, mat = get_embeddings_matrix(model, vocab, candidate_tokens)
    # Normalise all vectors to compute cosine sim cleanly
    mat_n = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    a_n = a_emb / (a_emb.norm() + 1e-12)
    b_n = b_emb / (b_emb.norm() + 1e-12)
    c_n = c_emb / (c_emb.norm() + 1e-12)

    sim_a = (mat_n @ a_n).cpu().numpy()
    sim_b = (mat_n @ b_n).cpu().numpy()
    sim_c = (mat_n @ c_n).cpu().numpy()

    # Shift to [0, 1] to avoid negative similarities breaking the multiplicative form
    sim_a = (sim_a + 1) / 2
    sim_b = (sim_b + 1) / 2
    sim_c = (sim_c + 1) / 2

    scores = (sim_b * sim_c) / (sim_a + epsilon)

    order = np.argsort(-scores)
    excluded = {a, b, c}
    out = []
    for idx in order:
        tok = resolved[idx]
        if tok in excluded:
            continue
        out.append((tok, float(scores[idx])))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Geometric diagnostics
# ---------------------------------------------------------------------------

def family_norm_stats(model, vocab, family_prefixes: dict[str, str]) -> dict:
    """For each token family (e.g., 'TEAM_', 'PLAYER_'), compute the mean
    L2-norm of the embeddings. Useful for diagnosing magnitude bias in
    analogy arithmetic.

    Args:
        family_prefixes: dict like {'team': 'TEAM_', 'player': 'PLAYER_'}.

    Returns:
        dict of {family_name: {'n': count, 'mean_norm': float, 'std_norm': float}}.
    """
    import torch
    stats = {}
    embedding_table = model.embeddings.token_embedding.weight.detach()
    for name, prefix in family_prefixes.items():
        ids = [vocab.token_to_id[t] for t in vocab.token_to_id if t.startswith(prefix)]
        if not ids:
            stats[name] = {"n": 0, "mean_norm": 0.0, "std_norm": 0.0}
            continue
        vecs = embedding_table[torch.tensor(ids)]
        norms = vecs.norm(dim=1).cpu().numpy()
        stats[name] = {
            "n": len(ids),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
        }
    return stats


def intra_family_cosine_distribution(model, vocab, prefix: str,
                                      sample_size: int = 200,
                                      seed: int = 42) -> dict:
    """Compute the distribution of pairwise cosine similarities within a
    token family. A 'cone-collapsed' embedding space (anisotropy) shows up
    here as a high mean similarity even between semantically unrelated tokens.

    Returns:
        dict with 'mean', 'median', 'p25', 'p75', 'min', 'max', 'n_pairs'.
    """
    import torch
    import random
    tokens = [t for t in vocab.token_to_id if t.startswith(prefix)]
    if len(tokens) < 2:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "min": 0.0, "max": 0.0, "n_pairs": 0}
    random.seed(seed)
    if len(tokens) > sample_size:
        tokens = random.sample(tokens, sample_size)
    _, mat = get_embeddings_matrix(model, vocab, tokens)
    mat_n = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    sim_matrix = (mat_n @ mat_n.T).cpu().numpy()
    # Upper triangle excluding diagonal
    iu = np.triu_indices(len(tokens), k=1)
    sims = sim_matrix[iu]
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "n_pairs": len(sims),
    }


def cluster_centroid(model, vocab, tokens: list[str], normalize: bool = True):
    """Compute the centroid embedding of a cluster of tokens (e.g., all
    CONMEBOL teams). If normalize=True, each vector is L2-normalised
    before averaging (avoids magnitude domination)."""
    import torch
    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    if normalize:
        mat = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    return mat.mean(dim=0)


# ---------------------------------------------------------------------------
# Dimensionality reduction (PCA + optional t-SNE)
# ---------------------------------------------------------------------------

@dataclass
class Reduction2D:
    """Result of a 2-D projection of embeddings."""
    tokens: list[str]
    coords: np.ndarray   # (N, 2)
    method: str          # 'pca' or 'tsne'
    explained_variance: Optional[tuple[float, float]] = None  # only for PCA


def pca_2d(model, vocab, tokens: Iterable[str]) -> Reduction2D:
    """Project embeddings to 2D via PCA. Returns Reduction2D."""
    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    X = mat.cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)   # centre
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T   # project on first 2 PCs

    total_var = (S ** 2).sum()
    ev = ((S[0] ** 2) / total_var, (S[1] ** 2) / total_var) if total_var > 0 else (0.0, 0.0)
    return Reduction2D(tokens=resolved, coords=coords, method="pca", explained_variance=ev)


def tsne_2d(
    model, vocab, tokens: Iterable[str],
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Reduction2D:
    """Project embeddings to 2D via t-SNE (uses sklearn).

    For visual clusters of teams/players. Less interpretable than PCA but
    typically much more readable in scatter plots.
    """
    from sklearn.manifold import TSNE
    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    X = mat.cpu().numpy()
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"t-SNE needs >= 4 points; got {n}")
    perp = min(perplexity, max(5.0, n / 4 - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, init="pca")
    coords = tsne.fit_transform(X)
    return Reduction2D(tokens=resolved, coords=coords, method="tsne")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def relative_similarity_score(
    model, vocab,
    anchor: str,
    positive: str,
    negative: str,
) -> dict:
    """Returns cos(anchor, positive), cos(anchor, negative), and Δ.

    Δ > 0 means the model places `positive` closer to `anchor` than `negative`,
    which is the expected behaviour for semantically aligned pairs (e.g.,
    anchor=PLAYER_MESSI, positive=TEAM_ARGENTINA, negative=TEAM_FRANCE).
    """
    a = get_token_embedding(model, vocab, anchor)
    p = get_token_embedding(model, vocab, positive)
    n = get_token_embedding(model, vocab, negative)
    cs_p = cosine_sim(a, p)
    cs_n = cosine_sim(a, n)
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "cos_positive": cs_p,
        "cos_negative": cs_n,
        "delta": cs_p - cs_n,
        "correct": cs_p > cs_n,
    }
