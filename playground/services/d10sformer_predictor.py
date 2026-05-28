"""D10Sformer-based predictor for the playground.

Envuelve el Transformer entrenado (fine-tune weighted) en la firma esperada
por el simulador Monte Carlo: `predict(team_a, team_b, venue) -> np.ndarray[3]`
con orden `[p_home_win, p_draw, p_away_win]`.

Uso:

    from services.d10sformer_predictor import D10SformerPredictor

    predictor = D10SformerPredictor(
        ckpt_path="checkpoints/finetune_weighted_15ep/best.pt",
        vocab_path="data/processed/vocab.json",
        team_features=load_team_features(),
        device="cpu",
    )
    probs = predictor.predict("Argentina", "France", venue="neutral")

Notas:
    - El forward pass se hace en torch.no_grad() y .eval() para inferencia pura.
    - max_seq_length=80, num_segments=8 — DEBE coincidir con el config del fine-tune.
    - Las secciones lineup/bench/events son None (no las tenemos para partidos
      futuros). El tokenizer las reemplaza por [MASK]; el modelo fue entrenado con
      feature-masking estocástico para ser robusto a esto.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Asegurar import desde src/
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.vocabulary import FootballVocab               # noqa: E402
from data.tokenizer import (                             # noqa: E402
    MatchTokenizer,
    MatchDocument,
    RollingFeatures,
)
from models.d10sformer import D10Sformer, D10SformerConfig  # noqa: E402
from inference.predictions import (  # noqa: E402
    NUM_SCORE_CLASSES,
    joint_score_probs_to_result_probs,
    score_logits_to_probs,
)


# ---------------------------------------------------------------------------
# Hyperparameters used during fine-tune (notebooks/04e_finetune_weighted.ipynb)
# Si re-entrenás con un config distinto, actualizá estos valores acá.
# ---------------------------------------------------------------------------
FT_D_MODEL = 256
FT_NUM_LAYERS = 6
FT_NUM_HEADS = 8
FT_D_FF = 1024
FT_MAX_SEQ_LENGTH = 80
FT_NUM_SEGMENTS = 8
FT_DROPOUT = 0.1
FT_TIE_MLM_WEIGHTS = True

# Resultado: 0 = home_win, 1 = draw, 2 = away_win (ver src/data/feature_engineering.py)
RESULT_HOME_WIN = 0
RESULT_DRAW = 1
RESULT_AWAY_WIN = 2


InferenceMode = str  # "result" | "score_derived"


class D10SformerPredictor:
    """Predictor que usa el D10Sformer fine-tuneado para inferencia W/D/L.

    La firma `predict(team_a, team_b, venue) -> np.ndarray[3]` cumple con la
    que espera `simulation.simulator.PrecomputedPredictor`, por lo que es un
    drop-in replacement del `logreg_predictor` del notebook 06.

    inference_mode:
        - "result": softmax sobre ResultHead (checkpoints 04d/04e).
        - "score_derived": softmax sobre ScoreHead (36 clases) y agrega a W/D/L
          (recomendado para checkpoints entrenados con LossSpec.finetune_score_primary).
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        vocab_path: str | Path,
        team_features: dict[str, dict],
        device: str | torch.device = "cpu",
        max_seq_length: int = FT_MAX_SEQ_LENGTH,
        tournament: str = "FIFA World Cup",
        inference_mode: InferenceMode = "result",
    ):
        if inference_mode not in ("result", "score_derived"):
            raise ValueError(
                f"inference_mode debe ser 'result' o 'score_derived', recibido: {inference_mode!r}"
            )
        self.inference_mode = inference_mode
        self.device = torch.device(device)
        self.max_seq_length = max_seq_length
        self.team_features = team_features
        self.tournament = tournament

        # 1) Vocab
        self.vocab = FootballVocab.load(Path(vocab_path))

        # 2) Tokenizer (mismo max_seq_length que en training)
        self.tokenizer = MatchTokenizer(self.vocab, max_seq_length=max_seq_length)

        # 3) Modelo (mismo config que en fine-tune)
        config = D10SformerConfig(
            vocab_size=len(self.vocab),
            d_model=FT_D_MODEL,
            num_layers=FT_NUM_LAYERS,
            num_heads=FT_NUM_HEADS,
            d_ff=FT_D_FF,
            max_seq_length=max_seq_length,
            num_segments=FT_NUM_SEGMENTS,
            dropout=FT_DROPOUT,
            pad_token_id=self.vocab.encode("[PAD]"),
            tie_mlm_weights=FT_TIE_MLM_WEIGHTS,
        )
        self.config = config
        self.model = D10Sformer(config)

        # 4) Cargar pesos
        ckpt = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            # Compatibilidad si alguien guardó solo el state_dict
            state = ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Diagnóstico útil — pero no abortar
            print(
                f"[D10SformerPredictor] load_state_dict: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
            if missing:
                print(f"  primeros missing: {missing[:5]}")
            if unexpected:
                print(f"  primeros unexpected: {unexpected[:5]}")

        self.model.to(self.device)
        self.model.eval()

        self.pad_id = self.vocab.encode("[PAD]")
        self._ckpt_step = ckpt.get("step", None)
        self._ckpt_val_loss = ckpt.get("best_val_loss", None)

    # ----------------------------- inference -----------------------------

    def _build_doc(
        self,
        team_a: str,
        team_b: str,
        venue: str = "neutral",
    ) -> MatchDocument:
        """Construye un MatchDocument sin lineups/eventos (lo que sabemos a priori)."""
        fa = self.team_features.get(team_a, {})
        fb = self.team_features.get(team_b, {})

        feats = RollingFeatures(
            home_elo=float(fa.get("elo")) if "elo" in fa else None,
            away_elo=float(fb.get("elo")) if "elo" in fb else None,
            home_form_pts=float(fa.get("form_pts")) if "form_pts" in fa else None,
            away_form_pts=float(fb.get("form_pts")) if "form_pts" in fb else None,
            home_recent_goals=float(fa.get("recent_goals")) if "recent_goals" in fa else None,
            away_recent_goals=float(fb.get("recent_goals")) if "recent_goals" in fb else None,
        )

        return MatchDocument(
            tournament=self.tournament,
            stage=None,
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            features=feats,
            lineup_a=None,
            bench_a=None,
            lineup_b=None,
            bench_b=None,
            events=None,
        )

    def _pad_to_max(self, token_ids: list[int], segment_ids: list[int]):
        """Pad/truncate hasta max_seq_length y crea attention_mask."""
        L = min(len(token_ids), self.max_seq_length)
        ids = list(token_ids[:L])
        segs = list(segment_ids[:L])
        attn = [1] * L

        # Pad
        n_pad = self.max_seq_length - L
        if n_pad > 0:
            ids.extend([self.pad_id] * n_pad)
            segs.extend([0] * n_pad)
            attn.extend([0] * n_pad)

        return ids, segs, attn

    def _forward(self, token_ids, segment_ids, attention_mask) -> dict:
        return self.model(
            token_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
        )

    def _result_probs_from_output(self, out: dict) -> np.ndarray:
        if self.inference_mode == "result":
            logits = out["result_logits"]
            if logits.dim() == 1:
                return F.softmax(logits, dim=-1).cpu().numpy().astype(float)
            return F.softmax(logits, dim=-1).cpu().numpy().astype(float)
        score_probs = score_logits_to_probs(out["score_logits"])
        return joint_score_probs_to_result_probs(score_probs)

    @torch.no_grad()
    def predict(
        self,
        team_a: str,
        team_b: str,
        venue: str = "neutral",
    ) -> np.ndarray:
        """Inferencia para un partido. Devuelve [p_home_win, p_draw, p_away_win]."""
        doc = self._build_doc(team_a, team_b, venue=venue)
        tok = self.tokenizer.tokenize(doc)

        ids, segs, attn = self._pad_to_max(tok.token_ids, tok.segment_ids)

        token_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segs], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        out = self._forward(token_ids, segment_ids, attention_mask)
        probs = self._result_probs_from_output(out)
        if probs.ndim == 2:
            return probs[0]
        return probs

    @torch.no_grad()
    def predict_score(
        self,
        team_a: str,
        team_b: str,
        venue: str = "neutral",
    ) -> np.ndarray:
        """Distribución conjunta sobre marcadores SCORE_h_a (36 clases, h,a ∈ 0..5)."""
        doc = self._build_doc(team_a, team_b, venue=venue)
        tok = self.tokenizer.tokenize(doc)
        ids, segs, attn = self._pad_to_max(tok.token_ids, tok.segment_ids)

        token_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segs], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        out = self._forward(token_ids, segment_ids, attention_mask)
        return score_logits_to_probs(out["score_logits"][0])

    @torch.no_grad()
    def predict_batch(
        self,
        pairs: list[tuple[str, str, str]],
    ) -> np.ndarray:
        """Inferencia batched. `pairs` es lista de (team_a, team_b, venue).

        Devuelve array (N, 3).
        """
        if not pairs:
            return np.zeros((0, 3), dtype=float)

        all_ids, all_segs, all_attn = [], [], []
        for team_a, team_b, venue in pairs:
            doc = self._build_doc(team_a, team_b, venue=venue)
            tok = self.tokenizer.tokenize(doc)
            ids, segs, attn = self._pad_to_max(tok.token_ids, tok.segment_ids)
            all_ids.append(ids)
            all_segs.append(segs)
            all_attn.append(attn)

        token_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(all_segs, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(all_attn, dtype=torch.long, device=self.device)

        out = self._forward(token_ids, segment_ids, attention_mask)
        return self._result_probs_from_output(out)

    # ----------------------------- introspection -----------------------------

    def describe(self) -> dict:
        return {
            "engine": "D10Sformer (fine-tune weighted)",
            "inference_mode": self.inference_mode,
            "num_score_classes": NUM_SCORE_CLASSES,
            "vocab_size": len(self.vocab),
            "params": self.model.num_parameters(),
            "d_model": self.config.d_model,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "max_seq_length": self.max_seq_length,
            "ckpt_step": self._ckpt_step,
            "ckpt_val_loss": self._ckpt_val_loss,
            "device": str(self.device),
        }

    def __call__(self, team_a: str, team_b: str, venue: str = "neutral") -> np.ndarray:
        """Permite usar la instancia directamente como callable Predictor."""
        return self.predict(team_a, team_b, venue=venue)
