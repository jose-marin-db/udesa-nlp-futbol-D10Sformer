#!/usr/bin/env python3
"""Genera notebooks/04f_finetune_score_primary.ipynb desde 04e (copia segura)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "04e_finetune_weighted.ipynb"
DST = ROOT / "notebooks" / "04f_finetune_score_primary.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = [
    "# 04f — Fine-tuning **score-first** (v2)\n",
    "\n",
    "**Objetivo:** `LossSpec.finetune_score_primary()` — marcador 36 clases; W/D/L se deriva en inferencia.\n",
    "\n",
    "- Datos: `DATA_ROOT` = `MyDrive/d10sformer`\n",
    "- Código/ckpt: `PROJECT_ROOT` = `MyDrive/d10sformer-v2`\n",
    "- Pretrain: `checkpoints_v1/pretrain_5ep/best.pt`\n",
]

# Quitar celda de class weights (solo cuenta train)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "train_counts = Counter()" in src or (
        "Distribución de clases W/D/L omitida" in src
    ):
        nb["cells"][i]["source"] = [
            "import pickle\n",
            "from data.vocabulary import FootballVocab\n",
            "from data.tokenizer import MatchTokenizer\n",
            "\n",
            "vocab = FootballVocab.load(VOCAB_PATH)\n",
            "tokenizer = MatchTokenizer(vocab, max_seq_length=80)\n",
            "print(f'Vocab: {len(vocab):,} tokens  |  PAD={vocab.decode(vocab.pad_id)!r}')\n",
            "\n",
            "with open(CORPUS_DIR / 'finetune_train.pkl', 'rb') as f:\n",
            "    finetune_docs = pickle.load(f)\n",
            "with open(CORPUS_DIR / 'val.pkl', 'rb') as f:\n",
            "    val_docs = pickle.load(f)\n",
            "with open(CORPUS_DIR / 'test.pkl', 'rb') as f:\n",
            "    test_docs = pickle.load(f)\n",
            "\n",
            "print(f'Fine-tune train: {len(finetune_docs):,}')\n",
            "print(f'Val:             {len(val_docs):,}')\n",
            "print(f'Test:            {len(test_docs):,}')\n",
        ]
    if "RESULT_VOCAB_TO_LOCAL" in src and "LabelMappedCollator" in src:
        nb["cells"][i]["source"] = [
            "from data.collator import MLMCollator, LabelMappedCollator\n",
            "\n",
            "ds_train = MatchDataset(finetune_docs, tokenizer)\n",
            "ds_val   = MatchDataset(val_docs, tokenizer)\n",
            "ds_test  = MatchDataset(test_docs, tokenizer)\n",
            "\n",
            "collator = LabelMappedCollator(MLMCollator(vocab, mlm_probability=0.15, seed=42))\n",
            "_t = collator([ds_train[0]])\n",
            "assert _t.score_labels.max() < 36\n",
            "\n",
            "BATCH_SIZE = 64\n",
            "train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)\n",
            "val_loader   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)\n",
            "test_loader  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)\n",
            "steps_per_epoch = len(train_loader)\n",
            "print(f'Train batches/época: {steps_per_epoch}')\n",
        ]
    if "run_name='finetune_weighted_15ep'" in src:
        src = src.replace("finetune_weighted_15ep", "finetune_score_15ep")
        src = src.replace(
            "loss_spec = LossSpec(\n"
            "    use_mlm=True,    lambda_mlm=0.2,\n"
            "    use_result=True, lambda_result=1.0,\n"
            "    use_score=True,  lambda_score=0.3,\n"
            "    result_class_weights=result_class_weights,   # ← NUEVO\n"
            ")\n"
            "print(f'\\nLossSpec: λ_mlm={loss_spec.lambda_mlm}, λ_result={loss_spec.lambda_result}, λ_score={loss_spec.lambda_score}')\n"
            "print(f'Result class weights: {result_class_weights}')\n",
            "loss_spec = LossSpec.finetune_score_primary()\n"
            "print(f'\\nLossSpec: λ_mlm={loss_spec.lambda_mlm}, λ_score={loss_spec.lambda_score}, use_result={loss_spec.use_result}')\n",
        )
        src = src.replace(
            "print(f'Result weights tensor: {trainer._result_weights_tensor}')",
            "print('Score-first: sin result weights tensor')",
        )
        nb["cells"][i]["source"] = [src]
    if "## 5. Fine-tuning (15 épocas con class weights)" in src:
        nb["cells"][i]["source"] = ["---\n", "## 5. Fine-tuning score-first (15 épocas)\n"]
    if "result_class_weights' in LossSpec" in src:
        nb["cells"][i]["source"] = [
            "assert hasattr(LossSpec, 'finetune_score_primary'), 'Subí trainer.py v2 a Drive'\n",
            "print('✓ trainer.py v2 OK')\n",
        ]

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {DST}")
