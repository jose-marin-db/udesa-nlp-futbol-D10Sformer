"""
Rutas del proyecto D10Sformer (v2) para Colab y ejecución local.

v2 separa:
  - PROJECT_ROOT: código, checkpoints nuevos, notebooks (carpeta d10sformer-v2 en Drive)
  - DATA_ROOT:    datos pesados compartidos con v1 (carpeta d10sformer en Drive)

En Colab, montá Drive y usá los nombres por defecto bajo MyDrive/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# IDs de carpetas en Google Drive (referencia; Colab usa paths por nombre)
DRIVE_FOLDER_ORIGINAL_DATA = "1fLNr0QUdJsFqtxPSzz6hsDaKGgi5Bx4F"  # MyDrive/d10sformer
DRIVE_FOLDER_V2_PROJECT = "1Xz1rbw8t8jF_6J5Ez-_vUb7MuPG69w-O"  # MyDrive/d10sformer-v2


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    src: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    corpus_dir: Path
    vocab_path: Path
    checkpoints: Path
    checkpoints_v1: Path
    notebooks: Path
    reports: Path
    logs: Path

    def ensure_dirs(self) -> None:
        """Crea solo carpetas de salida bajo PROJECT_ROOT (no duplica data/)."""
        for p in (
            self.checkpoints,
            self.reports,
            self.logs,
            self.project_root / "data" / "processed" / "corpus",
        ):
            p.mkdir(parents=True, exist_ok=True)


def resolve_colab_project_root(drive: Path | None = None) -> Path:
    """Detecta PROJECT_ROOT en Colab (soporta MyDrive/d10sformer-v2 anidado)."""
    drive = drive or Path("/content/drive/MyDrive")
    for candidate in (
        drive / "d10sformer-v2" / "d10sformer-v2",
        drive / "d10sformer-v2",
    ):
        if (candidate / "src").is_dir():
            return candidate.resolve()
    return (drive / "d10sformer-v2").resolve()


def resolve_colab_data_root(drive: Path | None = None) -> Path:
    drive = drive or Path("/content/drive/MyDrive")
    root = drive / "d10sformer"
    if (root / "data" / "processed" / "vocab.json").is_file():
        return root.resolve()
    raise FileNotFoundError(
        f"No encuentro {root / 'data' / 'processed' / 'vocab.json'}. "
        "DATA_ROOT debe ser la carpeta original d10sformer en MyDrive."
    )


def _default_colab_roots() -> tuple[Path, Path]:
    drive = Path("/content/drive/MyDrive")
    return resolve_colab_project_root(drive), resolve_colab_data_root(drive)


def collator_has_label_mapping(project_root: Path) -> bool:
    collator_py = project_root / "src" / "data" / "collator.py"
    if not collator_py.is_file():
        return False
    return "class LabelMappedCollator" in collator_py.read_text(encoding="utf-8")


def resolve_paths(
    *,
    in_colab: bool | None = None,
    project_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> ProjectPaths:
    """Resuelve rutas según entorno.

    Prioridad:
      1. Argumentos explícitos
      2. Variables de entorno D10S_PROJECT_ROOT / D10S_DATA_ROOT
      3. Colab: MyDrive/d10sformer-v2 + MyDrive/d10sformer
      4. Local: raíz del repo (.. desde notebooks/) para ambos
    """
    if in_colab is None:
        try:
            import google.colab  # noqa: F401

            in_colab = True
        except ImportError:
            in_colab = False

    env_project = os.environ.get("D10S_PROJECT_ROOT")
    env_data = os.environ.get("D10S_DATA_ROOT")

    if project_root is not None:
        proj = Path(project_root)
    elif env_project:
        proj = Path(env_project)
    elif in_colab:
        proj, _ = _default_colab_roots()
    else:
        proj = _guess_local_project_root()

    if data_root is not None:
        data = Path(data_root)
    elif env_data:
        data = Path(env_data)
    elif in_colab:
        _, data = _default_colab_roots()
    else:
        data = proj

    processed = data / "data" / "processed"
    return ProjectPaths(
        project_root=proj,
        data_root=data,
        src=proj / "src",
        data_raw=data / "data" / "raw",
        data_interim=data / "data" / "interim",
        data_processed=processed,
        corpus_dir=processed / "corpus",
        vocab_path=processed / "vocab.json",
        checkpoints=proj / "checkpoints",
        checkpoints_v1=data / "checkpoints",
        notebooks=proj / "notebooks",
        reports=proj / "reports",
        logs=proj / "logs",
    )


def _guess_local_project_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "notebooks" and (cwd.parent / "src").is_dir():
        return cwd.parent
    if (cwd / "src").is_dir():
        return cwd
    return cwd.parent


def ensure_paths(
    *,
    in_colab: bool | None = None,
    project_root: Path | str | None = None,
    data_root: Path | str | None = None,
    require_vocab: bool = False,
    require_corpus: bool = False,
) -> ProjectPaths:
    """Valida que existan rutas críticas y crea dirs de salida en v2."""
    paths = resolve_paths(
        in_colab=in_colab,
        project_root=project_root,
        data_root=data_root,
    )
    paths.ensure_dirs()

    if not paths.data_raw.exists():
        raise FileNotFoundError(
            f"No existe DATA_ROOT/raw: {paths.data_raw}\n"
            "Montá Drive y apuntá DATA_ROOT a la carpeta original "
            "'d10sformer' (id 1fLNr0QUdJsFqtxPSzz6hsDaKGgi5Bx4F)."
        )
    if require_vocab and not paths.vocab_path.exists():
        raise FileNotFoundError(
            f"Falta vocab.json en {paths.vocab_path}. "
            "Corré 02_tokenization.ipynb en la carpeta de datos (v1) o copiá el artefacto."
        )
    if require_corpus:
        for name in ("pretrain.pkl", "finetune_train.pkl", "val.pkl", "test.pkl"):
            p = paths.corpus_dir / name
            if not p.exists():
                raise FileNotFoundError(
                    f"Falta corpus {p}. Corré 04a_dataset_collator.ipynb contra DATA_ROOT."
                )
    return paths


def print_paths(paths: ProjectPaths) -> None:
    print("=== D10Sformer paths ===")
    print(f"  PROJECT_ROOT (código/ckpt v2): {paths.project_root}")
    print(f"  DATA_ROOT      (datos v1):     {paths.data_root}")
    print(f"  vocab:          {paths.vocab_path}  ({'OK' if paths.vocab_path.exists() else 'MISSING'})")
    print(f"  corpus:         {paths.corpus_dir}  ({'OK' if paths.corpus_dir.exists() else 'MISSING'})")
    print(f"  checkpoints:    {paths.checkpoints}")
