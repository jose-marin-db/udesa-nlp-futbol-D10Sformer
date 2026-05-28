#!/usr/bin/env python3
"""
Sube archivos del repo local a MyDrive/d10sformer-v2.

Requiere OAuth (una vez):
  pip install google-api-python-client google-auth-oauthlib
  python scripts/push_v2_to_drive.py --auth

Luego:
  python scripts/push_v2_to_drive.py
"""

from __future__ import annotations

import argparse
import mimetypes
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_FOLDER_ID = "1Xz1rbw8t8jF_6J5Ez-_vUb7MuPG69w-O"
TOKEN_PATH = ROOT / ".drive_upload_token.pickle"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# (ruta local relativa, carpeta destino relativa a d10sformer-v2)
UPLOAD_MANIFEST = [
    ("src/paths.py", "src"),
    ("src/data/collator.py", "src/data"),
    ("src/inference/__init__.py", "src/inference"),
    ("src/inference/predictions.py", "src/inference"),
    ("src/training/trainer.py", "src/training"),
    ("plan_v2.md", ""),
    ("DRIVE_SETUP.md", ""),
    ("data/README.md", "data"),
    ("configs/base_config.yaml", "configs"),
    ("playground/services/d10sformer_predictor.py", "playground/services"),
    ("tests/test_collator.py", "tests"),
    ("tests/test_inference_predictions.py", "tests"),
    ("tests/test_loss_spec.py", "tests"),
]

NOTEBOOK_GLOB = "notebooks/*.ipynb"


def _credentials(auth: bool):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN_PATH.exists():
        creds = pickle.loads(TOKEN_PATH.read_bytes())
    if auth or not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret = ROOT / "drive_client_secret.json"
            if not client_secret.exists():
                raise SystemExit(
                    "Falta drive_client_secret.json en la raíz del repo.\n"
                    "Creá credenciales OAuth Desktop en Google Cloud Console "
                    "y descargá el JSON como drive_client_secret.json,\n"
                    "o usá notebooks/99_upload_project_to_drive.ipynb en Colab."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_bytes(pickle.dumps(creds))
    return creds


def _folder_cache(service):
    cache: dict[tuple[str, str], str] = {}

    def ensure(parent_id: str, name: str) -> str:
        key = (parent_id, name)
        if key in cache:
            return cache[key]
        q = (
            f"'{parent_id}' in parents and name = '{name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        res = service.files().list(q=q, fields="files(id)").execute()
        files = res.get("files", [])
        if files:
            fid = files[0]["id"]
        else:
            meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
            fid = service.files().create(body=meta, fields="id").execute()["id"]
        cache[key] = fid
        return fid

    return ensure


def _upload_file(service, local: Path, parent_id: str) -> str:
    from googleapiclient.http import MediaFileUpload

    q = f"'{parent_id}' in parents and name = '{local.name}' and trashed = false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    mime, _ = mimetypes.guess_type(str(local))
    media = MediaFileUpload(str(local), mimetype=mime or "application/octet-stream", resumable=True)
    if existing:
        fid = existing[0]["id"]
        service.files().update(fileId=fid, media_body=media).execute()
        return f"updated {local.name}"
    body = {"name": local.name, "parents": [parent_id]}
    service.files().create(body=body, media_body=media, fields="id").execute()
    return f"created {local.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true")
    parser.add_argument("--folder-id", default=V2_FOLDER_ID)
    args = parser.parse_args()

    from googleapiclient.discovery import build

    service = build("drive", "v3", credentials=_credentials(args.auth))
    ensure = _folder_cache(service)

    for rel, dest in UPLOAD_MANIFEST:
        local = ROOT / rel
        if not local.exists():
            print(f"SKIP missing {rel}")
            continue
        parent = args.folder_id
        if dest:
            for part in dest.split("/"):
                parent = ensure(parent, part)
        print(_upload_file(service, local, parent))

    nb_parent = ensure(args.folder_id, "notebooks")
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        print(_upload_file(service, nb, nb_parent))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
