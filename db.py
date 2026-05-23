"""Supabase Storage-backed persistence.

We skip Postgres tables on purpose: creating them needs either the DB
password (not exposed in the dashboard without a click-through) or a
Personal Access Token. The service_role key, on the other hand, can
fully manage Storage. So every record lives as a pair of files in a
single bucket:

    world-cup/
      photos/{id}.jpg                          original group photo
      photos/{id}.json                         {id, created_at, width, height, source, face_count}
      faces/{photo_id}/{face_index}.jpg        512x512 crop
      faces/{photo_id}/{face_index}.json       {id, photo_id, face_index, bbox, attributes, ...}
      generations/{photo_id}/{id}.png          mirrored team photo
      generations/{photo_id}/{id}.json         {id, photo_id, country_code, prompt, model, ...}

All public functions are best-effort: any failure is logged and the
photo->generate flow continues.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

BUCKET = "world-cup"
FRAME_PATH = Path(__file__).resolve().parent / "svg" / "frame.png"

_client = None
_initialized = False
_bucket_ready = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_client():
    """Lazy singleton. Returns the Supabase client, or None if disabled."""
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        log.info("Supabase disabled: SUPABASE_URL or SUPABASE_KEY missing")
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        log.info("Supabase client initialized for %s", url)
        _ensure_bucket()
    except Exception as e:
        log.warning("Supabase init failed: %s: %s", e.__class__.__name__, e)
        _client = None
    return _client


def has_supabase() -> bool:
    return init_client() is not None


def public_url(storage_path: Optional[str]) -> Optional[str]:
    """Build the public URL for a storage_path. Returns None if disabled or
    no path. Doesn't validate that the object exists."""
    if not storage_path:
        return None
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not base:
        return None
    return f"{base}/storage/v1/object/public/{BUCKET}/{storage_path}"


def _rest_headers():
    key = os.environ.get("SUPABASE_KEY", "")
    return {"Authorization": f"Bearer {key}", "apikey": key}


def _ensure_bucket():
    """Idempotent — check, then create if missing. Called once at init."""
    global _bucket_ready
    if _bucket_ready:
        return
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        return
    headers = _rest_headers()
    try:
        r = requests.get(f"{url}/storage/v1/bucket/{BUCKET}", headers=headers, timeout=15)
        if r.status_code == 200:
            log.info("Supabase bucket '%s' exists", BUCKET)
            _bucket_ready = True
            return
    except Exception as e:
        log.warning("Bucket check failed: %s: %s", e.__class__.__name__, e)
        return
    try:
        r = requests.post(
            f"{url}/storage/v1/bucket",
            headers={**headers, "Content-Type": "application/json"},
            json={"id": BUCKET, "name": BUCKET, "public": True},
            timeout=15,
        )
        if r.status_code in (200, 201):
            log.info("Created Supabase bucket '%s' (public)", BUCKET)
            _bucket_ready = True
        elif r.status_code == 409:
            log.info("Bucket '%s' already existed", BUCKET)
            _bucket_ready = True
        else:
            log.warning("Bucket create returned %d: %s", r.status_code, r.text[:300])
    except Exception as e:
        log.warning("Bucket create failed: %s: %s", e.__class__.__name__, e)


def _data_uri_to_bytes(data_uri: str):
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError("not a data URI")
    header, _, b64 = data_uri.partition(",")
    if ";base64" not in header:
        raise ValueError("expected base64 data URI")
    mime = header[len("data:"):].split(";")[0] or "application/octet-stream"
    return base64.b64decode(b64), mime


def _ext_for_mime(mime: str) -> str:
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(mime, "bin")


def _upload_bytes(path: str, data: bytes, mime: str, upsert: bool = False) -> bool:
    """Upload via raw REST so we control upsert via the x-upsert header
    regardless of supabase-py version quirks."""
    client = init_client()
    if not client:
        return False
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    headers = {**_rest_headers(), "Content-Type": mime}
    if upsert:
        headers["x-upsert"] = "true"
    try:
        full = f"{url}/storage/v1/object/{BUCKET}/{path}"
        method = requests.put if upsert else requests.post
        r = method(full, headers=headers, data=data, timeout=(10, 60))
        if r.status_code in (200, 201):
            return True
        log.warning("storage upload %s -> %d: %s", path, r.status_code, r.text[:200])
        return False
    except Exception as e:
        log.warning("storage upload failed for %s: %s: %s", path, e.__class__.__name__, e)
        return False


def _upload_data_uri(path: str, data_uri: str, upsert: bool = False) -> bool:
    try:
        data, mime = _data_uri_to_bytes(data_uri)
    except Exception as e:
        log.warning("data URI decode failed for %s: %s", path, e)
        return False
    return _upload_bytes(path, data, mime, upsert=upsert)


def _upload_json(path: str, obj: dict, upsert: bool = False) -> bool:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return _upload_bytes(path, data, "application/json", upsert=upsert)


def _fetch_json(path: str) -> Optional[dict]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    try:
        r = requests.get(f"{url}/storage/v1/object/public/{BUCKET}/{path}",
                         headers=_rest_headers(), timeout=15)
        if r.ok:
            return r.json()
    except Exception as e:
        log.warning("fetch_json failed for %s: %s", path, e)
    return None


# ── Public API used by server.py ──────────────────────────────────────────────

def insert_photo(jpeg_data_uri: str, width: Optional[int], height: Optional[int],
                 source: Optional[str]) -> Optional[str]:
    """Persist a captured photo. Uploads the JPEG and a JSON sidecar.
    Returns the new photo_id (uuid), or None on failure."""
    client = init_client()
    if not client:
        return None
    try:
        photo_id = str(uuid.uuid4())
        try:
            _, mime = _data_uri_to_bytes(jpeg_data_uri)
        except Exception:
            mime = "image/jpeg"
        ext = _ext_for_mime(mime)
        img_path = f"photos/{photo_id}.{ext}"
        meta_path = f"photos/{photo_id}.json"
        _upload_data_uri(img_path, jpeg_data_uri)
        _upload_json(meta_path, {
            "id": photo_id,
            "created_at": _now_iso(),
            "image": img_path,
            "width": width if isinstance(width, int) else None,
            "height": height if isinstance(height, int) else None,
            "source": source if isinstance(source, str) else None,
            "face_count": None,
        })
        return photo_id
    except Exception as e:
        log.warning("insert_photo failed: %s: %s", e.__class__.__name__, e)
        return None


def insert_face(photo_id: Optional[str], face_index: Optional[int],
                bbox: Optional[dict], crop_data_uri: Optional[str],
                attributes: Optional[dict], attr_status: str = "ok",
                attr_error: Optional[str] = None,
                tokens: Optional[int] = None) -> Optional[str]:
    """Persist a face crop + its vision attributes."""
    client = init_client()
    if not client or not photo_id:
        return None
    try:
        face_id = str(uuid.uuid4())
        idx = face_index if isinstance(face_index, int) else 0
        img_path = None
        if crop_data_uri:
            try:
                _, mime = _data_uri_to_bytes(crop_data_uri)
            except Exception:
                mime = "image/jpeg"
            ext = _ext_for_mime(mime)
            img_path = f"faces/{photo_id}/{idx}.{ext}"
            if not _upload_data_uri(img_path, crop_data_uri, upsert=True):
                img_path = None
        meta_path = f"faces/{photo_id}/{idx}.json"
        _upload_json(meta_path, {
            "id": face_id,
            "created_at": _now_iso(),
            "photo_id": photo_id,
            "face_index": idx,
            "bbox": bbox if isinstance(bbox, dict) else None,
            "image": img_path,
            "attributes": attributes if isinstance(attributes, dict) else None,
            "attr_status": attr_status,
            "attr_error": attr_error,
            "tokens": tokens if isinstance(tokens, int) else None,
        }, upsert=True)
        return face_id
    except Exception as e:
        log.warning("insert_face failed: %s: %s", e.__class__.__name__, e)
        return None


def count_photos() -> int:
    """Return how many distinct photos exist (each photo is a .json sidecar
    in photos/). Returns 0 on failure or when Supabase is disabled."""
    if not init_client():
        return 0
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    try:
        r = requests.post(
            f"{url}/storage/v1/object/list/{BUCKET}",
            headers={**_rest_headers(), "Content-Type": "application/json"},
            json={"prefix": "photos/", "limit": 1000},
            timeout=15,
        )
        if not r.ok:
            return 0
        return sum(1 for item in r.json() if (item.get("name") or "").endswith(".json"))
    except Exception as e:
        log.warning("count_photos failed: %s: %s", e.__class__.__name__, e)
        return 0


def update_photo_face_count(photo_id: Optional[str], count: int) -> None:
    """Re-upload the photo JSON with face_count populated."""
    client = init_client()
    if not client or not photo_id:
        return
    meta_path = f"photos/{photo_id}.json"
    try:
        meta = _fetch_json(meta_path) or {"id": photo_id, "created_at": _now_iso()}
        meta["face_count"] = int(count)
        _upload_json(meta_path, meta, upsert=True)
    except Exception as e:
        log.warning("update_photo_face_count failed: %s: %s", e.__class__.__name__, e)


def _apply_frame(team_bytes: bytes) -> Optional[bytes]:
    """Overlay svg/frame.png onto the generated team photo using Pillow.
    Returns framed PNG bytes, or None if frame.png is missing / Pillow fails."""
    if not FRAME_PATH.is_file():
        return None
    try:
        from PIL import Image
        team = Image.open(BytesIO(team_bytes)).convert("RGBA")
        frame = Image.open(FRAME_PATH).convert("RGBA")
        if frame.size != team.size:
            frame = frame.resize(team.size, Image.LANCZOS)
        team.alpha_composite(frame)
        out = BytesIO()
        team.convert("RGB").save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("frame composite failed: %s: %s", e.__class__.__name__, e)
        return None


def _mirror_generated_image(generation_id: str, photo_id: Optional[str],
                            image_url: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (framed_path, raw_path). framed_path is the primary; raw_path
    keeps the original unframed image so we can re-frame later if needed."""
    try:
        if image_url.startswith("data:"):
            data, mime = _data_uri_to_bytes(image_url)
        else:
            r = requests.get(image_url, timeout=(10, 60))
            r.raise_for_status()
            mime = (r.headers.get("Content-Type") or "image/png").split(";")[0].strip()
            data = r.content

        prefix = f"generations/{photo_id or 'orphan'}/{generation_id}"
        raw_ext = _ext_for_mime(mime)
        raw_path = f"{prefix}_raw.{raw_ext}"
        _upload_bytes(raw_path, data, mime)

        framed = _apply_frame(data)
        if framed:
            framed_path = f"{prefix}.png"
            ok = _upload_bytes(framed_path, framed, "image/png")
            return (framed_path if ok else None, raw_path)
        # Frame failed — fall back to using the raw as the primary path.
        return (raw_path, raw_path)
    except Exception as e:
        log.warning("mirror_generated_image failed: %s: %s", e.__class__.__name__, e)
        return (None, None)


def insert_generation(photo_id: Optional[str], country_code: Optional[str],
                      country_name: Optional[str], n_people: Optional[int],
                      prompt: Optional[str], model: Optional[str],
                      elapsed: Optional[float], verified_count: Optional[int],
                      mismatch: Optional[bool], gmi_image_url: Optional[str],
                      error: Optional[str] = None,
                      tokens: Optional[int] = None) -> Optional[dict]:
    """Mirror the team photo (framed and raw) to Storage and write a JSON
    sidecar. Returns {id, storage_path, raw_path} or None on failure."""
    client = init_client()
    if not client:
        return None
    try:
        gen_id = str(uuid.uuid4())
        framed_path = raw_path = None
        if gmi_image_url and not error:
            framed_path, raw_path = _mirror_generated_image(gen_id, photo_id, gmi_image_url)
        meta_path = f"generations/{photo_id or 'orphan'}/{gen_id}.json"
        _upload_json(meta_path, {
            "id": gen_id,
            "created_at": _now_iso(),
            "photo_id": photo_id,
            "country_code": country_code or None,
            "country_name": country_name or None,
            "n_people": n_people if isinstance(n_people, int) else None,
            "prompt": prompt or None,
            "model": model or None,
            "elapsed_seconds": float(elapsed) if isinstance(elapsed, (int, float)) else None,
            "tokens": tokens if isinstance(tokens, int) else None,
            "verified_count": verified_count if isinstance(verified_count, int) else None,
            "mismatch": bool(mismatch) if mismatch is not None else False,
            "gmi_image_url": gmi_image_url or None,
            "image": framed_path,
            "image_raw": raw_path,
            "error": error or None,
        })
        return {"id": gen_id, "storage_path": framed_path, "raw_path": raw_path}
    except Exception as e:
        log.warning("insert_generation failed: %s: %s", e.__class__.__name__, e)
        return None
