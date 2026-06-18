import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import db

ROOT = Path(__file__).resolve().parent
SVG_DIR = ROOT / "svg"


def load_dotenv(path: Path) -> None:
    """Tiny .env loader. Sets keys in os.environ unless already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")
GMI_API_KEY = os.environ.get("GMI_API_KEY", "").strip()
GMI_CHAT_URL = "https://api.gmi-serving.com/v1/chat/completions"
GMI_QUEUE_URL = "https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey/requests"
# Gemini 3.1 flash-lite returns valid JSON in ~3s vs gpt-4o's ~5s on the same
# face attribute prompt — about 40% faster per call. Override via VISION_MODEL.
VISION_MODEL = os.environ.get("VISION_MODEL", "google/gemini-3.1-flash-lite-preview")
# gemini-3.1-flash-image-preview wins on the bench (scripts/bench_top5.py):
# avg 14.6s with consistent 85/100 identity, vs gpt-image-2-edit at 124s avg
# (variable 50-92 id, with frequent 60s+ outliers). Override via EDIT_MODEL env
# var to fall back to gpt-image-2-edit for max-identity at the cost of speed.
EDIT_MODEL = os.environ.get("EDIT_MODEL", "gpt-image-2-edit")

VISION_PROMPT = (
    "You are analyzing one cropped face. Return ONLY a JSON object with exactly these keys:\n"
    '  "apparent_age_band": one of "child","teen","20s","30s","40s","50s","60+"\n'
    '  "apparent_gender_presentation": one of "masculine","feminine","ambiguous"\n'
    '  "hair": short string like "short black", "long wavy blonde", "bald"\n'
    '  "facial_hair": one of "none","stubble","mustache","beard"\n'
    '  "glasses": boolean (true/false)\n'
    '  "skin_tone": one of "very light","light","medium","tan","brown","dark"\n'
    '  "distinguishing": at most 6 words, or empty string\n'
    "Do not describe clothing. Do not attempt to identify the person."
)

VERIFY_PROMPT = (
    'Count the distinct human people fully visible in this image. '
    'Return ONLY JSON: {"count": <integer>}'
)

IDENTITY_PROMPT = (
    "You are comparing two photos of a face. The FIRST image is the reference "
    "(ground truth). The SECOND image is a generated image. Score how well the "
    "generated face matches the SAME PERSON in the reference. Compare: face shape, "
    "jawline, eyes (shape and color), nose, mouth, hairline, hair color/texture, "
    "facial hair, skin tone, eyebrow shape, ears, age. Ignore: clothing, background, "
    "pose, expression, lighting. Return ONLY JSON: "
    '{"score": <0-100 integer>, "verdict": "<same|similar|different>", "notes": "<one short sentence>"}'
    " where 90+ = clearly the same person, 70-89 = strong resemblance, "
    "50-69 = some likeness, 30-49 = different but related features, 0-29 = different person."
)

app = Flask(__name__)


def gmi_headers():
    return {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json",
    }


def call_gmi_chat(messages, *, json_mode=True, max_tokens=300, temperature=0.2):
    """Returns (parsed_content, usage_dict). usage is {} if absent."""
    body = {
        "model": VISION_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(GMI_CHAT_URL, headers=gmi_headers(), json=body, timeout=(10, 60))
    r.raise_for_status()
    payload = r.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content) if json_mode else content
    return parsed, (payload.get("usage") or {})


@app.get("/")
def serve_index():
    return send_from_directory(ROOT, "index.html")


@app.get("/architecture")
def serve_architecture():
    return send_from_directory(ROOT, "architecture.html")


@app.get("/countries.json")
def serve_countries():
    return send_from_directory(ROOT, "countries.json", mimetype="application/json")


@app.get("/svg/<path:filename>")
def serve_svg(filename: str):
    if re.fullmatch(r"[A-Za-z0-9_-]+\.svg", filename):
        return send_from_directory(SVG_DIR, filename, mimetype="image/svg+xml")
    if re.fullmatch(r"[A-Za-z0-9_-]+\.png", filename):
        return send_from_directory(SVG_DIR, filename, mimetype="image/png")
    abort(404)


@app.get("/GDG-header.png")
def serve_header():
    return send_from_directory(ROOT, "GDG-header.png", mimetype="image/png")


@app.get("/test-photo.jpg")
def _test_photo():
    # Dev-only: lets the headless test driver fetch the local fixture
    # from the page without going through the OS file picker.
    return send_from_directory(".", "test-photo.jpg")


@app.get("/group-photo.png")
def _group_photo():
    # Dev fixture for multi-face detection testing.
    return send_from_directory(".", "group-photo.png")


@app.get("/api/header-exists")
def header_exists():
    return jsonify(exists=(ROOT / "GDG-header.png").is_file())


@app.get("/api/health")
def health():
    return jsonify(ok=True, has_key=bool(GMI_API_KEY), has_supabase=db.has_supabase())


@app.get("/api/stats")
def api_stats():
    """Aggregate counters surfaced on the result screen."""
    return jsonify(total_photos=db.count_photos())


@app.post("/api/photos")
def api_photos():
    """Persist a freshly-captured photo. Returns {photo_id} (or null on failure
    or when Supabase is disabled). Best-effort: never blocks the UX."""
    body = request.get_json(silent=True) or {}
    image = body.get("image")
    if not isinstance(image, str) or not image.startswith("data:image/"):
        return jsonify(error="image (data URI) required"), 400
    width = body.get("width") if isinstance(body.get("width"), int) else None
    height = body.get("height") if isinstance(body.get("height"), int) else None
    source = body.get("source") if isinstance(body.get("source"), str) else None
    photo_id = db.insert_photo(image, width, height, source)
    return jsonify(photo_id=photo_id)


@app.post("/api/identity-score")
def api_identity_score():
    """Compare a reference face vs a generated image and return a 0-100 match score."""
    body = request.get_json(silent=True) or {}
    reference = body.get("reference_image")  # data URI
    generated_url = body.get("generated_url")  # URL or data URI
    if not isinstance(reference, str) or not reference.startswith("data:image/"):
        return jsonify(error="reference_image (data URI) required"), 400
    if not isinstance(generated_url, str) or not generated_url:
        return jsonify(error="generated_url required"), 400
    # Use a stronger vision model for face comparison than the per-face attribute
    # extraction one — gpt-4o sees faces better than gemini-flash-lite.
    scoring_model = os.environ.get("IDENTITY_MODEL", "openai/gpt-4o")
    body_chat = {
        "model": scoring_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IDENTITY_PROMPT},
                    {"type": "image_url", "image_url": {"url": reference}},
                    {"type": "image_url", "image_url": {"url": generated_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 200,
        "temperature": 0.1,
    }
    try:
        r = requests.post(GMI_CHAT_URL, headers=gmi_headers(), json=body_chat, timeout=(10, 60))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return jsonify(result=json.loads(content), scoring_model=scoring_model)
    except requests.HTTPError as e:
        return jsonify(error=f"identity scoring error: {e.response.status_code} {e.response.text[:200]}"), 502
    except Exception as e:
        return jsonify(error=f"identity scoring failed: {e.__class__.__name__}: {e}"), 502


@app.post("/api/vision")
def api_vision():
    body = request.get_json(silent=True) or {}
    image = body.get("image")
    if not isinstance(image, str) or not image.startswith("data:image/"):
        return jsonify(error="image (data URI) required"), 400
    # Optional DB-tracking fields. Always present from the new frontend but
    # tolerated as missing so the endpoint still works in isolation.
    photo_id = body.get("photo_id") if isinstance(body.get("photo_id"), str) else None
    face_index = body.get("face_index") if isinstance(body.get("face_index"), int) else None
    bbox = body.get("bbox") if isinstance(body.get("bbox"), dict) else None
    try:
        attrs, usage = call_gmi_chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ],
            json_mode=True,
            max_tokens=200,
        )
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        face_id = db.insert_face(photo_id, face_index, bbox, image, attrs, "ok", None,
                                 tokens=tokens if isinstance(tokens, int) else None)
        return jsonify(attributes=attrs, face_id=face_id, tokens=tokens)
    except requests.HTTPError as e:
        err = f"GMI vision error: {e.response.status_code} {e.response.text[:300]}"
        db.insert_face(photo_id, face_index, bbox, image, None, "err", err)
        return jsonify(error=err), 502
    except Exception as e:
        err = f"vision failed: {e.__class__.__name__}: {e}"
        db.insert_face(photo_id, face_index, bbox, image, None, "err", err)
        return jsonify(error=err), 502


def build_edit_payload(model: str, prompt: str, reference_image_data_uri: str, quality: str = None) -> dict:
    """Authoritative payload shapes per the GMI /apikey/models schema endpoint.
    Verified 2026-05-23 against /api/v1/ie/requestqueue/apikey/models/{id}.

    KEY FIX: gemini models accept `image` as an ARRAY of URLs/data URIs (not
    `reference_image`). With the wrong key they silently fell back to
    text-to-image, which is why faces appeared to be 'invented'.
    """
    # ── Gemini image family: prompt + image (array). Multi-image supported. ──
    if model.startswith("gemini-"):
        return {"prompt": prompt, "image": [reference_image_data_uri]}

    # ── OpenAI gpt-image-2 EDIT (instruction edit of source image) ──
    if model.startswith("gpt-image-2-edit"):
        return {
            "prompt": prompt,
            "image": reference_image_data_uri,
            "size": "1024x1024",
            "quality": quality or os.environ.get("EDIT_QUALITY", "low"),
            "n": 1,
        }
    # ── gpt-image-2 (generate, no reference) ──
    if model == "gpt-image-2" or model == "gpt-image-2-generate":
        return {"prompt": prompt, "size": "1024x1024", "quality": quality or "low", "n": 1}
    # ── gpt-image-1.5 — text-to-image (does not accept image) ──
    if model.startswith("gpt-image-1"):
        return {"prompt": prompt, "size": "1024x1024", "quality": quality or "low", "n": 1}

    # ── Reve edit / remix ──
    if model.startswith("reve-edit"):
        return {"prompt": prompt, "reference_image": reference_image_data_uri}
    if model.startswith("reve-remix"):
        return {"prompt": prompt, "reference_images": [reference_image_data_uri]}

    # ── Bytedance SeedEdit (i2i) ──
    if model.startswith("seededit"):
        return {"prompt": prompt, "image": reference_image_data_uri}

    # ── Tencent Hunyuan i2i ──
    if model.startswith("hunyuan-image-to-image"):
        return {"prompt": prompt, "image": reference_image_data_uri}

    # ── BRIA Fibo Edit (instruction-based) ──
    if model == "bria-fibo-edit":
        return {"instruction": prompt, "images": [reference_image_data_uri]}
    # ── BRIA Fibo Restyle (requires style enum — not for our face-on-player use) ──
    if model == "bria-fibo-restyle":
        return {"image": reference_image_data_uri, "style": "photorealistic"}
    # ── Other bria-fibo variants (recolor/relight/reseason/restore) ──
    if model.startswith("bria-"):
        return {"prompt": prompt, "image": reference_image_data_uri}

    # ── flux-kontext-pro is text-to-image despite the name ──
    if model.startswith("flux-kontext"):
        return {"prompt": prompt, "aspect_ratio": "1:1"}

    # ── Default: assume openai-ish shape. ──
    return {"prompt": prompt, "image": reference_image_data_uri, "n": 1}


def needs_public_url(model: str) -> bool:
    """Some external providers (gemini, reve, seededit, bytedance) can't fetch
    data URIs — their fetchers expect HTTP(S) URLs. We auto-upload the data URI
    to a temp host (litterbox.catbox.moe, 1h retention) for these models."""
    return (model.startswith("gemini-")
            or model.startswith("reve-")
            or model.startswith("seededit"))


_url_cache = {}  # data_uri sha256 → public URL


def ensure_url(data_uri_or_url: str) -> str:
    """If `data_uri_or_url` is already an http(s) URL, return as-is. Otherwise
    upload the decoded bytes to litterbox.catbox.moe and return the URL.
    Caches by SHA256 so the same image is never re-uploaded within a session.
    """
    if not data_uri_or_url.startswith("data:"):
        return data_uri_or_url
    import hashlib
    header, _, b64data = data_uri_or_url.partition(",")
    mime = header.split(":", 1)[1].split(";", 1)[0] if ":" in header else "image/jpeg"
    ext = "jpg" if "jpeg" in mime else ("png" if "png" in mime else mime.split("/")[-1])
    raw = base64.b64decode(b64data)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    if digest in _url_cache:
        cached = _url_cache[digest]
        # litter.catbox.moe returns 405 on HEAD, so use a 1-byte GET to check liveness.
        try:
            r = requests.get(cached, headers={"Range": "bytes=0-0"}, timeout=5, stream=True)
            r.close()
            if r.status_code in (200, 206):
                return cached
        except Exception:
            pass

    filename = f"upload.{ext}"
    # Try litterbox (1h retention) twice, then catbox.moe (permanent) as fallback.
    attempts = [
        ("https://litterbox.catbox.moe/resources/internals/api.php", {"reqtype": "fileupload", "time": "1h"}),
        ("https://litterbox.catbox.moe/resources/internals/api.php", {"reqtype": "fileupload", "time": "1h"}),
        ("https://catbox.moe/user/api.php", {"reqtype": "fileupload"}),
    ]
    last_err = "no attempts ran"
    for endpoint, fields in attempts:
        try:
            r = requests.post(
                endpoint,
                data=fields,
                files={"fileToUpload": (filename, raw, mime)},
                timeout=60,
            )
            body = (r.text or "").strip()
            if r.status_code == 200 and body.startswith("https://"):
                _url_cache[digest] = body
                return body
            last_err = f"{endpoint} → HTTP {r.status_code} body={body[:200]!r}"
        except Exception as e:
            last_err = f"{endpoint} → {e.__class__.__name__}: {e}"
        time.sleep(0.5)
    raise RuntimeError(f"image host upload failed after retries: {last_err}")


def submit_image_edit(prompt: str, reference_image_data_uri: str, model: str = None, quality: str = None):
    use_model = model or EDIT_MODEL
    ref = reference_image_data_uri
    if needs_public_url(use_model):
        ref = ensure_url(ref)
    body = {
        "model": use_model,
        "payload": build_edit_payload(use_model, prompt, ref, quality),
    }
    # GMI's queue endpoint is synchronous-ish: it waits for the model. 180s
    # tolerates a slow run; if it actually times out we retry once before
    # giving up, because cold-starts on the GMI side cause occasional spikes.
    last_exc = None
    for attempt in range(2):
        try:
            return requests.post(GMI_QUEUE_URL, headers=gmi_headers(), json=body, timeout=(10, 180))
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            print(f"[submit_image_edit] ReadTimeout on attempt {attempt + 1}, retrying…", flush=True)
    raise last_exc


def extract_image_url(payload: dict):
    """Pull a usable image reference from GMI's reply.

    Different model families return different shapes:
    - OpenAI/BRIA/Reve/etc.: outcome.media_urls = [{url}]
    - Gemini: outcome.candidates[0].content.parts[*].inlineData.{mimeType, data}
      — base64 image inline in the response. We convert this to a data URI so
      downstream consumers (identity scoring, browser <img>) work uniformly.
    """
    outcome = payload.get("outcome") or {}

    urls = outcome.get("media_urls") or []
    if urls and isinstance(urls, list):
        first = urls[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
        if isinstance(first, str):
            return first

    candidates = outcome.get("candidates") or []
    for cand in candidates:
        content = (cand or {}).get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return f"data:{mime};base64,{inline['data']}"
    return None


def poll_request(request_id: str, *, budget_seconds: float = 120.0):
    """Returns (url, error, final_payload). final_payload is the last JSON
    body we got back (used by callers to scrape token usage if present)."""
    deadline = time.monotonic() + budget_seconds
    poll_url = f"{GMI_QUEUE_URL}/{request_id}"
    last_payload = {}
    while time.monotonic() < deadline:
        time.sleep(2.0)
        r = requests.get(poll_url, headers=gmi_headers(), timeout=(10, 30))
        if r.status_code >= 400:
            return None, f"poll error: {r.status_code} {r.text[:200]}", last_payload
        payload = r.json()
        last_payload = payload
        status = (payload.get("status") or "").lower()
        url = extract_image_url(payload)
        if url:
            return url, None, payload
        if status in {"failed", "error", "cancelled"}:
            return None, f"generation {status}: {payload.get('error') or payload}", payload
    return None, "generation timed out after 120s", last_payload


def verify_people_count(image_url: str):
    try:
        result, _usage = call_gmi_chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VERIFY_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            json_mode=True,
            max_tokens=50,
        )
        count = result.get("count")
        return int(count) if isinstance(count, (int, float)) else None
    except Exception:
        return None


@app.post("/api/generate")
def api_generate():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    reference = body.get("reference_image")
    country_code = (body.get("country_code") or "").strip()
    country_name = (body.get("country_name") or "").strip() or None
    n_people = body.get("n_people")
    model = (body.get("model") or "").strip() or None
    quality = (body.get("quality") or "").strip() or None
    skip_verify = bool(body.get("skip_verify"))
    photo_id = body.get("photo_id") if isinstance(body.get("photo_id"), str) else None
    if not prompt:
        return jsonify(error="prompt required"), 400
    if not isinstance(reference, str) or not (reference.startswith("data:image/") or reference.startswith("https://") or reference.startswith("http://")):
        return jsonify(error="reference_image must be a data URI or http(s) URL"), 400
    if not isinstance(n_people, int) or n_people < 1:
        return jsonify(error="n_people (int >= 1) required"), 400

    # Update the photo's face_count now that the user has committed to a count.
    db.update_photo_face_count(photo_id, n_people)

    used_model = model or EDIT_MODEL
    t0 = time.monotonic()
    try:
        r = submit_image_edit(prompt, reference, model=model, quality=quality)
        # Some edit models (notably gpt-image-2-edit) refuse group photos of real,
        # identifiable people with a 400 "Generation rejected". When that happens
        # and the caller didn't pin a specific model, fall back to the gemini image
        # model, which handles multi-face identity edits (and is faster). Override
        # the fallback target via FALLBACK_EDIT_MODEL.
        _rej = (r.text or "").lower()
        if (r.status_code == 400 and not model
                and any(w in _rej for w in ("reject", "policy", "blocked", "safety", "denied"))):
            fallback = os.environ.get("FALLBACK_EDIT_MODEL", "gemini-3.1-flash-image-preview")
            print(f"[api_generate] {used_model} rejected; falling back to {fallback}", flush=True)
            used_model = fallback
            r = submit_image_edit(prompt, reference, model=fallback, quality=quality)
        if r.status_code >= 400:
            err = f"GMI submit error: {r.status_code} {r.text[:800]}"
            db.insert_generation(photo_id, country_code or None, country_name, n_people, prompt,
                                 used_model, round(time.monotonic() - t0, 2), None, False, None, err)
            return jsonify(error=err, model=used_model), 502
        payload = r.json()
    except Exception as e:
        err = f"GMI submit failed: {e.__class__.__name__}: {e}"
        db.insert_generation(photo_id, country_code or None, country_name, n_people, prompt,
                             used_model, round(time.monotonic() - t0, 2), None, False, None, err)
        return jsonify(error=err, model=used_model), 502

    final_payload = payload
    image_url = extract_image_url(payload)
    if not image_url:
        request_id = payload.get("request_id")
        if not request_id:
            err = f"no media URL and no request_id in response: {json.dumps(payload)[:400]}"
            db.insert_generation(photo_id, country_code or None, country_name, n_people, prompt,
                                 used_model, round(time.monotonic() - t0, 2), None, False, None, err)
            return jsonify(error=err, model=used_model), 502
        image_url, poll_err, final_payload = poll_request(request_id)
        if not image_url:
            err = poll_err or "unknown polling error"
            db.insert_generation(photo_id, country_code or None, country_name, n_people, prompt,
                                 used_model, round(time.monotonic() - t0, 2), None, False, None, err)
            return jsonify(error=err, model=used_model), 502

    # GMI image-edit response shapes vary by model — try a few common locations
    # for token usage. Most image-gen models don't report tokens at all (returns None).
    usage = (final_payload.get("usage") or
             (final_payload.get("outcome") or {}).get("usage") or
             {}) if isinstance(final_payload, dict) else {}
    gen_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if not isinstance(gen_tokens, int):
        gen_tokens = None

    elapsed = round(time.monotonic() - t0, 2)
    verified = None if skip_verify else verify_people_count(image_url)
    mismatch = verified is not None and verified != n_people
    gen = db.insert_generation(photo_id, country_code or None, country_name, n_people, prompt,
                               used_model, elapsed, verified, mismatch, image_url, None,
                               tokens=gen_tokens) or {}
    result_url = db.public_url(gen.get("storage_path"))
    return jsonify(
        image_url=image_url,
        result_url=result_url,
        generation_id=gen.get("id"),
        tokens=gen_tokens,
        verified_count=verified,
        expected_count=n_people,
        mismatch=mismatch,
        country_code=country_code,
        model=used_model,
        elapsed=elapsed,
    )


@app.get("/api/download")
def api_download():
    url = request.args.get("url", "")
    name = request.args.get("name", "team")
    inline = request.args.get("inline", "0") == "1"
    if not url.startswith("https://storage.googleapis.com/"):
        abort(400, description="only storage.googleapis.com URLs allowed")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60] or "team"
    try:
        r = requests.get(url, stream=True, timeout=(10, 60))
        r.raise_for_status()
    except Exception as e:
        abort(502, description=f"download failed: {e}")
    content_type = r.headers.get("Content-Type", "image/png")
    ext = "png" if "png" in content_type else ("jpg" if "jpeg" in content_type else "png")
    headers = {"Content-Type": content_type}
    if not inline:
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}.{ext}"'
    return Response(r.iter_content(chunk_size=8192), headers=headers)


def main():
    if not GMI_API_KEY:
        print(
            "WARNING: GMI_API_KEY is not set. The UI will load, but /api/vision and /api/generate "
            "will fail. Export GMI_API_KEY before running for end-to-end use.",
            file=sys.stderr,
        )
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
