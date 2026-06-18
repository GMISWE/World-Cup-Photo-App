"""Generate synthetic group-photo test fixtures via GMI text-to-image.

Produces images with varying numbers of people (1..12 + a few extras) to test
the face-detection pipeline. Saves PNGs to test-fixtures/.

Usage: .venv/bin/python scripts/gen_test_images.py
"""
import base64
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-fixtures"
OUT.mkdir(exist_ok=True)


def load_key():
    for raw in (ROOT / ".env").read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:]
        if line.startswith("GMI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GMI_API_KEY", "")


KEY = load_key()
QUEUE = "https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey/requests"
MODEL = os.environ.get("GEN_MODEL", "gpt-image-2")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# (filename, requested_count, scene) — requested count is a target; the actual
# face count is established later by inspection. Scenes vary framing/pose.
SPECS = [
    ("p01", 1, "a single person standing at a tech conference booth, facing the camera, upper body visible"),
    ("p02", 2, "exactly 2 people standing side by side at a conference booth, facing the camera"),
    ("p03", 3, "exactly 3 people standing in a row at a conference booth, facing the camera"),
    ("p04", 4, "exactly 4 people standing shoulder to shoulder at a trade show booth, facing the camera"),
    ("p05", 5, "exactly 5 coworkers standing in a row at a conference booth, smiling at the camera"),
    ("p06", 6, "exactly 6 people posing together as a team at a tech expo booth, facing the camera"),
    ("p07", 7, "exactly 7 people standing as a group at a conference booth, all facing the camera"),
    ("p08", 8, "exactly 8 people posing as a team photo at a trade show, two loose rows, facing the camera"),
    ("p09", 9, "exactly 9 people in a group team photo at a tech conference, facing the camera"),
    ("p10", 10, "exactly 10 people posing for a large team photo at an expo, two rows, facing the camera"),
    ("p11", 11, "exactly 11 people in a big group photo at a tech conference booth, two rows, facing the camera"),
    ("p12", 12, "exactly 12 people posing for a large company team photo, two rows, all facing the camera"),
    ("x03", 3, "a candid photo of 3 friends at a hackathon, seated at a table facing the camera"),
    ("x07", 7, "7 hackathon participants standing in a casual group, varied heights, facing the camera"),
    ("x11", 11, "11 conference attendees crowded together for a group selfie, faces clearly visible"),
]

STYLE = ", ultra-photorealistic candid photograph, natural lighting, faces clearly visible, 35mm, sharp focus"


def submit(prompt):
    body = {"model": MODEL, "payload": {"prompt": prompt, "size": "1024x1024", "quality": "low", "n": 1}}
    r = requests.post(QUEUE, headers=HEADERS, json=body, timeout=(10, 180))
    r.raise_for_status()
    return r.json()


def extract_url(payload):
    outcome = payload.get("outcome") or {}
    urls = outcome.get("media_urls") or []
    if urls:
        first = urls[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"], None
        if isinstance(first, str):
            return first, None
    for cand in outcome.get("candidates") or []:
        for part in ((cand or {}).get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return None, inline["data"]
    return None, None


def poll(request_id, budget=180):
    deadline = time.monotonic() + budget
    url = f"{QUEUE}/{request_id}"
    while time.monotonic() < deadline:
        time.sleep(2.5)
        r = requests.get(url, headers=HEADERS, timeout=(10, 30))
        if r.status_code >= 400:
            return None, None
        p = r.json()
        u, b64 = extract_url(p)
        if u or b64:
            return u, b64
        if (p.get("status") or "").lower() in {"failed", "error", "cancelled"}:
            return None, None
    return None, None


def gen(spec):
    name, count, scene = spec
    prompt = scene + STYLE
    try:
        payload = submit(prompt)
        u, b64 = extract_url(payload)
        if not u and not b64:
            rid = payload.get("request_id")
            if not rid:
                return (name, count, f"ERR no request_id: {json.dumps(payload)[:200]}")
            u, b64 = poll(rid)
        if u:
            img = requests.get(u, timeout=(10, 60)).content
        elif b64:
            img = base64.b64decode(b64)
        else:
            return (name, count, "ERR no image after poll")
        (OUT / f"{name}.png").write_bytes(img)
        return (name, count, f"ok {len(img)} bytes")
    except Exception as e:
        return (name, count, f"ERR {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    print(f"model={MODEL} out={OUT}")
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(gen, SPECS):
            print(f"  {res[0]} (req {res[1]:>2}): {res[2]}", flush=True)
    print(f"done in {round(time.monotonic()-t0,1)}s")
