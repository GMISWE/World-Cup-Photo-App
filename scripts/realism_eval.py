"""Prompt-evolution A/B for output realism.

Builds a 3-face collage from a real source photo (the same way the app does),
then generates the same Brazil team with several STYLE/realism prompt variants
via the app's /api/generate (gemini image model). Saves each output so they can
be compared side by side, and reports the people-count verification.

Usage: .venv/bin/python scripts/realism_eval.py
"""
import base64, importlib.util, io, json, sys
from pathlib import Path
import cv2, numpy as np, requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-fixtures" / "realism"
OUT.mkdir(parents=True, exist_ok=True)
SRC = "/Users/roan/Downloads/05106051-9f97-457a-ad21-d2901426a709.JPG"  # booth, 3 faces
BASE = "http://127.0.0.1:5050"

# reuse the detection multipass
spec = importlib.util.spec_from_file_location("ds", str(ROOT / "scripts/detect_sweep.py"))
ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)


def crop_face(rgb, bb):
    H, W = rgb.shape[:2]
    x, y, w, h, _ = bb
    cx, cy = x + w / 2, y + h / 2
    side = max(w * (1 + 2 * 0.35), h * (1 + 0.45 + 0.55))
    side = min(side, W, H)
    sx = int(max(0, min(cx - side / 2, W - side)))
    sy = int(max(0, min(cy - side / 2 - h * 0.06, H - side)))
    crop = rgb[sy:sy + int(side), sx:sx + int(side)]
    return cv2.resize(crop, (512, 512))


def build_collage(crops):
    W = H = 768; cols = len(crops); gutter = 10; labelh = 26
    cellW = (W - gutter * (cols + 1)) // cols
    cellH = H - gutter * 2
    tileH = cellH - labelh
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for i, c in enumerate(crops):
        im = Image.fromarray(c).resize((cellW, tileH))
        x = gutter + i * (cellW + gutter); y = gutter
        canvas.paste(im, (x, y))
    buf = io.BytesIO(); canvas.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


GUIDE = (
    "- Player 1 — back row (standing): 20s feminine, long straight light brown hair, no facial hair, no glasses, light skin, small mole on chin.\n"
    "- Player 2 — back row (standing): 20s masculine, short black hair, no facial hair, wearing glasses, light skin, prominent smile.\n"
    "- Player 3 — back row (standing): 20s feminine, long dark hair, no facial hair, no glasses, light skin, wearing a baseball cap."
)

def prompt(style):
    return "\n".join([
        "OFFICIAL FIFA PRE-MATCH TEAM PHOTOGRAPH of the Brazil national football team, posing on the pitch of a packed modern football stadium just before kickoff. Wide horizontal photograph, taken from chest level, full squad centered.",
        "CRITICAL CONSTRAINT: The photo must contain EXACTLY 3 players — no goalkeeper, no coaches, no referees, no extra people, no missing people. Count: 3.",
        "KIT: yellow Brazil home jersey with green trim on collar and sleeves, blue shorts, white/blue socks. Modern athletic fit, short sleeves, national crest on left chest. No sponsor text on the shirt front.",
        "FORMATION: A single row of exactly 3 players standing shoulder-to-shoulder, arms behind their backs.",
        "REFERENCE IMAGE: a white-background grid of 3 labeled portrait tiles, Player 1 to Player 3. PRESERVE THE EXACT IDENTITY of each person — face shape, eyes, nose, mouth, jawline, hairstyle, hair color, skin tone, glasses, moles, cap. Do not blend or invent faces. Player 1 leftmost, Player 3 rightmost.",
        "PER-PLAYER GUIDE:", GUIDE,
        "EXPRESSION: light, gentle, closed-mouth smile, eyes relaxed looking at camera.",
        "BACKGROUND: blurred view of a large floodlit football stadium with a packed crowd. Green grass pitch.",
        style,
    ])

VARIANTS = {
    "v0_baseline": "STYLE: ultra-photorealistic sports photography, 35mm full-frame, 50mm prime lens, f/4, ISO 400, mild depth of field, players sharp and crowd softly blurred. No text, no captions, no scoreboards, no flags overlaid in foreground.",
    "v1_skin": "STYLE: ultra-photorealistic broadcast sports photography shot on a Sony A1 with a 70-200mm f/2.8 lens, players tack-sharp and crowd softly blurred. LIFELIKE SKIN: natural skin texture with visible pores and fine detail, subsurface scattering, realistic blemishes and stubble, NOT smooth/waxy/plastic, no CGI or 3D-render look. Accurate neutral white balance under stadium floodlights, high dynamic range, subtle photographic film grain. No text, captions, scoreboards, or overlaid graphics.",
    "v2_filmic": "STYLE: photorealistic editorial sports portrait, Canon EOS R5 + 85mm f/1.8, golden-hour stadium floodlight, cinematic but natural color grade. Skin shows real pores, micro-shadows and natural specular highlights (no airbrushing, no plastic sheen, no uncanny smoothness). Fine 35mm film grain, true-to-life eyes with natural catchlights, sharp individual hair strands. Absolutely no AI artifacts, extra fingers, warped features, text, or watermarks.",
}


def generate(style):
    body = {"country_code": "BR", "country_name": "Brazil", "prompt": prompt(style),
            "reference_image": COLLAGE, "n_people": 3,
            "model": "gemini-3.1-flash-image-preview", "skip_verify": False}
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=(10, 200))
    return r.status_code, r.json()


if __name__ == "__main__":
    rgb = ds.load_rgb.__wrapped__ if hasattr(ds.load_rgb, "__wrapped__") else None
    bgr = cv2.imread(SRC); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = ds.multipass(rgb, 4, 2, 0.25, 0.5, 0.35)
    boxes = sorted(boxes, key=lambda b: b[0])[:3]
    print(f"source faces: {len(boxes)}")
    crops = [crop_face(rgb, b) for b in boxes]
    COLLAGE = build_collage(crops)
    Image.open(io.BytesIO(base64.b64decode(COLLAGE.split(',')[1]))).save(OUT / "_collage.jpg")
    for name, style in VARIANTS.items():
        st, j = generate(style)
        if st == 200 and j.get("image_url"):
            url = j["image_url"]
            img = requests.get(url, timeout=(10, 60)).content if url.startswith("http") else base64.b64decode(url.split(",", 1)[1])
            (OUT / f"{name}.png").write_bytes(img)
            print(f"  {name}: ok  verified={j.get('verified_count')}/{j.get('expected_count')} model={j.get('model')} {j.get('elapsed')}s")
        else:
            print(f"  {name}: ERR {st} {str(j)[:160]}")
