"""Independent ground-truth: ask a GMI vision model to count people per fixture."""
import base64, json, os, time
from pathlib import Path
import concurrent.futures as cf
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-fixtures"

def load_key():
    for raw in (ROOT/".env").read_text().splitlines():
        line=raw.strip()
        if line.startswith("export "): line=line[7:]
        if line.startswith("GMI_API_KEY="): return line.split("=",1)[1].strip().strip('"').strip("'")
    return os.environ.get("GMI_API_KEY","")

KEY=load_key()
CHAT="https://api.gmi-serving.com/v1/chat/completions"
H={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"}
MODEL=os.environ.get("COUNT_MODEL","openai/gpt-4o")
PROMPT=('Count the number of distinct human faces clearly visible in this image '
        '(a face counts if both eyes are visible). Return ONLY JSON: {"faces": <integer>}.')

names=["p01","p02","p03","p04","p05","p06","p07","p08","p09","p10","p11","p12","x03","x07","x11"]

def count(n):
    b64=base64.b64encode((OUT/f"{n}.png").read_bytes()).decode()
    uri=f"data:image/png;base64,{b64}"
    body={"model":MODEL,"messages":[{"role":"user","content":[
        {"type":"text","text":PROMPT},{"type":"image_url","image_url":{"url":uri}}]}],
        "response_format":{"type":"json_object"},"max_tokens":50,"temperature":0}
    try:
        r=requests.post(CHAT,headers=H,json=body,timeout=(10,60)); r.raise_for_status()
        j=json.loads(r.json()["choices"][0]["message"]["content"])
        return (n,j.get("faces"))
    except Exception as e:
        return (n,f"ERR {e.__class__.__name__}: {str(e)[:80]}")

if __name__=="__main__":
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        res=dict(ex.map(count,names))
    for n in names: print(f"  {n}: {res[n]}")
