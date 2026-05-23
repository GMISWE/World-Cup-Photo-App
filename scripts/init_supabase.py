"""One-shot Supabase sanity check.

The bucket is auto-created on first server start (see db.init_client).
This script just confirms the credentials in .env actually work and the
bucket is present + public. Safe to re-run.

Required .env entries:
  SUPABASE_URL   -- https://{ref}.supabase.co
  SUPABASE_KEY   -- service_role (sb_secret_...) key

Run:  python scripts/init_supabase.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env (same pattern as server.py)
for raw in (ROOT / ".env").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    if line.startswith("export "):
        line = line[7:].lstrip()
    k, _, v = line.partition("=")
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(ROOT))
import db


def main():
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print("ERROR: SUPABASE_URL and SUPABASE_KEY required in .env", file=sys.stderr)
        sys.exit(1)
    print(f"-> Initializing Supabase client ...")
    client = db.init_client()
    if not client:
        print("ERROR: Supabase client failed to initialize (see log output above)", file=sys.stderr)
        sys.exit(1)
    if not db._bucket_ready:
        print("ERROR: bucket setup did not complete", file=sys.stderr)
        sys.exit(1)
    print()
    print("READY -- Supabase storage ready.")
    print(f"   URL    : {os.environ.get('SUPABASE_URL')}")
    print(f"   Bucket : {db.BUCKET} (public)")
    print(f"   Schema : storage-based (photos/, faces/, generations/)")


if __name__ == "__main__":
    main()
