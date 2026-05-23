-- One-shot Supabase schema bootstrap for the World Cup 2026 photo app.
-- Paste this entire file into the Supabase SQL Editor and click Run.
-- Idempotent: safe to re-run.

-- 1) Tables -------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS photos (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   timestamptz NOT NULL DEFAULT now(),
    storage_path text,
    width        integer,
    height       integer,
    source       text,
    face_count   integer
);
CREATE INDEX IF NOT EXISTS photos_created_at_idx ON photos (created_at DESC);

CREATE TABLE IF NOT EXISTS faces (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   timestamptz NOT NULL DEFAULT now(),
    photo_id     uuid NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    face_index   integer,
    bbox         jsonb,
    storage_path text,
    attributes   jsonb,
    attr_status  text,
    attr_error   text
);
CREATE INDEX IF NOT EXISTS faces_photo_id_idx ON faces (photo_id);
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'faces_photo_face_unique') THEN
        ALTER TABLE faces ADD CONSTRAINT faces_photo_face_unique UNIQUE (photo_id, face_index);
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS generations (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       timestamptz NOT NULL DEFAULT now(),
    photo_id         uuid REFERENCES photos(id) ON DELETE CASCADE,
    country_code     text,
    country_name     text,
    n_people         integer,
    prompt           text,
    model            text,
    elapsed_seconds  numeric,
    verified_count   integer,
    mismatch         boolean DEFAULT false,
    gmi_image_url    text,
    storage_path     text,
    error            text
);
CREATE INDEX IF NOT EXISTS generations_photo_id_idx  ON generations (photo_id);
CREATE INDEX IF NOT EXISTS generations_created_at_idx ON generations (created_at DESC);

-- 2) Storage bucket -----------------------------------------------------------
-- Public read so generated <img src=...> tags load directly, matching the
-- existing GCS posture. The backend uses the secret key (service_role) so RLS
-- on storage.objects doesn't block its uploads.

INSERT INTO storage.buckets (id, name, public)
VALUES ('world-cup', 'world-cup', true)
ON CONFLICT (id) DO UPDATE SET public = true;
