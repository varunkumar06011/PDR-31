# PDR-31 Wound Segmentation Backend

FastAPI backend for wound segmentation and area measurement. Uses a U-Net
(EfficientNet-B0 encoder) via `segmentation_models_pytorch`, OpenCV for
calibration marker detection, and Supabase for data storage.

> **Disclaimer:** This is a portfolio/demo project — not a certified medical
> device. Segmentation accuracy is demo-grade unless a trained checkpoint is
> provided via `MODEL_CHECKPOINT_PATH`.

---

## Run Locally

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy .env.example to .env and fill in your Supabase credentials
cp .env.example .env

# (Optional) Set MODEL_CHECKPOINT_PATH to a trained .pth file
# MODEL_CHECKPOINT_PATH=./checkpoints/unet_wound.pth

# Start the server
uvicorn app.main:app --reload
```

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Test the model in isolation

Place a test image in `/backend/samples/` and run:
```bash
python -m app.model
```
Output is saved to `samples/output_test.png`.

---

## Deploy to Railway

1. Go to [railway.app](https://railway.app) and create a new project.
2. Connect your GitHub repo.
3. Set the **root directory** to `/backend`.
4. Railway should auto-detect the `Procfile`:
   ```
   web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables:
   - `SUPABASE_URL` — your Supabase project URL
   - `SUPABASE_KEY` — your Supabase anon/service key
   - `MODEL_CHECKPOINT_PATH` — (optional) path to a trained checkpoint
6. Deploy. Railway will install `requirements.txt` and start the server.
7. Confirm the healthcheck: `https://<your-app>.up.railway.app/health`

---

## Supabase Setup

### 1. Create the `visits` table

Go to your Supabase dashboard → **SQL Editor** → paste and run:

```sql
CREATE TABLE visits (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id  TEXT NOT NULL,
    visit_date  DATE NOT NULL,
    area_mm2    NUMERIC NOT NULL,
    overlay_url TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_visits_patient_date
    ON visits (patient_id, visit_date);
```

### 2. Create the `wound-overlays` storage bucket (public)

1. Go to your **Supabase project dashboard**
2. Click **Storage** in the left sidebar
3. Click **New bucket**
4. Name it: `wound-overlays`
5. Toggle **Public bucket** to **ON**
6. Click **Create bucket**

This makes uploaded overlay images publicly accessible via their URL.

### 3. Get your Supabase credentials

- **Project URL**: Dashboard → Settings → API → Project URL
- **API Key**: Dashboard → Settings → API → `anon` or `service_role` key

Set these as `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` (local) or
Railway environment variables (deployed).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Healthcheck — returns `{"status":"ok"}` |
| `POST` | `/segment` | Upload image + patient_id + visit_date → returns area + overlay URL |
| `GET`  | `/trajectory/{patient_id}` | Returns all visits for a patient as JSON list |

### POST /segment

Multipart form data:
- `file` — image file (PNG/JPG)
- `patient_id` — string (e.g. "P001")
- `visit_date` — ISO date string (e.g. "2025-08-13")

Response:
```json
{
  "patient_id": "P001",
  "visit_date": "2025-08-13",
  "area_mm2": 142.3,
  "overlay_url": "https://...supabase.co/storage/v1/object/public/wound-overlays/..."
}
```

On calibration failure (no marker detected):
```json
{
  "detail": "Calibration marker not detected — ensure a clear circular marker is visible in the photo."
}
```
(HTTP 422)
