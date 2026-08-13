"""Supabase database + storage operations."""

import os
import uuid
import logging

from dotenv import load_dotenv
from supabase import create_client

logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# IMPORTANT: You must create the "wound-overlays" storage bucket in the
# Supabase dashboard before using this code:
#   1. Go to your Supabase project dashboard
#   2. Click "Storage" in the left sidebar
#   3. Click "New bucket"
#   4. Name it "wound-overlays"
#   5. Toggle "Public bucket" to ON (so uploaded images are publicly readable)
#   6. Click "Create bucket"
BUCKET_NAME = "wound-overlays"


def upload_overlay_image(image_bytes: bytes, filename: str) -> str:
    """Upload overlay image to Supabase Storage and return the public URL.

    Args:
        image_bytes: PNG bytes of the overlay image.
        filename: Path/filename within the bucket (e.g. "P001/2025-08-13/abc.png").

    Returns:
        Public URL of the uploaded image.

    Raises:
        RuntimeError: If Supabase client is not initialized.
        Exception: If the bucket doesn't exist or upload fails, with a clear message.
    """
    if supabase is None:
        raise RuntimeError("Supabase client not initialized — check SUPABASE_URL and SUPABASE_KEY env vars.")

    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/png"},
        )
    except Exception as e:
        err_str = str(e).lower()
        if "not found" in err_str or "bucket" in err_str or "404" in err_str:
            raise RuntimeError(
                f"Storage bucket '{BUCKET_NAME}' not found — "
                f"create it in the Supabase dashboard: Storage > New bucket > "
                f"name it '{BUCKET_NAME}' > toggle Public > Create."
            ) from e
        raise RuntimeError(f"Failed to upload overlay image to Supabase Storage: {e}") from e

    overlay_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
    return overlay_url


def create_visit(patient_id: str, visit_date: str, area_mm2: float, overlay_url: str) -> dict:
    """Insert a visit row into the Supabase 'visits' table.

    Args:
        patient_id: Patient identifier.
        visit_date: ISO date string (e.g. "2025-08-13").
        area_mm2: Measured wound area in mm².
        overlay_url: Public URL of the uploaded overlay image.

    Returns:
        The inserted row as a dict.

    Raises:
        RuntimeError: If Supabase client is not initialized.
        Exception: If the insert fails, with a clear message.
    """
    if supabase is None:
        raise RuntimeError("Supabase client not initialized — check SUPABASE_URL and SUPABASE_KEY env vars.")

    row = {
        "patient_id": patient_id,
        "visit_date": visit_date,
        "area_mm2": area_mm2,
        "overlay_url": overlay_url,
    }

    try:
        insert_resp = supabase.table("visits").insert(row).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to insert visit into Supabase: {e}") from e

    return insert_resp.data[0] if insert_resp.data else row


def get_trajectory(patient_id: str) -> list[dict]:
    """Return all visit rows for a patient, ordered by visit_date ascending.

    Args:
        patient_id: Patient identifier.

    Returns:
        List of visit dicts ordered by visit_date ascending.
        Returns empty list if no visits exist.
    """
    if supabase is None:
        raise RuntimeError("Supabase client not initialized — check SUPABASE_URL and SUPABASE_KEY env vars.")

    resp = (
        supabase.table("visits")
        .select("*")
        .eq("patient_id", patient_id)
        .order("visit_date", desc=False)
        .execute()
    )
    return resp.data if resp.data else []
