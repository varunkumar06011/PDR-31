import io
import os
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

from app.model import load_model, predict_mask, overlay_mask
from app.calibrate import detect_marker, pixels_to_mm2
from app.db import upload_overlay_image, create_visit, get_trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDR-31 Wound Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    logger.info("App starting — model will load on first request.")


@app.get("/health")
def health():
    """Healthcheck endpoint for Railway."""
    return {"status": "ok"}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    visit_date: str = Form(...),
):
    """Run wound segmentation, calibrate area, save to Supabase, return results."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    from app.model import _get_model
    model = _get_model()

    mask = predict_mask(model, image)

    try:
        marker_diameter_px = detect_marker(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    area_mm2 = pixels_to_mm2(mask, marker_diameter_px)

    overlay = overlay_mask(image, mask)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_bytes = buf.getvalue()

    filename = f"{patient_id}/{visit_date}/{uuid.uuid4().hex}.png"

    try:
        overlay_url = upload_overlay_image(overlay_bytes, filename)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload overlay image: {e}")

    try:
        visit = create_visit(patient_id, visit_date, area_mm2, overlay_url)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save visit: {e}")

    return {
        "patient_id": patient_id,
        "visit_date": visit_date,
        "area_mm2": area_mm2,
        "overlay_url": overlay_url,
    }


@app.get("/trajectory/{patient_id}")
def trajectory(patient_id: str):
    """Return all visits for a patient, ordered by date. Empty list if none."""
    try:
        data = get_trajectory(patient_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return data if data else []
