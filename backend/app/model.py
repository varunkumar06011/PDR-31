"""Model loading, inference, and mask overlay using segmentation_models_pytorch."""

import os
import sys
import logging

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

_MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: str | None = None) -> torch.nn.Module:
    """Load a U-Net model with EfficientNet-B3 encoder.

    If checkpoint_path is provided and the file exists, load weights from it.
    Otherwise use imagenet-initialized encoder only (demo-level accuracy).
    Model is cached in a global and loaded once at startup.
    """
    global _MODEL

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning(
            "Running with imagenet-initialized encoder only — "
            "segmentation accuracy is demo-grade, not clinical."
        )

    model.to(DEVICE)
    model.eval()
    _MODEL = model
    return model


def _get_model() -> torch.nn.Module:
    if _MODEL is None:
        checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
        load_model(checkpoint_path)
    return _MODEL


def _preprocess(image: Image.Image) -> torch.Tensor:
    """Resize to 512x512, normalize with imagenet mean/std, convert to tensor."""
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # normalize
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    return tensor


def predict_mask(model: torch.nn.Module, image: Image.Image) -> np.ndarray:
    """Run inference on a PIL image and return a binary mask.

    Steps:
      1. Resize to 512x512, normalize with imagenet mean/std.
      2. Run forward pass under torch.no_grad().
      3. Apply sigmoid + threshold at 0.5.
      4. Resize mask back to original image dimensions.

    Returns:
        np.ndarray of shape (H, W) with dtype uint8, values 0 or 255.
    """
    original_size = image.size  # (W, H)

    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask_small = (probs >= 0.5).astype(np.uint8) * 255

    mask_full = cv2.resize(
        mask_small,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask_full


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 255, 255),
    alpha: float = 0.55,
) -> Image.Image:
    """Composite a semi-transparent colored mask onto the original image.

    Args:
        image: Original PIL image.
        mask: Binary mask array (H, W), values 0 or 255.
        color: RGB tuple for the overlay color.
        alpha: Blend factor for the overlay (0-1).

    Returns:
        PIL Image with mask overlay composited.
    """
    img_np = np.array(image.convert("RGB"))
    mask_bool = mask > 127

    overlay = img_np.copy()
    overlay[mask_bool] = color

    beta = 1.0 - alpha
    result = cv2.addWeighted(img_np, beta, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return Image.fromarray(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    sample_dir = os.path.abspath(sample_dir)

    sample_files = [
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.isdir(sample_dir) else []

    if not sample_files:
        print(f"No sample images found in {sample_dir}")
        print("Place a test image (e.g. wound1.jpg) in /backend/samples/ and re-run.")
        sys.exit(1)

    sample_path = os.path.join(sample_dir, sample_files[0])
    print(f"Running inference on: {sample_path}")

    model = load_model()
    img = Image.open(sample_path)
    mask = predict_mask(model, img)
    overlaid = overlay_mask(img, mask)

    output_path = os.path.join(sample_dir, "output_test.png")
    overlaid.save(output_path)
    print(f"Overlay saved to: {output_path}")
    print(f"Mask pixel count: {np.count_nonzero(mask)}")
