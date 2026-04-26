"""
utils.py
OrbitalMind Utilities v2.0
- Synthetic satellite image generation (5 realistic scene types including Flood)
- Bandwidth computation
- JSON formatting
"""

import numpy as np
import json
from typing import Dict, Any


def generate_sample_image(scene_type: str) -> np.ndarray:
    """
    Generate realistic 256×256 RGB synthetic satellite images.
    Includes new Flood Scene and Post-Disaster scenes.
    """
    np.random.seed(42)
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.float32)

    if scene_type == "Agricultural":
        base_green = np.random.normal(0.38, 0.08, (h, w))
        base_red   = np.random.normal(0.22, 0.06, (h, w))
        base_blue  = np.random.normal(0.15, 0.04, (h, w))
        for _ in range(12):
            cx = np.random.randint(30, 220)
            cy = np.random.randint(30, 220)
            r_size = np.random.randint(15, 45)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            mask = ((xx - cx)**2 + (yy - cy)**2) < r_size**2
            if np.random.rand() > 0.5:
                base_green[mask] *= 0.65
                base_red[mask]   *= 1.35
            else:
                base_green[mask] *= 1.25
                base_red[mask]   *= 0.80
        img[:, :, 0] = np.clip(base_red, 0, 1)
        img[:, :, 1] = np.clip(base_green, 0, 1)
        img[:, :, 2] = np.clip(base_blue, 0, 1)

    elif scene_type == "Urban / Coastal":
        base = np.random.normal(0.45, 0.12, (h, w))
        water_mask = np.zeros((h, w), dtype=bool)
        water_mask[140:, :110] = True
        for row in range(130, 170):
            col = int(110 + 15 * np.sin((row - 130) * 0.25))
            water_mask[row, :max(0, col)] = True
        img[:, :, 0] = np.clip(base * 0.85, 0, 1)
        img[:, :, 1] = np.clip(base * 0.82, 0, 1)
        img[:, :, 2] = np.clip(base * 0.90, 0, 1)
        img[water_mask, 0] = 0.05 + np.random.normal(0, 0.02, water_mask.sum())
        img[water_mask, 1] = 0.12 + np.random.normal(0, 0.02, water_mask.sum())
        img[water_mask, 2] = 0.35 + np.random.normal(0, 0.03, water_mask.sum())

    elif scene_type == "Forest / Wildfire":
        base_green = np.random.normal(0.30, 0.07, (h, w))
        base_red   = np.random.normal(0.18, 0.05, (h, w))
        base_blue  = np.random.normal(0.12, 0.04, (h, w))
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        scar_mask = (((xx-160)**2 / 60**2) + ((yy-80)**2 / 45**2)) < 1
        base_red[scar_mask]   = 0.55 + np.random.normal(0, 0.04, scar_mask.sum())
        base_green[scar_mask] = 0.08 + np.random.normal(0, 0.02, scar_mask.sum())
        base_blue[scar_mask]  = 0.06 + np.random.normal(0, 0.01, scar_mask.sum())
        img[:, :, 0] = np.clip(base_red, 0, 1)
        img[:, :, 1] = np.clip(base_green, 0, 1)
        img[:, :, 2] = np.clip(base_blue, 0, 1)

    elif scene_type == "Flood Scene":
        # Majority water inundation with visible field / road remnants
        base_green = np.random.normal(0.15, 0.05, (h, w))
        base_red   = np.random.normal(0.10, 0.04, (h, w))
        base_blue  = np.random.normal(0.30, 0.06, (h, w))

        # Large contiguous flood body (lower 60%)
        flood_body = np.zeros((h, w), dtype=bool)
        flood_body[100:, :] = True
        # Irregular flood boundary
        for col in range(w):
            boundary = int(100 + 20 * np.sin(col * 0.08) + 10 * np.cos(col * 0.15))
            flood_body[max(0, boundary):, col] = True

        # Secondary flood patches (upper scene)
        for _ in range(5):
            cx = np.random.randint(10, 200)
            cy = np.random.randint(20, 90)
            r  = np.random.randint(10, 28)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            flood_body |= ((xx - cx)**2 + (yy - cy)**2) < r**2

        # Flood water appearance: low red, higher blue, turbid brownish
        img[:, :, 0] = np.clip(base_red, 0, 1)
        img[:, :, 1] = np.clip(base_green, 0, 1)
        img[:, :, 2] = np.clip(base_blue, 0, 1)
        img[flood_body, 0] = 0.08 + np.random.normal(0, 0.02, flood_body.sum())
        img[flood_body, 1] = 0.14 + np.random.normal(0, 0.02, flood_body.sum())
        img[flood_body, 2] = 0.38 + np.random.normal(0, 0.04, flood_body.sum())

        # Visible remnant roads / embankments (slightly brighter)
        for i in range(0, h, 30):
            img[i:i+3, :, :] = np.clip(img[i:i+3, :, :] * 1.8, 0, 1)

    elif scene_type == "Post-Disaster / Mixed":
        # Mix: partial flood + burn scar + rubble
        base = np.random.normal(0.35, 0.10, (h, w, 3))
        # Flood zone left half
        base[120:, :100, 2] = 0.40
        base[120:, :100, 0] = 0.08
        base[120:, :100, 1] = 0.13
        # Burn scar top-right
        base[:80, 150:, 0] = 0.50
        base[:80, 150:, 1] = 0.07
        base[:80, 150:, 2] = 0.05
        img = np.clip(base, 0, 1)

    else:
        img = np.random.uniform(0.1, 0.8, (h, w, 3)).astype(np.float32)

    # Spatial smoothing for realism
    try:
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], sigma=1.5)
    except ImportError:
        pass  # Skip if scipy not available

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def compute_bandwidth_saving(img_array: np.ndarray, output_json: Dict) -> Dict:
    raw_bytes  = img_array.nbytes
    json_str   = json.dumps(output_json, separators=(",", ":"))
    json_bytes = len(json_str.encode("utf-8"))
    saving_pct = (1.0 - json_bytes / raw_bytes) * 100
    return {
        "raw_kb": raw_bytes / 1024,
        "output_bytes": json_bytes,
        "saving_pct": saving_pct,
        "compression_ratio": raw_bytes / max(json_bytes, 1),
    }


def format_output_json(result: Dict[str, Any]) -> str:
    payload = result.get("output_json", {})
    return json.dumps(payload, indent=2, ensure_ascii=False)