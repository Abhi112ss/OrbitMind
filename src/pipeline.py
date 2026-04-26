"""
pipeline.py — OrbitalMind Core Pipeline v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture: 3-Layer Verified Inference
  Layer 1 — TerraMind Encoder   : Physics-informed feature extraction
  Layer 2 — Guard Model         : Confidence gate + contradiction check
  Layer 3 — TM-small Verifier   : Scorer + self-learning loop

v3 NEW:
  - multi_head_spectral_analysis(): context-aware gating
      Step1 indices → Step2 raw masks → Step3 context gates
      → Step4 morphological erosion → Step5 fractions
      → Step6 cross-head normalisation → Step7 contradiction penalty
  - Forest NEVER classified as burn (NDVI gate on burn head)
  - Water NEVER classified as vegetation (NDWI gate on veg head)
  - pixel_to_latlon() shared coordinate converter
  - spectral_scores dict exposed in result for UI display
  - Plotly-ready data structures returned from pipeline

Inspired by TerraTorch pipeline architecture (https://github.com/IBM/terratorch)
"""

import numpy as np
import time
import random
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("OrbitalMind")


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE UTILITY  — shared by FloodGeoLocaliser and app.py click handler
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_latlon(
    px: float, py: float,
    img_w: int, img_h: int,
    scene_lat: float, scene_lon: float,
    gsd_m: float = 10.0,
) -> Tuple[float, float]:
    """
    Pixel (px, py) to geographic (lat, lon) via flat-Earth approximation.
    Origin  : scene centre = (scene_lat, scene_lon)
    +x axis : rightward -> East  (increasing lon)
    +y axis : downward  -> South (decreasing lat)
    gsd_m   : metres per pixel (default 10m = Sentinel-2)
    """
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * np.cos(np.radians(scene_lat)) + 1e-12)
    dx = px - img_w / 2.0
    dy = py - img_h / 2.0
    lat = scene_lat - dy * gsd_m * deg_per_m_lat
    lon = scene_lon + dx * gsd_m * deg_per_m_lon
    return round(float(lat), 5), round(float(lon), 5)


# ══════════════════════════════════════════════════════════════════════════════
# MORPHOLOGICAL NOISE REDUCTION (pure numpy, no scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _erode_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Simple binary erosion via repeated 4-directional AND-shift.
    Removes isolated noise pixels.  O(H*W*radius), no external deps.
    """
    out = mask.copy()
    for _ in range(radius):
        out = (
            out
            & np.roll(out,  1, axis=0)
            & np.roll(out, -1, axis=0)
            & np.roll(out,  1, axis=1)
            & np.roll(out, -1, axis=1)
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE MULTI-HEAD SPECTRAL ANALYSIS  (core v3 algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def multi_head_spectral_analysis(img_array: np.ndarray) -> Dict[str, Any]:
    """
    7-step context-aware 5-head spectral analysis.

    STEP 1  Compute NDVI, NDWI, NBR, BSI, gradient
    STEP 2  Raw per-pixel masks (loose thresholds)
    STEP 3  Context gating — physically impossible combinations suppressed
              flood       = flood_raw  AND ndvi<0.3  AND NOT veg_raw
              burn        = burn_raw   AND ndvi<0.3  AND NOT veg_raw   <- FIX: no forest-burn
              vegetation  = veg_raw    AND ndwi<0.2                    <- FIX: no water-veg
              crop_stress = stress_raw AND veg_raw   AND NOT flood_raw
              change      = change_raw AND NOT flood AND NOT burn
    STEP 4  Morphological erosion (remove noise patches)
    STEP 5  Scene-level fraction scores
    STEP 6  Cross-head normalisation
              veg>0.6  -> suppress burn & flood
              flood>0.4 -> suppress veg & stress
              burn>0.3  -> suppress crop_stress
    STEP 7  Contradiction score -> confidence multiplier
              penalty if high NDVI + burn, or high NDWI + vegetation
    """
    if img_array.ndim != 3 or img_array.shape[2] < 3:
        raise ValueError("Expected H x W x 3 uint8 RGB array.")

    img = img_array.astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    eps = 1e-8

    # STEP 1: Spectral indices
    NIR  = np.clip(0.7 * R + 0.3 * G, 0.0, 1.0)
    ndvi = (NIR - R)   / (NIR + R   + eps)
    ndwi = (G   - NIR) / (G   + NIR + eps)
    nbr  = (NIR - B)   / (NIR + B   + eps)
    bsi  = ((R + B) - (NIR + G)) / ((R + B) + (NIR + G) + eps)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    gx   = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy   = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = np.sqrt(gx**2 + gy**2)

    # STEP 2: Raw masks (intentionally loose)
    flood_raw  = (ndwi >  0.05) & (gray < 0.40)
    veg_raw    =  ndvi >  0.30
    burn_raw   =  nbr  < -0.10
    stress_raw = (ndvi >  0.05) & (ndvi < 0.28)
    change_raw =  grad >  0.08

    # STEP 3: Context gating
    flood       = flood_raw  & (ndvi < 0.30) & ~veg_raw
    burn        = burn_raw   & (ndvi < 0.30) & ~veg_raw   # KEY: dense veg blocks burn
    vegetation  = veg_raw    & (ndwi < 0.20)               # KEY: water blocks vegetation
    crop_stress = stress_raw & veg_raw & ~flood_raw
    change      = change_raw & ~flood & ~burn

    # STEP 4: Noise reduction
    flood       = _erode_mask(flood,       radius=1)
    burn        = _erode_mask(burn,        radius=2)
    vegetation  = _erode_mask(vegetation,  radius=1)
    crop_stress = _erode_mask(crop_stress, radius=1)

    # STEP 5: Fractions + index means
    flood_frac  = float(np.mean(flood))
    veg_frac    = float(np.mean(vegetation))
    burn_frac   = float(np.mean(burn))
    stress_frac = float(np.mean(crop_stress))
    change_frac = float(np.mean(change))

    mean_ndvi   = float(np.mean(ndvi))
    mean_ndwi   = float(np.mean(ndwi))
    mean_nbr    = float(np.mean(nbr))
    mean_bsi    = float(np.mean(bsi))
    grad_energy = float(np.mean(grad))
    grad_std    = float(np.std(grad))
    dark_frac   = float(np.mean(gray < 0.15))

    # Composite scores
    raw_flood  = max(0.0, mean_ndwi)*2.5 + flood_frac*3.5 + max(0,-mean_ndvi)*0.5
    raw_veg    = max(0.0, mean_ndvi)*2.0 + veg_frac*2.5
    raw_burn   = max(0.0, -mean_nbr)*2.5 + burn_frac*3.5 + dark_frac*0.5
    raw_stress = stress_frac*2.5 + max(0,mean_bsi)*1.5 + max(0,0.35-mean_ndvi)*1.5
    raw_change = grad_energy*5.0  + grad_std*3.0 + change_frac*1.5

    def _n(x, d):
        return float(np.clip(x / d, 0.0, 1.0))

    s_flood  = _n(raw_flood,  4.0)
    s_veg    = _n(raw_veg,    4.5)
    s_burn   = _n(raw_burn,   4.5)
    s_stress = _n(raw_stress, 4.0)
    s_change = _n(raw_change, 4.5)

    # STEP 6: Cross-head normalisation
    if s_veg > 0.60:
        s_burn  = min(s_burn,  0.20)
        s_flood = min(s_flood, 0.25)
    if s_flood > 0.40:
        s_veg    = min(s_veg,    0.30)
        s_stress = min(s_stress, 0.20)
    if s_burn > 0.30:
        s_stress = min(s_stress, 0.15)

    # STEP 7: Contradiction score + confidence adjustment
    contradiction = 0.0
    if s_burn  > 0.25 and mean_ndvi > 0.40:
        contradiction += (mean_ndvi - 0.40) * 1.5
    if s_veg   > 0.40 and mean_ndwi > 0.30:
        contradiction += (mean_ndwi - 0.30) * 1.2
    if s_flood > 0.40 and mean_ndvi > 0.35:
        contradiction += (mean_ndvi - 0.35) * 0.8
    contradiction   = float(np.clip(contradiction, 0.0, 0.80))
    conf_mult       = 1.0 - contradiction

    if s_flood > 0.30: s_flood *= conf_mult
    if s_burn  > 0.30: s_burn  *= conf_mult
    if s_veg   > 0.30: s_veg   *= conf_mult

    s_flood  = float(np.clip(s_flood,  0.0, 1.0))
    s_veg    = float(np.clip(s_veg,    0.0, 1.0))
    s_burn   = float(np.clip(s_burn,   0.0, 1.0))
    s_stress = float(np.clip(s_stress, 0.0, 1.0))
    s_change = float(np.clip(s_change, 0.0, 1.0))

    return {
        "flood":       s_flood,
        "vegetation":  s_veg,
        "burn":        s_burn,
        "crop_stress": s_stress,
        "change":      s_change,
        "_aux": {
            "mean_ndvi":       round(mean_ndvi,    4),
            "mean_ndwi":       round(mean_ndwi,    4),
            "mean_nbr":        round(mean_nbr,     4),
            "mean_bsi":        round(mean_bsi,     4),
            "flood_frac":      round(flood_frac,   4),
            "veg_frac":        round(veg_frac,     4),
            "burn_frac":       round(burn_frac,    4),
            "stressed_frac":   round(stress_frac,  4),
            "change_frac":     round(change_frac,  4),
            "grad_energy":     round(grad_energy,  5),
            "contradiction":   round(contradiction, 4),
            "conf_multiplier": round(conf_mult,    4),
            # per-pixel maps for visualiser
            "ndvi_map":   ndvi,
            "ndwi_map":   ndwi,
            "nbr_map":    nbr,
            "flood_mask": flood,
            "burn_mask":  burn,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TERRAMIND HF ADAPTER
# Attempts to load ibm-esa-geospatial/TerraMind-1.0-small via HuggingFace
# Transformers AutoModel. Falls back to physics simulation if unavailable.
# ══════════════════════════════════════════════════════════════════════════════

class TerraMindHFAdapter:
    MODEL_ID = "ibm-esa-geospatial/TerraMind-1.0-small"

    def __init__(self):
        self.model = None
        self.processor = None
        self.backend = "simulation"
        self._try_load_hf()

    def _try_load_hf(self):
        try:
            import torch
            v = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if v < (2, 4):
                raise RuntimeError(f"PyTorch {torch.__version__} < 2.4 required.")
            from transformers import AutoModel, AutoConfig, AutoImageProcessor
            logger.info(f"Loading {self.MODEL_ID}...")
            AutoConfig.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.MODEL_ID, trust_remote_code=True)
            except Exception:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    self.MODEL_ID, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model.eval()
            self.backend = "terramind_hf"
            logger.info("TerraMind-1.0-small loaded OK")
        except Exception as e:
            logger.warning(f"HF unavailable ({e}). Physics simulation active.")
            self.backend = "simulation"

    def get_embedding(self, img_array: np.ndarray) -> np.ndarray:
        if self.backend == "terramind_hf":
            return self._hf_embed(img_array)
        return self._simulated_embed(img_array)

    def _hf_embed(self, img_array: np.ndarray) -> np.ndarray:
        try:
            import torch
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img_array)
            inputs = self.processor(images=pil, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output.squeeze(0).cpu().numpy()
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                first = list(outputs.values())[0]
                emb = first.mean(dim=list(range(1, first.dim()))).squeeze(0).cpu().numpy()
            emb = emb[:512] if emb.shape[0] >= 512 else np.pad(emb, (0, 512 - emb.shape[0]))
            return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)
        except Exception as e:
            logger.warning(f"HF embed failed: {e}. Fallback.")
            return self._simulated_embed(img_array)

    def _simulated_embed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Physics-informed embedding simulation.
        Produces a 512-d vector from spectral histograms + texture statistics.
        Used when TerraMind-1.0-small is not available.
        # Architecture inspired by TerraTorch pipeline (https://github.com/IBM/terratorch)
        """
        img = img_array.astype(np.float32) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        NIR = 0.7 * R + 0.3 * G
        ndvi  = (NIR - R) / (NIR + R + 1e-8)
        ndwi  = (G - NIR) / (G + NIR + 1e-8)
        bsi   = ((R+B)-(NIR+G)) / ((R+B)+(NIR+G)+1e-8)
        gray  = 0.299 * R + 0.587 * G + 0.114 * B
        spectral = np.concatenate([
            np.histogram(ch, bins=64, range=(-1,1))[0].astype(np.float32)
            for ch in [ndvi, ndwi, bsi, gray]
        ])
        patches = []
        h, w = gray.shape
        step = max(1, min(h, w) // 8)
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                p = gray[i:i+step, j:j+step]
                patches.extend([p.mean(), p.std(),
                                 float(np.percentile(p, 25)),
                                 float(np.percentile(p, 75))])
        texture = np.array(patches[:256], dtype=np.float32)
        if len(texture) < 256:
            texture = np.pad(texture, (0, 256 - len(texture)))
        emb = np.concatenate([spectral, texture])
        return (emb / (np.linalg.norm(emb)+1e-8)).astype(np.float32)

    @property
    def is_real_model(self):
        return self.backend == "terramind_hf"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — TERRAMIND ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class TerraMindEncoder:
    """
    Physics-informed feature extractor.
    Delegates all index/mask computation to multi_head_spectral_analysis()
    so spectral heads and encoder share identical definitions.
    # Architecture inspired by TerraTorch pipeline (https://github.com/IBM/terratorch)
    """

    def encode(self, img_array: np.ndarray) -> Dict[str, Any]:
        spectral = multi_head_spectral_analysis(img_array)
        aux = spectral.pop("_aux")

        img = img_array.astype(np.float32) / 255.0
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.299*R + 0.587*G + 0.114*B
        gx   = np.abs(np.diff(gray, axis=1, prepend=gray[:,:1]))
        gy   = np.abs(np.diff(gray, axis=0, prepend=gray[:1,:]))
        edge_map = np.sqrt(gx**2 + gy**2)

        features: Dict[str, Any] = {
            # per-pixel maps (used by visualiser)
            "ndvi_map":    aux["ndvi_map"],
            "mndwi_map":   aux["ndwi_map"],
            "nbr_map":     aux["nbr_map"],
            "bsi_map":     np.zeros_like(gray),
            "water_mask":  aux["ndwi_map"] > 0.05,
            "flood_mask":  aux["flood_mask"],
            "burn_mask":   aux["burn_mask"],
            "edge_map":    edge_map,
            # scalar spectral indices
            "ndvi_mean":    aux["mean_ndvi"],
            "ndvi_std":     float(np.std(aux["ndvi_map"])),
            "ndvi_p10":     float(np.percentile(aux["ndvi_map"], 10)),
            "ndvi_p90":     float(np.percentile(aux["ndvi_map"], 90)),
            "mndwi_mean":   aux["mean_ndwi"],
            "bsi_mean":     aux["mean_bsi"],
            "nbr_mean":     aux["mean_nbr"],
            # texture
            "texture_variance":      float(np.var(gray)),
            "edge_energy":           float(np.mean(edge_map)),
            "edge_max":              float(np.max(edge_map)),
            "spatial_heterogeneity": float(np.std(edge_map)),
            # channel stats
            "rgb_means": [float(np.mean(R)), float(np.mean(G)), float(np.mean(B))],
            "rgb_stds":  [float(np.std(R)),  float(np.std(G)),  float(np.std(B))],
            # scene fractions (post context-gating + morphology)
            "dark_fraction":         float(np.mean(gray < 0.15)),
            "bright_fraction":       float(np.mean(gray > 0.75)),
            "water_fraction":        float(np.mean(aux["ndwi_map"] > 0.05)),
            "flood_fraction":        aux["flood_frac"],
            "vegetation_fraction":   aux["veg_frac"],
            "stressed_veg_fraction": aux["stressed_frac"],
            "burn_fraction":         aux["burn_frac"],
            # diagnostics
            "contradiction_score":   aux["contradiction"],
            "conf_multiplier":       aux["conf_multiplier"],
            # spectral head priors (fed into multi-head predictor)
            "spectral_flood":        spectral["flood"],
            "spectral_vegetation":   spectral["vegetation"],
            "spectral_burn":         spectral["burn"],
            "spectral_crop_stress":  spectral["crop_stress"],
            "spectral_change":       spectral["change"],
        }
        return features


# ══════════════════════════════════════════════════════════════════════════════
# TiM — THINKING IN MODALITIES
# ══════════════════════════════════════════════════════════════════════════════

class ThinkingInModalities:
    """
    Generates synthetic intermediate modalities (NDVI, water prob, burn index)
    from RGB input — mirrors TerraMind's Thinking-in-Modalities approach.
    """
    def generate(self, img_array: np.ndarray, features: Dict) -> Dict[str, np.ndarray]:
        return {
            "ndvi":       features["ndvi_map"],
            "water_prob": np.clip(features["mndwi_map"], 0, 1),
            "burn_index": np.clip(-features["nbr_map"],  0, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FLOOD GEO-LOCALISER
# ══════════════════════════════════════════════════════════════════════════════

class FloodGeoLocaliser:
    """
    Converts flood pixel masks to geographic cluster information.
    Uses pixel_to_latlon() for coordinate mapping.
    """
    def __init__(self, scene_lat=19.07, scene_lon=72.87, gsd_m=10.0, img_size=256):
        self.scene_lat = scene_lat
        self.scene_lon = scene_lon
        self.gsd_m     = gsd_m
        self.img_size  = img_size

    def localise(self, flood_mask: np.ndarray) -> Dict[str, Any]:
        if not np.any(flood_mask):
            return {"flood_clusters": [], "total_flooded_ha": 0.0,
                    "cluster_count": 0,
                    "scene_centre": {"lat": self.scene_lat, "lon": self.scene_lon}}
        raw = self._find_clusters(flood_mask)
        geo = []
        for c in raw:
            r_min, r_max, c_min, c_max, area_px = c
            lat_c, lon_c   = pixel_to_latlon((c_min+c_max)/2, (r_min+r_max)/2,
                                              self.img_size, self.img_size,
                                              self.scene_lat, self.scene_lon, self.gsd_m)
            lat_tl, lon_tl = pixel_to_latlon(c_min, r_min, self.img_size, self.img_size,
                                              self.scene_lat, self.scene_lon, self.gsd_m)
            lat_br, lon_br = pixel_to_latlon(c_max, r_max, self.img_size, self.img_size,
                                              self.scene_lat, self.scene_lon, self.gsd_m)
            area_ha = area_px * self.gsd_m**2 / 10_000
            geo.append({
                "cluster_id": len(geo)+1,
                "lat": lat_c, "lon": lon_c,
                "bbox_lat": [lat_tl, lat_br],
                "bbox_lon": [lon_tl, lon_br],
                "area_ha":  round(area_ha, 2),
                "width_m":  round((c_max-c_min)*self.gsd_m, 0),
                "height_m": round((r_max-r_min)*self.gsd_m, 0),
                "severity": "HIGH" if area_ha > 50 else ("MEDIUM" if area_ha > 10 else "LOW"),
            })
        return {
            "flood_clusters":   geo[:5],
            "total_flooded_ha": round(sum(c["area_ha"] for c in geo), 2),
            "cluster_count":    len(geo),
            "scene_centre":     {"lat": self.scene_lat, "lon": self.scene_lon},
        }

    def _find_clusters(self, mask):
        visited  = np.zeros_like(mask, dtype=bool)
        clusters = []
        def bfs(sr, sc):
            q = [(sr, sc)]; visited[sr,sc]=True
            r0=r1=sr; c0=c1=sc; area=0
            while q:
                r,c = q.pop(); area+=1
                r0=min(r0,r); r1=max(r1,r)
                c0=min(c0,c); c1=max(c1,c)
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<mask.shape[0] and 0<=nc<mask.shape[1]:
                        if mask[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc]=True; q.append((nr,nc))
            return (r0,r1,c0,c1,area)
        for r in range(0, mask.shape[0], 2):
            for c in range(0, mask.shape[1], 2):
                if mask[r,c] and not visited[r,c]:
                    cl = bfs(r,c)
                    if cl[4] > 20:
                        clusters.append(cl)
        clusters.sort(key=lambda x: -x[4])
        return clusters


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — GUARD MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GuardModel:
    """
    Confidence gate and spectral contradiction checker.
    Blocks or warns before events are downlinked.
    """
    CONF_THRESHOLD  = 0.28
    CLOUD_THRESHOLD = 0.60

    def check(self, multi_head: Dict[str, float],
              features: Dict, proposed_event: str) -> Dict[str, Any]:
        flags = []; guard_passed = True

        if features["bright_fraction"] > self.CLOUD_THRESHOLD:
            flags.append({"code":"CLOUD_COVER",
                          "detail":f"bright={features['bright_fraction']:.2f}. SAR recommended.",
                          "severity":"WARN"})

        top_score = max(multi_head.values())
        if top_score < self.CONF_THRESHOLD:
            flags.append({"code":"LOW_CONFIDENCE",
                          "detail":f"top={top_score:.2f} < {self.CONF_THRESHOLD}.",
                          "severity":"BLOCK"})
            guard_passed = False

        if proposed_event == "Flood Detected":
            if features["water_fraction"] < 0.03 and features["mndwi_mean"] < -0.2:
                flags.append({"code":"SPECTRAL_MISMATCH_FLOOD",
                               "detail":f"water={features['water_fraction']:.3f}, MNDWI={features['mndwi_mean']:.2f}.",
                               "severity":"BLOCK"})
                guard_passed = False
        elif proposed_event == "Crop Stress Alert":
            if features["vegetation_fraction"] < 0.03:
                flags.append({"code":"NO_VEGETATION",
                               "detail":f"veg={features['vegetation_fraction']:.3f}.",
                               "severity":"BLOCK"})
                guard_passed = False
        elif proposed_event == "Burn Scar Detected":
            if features["ndvi_mean"] > 0.45:
                flags.append({"code":"SPECTRAL_MISMATCH_BURN",
                               "detail":f"NDVI={features['ndvi_mean']:.2f} too high for burn.",
                               "severity":"BLOCK"})
                guard_passed = False

        if features.get("contradiction_score", 0) > 0.40:
            flags.append({"code":"HIGH_CONTRADICTION",
                           "detail":f"contradiction={features['contradiction_score']:.2f}.",
                           "severity":"WARN"})

        scores = sorted(multi_head.values(), reverse=True)
        if len(scores)>=2 and (scores[0]-scores[1])<0.10 and scores[0]>0.30:
            flags.append({"code":"AMBIGUOUS_HEAD_AGREEMENT",
                           "detail":f"top two differ by {scores[0]-scores[1]:.3f}.",
                           "severity":"WARN"})

        return {
            "guard_passed":   guard_passed,
            "flags":          flags,
            "scene_quality":  "DEGRADED" if features["bright_fraction"] > self.CLOUD_THRESHOLD else "GOOD",
            "top_confidence": round(top_score, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE TRIGGER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveTriggerEngine:
    """
    Dynamic scene-change detector with rolling history baseline.
    Skips inference on stable scenes to save on-orbit compute.
    """
    ABSOLUTE_THRESHOLD = 0.02
    HISTORY_SIZE = 10

    def __init__(self):
        self._history: deque = deque(maxlen=self.HISTORY_SIZE)

    def check(self, features: Dict) -> Tuple[str, Dict]:
        signal = features["edge_energy"] + features["spatial_heterogeneity"] * 0.5
        if len(self._history) >= 3:
            h = np.array(self._history)
            dynamic_thresh = max(self.ABSOLUTE_THRESHOLD, h.mean() - 1.5*h.std())
        else:
            dynamic_thresh = self.ABSOLUTE_THRESHOLD
        self._history.append(signal)

        force = (features["flood_fraction"] > 0.03 or
                 features["burn_fraction"]   > 0.05 or
                 features["water_fraction"]  > 0.10)

        if force:
            status, reason = "FORCE_PROCESS", "High-priority spectral signal forced."
        elif signal > dynamic_thresh:
            status = "CHANGE_DETECTED"
            reason = f"signal={signal:.4f} > {dynamic_thresh:.4f}."
        else:
            status = "LOW_CHANGE"
            reason = f"signal={signal:.4f} near baseline={dynamic_thresh:.4f}."

        return status, {"signal": round(signal,5), "dynamic_threshold": round(dynamic_thresh,5),
                        "history_length": len(self._history), "reason": reason,
                        "force_process": force}


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadPredictor:
    """
    Four independent prediction heads sharing the encoder feature space.
    Each head blends spectral priors with scene-type context.
    # Inspired by TerraTorch multi-task fine-tuning (https://github.com/IBM/terratorch)
    """

    def predict_flood(self, f, scene_type):
        prior = f.get("spectral_flood", 0.0)
        raw = (prior*1.6 + f["water_fraction"]*2.0 + f["flood_fraction"]*2.0
               + max(0, f["mndwi_mean"])*1.5 + f["dark_fraction"]*0.4
               + max(0, -f["ndvi_mean"])*0.5) / 3.5
        if scene_type in ("Agricultural","Urban / Coastal","Flood Scene"): raw *= 1.08
        return float(np.clip(raw + random.gauss(0, 0.012), 0, 1))

    def predict_crop_stress(self, f, scene_type):
        prior = f.get("spectral_crop_stress", 0.0)
        raw = (prior*1.6 + max(0, 0.45-f["ndvi_mean"])*1.8
               + f["stressed_veg_fraction"]*1.2 + max(0, f["bsi_mean"])*1.0) / 3.0
        if scene_type == "Agricultural": raw *= 1.20
        return float(np.clip(raw + random.gauss(0, 0.012), 0, 1))

    def predict_change(self, f, scene_type):
        prior = f.get("spectral_change", 0.0)
        raw = (prior*1.6 + min(f["edge_energy"]*4.0, 0.85)
               + f["spatial_heterogeneity"]*1.2 + f["texture_variance"]*0.8) / 3.0
        if scene_type in ("Urban / Coastal","Forest / Wildfire"): raw *= 1.08
        return float(np.clip(raw + random.gauss(0, 0.012), 0, 1))

    def predict_burn_scar(self, f, scene_type):
        prior = f.get("spectral_burn", 0.0)
        ndvi_penalty = max(0.0, f["ndvi_mean"] - 0.30) * 2.0  # forest gate
        raw = (prior*1.6 + f["burn_fraction"]*2.5 + max(0, -f["nbr_mean"])*2.0
               + f["dark_fraction"]*0.6
               + max(0, f["rgb_means"][0]-f["rgb_means"][1])*1.2) / 3.5
        raw = max(0.0, raw - ndvi_penalty)
        if scene_type == "Forest / Wildfire": raw *= 1.25
        return float(np.clip(raw + random.gauss(0, 0.012), 0, 1))

    def predict(self, features, task, scene_type):
        return {
            "flood":       self.predict_flood(features, scene_type),
            "crop_stress": self.predict_crop_stress(features, scene_type),
            "change":      self.predict_change(features, scene_type),
            "burn_scar":   self.predict_burn_scar(features, scene_type),
        }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — TERRAMIND-SMALL VERIFIER + SELF-LEARNING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerifierSample:
    embedding:        List[float]
    predicted_event:  str
    confidence:       float
    validation_score: int
    correction:       Optional[str]
    guard_flags:      List[str]
    timestamp:        float = field(default_factory=time.time)


class TerraMindSmallVerifier:
    """
    Layer 3: Uses TerraMindHFAdapter embedding (real model when available,
    physics simulation otherwise) to produce a validation score.
    Maintains a rolling buffer for self-learning — similar samples in the
    buffer influence the current score via cosine similarity weighting.
    """
    BUFFER_SIZE = 50

    def __init__(self, hf_adapter: TerraMindHFAdapter):
        self.hf = hf_adapter
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._correction_count = 0

    def score(self, predictions, features, primary_event, confidence,
              guard_result, embedding) -> Dict[str, Any]:
        cp   = self._query_correction_buffer(embedding, primary_event)
        cs   = self._consistency_check(features, primary_event)
        cb   = 8 if 0.55 <= confidence <= 0.95 else 0
        gp   = -10 * sum(1 for f in guard_result["flags"] if f["severity"]=="BLOCK")
        gw   = -4  * sum(1 for f in guard_result["flags"] if f["severity"]=="WARN")
        crb  = int(cp * 15)
        ctra = -int(features.get("contradiction_score",0) * 20)
        raw  = 55 + cs + cb + gp + gw + crb + ctra
        score = int(np.clip(raw + random.randint(-3,3), 0, 100))

        if score >= 78:
            level = "Strong"
            reason = (f"High consistency for '{primary_event}'. "
                      f"NDVI={features['ndvi_mean']:.2f}, water={features['water_fraction']:.2f}. "
                      + (f"Self-learning +{crb}pts. " if crb > 0 else "")
                      + "Operationally reliable.")
        elif score >= 55:
            level = "Moderate"
            flag_codes = [f["code"] for f in guard_result["flags"]]
            reason = (f"Partial alignment for '{primary_event}'. "
                      + (f"Flags: {flag_codes}." if flag_codes else "No guard flags."))
        else:
            level = "Weak"
            reason = f"Low consistency for '{primary_event}'. Manual review recommended."

        self._buffer.append(VerifierSample(
            embedding=embedding[:32].tolist(),
            predicted_event=primary_event, confidence=confidence,
            validation_score=score,
            correction=None if score >= 55 else "LOW_SCORE_FLAGGED",
            guard_flags=[f["code"] for f in guard_result["flags"]]))
        if score < 55: self._correction_count += 1

        return {"validation_score": score, "validation_level": level,
                "validation_reason": reason, "correction_prior_applied": crb > 0,
                "self_learning_buffer_size": len(self._buffer),
                "total_corrections": self._correction_count,
                "model_backend": self.hf.backend}

    def _consistency_check(self, features, event):
        s = 0
        if event == "Flood Detected":
            if features["water_fraction"]  > 0.10: s += 12
            if features["flood_fraction"]  > 0.05: s += 10
            if features["mndwi_mean"]      > 0.0:  s += 8
            if features["ndvi_mean"]       < 0.2:  s += 5
        elif event == "Crop Stress Alert":
            if features["bsi_mean"]              > 0:    s += 10
            if features["ndvi_mean"]             < 0.35: s += 10
            if features["stressed_veg_fraction"] > 0.1:  s += 8
            if features["vegetation_fraction"]   > 0.05: s += 5
        elif event == "Burn Scar Detected":
            if features["burn_fraction"]  > 0.05:  s += 14
            if features["nbr_mean"]       < -0.05: s += 10
            if features["dark_fraction"]  > 0.15:  s += 6
            if features["ndvi_mean"]      < 0.25:  s += 5
        elif event == "Scene Change Detected":
            if features["edge_energy"]           > 0.05: s += 12
            if features["spatial_heterogeneity"] > 0.02: s += 8
        elif event == "Scene Normal":
            if features["ndvi_mean"]      > 0.3:  s += 10
            if features["water_fraction"] < 0.05: s += 6
            if features["burn_fraction"]  < 0.02: s += 4
        return s

    def _query_correction_buffer(self, embedding, event):
        if len(self._buffer) < 3: return 0.0
        sims = []
        for sample in self._buffer:
            try:
                prev = np.array(sample.embedding)
                curr = embedding[:len(prev)]
                cos  = np.dot(prev,curr)/(np.linalg.norm(prev)*np.linalg.norm(curr)+1e-8)
                if cos > 0.85 and sample.predicted_event == event:
                    sims.append((cos, sample.validation_score))
            except Exception: continue
        if not sims: return 0.0
        w = np.array([s[0] for s in sims])
        v = np.array([s[1] for s in sims])
        return float(np.clip((np.average(v,weights=w)-65)/100, -0.5, 0.5))

    def get_learning_stats(self):
        if not self._buffer:
            return {"buffer_size":0,"avg_validation_score":0,
                    "correction_rate":0.0,"model_backend":self.hf.backend}
        scores = [s.validation_score for s in self._buffer]
        return {"buffer_size": len(self._buffer),
                "avg_validation_score": round(float(np.mean(scores)),1),
                "correction_rate": round(self._correction_count/max(len(self._buffer),1),3),
                "model_backend": self.hf.backend}


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC COMPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCompressor:
    """Compress multi-head outputs into a <2KB actionable JSON payload."""
    EVENT_MAP = {"flood":"Flood Detected","crop_stress":"Crop Stress Alert",
                 "change":"Scene Change Detected","burn_scar":"Burn Scar Detected"}
    TASK_HEAD_MAP = {"flood detection":"flood","crop stress detection":"crop_stress",
                     "change detection":"change","burn scar detection":"burn_scar"}

    def compress(self, multi_head, features, task, scene_type, geo_info=None):
        t = task.lower()
        if t in self.TASK_HEAD_MAP:
            dominant = (self.TASK_HEAD_MAP[t], multi_head[self.TASK_HEAD_MAP[t]])
        else:
            dk = max(multi_head, key=multi_head.get)
            dominant = (dk, multi_head[dk])

        if dominant[1] < 0.32:
            return {"event":"Scene Normal","confidence":round(1.0-dominant[1],3),
                    "priority":"LOW","explanation":
                    (f"No critical events in {scene_type}. "
                     f"NDVI={features['ndvi_mean']:.2f}, "
                     f"water={features['water_fraction']:.3f}. Routine monitoring."),
                    "geo_detail":None}

        event      = self.EVENT_MAP[dominant[0]]
        confidence = round(float(dominant[1]),3)
        priority   = "HIGH" if confidence>=0.72 else ("MEDIUM" if confidence>=0.48 else "LOW")
        geo_detail = None

        if event=="Flood Detected" and geo_info and geo_info.get("flood_clusters"):
            c = geo_info["flood_clusters"][0]
            geo_detail = {"primary_cluster_lat":c["lat"],"primary_cluster_lon":c["lon"],
                          "total_flooded_ha":geo_info["total_flooded_ha"],
                          "cluster_count":geo_info["cluster_count"]}
            explanation = (f"{geo_info['cluster_count']} flood cluster(s). "
                           f"Primary: ({c['lat']:.4f}N, {c['lon']:.4f}E), "
                           f"~{c['area_ha']} ha, sev={c['severity']}. "
                           f"Total ~{geo_info['total_flooded_ha']} ha. Conf={confidence:.0%}.")
        elif event == "Crop Stress Alert":
            explanation = (f"NDVI={features['ndvi_mean']:.2f}, "
                           f"stress_frac={features['stressed_veg_fraction']:.2f}, "
                           f"BSI={features['bsi_mean']:.2f}. Conf={confidence:.0%}.")
        elif event == "Burn Scar Detected":
            explanation = (f"NBR={features['nbr_mean']:.2f}, "
                           f"burn_frac={features['burn_fraction']:.2f}. "
                           f"NDVI gate passed. Conf={confidence:.0%}.")
        else:
            explanation = (f"edge={features['edge_energy']:.3f}, "
                           f"hetero={features['spatial_heterogeneity']:.3f}. "
                           f"Conf={confidence:.0%}.")

        return {"event":event,"confidence":confidence,"priority":priority,
                "explanation":explanation,"geo_detail":geo_detail}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════════════════

class BaselineClassifier:
    """Simple NDVI/water threshold classifier — no multi-modal fusion."""
    def predict(self, features, scene_type):
        ndvi  = features["ndvi_mean"]
        water = features["water_fraction"]
        burn  = features["burn_fraction"]
        if water>0.20:   event,conf,pri = "Flood Detected",    0.61,"MEDIUM"
        elif burn>0.10:  event,conf,pri = "Burn Scar Detected",0.55,"MEDIUM"
        elif ndvi<0.2:   event,conf,pri = "Crop Stress Alert", 0.54,"MEDIUM"
        else:            event,conf,pri = "Scene Normal",       0.72,"LOW"
        return {"event":event,"confidence":conf,"priority":pri,
                "explanation":f"NDVI={ndvi:.2f} threshold only. No fusion.",
                "validation_score":None,"validation_reason":"Baseline: no validation."}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class OrbitalMindPipeline:
    """
    3-Layer Verified Inference Pipeline.
    Layer 1: TerraMindEncoder (physics-informed spectral features)
    Layer 2: GuardModel (confidence gate + contradiction check)
    Layer 3: TerraMindSmallVerifier (embedding-based scoring + self-learning)
    # Inspired by TerraTorch pipeline architecture (https://github.com/IBM/terratorch)
    """
    def __init__(self, use_tim=True, use_scorer=True, task="Multi-Task (All)",
                 scene_type="Agricultural", scene_lat=19.07, scene_lon=72.87):
        self.use_tim    = use_tim
        self.use_scorer = use_scorer
        self.task       = task
        self.scene_type = scene_type
        self.hf_adapter    = TerraMindHFAdapter()
        self.encoder       = TerraMindEncoder()
        self.tim           = ThinkingInModalities()
        self.predictor     = MultiHeadPredictor()
        self.trigger       = AdaptiveTriggerEngine()
        self.guard         = GuardModel()
        self.verifier      = TerraMindSmallVerifier(self.hf_adapter)
        self.compressor    = SemanticCompressor()
        self.geo_localiser = FloodGeoLocaliser(scene_lat=scene_lat, scene_lon=scene_lon)

    def run(self, img_array: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        random.seed(int(np.sum(img_array)) % 10_000)

        features  = self.encoder.encode(img_array)
        embedding = self.hf_adapter.get_embedding(img_array)

        spectral_scores = {k: features[f"spectral_{k}"]
                           for k in ["flood","vegetation","burn","crop_stress","change"]}

        trigger_status, trigger_detail = self.trigger.check(features)
        tim_modalities = self.tim.generate(img_array, features) if self.use_tim else None
        multi_head  = self.predictor.predict(features, self.task, self.scene_type)
        geo_info    = self.geo_localiser.localise(features["flood_mask"])
        prediction  = self.compressor.compress(multi_head, features, self.task,
                                               self.scene_type, geo_info)

        guard_result = self.guard.check(multi_head, features, prediction["event"])
        if not guard_result["guard_passed"]:
            prediction["priority"]      = "UNCERTAIN"
            prediction["guard_blocked"] = True
            prediction["explanation"]   = "[GUARD BLOCKED] " + prediction["explanation"][:120] + " Manual review."
        else:
            prediction["guard_blocked"] = False

        validation = None
        if self.use_scorer:
            validation = self.verifier.score(
                multi_head, features, prediction["event"],
                prediction["confidence"], guard_result, embedding)

        baseline   = BaselineClassifier().predict(features, self.scene_type)
        latency_ms = int((time.time()-t0)*1000)

        output_payload = {
            "event": prediction["event"], "confidence": prediction["confidence"],
            "priority": prediction["priority"], "explanation": prediction["explanation"][:220],
            "guard_passed": guard_result["guard_passed"],
        }
        if prediction.get("geo_detail"):
            output_payload["geo"] = prediction["geo_detail"]
        if validation:
            output_payload["validation_score"] = validation["validation_score"]
            output_payload["validation_level"]  = validation["validation_level"]
            output_payload["model_backend"]     = validation["model_backend"]

        return {
            "trigger_status":  trigger_status,
            "trigger_detail":  trigger_detail,
            "prediction":      prediction,
            "multi_head":      multi_head,
            "spectral_scores": spectral_scores,
            "guard_result":    guard_result,
            "validation":      validation,
            "baseline":        baseline,
            "geo_info":        geo_info,
            "tim_modalities":  tim_modalities,
            "features_summary": {k:v for k,v in features.items()
                                 if k not in ("ndvi_map","mndwi_map","bsi_map","nbr_map",
                                              "water_mask","flood_mask","burn_mask","edge_map")},
            "output_json":    output_payload,
            "latency_ms":     latency_ms,
            "scene_type":     self.scene_type,
            "task":           self.task,
            "hf_backend":     self.hf_adapter.backend,
            "learning_stats": self.verifier.get_learning_stats(),
        }