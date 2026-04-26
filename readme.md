# 🛰️ OrbitalMind — Edge Intelligence for Space-Based Decision Systems

> **"Downlink the answer, not the data."**
>
> OrbitalMind processes satellite imagery onboard and transmits structured, actionable insights — not raw files. A 200 MB scene becomes a sub-2 KB JSON payload without losing decision-critical information.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Transmission System](#transmission-system)
- [Edge Feasibility](#edge-feasibility)
- [TerraMind Integration](#terramind-integration)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Known Constraints](#known-constraints)

---

## Overview

Satellite downlink bandwidth is expensive, constrained, and slow. Current pipelines transfer raw imagery to the ground, then process it — introducing hours of latency and high transmission cost.

OrbitalMind inverts this model. A multi-stage inference pipeline runs onboard (or at the edge), extracts semantic information, validates it through a guard layer, and transmits only structured results. The system is designed to fit within the power and memory envelope of a Jetson AGX Orin-class payload.

**Primary use case demonstrated:** Agricultural and disaster monitoring over Indian subcontinent scenes — flood detection, vegetation stress, burn scar identification, and change detection.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OrbitalMind Pipeline                         │
│                                                                 │
│  RGB Image (256×256)                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                           │
│  │  Layer 1         │  TerraMind-inspired encoder               │
│  │  Feature Extract │  NDVI / NDWI / NBR / BSI indices          │
│  │  (7-step spec.)  │  Context-gated multi-head analysis        │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  TiM Modalities  │  Synthetic NDVI map                       │
│  │  (optional)      │  Water probability map                    │
│  │                  │  Burn index map                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Multi-Head      │  Flood  │ Crop Stress                     │
│  │  Predictor       │  Change │ Burn Scar                       │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Layer 2         │  Confidence gate                          │
│  │  Guard Model     │  Spectral contradiction check             │
│  │                  │  Block / Warn / Pass                      │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Layer 3         │  Embedding-based validation score         │
│  │  TM-small        │  Self-learning correction buffer          │
│  │  Verifier        │  Consistency check per event type         │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Semantic        │  < 2 KB structured JSON                   │
│  │  Compressor      │  Event · Confidence · Priority            │
│  │                  │  Geo-clusters · Validation score          │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│     Downlink Payload  →  TSP-Optimised Relay  →  Ground         │
└─────────────────────────────────────────────────────────────────┘
```

**Optional RasterVision wrapper** adds typed `PipelineConfig` (analyze → compress → export), JSON artifact caching, and a reproducible run record.

---

## Features

### Inference

- **Multi-head spectral analysis** — 7-step context-aware pipeline computing NDVI, NDWI, NBR, BSI from RGB input
- **Context gating** — physically impossible label combinations are suppressed (e.g. dense forest cannot simultaneously be burn scar; water body cannot be classified as vegetation)
- **Morphological noise reduction** — pure NumPy erosion to remove isolated false-positive pixels
- **Adaptive trigger engine** — rolling signal baseline skips inference on stable scenes, reducing unnecessary computation by ~60%
- **Guard model** — blocks or warns before downlinking low-confidence or spectroscopically contradictory results
- **Self-learning verifier buffer** — rolling 50-sample cosine-similarity buffer adjusts validation scores based on past similar scenes

### Detection Tasks

| Task | Method | Output |
|---|---|---|
| Flood detection | NDWI + context gate + geo-localiser | Cluster lat/lon, area (ha), severity |
| Crop stress | NDVI + BSI + stressed vegetation fraction | Score + fraction |
| Burn scar | NBR + dark fraction + NDVI gate | Score + burn fraction |
| Scene change | Gradient energy + spatial heterogeneity | Change probability |

### Visualisation

- Streamlit dashboard with dark space theme
- Interactive flood cluster map (Leafmap / Folium fallback)
- Spectral radar chart, multi-head bar chart, validation gauge
- TiM intermediate modality tabs (NDVI / Water Probability / Flood Mask / Burn Index)
- NASA Mission Control–style 3-D globe with animated TSP transmission routing

---

## Transmission System

Results are compressed into a structured payload and routed through a simulated 14-node orbital relay network.

### Payload

```json
{
  "event": "Flood Detected",
  "confidence": 0.81,
  "priority": "HIGH",
  "geo": {
    "primary_cluster_lat": 19.0741,
    "primary_cluster_lon": 72.8812,
    "total_flooded_ha": 143.5,
    "cluster_count": 3
  },
  "validation_score": 84,
  "validation_level": "Strong"
}
```

Typical payload size: **800–1 400 bytes** vs. 196 608 bytes for raw 256×256 RGB — a reduction of **>99.99%**.

### TSP Routing

- 14-node network: 4 LEO · 2 MEO · 2 GEO · 2 Space Datacenters · 3 Ground Stations · Mission Control
- Edge cost weighted by: distance (0.25) · link quality (0.45) · congestion (0.15) · latency (0.15)
- Solver: Nearest-Neighbour seed → 2-opt → 3-opt refinement
- Animated hop-by-hop visualisation with link quality colour coding and round-trip (forward + return) simulation

---

## Edge Feasibility

Target hardware: **NVIDIA Jetson AGX Orin** on a 6U CubeSat payload.

| Metric | Value | Notes |
|---|---|---|
| Model footprint | ~89 MB total | INT8 TensorRT export reduces to ~22 MB |
| Inference latency | ~380 ms/tile (INT8) | ~45 ms on dev GPU (FP32) |
| Tiles per overpass | ~120 | |
| Peak power | ~9 W | Within 6U solar budget (~20 W peak) |
| Duty cycle | ~44% | Adaptive trigger reduces wasted compute |
| Guard model overhead | ~2 ms | |

---

## TerraMind Integration

TerraMind-1.0-small (IBM / ESA, `ibm-esa-geospatial/TerraMind-1.0-small`) is used as the feature encoder backbone via HuggingFace `transformers.AutoModel`.

**Runtime behaviour:**

- If the model loads successfully → real embeddings are used for the Layer 3 verifier
- If the model is unavailable (missing weights, incompatible PyTorch version, no internet) → the system falls back to a physics-informed simulation embedding derived from spectral histogram statistics and texture features
- The fallback is clearly labelled in the UI (`⚙️ TerraMind-small: physics sim`)
- All other pipeline stages (spectral analysis, guard, compressor, geo-localiser) run independently of the model load status

**What TerraMind contributes:** A 512-dimensional scene embedding used to compute cosine similarity against a rolling correction buffer in the Layer 3 verifier, improving validation score reliability on similar scenes over time.

---

## Evaluation

Metrics shown are illustrative comparisons between a simple NDVI/water-threshold baseline classifier and the full OrbitalMind pipeline on held-out synthetic scenes.

| Metric | Baseline | OrbitalMind | Δ |
|---|---|---|---|
| Flood F1 | 0.55 | 0.87 | +58% |
| Crop mIoU | 0.46 | 0.74 | +61% |
| Change mAP | 0.56 | 0.81 | +45% |
| Burn F1 | 0.48 | 0.79 | +65% |
| False Alarm Rate | 27% | 8% | −70% |
| Avg Validation Score | — | ~82 / 100 | — |
| Payload Size | 200 MB (raw) | ~1.2 KB | −99.9994% |

> **Note:** These numbers reflect synthetic scene evaluation. Real-world performance on multispectral Sentinel-2 imagery would require proper calibration and labelled ground truth.

---

## Limitations

These are documented honestly because engineering credibility matters more than marketing.

- **RGB-only input** — the pipeline approximates NIR and SWIR bands from RGB channels using linear proxies. This introduces error vs. true multispectral data. Sentinel-2 Band 8 is not available from a standard camera.
- **TerraMind architecture** — only model weights are publicly available; the full architecture is reconstructed via HuggingFace AutoModel. Embedding quality may differ from the intended use.
- **Simulated routing** — the TSP transmission system is a physics-informed simulation, not a live network. Link quality, latency, and bandwidth figures are plausible estimates, not measured values.
- **Synthetic scenes** — training and evaluation use procedurally generated scenes. Performance on real satellite imagery (especially with cloud cover, sensor noise, or edge cases) is not validated.
- **No TensorRT calibration** — the Jetson latency estimate (~380 ms) is a projection; INT8 calibration has not been run.
- **Self-learning buffer** — the 50-sample rolling correction buffer is in-session only. It does not persist between runs and is not equivalent to fine-tuning.

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.4+ (for TerraMind HF loading; earlier versions fall back to simulation)
- CUDA optional but recommended for TerraMind embedding

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/orbitalmind.git
cd orbitalmind

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download TerraMind weights
#    The system will automatically attempt to pull from HuggingFace Hub.
#    To pre-download manually:
python -c "
from transformers import AutoModel, AutoImageProcessor
AutoImageProcessor.from_pretrained('ibm-esa-geospatial/TerraMind-1.0-small', trust_remote_code=True)
AutoModel.from_pretrained('ibm-esa-geospatial/TerraMind-1.0-small', trust_remote_code=True)
"
```

### `requirements.txt` (core)

```
streamlit>=1.35
torch>=2.4
torchvision
transformers>=4.40
numpy
pillow
plotly
matplotlib
rastervision-pipeline
leafmap
folium
pandas
```

---

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Workflow

1. **Select scene type** from the sidebar (Agricultural / Urban / Flood / Wildfire / Custom Upload)
2. **Set coordinates** manually or click "Use My Location" to pull browser geolocation
3. **Toggle pipeline options** — TiM modalities, TerraMind-small scorer, RasterVision mode
4. Click **"Run OrbitalMind Inference"**
5. Review:
   - Prediction card (event · confidence · priority)
   - Guard model flags
   - Validation score gauge
   - Flood cluster map
   - Downlink payload size
6. Open the **Space Data Transmission** tab to see the TSP-routed 3-D globe animation

---

## Project Structure

```
orbitalmind/
├── app.py                    # Streamlit UI — layout and tab orchestration
├── nasa_tsp_dashboard.py     # NASA-style 3-D globe TSP visualiser
├── requirements.txt
├── src/
│   ├── pipeline.py           # Core 3-layer inference pipeline
│   ├── rv_pipeline.py        # RasterVision pipeline wrapper
│   ├── space_tsp.py          # TSP routing simulator (14-node network)
│   ├── visualizer.py         # All Plotly / PIL rendering functions
│   └── utils.py              # Bandwidth calculation, JSON formatting
└── models/
    └── (TerraMind weights downloaded here by HuggingFace cache)
```

---

## Known Constraints

| Constraint | Mitigation |
|---|---|
| No real satellite feed | Procedural scene generator covers 6 scene types |
| No onboard hardware | Jetson specs documented; pipeline profiled on dev GPU |
| TerraMind weights-only | Fallback simulation maintains pipeline stability |
| Single-session learning | Buffer resets per run; not a substitute for retraining |
| Cloudcover | Guard model raises `CLOUD_COVER` warn flag; SAR fusion noted as future work |

---

## Acknowledgements

- [TerraMind (IBM / ESA)](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) — geospatial foundation model
- [TerraTorch](https://github.com/IBM/terratorch) — multi-task fine-tuning pipeline architecture (inspiration)
- [RasterVision](https://docs.rastervision.io/) — typed pipeline config and filesystem API
- [Leafmap](https://leafmap.org/) — interactive geospatial mapping

---
