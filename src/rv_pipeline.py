"""
rv_pipeline.py — OrbitalMind RasterVision Integration v1.0
════════════════════════════════════════════════════════════
Wraps the OrbitalMind 3-layer inference pipeline inside a proper
RasterVision Pipeline so that:

  1. Scene configuration is strongly typed & serialisable (Config / PipelineConfig)
  2. Inference is broken into 3 discrete RV commands that can be run
     independently, cached, or parallelised:
        • analyze   — spectral feature extraction + multi-head predictions
        • compress  — semantic compressor → <2 KB JSON payload
        • export    — write final downlink artifact + bandwidth report
  3. Results and intermediate state are persisted as JSON files under
     root_uri (local or cloud) using RasterVision's FileSystem API
  4. The RVPipelineRunner class is importable by app.py and Streamlit
     to run inference end-to-end and return the same result dict that
     the original OrbitalMindPipeline.run() returned.

Usage from app.py:
    from src.rv_pipeline import RVPipelineRunner
    runner = RVPipelineRunner(
        scene_type=scene_type, task=task,
        scene_lat=lat, scene_lon=lon,
        use_tim=True, use_scorer=True,
    )
    result = runner.run(img_array)          # returns same dict as before
    rv_cfg  = runner.last_config            # RV PipelineConfig (serialisable)
    rv_cmds = runner.last_commands_run      # list of command names executed
"""

from __future__ import annotations
from src.space_tsp import TransmissionSimulator
sim = TransmissionSimulator()

import os
import json
import tempfile
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

# ── RasterVision imports ─────────────────────────────────────────────
from rastervision.pipeline.config import Config, Field, register_config
from rastervision.pipeline.pipeline import Pipeline
from rastervision.pipeline.pipeline_config import PipelineConfig

from rastervision.pipeline.file_system.utils import (
    json_to_file,
    file_to_json,
    make_dir,
    file_exists,
)

logger = logging.getLogger("OrbitalMind.RV")

# All configs registered under the built-in 'rastervision.pipeline' plugin
# because our code lives outside the rastervision.* namespace.
_RV_PLUGIN = "rastervision.pipeline"


# ══════════════════════════════════════════════════════════════════════
# TYPED CONFIGURATION SCHEMAS
# ══════════════════════════════════════════════════════════════════════


class OMSceneConfig(Config):
    """
    Strongly-typed description of the satellite scene being analysed.
    Serialisable to/from JSON via RasterVision Config machinery.
    """
    scene_type: str   = Field("Agricultural",      description="Synthetic scene type")
    scene_lat:  float = Field(19.07,               description="Scene centre latitude (WGS-84)")
    scene_lon:  float = Field(72.87,               description="Scene centre longitude (WGS-84)")
    gsd_m:      float = Field(10.0,                description="Ground sample distance (m/pixel)")
    img_width:  int   = Field(256,                 description="Image width in pixels")
    img_height: int   = Field(256,                 description="Image height in pixels")



class OMInferenceConfig(PipelineConfig):
    """
    Top-level RasterVision PipelineConfig for one OrbitalMind inference run.
    Includes the scene description plus all model toggles and thresholds.
    Can be serialised to JSON and replayed later for reproducibility.
    """
    scene:              OMSceneConfig = Field(default_factory=OMSceneConfig)
    task:               str           = Field("Multi-Task (All)",  description="Primary inference task")
    use_tim:            bool          = Field(True,                description="Enable Thinking-in-Modalities")
    use_scorer:         bool          = Field(True,                description="Enable TerraMind-small verifier")
    confidence_threshold: float       = Field(0.32,               description="Minimum confidence to flag an event")
    adaptive_trigger:   bool          = Field(True,                description="Enable adaptive trigger engine")

    def build(self, tmp_dir: str) -> "OrbitalMindRVPipeline":
        return OrbitalMindRVPipeline(self, tmp_dir)


# ══════════════════════════════════════════════════════════════════════
# RASTERVISION PIPELINE — 3 discrete commands
# ══════════════════════════════════════════════════════════════════════

class OrbitalMindRVPipeline(Pipeline):
    """
    RasterVision Pipeline with 3 sequential commands:

    1. analyze  — TerraMindEncoder + MultiHeadPredictor + Guard
    2. compress — SemanticCompressor → <2 KB JSON
    3. export   — write final artifact + bandwidth metrics

    Intermediate results are written as JSON files under
    self.config.root_uri so each stage can be cached or re-run
    independently (standard RV pattern).
    """
    commands:       List[str] = ["analyze", "compress", "export"]
    split_commands: List[str] = []
    gpu_commands:   List[str] = []

    # ------------------------------------------------------------------
    # COMMAND 1 — spectral analysis + multi-head predictions
    # ------------------------------------------------------------------
    def analyze(self):
        """
        Layer 1: TerraMindEncoder → multi_head_spectral_analysis
        Layer 2: GuardModel
        Layer 3: TerraMindSmallVerifier (if use_scorer)

        Reads img_array from the run-time context (set by _inject_image).
        Writes analyze_result.json under root_uri.
        """
        logger.info("[RV][analyze] Starting spectral analysis…")
        cfg   = self.config
        scene = cfg.scene
        img   = self._get_image()

        # Late import to avoid circular deps; pipeline.py lives in src/
        from src.pipeline import (
            TerraMindEncoder,
            TerraMindHFAdapter,
            MultiHeadPredictor,
            GuardModel,
            AdaptiveTriggerEngine,
            FloodGeoLocaliser,
            ThinkingInModalities,
            TerraMindSmallVerifier,
            multi_head_spectral_analysis,
        )

        t0 = time.time()

        encoder   = TerraMindEncoder()
        hf        = TerraMindHFAdapter()
        predictor = MultiHeadPredictor()
        guard     = GuardModel()
        trigger   = AdaptiveTriggerEngine()
        geo_loc   = FloodGeoLocaliser(
            scene_lat=scene.scene_lat,
            scene_lon=scene.scene_lon,
            gsd_m=scene.gsd_m,
            img_size=scene.img_width,
        )
        tim = ThinkingInModalities()
        verifier = TerraMindSmallVerifier(hf)

        features  = encoder.encode(img)
        embedding = hf.get_embedding(img)

        spectral_scores = {k: features[f"spectral_{k}"]
                           for k in ["flood", "vegetation", "burn", "crop_stress", "change"]}

        trigger_status, trigger_detail = trigger.check(features)

        tim_modalities = None
        if cfg.use_tim:
            tim_modalities = tim.generate(img, features)

        multi_head = predictor.predict(features, cfg.task, scene.scene_type)
        geo_info   = geo_loc.localise(features["flood_mask"])

        # Determine dominant event for guard check
        from src.pipeline import SemanticCompressor
        _sc    = SemanticCompressor()
        _pred  = _sc.compress(multi_head, features, cfg.task, scene.scene_type, geo_info)
        proposed_event = _pred["event"]

        guard_result = guard.check(multi_head, features, proposed_event)

        validation = None
        if cfg.use_scorer:
            validation = verifier.score(
                multi_head, features, proposed_event,
                _pred["confidence"], guard_result, embedding,
            )

        latency_ms = int((time.time() - t0) * 1000)

        # Strip numpy arrays before serialising
        features_safe = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in features.items()
                         if k not in ("ndvi_map", "mndwi_map", "bsi_map", "nbr_map",
                                      "water_mask", "flood_mask", "burn_mask", "edge_map")}

        analyze_result = {
            "trigger_status":  trigger_status,
            "trigger_detail":  trigger_detail,
            "multi_head":      multi_head,
            "spectral_scores": spectral_scores,
            "guard_result":    guard_result,
            "validation":      validation,
            "geo_info":        {
                "flood_clusters":   geo_info.get("flood_clusters", []),
                "total_flooded_ha": geo_info.get("total_flooded_ha", 0.0),
                "cluster_count":    geo_info.get("cluster_count", 0),
                "scene_centre":     geo_info.get("scene_centre", {}),
            },
            "features_summary": features_safe,
            "latency_ms":      latency_ms,
            "hf_backend":      hf.backend,
            "learning_stats":  verifier.get_learning_stats(),
            "scene_type":      scene.scene_type,
            "task":            cfg.task,
        }

        make_dir(cfg.root_uri)
        json_to_file(analyze_result, os.path.join(cfg.root_uri, "analyze_result.json"))
        logger.info(f"[RV][analyze] Done in {latency_ms} ms — "
                    f"trigger={trigger_status}, guard={'PASS' if guard_result['guard_passed'] else 'BLOCK'}")

    # ------------------------------------------------------------------
    # COMMAND 2 — semantic compression → <2 KB payload
    # ------------------------------------------------------------------
    def compress(self):
        """
        Reads analyze_result.json, runs SemanticCompressor,
        applies confidence threshold, writes compressed.json.
        """
        logger.info("[RV][compress] Building downlink payload…")
        cfg = self.config
        ar  = file_to_json(os.path.join(cfg.root_uri, "analyze_result.json"))

        from src.pipeline import SemanticCompressor, FloodGeoLocaliser
        import random

        random.seed(0)
        sc = SemanticCompressor()

        multi_head = ar["multi_head"]
        features   = ar["features_summary"]
        geo_info   = ar["geo_info"]
        guard      = ar["guard_result"]

        prediction = sc.compress(multi_head, features, cfg.task,
                                  cfg.scene.scene_type, geo_info)

        if not guard["guard_passed"]:
            prediction["priority"]      = "UNCERTAIN"
            prediction["guard_blocked"] = True
            prediction["explanation"]   = (
                "[GUARD BLOCKED] " + prediction["explanation"][:120] + " Manual review."
            )
        else:
            prediction["guard_blocked"] = False

        # Apply RV-level confidence threshold
        if prediction["confidence"] < cfg.confidence_threshold:
            prediction["priority"]     = "LOW"
            prediction["rv_filtered"]  = True
            prediction["explanation"] += (
                f" [RV] Conf {prediction['confidence']:.2f} < "
                f"threshold {cfg.confidence_threshold:.2f}."
            )
        else:
            prediction["rv_filtered"] = False

        # Build final output JSON (<2 KB)
        output_payload = {
            "event":        prediction["event"],
            "confidence":   prediction["confidence"],
            "priority":     prediction["priority"],
            "explanation":  prediction["explanation"][:220],
            "guard_passed": guard["guard_passed"],
            "rv_filtered":  prediction["rv_filtered"],
        }
        if prediction.get("geo_detail"):
            output_payload["geo"] = prediction["geo_detail"]
        if ar.get("validation"):
            output_payload["validation_score"] = ar["validation"]["validation_score"]
            output_payload["validation_level"]  = ar["validation"]["validation_level"]
            output_payload["model_backend"]     = ar["validation"]["model_backend"]

        compressed = {
            "prediction":   prediction,
            "output_json":  output_payload,
        }

        json_to_file(compressed, os.path.join(cfg.root_uri, "compressed.json"))
        logger.info(f"[RV][compress] event={prediction['event']}, "
                    f"conf={prediction['confidence']:.2f}, priority={prediction['priority']}")

    # ------------------------------------------------------------------
    # COMMAND 3 — export final artifact + bandwidth report
    # ------------------------------------------------------------------
    def export(self):
        """
        Reads compressed.json + analyze_result.json, writes:
          - downlink_payload.json  (the <2 KB artifact)
          - bandwidth_report.json  (raw vs compressed size comparison)
          - rv_config.json         (the full PipelineConfig for reproducibility)
        """
        logger.info("[RV][export] Writing final artifacts…")
        cfg  = self.config
        ar   = file_to_json(os.path.join(cfg.root_uri, "analyze_result.json"))
        comp = file_to_json(os.path.join(cfg.root_uri, "compressed.json"))

        output_payload = comp["output_json"]

        # Downlink artifact
        dl_path = os.path.join(cfg.root_uri, "downlink_payload.json")
        json_to_file(output_payload, dl_path)

        # Bandwidth report
        raw_bytes    = cfg.scene.img_width * cfg.scene.img_height * 3   # uint8 RGB
        json_str     = json.dumps(output_payload, separators=(",", ":"))
        json_bytes   = len(json_str.encode("utf-8"))
        saving_pct   = (1.0 - json_bytes / raw_bytes) * 100
        bw_report    = {
            "raw_bytes":        raw_bytes,
            "raw_kb":           raw_bytes / 1024,
            "output_bytes":     json_bytes,
            "saving_pct":       round(saving_pct, 6),
            "compression_ratio": round(raw_bytes / max(json_bytes, 1), 1),
            "within_2kb_limit":  json_bytes <= 2048,
        }
        json_to_file(bw_report, os.path.join(cfg.root_uri, "bandwidth_report.json"))

        # RV config snapshot for reproducibility
        rv_cfg_dict = cfg.model_dump()
        json_to_file(rv_cfg_dict, os.path.join(cfg.root_uri, "rv_config.json"))

        logger.info(
            f"[RV][export] Payload {json_bytes} B "
            f"({saving_pct:.4f}% saving). "
            f"{'✅ within 2KB' if json_bytes <= 2048 else '❌ exceeds 2KB'}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_image(self) -> np.ndarray:
        """Retrieve the injected image array from tmp_dir."""
        img_path = os.path.join(self.tmp_dir, "_rv_img.npy")
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"[RV] Image not found at {img_path}. "
                "Call _inject_image() before running the pipeline."
            )
        return np.load(img_path)

    def _inject_image(self, img_array: np.ndarray):
        """Persist img_array to tmp_dir so pipeline commands can load it."""
        make_dir(self.tmp_dir)
        np.save(os.path.join(self.tmp_dir, "_rv_img.npy"), img_array)


# ══════════════════════════════════════════════════════════════════════
# HIGH-LEVEL RUNNER — called by app.py
# ══════════════════════════════════════════════════════════════════════

class RVPipelineRunner:
    """
    Drop-in wrapper around OrbitalMindRVPipeline for use in app.py.

    Provides the same interface as the old OrbitalMindPipeline:
        runner = RVPipelineRunner(scene_type=..., task=..., ...)
        result = runner.run(img_array)   # returns the full result dict

    Additionally exposes:
        runner.last_config         → OMInferenceConfig (serialisable)
        runner.last_commands_run   → ['analyze', 'compress', 'export']
        runner.last_root_uri       → path to all intermediate JSON files
    """

    def __init__(
        self,
        scene_type:   str   = "Agricultural",
        task:         str   = "Multi-Task (All)",
        scene_lat:    float = 19.07,
        scene_lon:    float = 72.87,
        use_tim:      bool  = True,
        use_scorer:   bool  = True,
        conf_threshold: float = 0.32,
        root_uri:     Optional[str] = None,
    ):
        self.scene_type    = scene_type
        self.task          = task
        self.scene_lat = scene_lat
        self.scene_lon = scene_lon
        self.use_tim       = use_tim
        self.use_scorer    = use_scorer
        self.conf_threshold = conf_threshold
        self.root_uri      = root_uri          # None → use a temp dir

        self.last_config       = None
        self.last_commands_run = []
        self.last_root_uri     = None

    def run(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Run the full 3-command RV pipeline and return a unified result dict
        identical in shape to what OrbitalMindPipeline.run() returned,
        plus an extra 'rv' key with RasterVision-specific metadata.
        """
        # Create or reuse a working directory
        _tmpdir_ctx = None
        if self.root_uri:
            root = self.root_uri
            make_dir(root)
        else:
            _tmpdir_ctx = tempfile.TemporaryDirectory()
            root = _tmpdir_ctx.name

        self.last_root_uri = root

        scene_cfg = OMSceneConfig(
            scene_type=self.scene_type,
            scene_lat=self.scene_lat,
            scene_lon=self.scene_lon,
        )
        inf_cfg = OMInferenceConfig(
            root_uri=root,
            scene=scene_cfg,
            task=self.task,
            use_tim=self.use_tim,
            use_scorer=self.use_scorer,
            confidence_threshold=self.conf_threshold,
        )
        self.last_config = inf_cfg

        # Build the RV pipeline instance
        pipe = inf_cfg.build(root)

        # Inject the image so commands can load it
        pipe._inject_image(img_array)

        # Run all three commands sequentially
        self.last_commands_run = []
        for cmd in pipe.commands:
            logger.info(f"[RV] Running command: {cmd}")
            getattr(pipe, cmd)()
            self.last_commands_run.append(cmd)

        # Read back all produced artifacts
        analyze_result = file_to_json(os.path.join(root, "analyze_result.json"))
        compressed     = file_to_json(os.path.join(root, "compressed.json"))
        bw_report      = file_to_json(os.path.join(root, "bandwidth_report.json"))
        rv_cfg_dict    = file_to_json(os.path.join(root, "rv_config.json"))

        # Assemble unified result dict (same shape as original pipeline)
        result = {
            "trigger_status":   analyze_result["trigger_status"],
            "trigger_detail":   analyze_result["trigger_detail"],
            "prediction":       compressed["prediction"],
            "multi_head":       analyze_result["multi_head"],
            "spectral_scores":  analyze_result["spectral_scores"],
            "guard_result":     analyze_result["guard_result"],
            "validation":       analyze_result.get("validation"),
            "baseline":         None,   # baseline not run in RV mode
            "geo_info":         analyze_result["geo_info"],
            "tim_modalities":   None,   # numpy arrays not persisted to JSON
            "features_summary": analyze_result["features_summary"],
            "output_json":      compressed["output_json"],
            "latency_ms":       analyze_result["latency_ms"],
            "scene_type":       self.scene_type,
            "task":             self.task,
            "hf_backend":       analyze_result["hf_backend"],
            "learning_stats":   analyze_result["learning_stats"],
            # RasterVision-specific metadata
            "rv": {
                "enabled":        True,
                "commands_run":   self.last_commands_run,
                "root_uri":       root,
                "config":         rv_cfg_dict,
                "bandwidth":      bw_report,
                "config_type":    "OMInferenceConfig",
                "rv_filtered":    compressed["prediction"].get("rv_filtered", False),
            },
        }
        tx = sim.simulate(
            payload_bytes=1200,  # keep constant for demo
            scene_lat=self.scene_lat,
            scene_lon=self.scene_lon
        )

        result["transmission"] = tx

        # Clean up temp dir only if we created one
        if _tmpdir_ctx:
            _tmpdir_ctx.cleanup()

        return result

    def get_config_json(self) -> str:
        """Return the last RV PipelineConfig as a pretty-printed JSON string."""
        if self.last_config is None:
            return "{}"
        return json.dumps(self.last_config.model_dump(), indent=2)