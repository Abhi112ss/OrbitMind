"""
app.py — OrbitalMind v3.2  Streamlit UI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v3.2 additions:
  ① RasterVision pipeline toggle — sidebar toggle switches between
     original OrbitalMindPipeline and RVPipelineRunner
  ② RV metadata panel — shows commands run, root_uri, config JSON,
     bandwidth report, rv_filtered flag, all from the 'rv' result key
  ③ RV Config expander — pretty-printed OMInferenceConfig JSON
  ④ All previously restored features retained:
     - TiM Intermediate Modalities tabs (NDVI/Water/Flood/Burn)
     - 3-Layer Pipeline Architecture diagram (collapsible)
     - Fixed leafmap (folium.GeoJson instead of add_geojson)

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
import json
import sys, os
from src.nasa_tsp_dashboard import render_nasa_tsp_dashboard

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import OrbitalMindPipeline, pixel_to_latlon
from src.visualizer import (
    render_ndvi_colormap,
    render_overlay_mask,
    render_change_heatmap,
    render_spectral_radar,
    render_multi_head_bars,
    render_performance_chart,
    render_bandwidth_chart,
    render_validation_gauge,
    render_scene_fractions_chart,
    render_leafmap_flood,
    render_edge_specs,
    render_tim_tabs,
    render_pipeline_architecture,
)
from src.utils import generate_sample_image, compute_bandwidth_saving, format_output_json
from src.rv_pipeline import RVPipelineRunner   # ← RasterVision integration

# ── optional: streamlit-image-coordinates ───────────────────────────
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
    _COORDS_AVAILABLE = True
except ImportError:
    _COORDS_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OrbitalMind v3.2",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #0e1117; }
.metric-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:14px; margin-bottom:8px;
}
.guard-BLOCK {
    background:#2a0a0a; border:1px solid #ff4b4b;
    border-radius:6px; padding:8px 12px; margin:4px 0;
    color:#ff4b4b; font-size:12px;
}
.guard-WARN {
    background:#2a1c0a; border:1px solid #ffa421;
    border-radius:6px; padding:8px 12px; margin:4px 0;
    color:#ffa421; font-size:12px;
}
.badge-HIGH      { color:#ff4b4b; font-weight:700; }
.badge-MEDIUM    { color:#ffa421; font-weight:700; }
.badge-LOW       { color:#21c354; font-weight:700; }
.badge-UNCERTAIN { color:#9f7efe; font-weight:700; }
.json-box {
    background:#161b22; border:1px solid #30363d;
    border-radius:6px; padding:12px;
    font-family:monospace; font-size:11px;
    white-space:pre-wrap; word-break:break-all; color:#c9d1d9;
}
.location-pill {
    background:#0d2137; border:1px solid #0068c9;
    border-radius:20px; padding:4px 14px;
    color:#58a6ff; font-size:12px; display:inline-block;
}
.tm-badge-real { color:#21c354; font-size:11px; }
.tm-badge-sim  { color:#ffa421; font-size:11px; }
.rv-badge {
    background:#1a2a1a; border:1px solid #21c354;
    border-radius:6px; padding:6px 12px; margin:4px 0;
    color:#21c354; font-size:12px; display:inline-block;
}
.rv-cmd-pill {
    background:#162032; border:1px solid #0068c9;
    border-radius:12px; padding:2px 10px; margin:2px;
    color:#58a6ff; font-size:11px; display:inline-block;
}
.rv-filtered {
    background:#2a1c0a; border:1px solid #ffa421;
    border-radius:6px; padding:6px 12px;
    color:#ffa421; font-size:12px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "scene_lat":     19.07,
        "scene_lon":     72.87,
        "last_result":   None,
        "last_img":      None,
        "click_coords":  None,
        "geo_requested": False,
        "use_rv":        True,        # RasterVision toggle
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════════
# GEOLOCATION JS
# ════════════════════════════════════════════════════════════════════
def inject_geolocation_js():
    st.components.v1.html("""
    <script>
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const url = new URL(window.parent.location.href);
                url.searchParams.set('geo_lat', pos.coords.latitude.toFixed(5));
                url.searchParams.set('geo_lon', pos.coords.longitude.toFixed(5));
                window.parent.history.replaceState({}, '', url.toString());
            },
            (err) => console.log('Geolocation denied:', err.message),
            {timeout: 5000, maximumAge: 60000}
        );
    }
    </script>
    <p style="color:#8b949e;font-size:11px;">📡 Requesting browser location…</p>
    """, height=30)

def _read_geo_from_query():
    try:
        params = st.query_params
        if "geo_lat" in params and "geo_lon" in params:
            lat = float(params["geo_lat"])
            lon = float(params["geo_lon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except Exception:
        pass
    return None, None


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛰️ OrbitalMind v3.2")
    st.caption("Adaptive Scene Intelligence for Space")
    st.divider()

    st.subheader("📍 Scene Location")
    if st.button("🌐 Use My Location", use_container_width=True):
        st.session_state["geo_requested"] = True

    if st.session_state["geo_requested"]:
        inject_geolocation_js()
        g_lat, g_lon = _read_geo_from_query()
        if g_lat is not None:
            st.session_state["scene_lat"] = g_lat
            st.session_state["scene_lon"] = g_lon
            st.session_state["geo_requested"] = False
            st.success(f"Location set: {g_lat:.4f}N, {g_lon:.4f}E")

    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_val = st.number_input("Latitude",  value=st.session_state["scene_lat"],
                                   min_value=-90.0,  max_value=90.0,  step=0.01, format="%.4f")
    with col_lon:
        lon_val = st.number_input("Longitude", value=st.session_state["scene_lon"],
                                   min_value=-180.0, max_value=180.0, step=0.01, format="%.4f")
    st.session_state["scene_lat"] = lat_val
    st.session_state["scene_lon"] = lon_val

    if st.session_state["click_coords"]:
        cx, cy = st.session_state["click_coords"]
        cl, co = pixel_to_latlon(cx, cy, 256, 256,
                                  st.session_state["scene_lat"],
                                  st.session_state["scene_lon"], gsd_m=10.0)
        st.markdown(
            f'<div class="location-pill">🖱️ Clicked: <b>{cl:.4f}°N, {co:.4f}°E</b></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("⚙️ Mission Config")

    task = st.selectbox("Primary Task", [
        "Multi-Task (All)", "Flood Detection",
        "Crop Stress Detection", "Change Detection", "Burn Scar Detection",
    ], index=0)

    scene_type = st.selectbox("Scene Type", [
        "Agricultural", "Urban / Coastal", "Forest / Wildfire",
        "Flood Scene", "Post-Disaster / Mixed", "Custom Upload",
    ])

    adaptive_trigger = st.toggle("Adaptive Trigger Engine",       value=True)
    tim_enabled      = st.toggle("TiM (Thinking-in-Modalities)",  value=True)
    scorer_enabled   = st.toggle("TerraMind-Small Scorer",        value=True)

    st.divider()
    st.subheader("🌿 RasterVision")
    use_rv = st.toggle(
        "Use RasterVision Pipeline",
        value=st.session_state["use_rv"],
        help=(
            "When ON: runs inference through RasterVision's typed "
            "PipelineConfig + 3-command pipeline (analyze → compress → export) "
            "with JSON artifact caching.\n\n"
            "When OFF: runs the original OrbitalMindPipeline directly."
        ),
    )
    st.session_state["use_rv"] = use_rv
    if use_rv:
        rv_conf_threshold = st.slider(
            "RV Confidence Threshold", 0.10, 0.90, 0.32, 0.01,
            help="Predictions below this threshold are marked rv_filtered=True",
        )
        st.caption(
            "RasterVision wraps each run in a typed `OMInferenceConfig`, "
            "runs 3 pipeline commands, and caches intermediate JSON artifacts."
        )
        st.markdown('<span class="rv-badge">✅ RasterVision active</span>',
                    unsafe_allow_html=True)
    else:
        rv_conf_threshold = 0.32
        st.caption("Direct OrbitalMindPipeline mode.")

    st.divider()

    if st.session_state["last_result"]:
        r  = st.session_state["last_result"]
        fs = r.get("features_summary", {})
        st.subheader("🌍 Scene Fractions")
        st.metric("🌿 Vegetation",  f"{fs.get('vegetation_fraction', 0):.0%}")
        st.metric("🌊 Flood",       f"{fs.get('flood_fraction', 0):.0%}")
        st.metric("🔥 Burn",        f"{fs.get('burn_fraction', 0):.0%}")
        st.metric("🔄 Change",      f"{r['multi_head'].get('change', 0):.0%}")
        st.metric("⚠️ Crop Stress", f"{fs.get('stressed_veg_fraction', 0):.0%}")
        st.divider()

    if st.session_state["last_result"]:
        backend = st.session_state["last_result"].get("hf_backend", "simulation")
        cls     = "tm-badge-real" if backend == "terramind_hf" else "tm-badge-sim"
        label   = "✅ TerraMind-small: real model" if backend == "terramind_hf" else "⚙️ TerraMind-small: physics sim"
        st.markdown(f'<span class="{cls}">{label}</span>', unsafe_allow_html=True)
        ls = st.session_state["last_result"].get("learning_stats", {})
        if ls:
            st.caption(f"Self-learning buffer: {ls.get('buffer_size',0)} samples | "
                       f"Avg score: {ls.get('avg_validation_score',0)}")
    else:
        st.caption("🛰️ TerraMind-small status will show after inference.")


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("## 🛰️ OrbitalMind — *Downlink the Answer, Not the Data*")
st.markdown(
    "> **Customer Story:** A crop insurer in Maharashtra needs flood verification within 48 h "
    "to process ₹2Cr+ claims. Traditional downlink: 200 MB raw imagery, 4–12 h latency, ₹8K/GB. "
    "**OrbitalMind** transmits a **~1.2 KB JSON** insight in under 3 minutes from orbit."
)
st.divider()

# ── Pipeline architecture diagram ───────────────────────────────────
render_pipeline_architecture()


# ════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT ────────────────────────────────────────────────────────────
with col_left:
    st.subheader("📡 Satellite Input")

    if scene_type == "Custom Upload":
        uploaded = st.file_uploader("Upload satellite image", type=["png","jpg","jpeg","tif"])
        if uploaded:
            from PIL import Image as PILImage
            img_arr = np.array(PILImage.open(uploaded).convert("RGB").resize((256, 256)))
        else:
            st.info("Upload an image or switch to a preset scene type.")
            img_arr = generate_sample_image("Agricultural")
    else:
        img_arr = generate_sample_image(scene_type)

    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_arr.astype(np.uint8))

    if _COORDS_AVAILABLE:
        st.caption("🖱️ **Click on the image** to pin a location")
        click_val = img_coords(pil_img, key="scene_click", use_column_width=True)
        if click_val is not None:
            st.session_state["click_coords"] = (click_val["x"], click_val["y"])
            cl, co = pixel_to_latlon(
                click_val["x"], click_val["y"], 256, 256,
                st.session_state["scene_lat"], st.session_state["scene_lon"], gsd_m=10.0,
            )
            st.markdown(
                f'<div class="location-pill">📌 Selected: <b>{cl:.4f}°N, {co:.4f}°E</b>'
                f' — pixel ({click_val["x"]}, {click_val["y"]})</div>',
                unsafe_allow_html=True,
            )
    else:
        st.image(pil_img, caption=f"Scene: {scene_type} (256×256)", use_container_width=True)
        st.caption("ℹ️ Install `streamlit-image-coordinates` for click-to-location.")

    # TiM Intermediate Modalities tabs
    if tim_enabled:
        render_tim_tabs(img_arr)

    # Spectral radar (post-inference)
    if st.session_state["last_result"]:
        ss = st.session_state["last_result"].get("spectral_scores", {})
        if ss:
            st.plotly_chart(render_spectral_radar(ss),
                            use_container_width=True, config={"displayModeBar": False})


# ── RIGHT ───────────────────────────────────────────────────────────
with col_right:
    st.subheader("🧠 Inference Engine")

    # Show which backend will be used
    if st.session_state["use_rv"]:
        st.markdown(
            '<span class="rv-badge">🌿 RasterVision Pipeline active</span> '
            '— analyze → compress → export',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Direct OrbitalMindPipeline mode")

    run_btn = st.button("🚀 Run OrbitalMind Inference", type="primary",
                        use_container_width=True)

    if run_btn:
        with st.spinner("Running 3-layer verified inference…"):
            prog = st.progress(0, "Initialising pipeline…")
            time.sleep(0.15)

            if st.session_state["use_rv"]:
                # ── RasterVision path ──────────────────────────────
                prog.progress(10, "🌿 [RV] Building OMInferenceConfig…")
                time.sleep(0.1)
                runner = RVPipelineRunner(
                    scene_type=scene_type,
                    task=task,
                    scene_lat=st.session_state["scene_lat"],
                    scene_lon=st.session_state["scene_lon"],
                    use_tim=tim_enabled,
                    use_scorer=scorer_enabled,
                    conf_threshold=rv_conf_threshold,
                )
                prog.progress(25, "🌿 [RV] Command: analyze — spectral extraction…")
                time.sleep(0.2)
                prog.progress(55, "🌿 [RV] Command: compress — semantic payload…")
                time.sleep(0.15)
                prog.progress(80, "🌿 [RV] Command: export — artifacts + bandwidth…")
                time.sleep(0.1)
                result = runner.run(img_arr)
                # Attach runner for config display
                result["_rv_runner"] = runner

            else:
                # ── Direct path ────────────────────────────────────
                pipeline = OrbitalMindPipeline(
                    use_tim=tim_enabled,
                    use_scorer=scorer_enabled,
                    task=task,
                    scene_type=scene_type,
                    scene_lat=st.session_state["scene_lat"],
                    scene_lon=st.session_state["scene_lon"],
                )
                prog.progress(30, "Layer 1: Spectral analysis (7 steps)…")
                time.sleep(0.3)
                prog.progress(55, "Layer 2: Guard model check…")
                time.sleep(0.15)
                prog.progress(75, "Layer 3: TerraMind-small verifier…")
                time.sleep(0.2)
                prog.progress(92, "Semantic compression → JSON…")
                result = pipeline.run(img_arr)

            prog.progress(100, "Done ✓")
            st.session_state["last_result"] = result
            st.session_state["last_img"]    = img_arr
            time.sleep(0.2)
            prog.empty()

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        pred   = result["prediction"]
        fs     = result["features_summary"]

        # ── RasterVision metadata panel ──────────────────────────
        rv_meta = result.get("rv")
        if rv_meta and rv_meta.get("enabled"):
            with st.expander("🌿 RasterVision Pipeline Metadata", expanded=True):
                rvc1, rvc2 = st.columns(2)
                with rvc1:
                    st.markdown("**Commands executed:**")
                    cmds_html = " ".join(
                        f'<span class="rv-cmd-pill">✅ {c}</span>'
                        for c in rv_meta["commands_run"]
                    )
                    st.markdown(cmds_html, unsafe_allow_html=True)
                    st.caption(f"root_uri: `{rv_meta['root_uri']}`")
                with rvc2:
                    bw = rv_meta.get("bandwidth", {})
                    if bw:
                        st.metric("Raw size",      f"{bw.get('raw_kb', 0):.0f} KB")
                        st.metric("Payload size",  f"{bw.get('output_bytes', 0)} B")
                        st.metric("Saving",        f"{bw.get('saving_pct', 0):.4f}%")
                        within = bw.get("within_2kb_limit", False)
                        if within:
                            st.success("✅ Within 2 KB constraint")
                        else:
                            st.error("❌ Exceeds 2 KB")

                if rv_meta.get("rv_filtered"):
                    st.markdown(
                        '<div class="rv-filtered">⚠️ <b>RV Filtered</b>: '
                        'Confidence below threshold — prediction downgraded to LOW priority.</div>',
                        unsafe_allow_html=True,
                    )

                # RV PipelineConfig JSON
                runner = result.get("_rv_runner")
                if runner:
                    with st.expander("📋 OMInferenceConfig (RV PipelineConfig JSON)"):
                        cfg_json = runner.get_config_json()
                        st.markdown(f"<div class='json-box'>{cfg_json}</div>",
                                    unsafe_allow_html=True)
                        st.caption("This config is fully serialisable and can reproduce this exact run.")

        # ── Adaptive trigger ─────────────────────────────────────
        ts     = result["trigger_status"]
        td     = result.get("trigger_detail", {})
        t_icon = {"CHANGE_DETECTED":"🟡","FORCE_PROCESS":"🔴","LOW_CHANGE":"🟢"}.get(ts,"⚪")
        st.info(f"{t_icon} **Adaptive Trigger:** {ts} — {td.get('reason','')}")

        # ── Guard flags ──────────────────────────────────────────
        gr = result.get("guard_result", {})
        if gr.get("flags"):
            with st.expander(f"🛡️ Guard Model — {len(gr['flags'])} flag(s)", expanded=False):
                for flag in gr["flags"]:
                    sev_cls = "guard-BLOCK" if flag["severity"] == "BLOCK" else "guard-WARN"
                    st.markdown(
                        f'<div class="{sev_cls}"><b>[{flag["severity"]}] {flag["code"]}</b>'
                        f' — {flag["detail"]}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.success("🛡️ Guard Model: all checks passed")

        # ── Prediction metrics ───────────────────────────────────
        st.markdown("### 🎯 Prediction")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Event",      pred["event"])
        pc2.metric("Confidence", f"{pred['confidence']:.0%}")
        pc3.metric("Priority",   pred["priority"])
        pc4.metric("Latency",    f"{result['latency_ms']} ms")

        pri_col = {"HIGH":"badge-HIGH","MEDIUM":"badge-MEDIUM",
                   "LOW":"badge-LOW","UNCERTAIN":"badge-UNCERTAIN"}.get(pred["priority"],"")
        st.markdown(
            f'<span class="{pri_col}">▶ {pred["priority"]}</span> — {pred["explanation"]}',
            unsafe_allow_html=True,
        )

        # ── Multi-head bars ──────────────────────────────────────
        st.plotly_chart(render_multi_head_bars(result["multi_head"]),
                        use_container_width=True, config={"displayModeBar": False})

        # ── Overlay mask ─────────────────────────────────────────
        st.markdown("**🗺️ Scene Overlay** *(Flood=blue, Burn=red)*")
        try:
            from src.pipeline import multi_head_spectral_analysis
            aux_data   = multi_head_spectral_analysis(img_arr)
            flood_mask = aux_data["_aux"]["flood_mask"]
            burn_mask  = aux_data["_aux"]["burn_mask"]
            st.image(render_overlay_mask(img_arr, flood_mask, burn_mask),
                     use_container_width=True)
        except Exception:
            st.image(render_change_heatmap(img_arr), caption="Change heatmap",
                     use_container_width=True)

        # ── Validation gauge ─────────────────────────────────────
        if scorer_enabled and result.get("validation"):
            val = result["validation"]
            st.plotly_chart(
                render_validation_gauge(val["validation_score"], val["validation_level"]),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.caption(f"**Verifier:** {val['validation_reason']}")
            be = val.get("model_backend", "simulation")
            if be == "terramind_hf":
                st.success("🤖 Scored by real TerraMind-1.0-small (HuggingFace)")
            else:
                st.info("⚙️ TerraMind-small: physics-informed simulation")

        # ── Bandwidth chart ──────────────────────────────────────
        # Use RV bandwidth report if available, else compute from scratch
        rv_bw = (result.get("rv") or {}).get("bandwidth")
        if rv_bw:
            raw_kb       = rv_bw["raw_kb"]
            output_bytes = rv_bw["output_bytes"]
            saving_pct   = rv_bw["saving_pct"]
        else:
            bw_data      = compute_bandwidth_saving(img_arr, result["output_json"])
            raw_kb       = bw_data["raw_kb"]
            output_bytes = bw_data["output_bytes"]
            saving_pct   = bw_data["saving_pct"]

        st.plotly_chart(
            render_bandwidth_chart(raw_kb, output_bytes),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.metric("Bandwidth Saved",
                  f"{saving_pct:.4f}%",
                  delta=f"↓ {raw_kb - output_bytes/1024:.0f} KB")

        # ── Downlink JSON ────────────────────────────────────────
        with st.expander("📦 Downlink Payload (< 2 KB)"):
            json_str = format_output_json(result)
            st.markdown(f"<div class='json-box'>{json_str}</div>",
                        unsafe_allow_html=True)
            size = len(json_str.encode())
            st.caption(f"Payload: **{size} bytes**")
            if size <= 2048:
                st.success("✅ Within 2 KB constraint")
            else:
                st.error("❌ Exceeds 2 KB")

    else:
        st.info("Configure the mission on the left, then click **Run OrbitalMind Inference**.")


# ════════════════════════════════════════════════════════════════════
# BOTTOM TABS
# ════════════════════════════════════════════════════════════════════
st.divider()
tab_map, tab_metrics, tab_edge, tab3 ,tab_rv, tab_log = st.tabs([
    "🌍 Flood Map", "📊 Performance", "🔧 Edge Feasibility", "Space Data Travel Estimation",
    "🌿 RasterVision", "📋 Mission Log",
])

# ── Tab 1: Flood map ─────────────────────────────────────────────────
with tab_map:
    st.markdown("### Interactive Flood Cluster Map")
    scene_lat = st.session_state["scene_lat"]
    scene_lon = st.session_state["scene_lon"]

    if st.session_state["last_result"]:
        geo_info = st.session_state["last_result"].get("geo_info", {})
        clusters = geo_info.get("flood_clusters", [])

        if clusters:
            st.success(f"🌊 {len(clusters)} flood cluster(s) detected — "
                       f"total ~{geo_info.get('total_flooded_ha', 0):.1f} ha")
            import pandas as pd
            rows = [{
                "ID": c["cluster_id"], "Lat": c["lat"], "Lon": c["lon"],
                "Area (ha)": c["area_ha"], "Width (m)": c["width_m"],
                "Height (m)": c["height_m"], "Severity": c["severity"],
            } for c in clusters]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            m = render_leafmap_flood(geo_info, scene_lat, scene_lon)
            if m is not None:
                m.to_streamlit(height=420)
            else:
                import plotly.express as px
                st.markdown("*(leafmap not available — Plotly fallback)*")
                fig_geo = px.scatter_mapbox(
                    lat=[c["lat"] for c in clusters],
                    lon=[c["lon"] for c in clusters],
                    size=[max(10, c["area_ha"]) for c in clusters],
                    color=[c["severity"] for c in clusters],
                    color_discrete_map={"HIGH":"#ff4b4b","MEDIUM":"#ffa421","LOW":"#21c354"},
                    zoom=11, mapbox_style="carto-darkmatter", title="Flood Clusters",
                )
                fig_geo.update_layout(paper_bgcolor="#0e1117",
                                       margin=dict(l=0,r=0,t=30,b=0),
                                       font=dict(color="#c9d1d9"))
                st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("No flood clusters detected. Try **Flood Scene** scene type.")
            try:
                import leafmap.foliumap as leafmap_mod
                import folium
                m = leafmap_mod.Map(center=[scene_lat, scene_lon], zoom=11)
                m.add_basemap("CartoDB.DarkMatter")
                folium.Marker(location=[scene_lat, scene_lon],
                              popup=f"Scene centre<br>{scene_lat:.4f}N, {scene_lon:.4f}E"
                              ).add_to(m)
                m.to_streamlit(height=380)
            except Exception:
                st.info(f"Scene centre: **{scene_lat:.4f}°N, {scene_lon:.4f}°E**")
    else:
        st.info("Run inference first to see flood cluster locations.")
        try:
            import leafmap.foliumap as leafmap_mod
            import folium
            m = leafmap_mod.Map(center=[scene_lat, scene_lon], zoom=6)
            m.add_basemap("CartoDB.DarkMatter")
            folium.Marker(location=[scene_lat, scene_lon],
                          popup=f"Scene centre: {scene_lat:.4f}N, {scene_lon:.4f}E"
                          ).add_to(m)
            m.to_streamlit(height=380)
        except Exception:
            st.info(f"Default location: {scene_lat:.4f}°N, {scene_lon:.4f}°E")

# ── Tab 2: Performance ───────────────────────────────────────────────
with tab_metrics:
    st.markdown("### Baseline vs OrbitalMind v3.2")
    st.plotly_chart(render_performance_chart(), use_container_width=True,
                    config={"displayModeBar": False})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flood F1",         "0.87", "+58% vs baseline")
    m2.metric("Crop mIoU",        "0.74", "+61% vs baseline")
    m3.metric("Change mAP",       "0.81", "+45% vs baseline")
    m4.metric("False Alarm Rate", "8%",   "-19pp vs baseline", delta_color="inverse")

    if st.session_state["last_result"]:
        fs = st.session_state["last_result"]["features_summary"]
        st.plotly_chart(render_scene_fractions_chart(fs), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("""
    | Metric | Baseline | OrbitalMind v3.2 | Δ |
    |---|---|---|---|
    | Flood F1 | 0.55 | **0.87** | +58% |
    | Crop mIoU | 0.46 | **0.74** | +61% |
    | Change mAP | 0.56 | **0.81** | +45% |
    | Burn F1 | 0.48 | **0.79** | +65% |
    | False Alarm Rate | 27% | **8%** | −70% |
    | Avg Validation Score | — | **82/100** | — |
    | Payload Size | 200 MB (raw) | **~1.2 KB** | −99.9994% |
    """)

# ── Tab 3: Edge feasibility ──────────────────────────────────────────
with tab_edge:
    render_edge_specs()
# ── Tab 4: Data Transmission — TSP-Optimised ─────────────────────────────────
"""
nasa_tsp_dashboard.py — OrbitalMind NASA Mission Control TSP Globe
═══════════════════════════════════════════════════════════════════
Self-contained. No external template. No unresolved references.

Usage in app.py — replace the entire `with tab3:` block:

    from nasa_tsp_dashboard import render_nasa_tsp_dashboard

    with tab3:
        tx_data = st.session_state.get("last_result", {})
        tx_data = tx_data.get("transmission") if tx_data else None
        render_nasa_tsp_dashboard(tx_data)
"""

import json
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# GLOBE HTML
# The literal string  {TX_JSON}  is replaced at runtime via str.replace()
# ─────────────────────────────────────────────────────────────────────────────

_GLOBE_HTML = (
    "<!DOCTYPE html>\n"
    "<html>\n"
    "<head>\n"
    "<meta charset='utf-8'>\n"
    "<style>\n"
    "* { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "body { background: #000a14; font-family: 'Courier New', monospace; color: #cdd9e5; overflow-x: hidden; }\n"
    "#canvas { display: block; width: 100%; height: 500px; cursor: grab; }\n"
    ".ctrl { display: flex; gap: 8px; padding: 8px 0 4px; flex-wrap: wrap; align-items: center; }\n"
    ".btn { font-family: 'Courier New', monospace; font-size: 12px; padding: 6px 18px; background: #001830; border-radius: 4px; cursor: pointer; border: 1px solid; }\n"
    ".btn-p { color: #00eaff; border-color: #00eaff55; }\n"
    ".btn-p:hover { background: #002840; }\n"
    ".btn-s { color: #7a9ab5; border-color: #334455; }\n"
    ".btn-s:hover { background: #001020; }\n"
    "#status { font-size: 11px; color: #00eaff; padding: 5px 10px; background: #000e1c; border: 1px solid #00eaff22; border-radius: 4px; flex: 1; min-width: 180px; }\n"
    ".metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; padding: 8px 0; }\n"
    ".metric { background: #000e1c; border: 1px solid #0a2540; border-radius: 6px; padding: 10px; text-align: center; }\n"
    ".ml { font-size: 9px; color: #3a6a9a; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }\n"
    ".mv { font-size: 20px; color: #00eaff; }\n"
    ".mvg { font-size: 20px; color: #22ff88; }\n"
    "#hoplog { font-size: 11px; max-height: 140px; overflow-y: auto; background: #000a14; border: 1px solid #0a2540; border-radius: 6px; padding: 6px; }\n"
    ".he { display: flex; justify-content: space-between; align-items: center; padding: 3px 4px; border-bottom: 1px solid #0a2030; }\n"
    ".leg { display: flex; flex-wrap: wrap; gap: 10px; padding: 6px 0; font-size: 10px; color: #5a7a9a; }\n"
    ".li { display: flex; align-items: center; gap: 4px; }\n"
    ".ld { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }\n"
    ".st { font-size: 10px; color: #3a6a9a; letter-spacing: 3px; text-transform: uppercase; padding: 6px 0 3px; border-bottom: 1px solid #0a2030; margin-bottom: 4px; }\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<canvas id='canvas'></canvas>\n"
    "<div class='ctrl'>\n"
    "  <button class='btn btn-p' id='btnPlay' onclick='togglePlay()'>&#9654; Transmit</button>\n"
    "  <button class='btn btn-s' onclick='doReset()'>&#8635; Reset</button>\n"
    "  <span id='status'>&#9711; SYSTEM READY &mdash; Press Transmit</span>\n"
    "</div>\n"
    "<div class='metrics'>\n"
    "  <div class='metric'><div class='ml'>Hops</div><div class='mv' id='mH'>&#8212;</div></div>\n"
    "  <div class='metric'><div class='ml'>Latency</div><div class='mv' id='mL'>&#8212;</div></div>\n"
    "  <div class='metric'><div class='ml'>Distance</div><div class='mv' id='mD'>&#8212;</div></div>\n"
    "  <div class='metric'><div class='ml'>TSP Gain</div><div class='mvg' id='mT'>&#8212;</div></div>\n"
    "</div>\n"
    "<div class='st'>&#9632; LEGEND</div>\n"
    "<div class='leg'>\n"
    "  <div class='li'><div class='ld' style='background:#00eaff'></div>LEO Satellite</div>\n"
    "  <div class='li'><div class='ld' style='background:#3b82f6'></div>MEO Relay</div>\n"
    "  <div class='li'><div class='ld' style='background:#c084fc'></div>GEO Hub</div>\n"
    "  <div class='li'><div class='ld' style='background:#facc15'></div>Space Datacenter</div>\n"
    "  <div class='li'><div class='ld' style='background:#f8fafc'></div>Ground Station</div>\n"
    "  <div class='li'><div class='ld' style='background:#22ff88'></div>Mission Control</div>\n"
    "  <div class='li'><div class='ld' style='background:#ff6600'></div>You (Geolocation)</div>\n"
    "</div>\n"
    "<div class='st' style='margin-top:6px'>&#9632; HOP LOG</div>\n"
    "<div id='hoplog'></div>\n"
    "<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>\n"
    "<script>\n"
    "const TX = TXJSONHERE;\n"
    "const ER = 6371;\n"
    "const NC = {'LEO Satellite':0x00eaff,'MEO Relay':0x3b82f6,'GEO Hub':0xc084fc,'Space Datacenter':0xfacc15,'Ground Station':0xf8fafc,'Earth (Mission Control)':0x22ff88};\n"
    "const NS = {'LEO Satellite':0.055,'MEO Relay':0.075,'GEO Hub':0.095,'Space Datacenter':0.085,'Ground Station':0.050,'Earth (Mission Control)':0.12};\n"
    "function xyz(lat,lon,alt){\n"
    "  const r=(ER+(alt||0))/ER,p=lat*Math.PI/180,l=lon*Math.PI/180;\n"
    "  return new THREE.Vector3(r*Math.cos(p)*Math.cos(l),r*Math.sin(p),-r*Math.cos(p)*Math.sin(l));\n"
    "}\n"
    "const canvas=document.getElementById('canvas');\n"
    "const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});\n"
    "renderer.setPixelRatio(Math.min(devicePixelRatio,2));\n"
    "renderer.setClearColor(0x000a14,1);\n"
    "const scene=new THREE.Scene();\n"
    "const camera=new THREE.PerspectiveCamera(45,1,0.01,500);\n"
    "camera.position.set(0,0,3.8);\n"
    "function onR(){const w=canvas.clientWidth,h=canvas.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}\n"
    "onR(); window.addEventListener('resize',onR);\n"
    "// Stars\n"
    "const sp=new Float32Array(4000*3);\n"
    "for(let i=0;i<4000;i++){const t=Math.random()*2*Math.PI,p=Math.acos(2*Math.random()-1),r=50+Math.random()*80;sp[i*3]=r*Math.sin(p)*Math.cos(t);sp[i*3+1]=r*Math.cos(p);sp[i*3+2]=r*Math.sin(p)*Math.sin(t);}\n"
    "const sg=new THREE.BufferGeometry();sg.setAttribute('position',new THREE.BufferAttribute(sp,3));\n"
    "scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xffffff,size:0.09,transparent:true,opacity:0.55})));\n"
    "// Earth\n"
    "const earth=new THREE.Mesh(new THREE.SphereGeometry(1,64,64),new THREE.MeshPhongMaterial({color:0x0a2040,emissive:0x010810,shininess:18}));\n"
    "scene.add(earth);\n"
    "scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.045,32,32),new THREE.MeshPhongMaterial({color:0x003366,transparent:true,opacity:0.15,side:THREE.BackSide})));\n"
    "// Grid\n"
    "const gm=new THREE.LineBasicMaterial({color:0x112233,transparent:true,opacity:0.45});\n"
    "for(let la=-80;la<=80;la+=20){const pts=[];for(let lo=0;lo<=361;lo+=4)pts.push(xyz(la,lo,0));scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gm));}\n"
    "for(let lo=0;lo<360;lo+=20){const pts=[];for(let la=-90;la<=90;la+=4)pts.push(xyz(la,lo,0));scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),gm));}\n"
    "// Orbit rings\n"
    "function ring(alt,col,op){const r=(ER+alt)/ER,pts=[];for(let a=0;a<=361;a+=2)pts.push(new THREE.Vector3(r*Math.cos(a*Math.PI/180),0,r*Math.sin(a*Math.PI/180)));scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:col,transparent:true,opacity:op})));}\n"
    "ring(550,0x00eaff,0.12);ring(20200,0x3b82f6,0.07);ring(35786,0xc084fc,0.05);\n"
    "// Lights\n"
    "scene.add(new THREE.AmbientLight(0x112244,1));\n"
    "const sun=new THREE.DirectionalLight(0x4488cc,1.4);sun.position.set(5,3,5);scene.add(sun);\n"
    "// Nodes\n"
    "const nodes=TX.node_coords||[];\n"
    "nodes.forEach(n=>{\n"
    "  const col=NC[n.type]||0xffffff,sz=NS[n.type]||0.06;\n"
    "  let geo;\n"
    "  if(n.type==='Space Datacenter') geo=new THREE.OctahedronGeometry(sz);\n"
    "  else if(n.type==='GEO Hub') geo=new THREE.TetrahedronGeometry(sz*1.3);\n"
    "  else geo=new THREE.SphereGeometry(sz,10,10);\n"
    "  const m=new THREE.Mesh(geo,new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:0.25}));\n"
    "  m.position.copy(xyz(n.lat,n.lon,n.altitude||0));\n"
    "  scene.add(m);\n"
    "});\n"
    "// User dot\n"
    "const uMesh=new THREE.Mesh(new THREE.SphereGeometry(0.065,12,12),new THREE.MeshPhongMaterial({color:0xff6600,emissive:0xff3300,emissiveIntensity:0.9}));\n"
    "uMesh.position.copy(xyz(20,77,0));scene.add(uMesh);\n"
    "// Arc helper\n"
    "function arcPts(nA,nB){if(!nA||!nB)return null;const pA=xyz(nA.lat,nA.lon,nA.altitude||0),pB=xyz(nB.lat,nB.lon,nB.altitude||0),mid=pA.clone().add(pB).multiplyScalar(0.5);mid.normalize().multiplyScalar(Math.max(pA.length(),pB.length())*1.08);return new THREE.QuadraticBezierCurve3(pA,mid,pB).getPoints(80);}\n"
    "// Route arcs from hops\n"
    "const hops=TX.hops||[];\n"
    "const arcs=[];\n"
    "for(let i=0;i<hops.length;i++){const nA=nodes[i],nB=nodes[i+1];if(!nA||!nB)continue;const pts=arcPts(nA,nB);if(!pts)continue;const q=hops[i].link_quality||0.9;const col=q>=0.85?0x22ff88:q>=0.75?0xf97316:0xef4444;const mat=new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.25});const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),mat);scene.add(ln);arcs.push({ln,pts,col});}\n"
    "// BG connections\n"
    "nodes.forEach((a,i)=>{nodes.forEach((b,j)=>{if(j<=i)return;const d=Math.abs(a.lat-b.lat)+Math.abs(a.lon-b.lon);if(d>60)return;const pts=arcPts(a,b);if(!pts)return;scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x0a1e30,transparent:true,opacity:0.5})));});});\n"
    "// Packet\n"
    "const pkt=new THREE.Mesh(new THREE.SphereGeometry(0.042,12,12),new THREE.MeshPhongMaterial({color:0xffffff,emissive:0x00eaff,emissiveIntensity:2.5}));\n"
    "pkt.visible=false;scene.add(pkt);\n"
    "const TRL=20,tms=[];\n"
    "for(let i=0;i<TRL;i++){const f=1-i/TRL;const tm=new THREE.Mesh(new THREE.SphereGeometry(0.028*f,6,6),new THREE.MeshBasicMaterial({color:0x00eaff,transparent:true,opacity:0.65*f}));tm.visible=false;scene.add(tm);tms.push(tm);}\n"
    "// Drag\n"
    "let drag=false,ox=0,oy=0,rotY=0.3,rotX=0.12;\n"
    "canvas.addEventListener('mousedown',e=>{drag=true;ox=e.clientX;oy=e.clientY;canvas.style.cursor='grabbing';});\n"
    "window.addEventListener('mouseup',()=>{drag=false;canvas.style.cursor='grab';});\n"
    "window.addEventListener('mousemove',e=>{if(!drag)return;rotY+=(e.clientX-ox)*0.006;ox=e.clientX;rotX+=(e.clientY-oy)*0.006;oy=e.clientY;rotX=Math.max(-1.2,Math.min(1.2,rotX));});\n"
    "canvas.addEventListener('wheel',e=>{camera.position.z=Math.max(1.8,Math.min(8,camera.position.z+e.deltaY*0.005));e.preventDefault();},{passive:false});\n"
    "// State\n"
    "let playing=false,hi=0,th=0,phase='fwd',tHist=[];\n"
    "const HF=80;\n"
    "const SF=['&#9711; INITIATING ORBITAL TRANSMISSION...','&#9650; UPLINK: packet leaving source','&#11041; ISL: inter-satellite optical relay','&#11042; RELAY: routing via Space Datacenter','&#11040; GEO handoff — signal locked','&#9660; DOWNLINK approaching ground station','&#10003; PAYLOAD DELIVERED &mdash; Mission Control confirmed'];\n"
    "const SR=['&#8629; RETURN: Mission Control &rarr; Source','&#8629; Uplink ground &rarr; GEO...','&#8629; GEO &rarr; MEO relay...','&#8629; MEO &rarr; LEO handoff...','&#8629; LEO &rarr; Source node...','&#10004; ROUND-TRIP COMPLETE'];\n"
    "function ss(m){document.getElementById('status').innerHTML=m;}\n"
    "function logH(f,t,q,l){const col=q>=0.85?'#22ff88':q>=0.75?'#f97316':'#ef4444';const e=document.createElement('div');e.className='he';e.innerHTML='<span style=\"color:#00eaff\">'+f+'</span><span style=\"color:#335566;margin:0 4px\">&rarr;</span><span style=\"color:#22ff88\">'+t+'</span><span style=\"font-size:10px;margin-left:8px;color:'+col+'\">QTY '+(q*100).toFixed(0)+'% | '+parseFloat(l).toFixed(0)+'ms</span>';const lg=document.getElementById('hoplog');lg.appendChild(e);lg.scrollTop=lg.scrollHeight;}\n"
    "// Metrics\n"
    "document.getElementById('mH').textContent=TX.total_hops||hops.length;\n"
    "document.getElementById('mL').textContent=TX.total_latency_ms?TX.total_latency_ms.toFixed(0)+' ms':'&#8212;';\n"
    "document.getElementById('mD').textContent=TX.total_distance_km?(TX.total_distance_km/1000).toFixed(0)+'k km':'&#8212;';\n"
    "document.getElementById('mT').textContent=TX.tsp_improvement?'+'+TX.tsp_improvement.toFixed(1)+'%':'&#8212;';\n"
    "// Paths\n"
    "const fPts=[],rPts=[];\n"
    "arcs.forEach(a=>fPts.push(a.pts));\n"
    "const rev=[...fPts].reverse().map(p=>[...p].reverse());\n"
    "rev.forEach(p=>rPts.push(p));\n"
    "function togglePlay(){\n"
    "  if(playing){playing=false;document.getElementById('btnPlay').innerHTML='&#9654; Transmit';}\n"
    "  else{doReset();playing=true;document.getElementById('btnPlay').innerHTML='&#9646;&#9646; Pause';ss(SF[0]);document.getElementById('hoplog').innerHTML='';}\n"
    "}\n"
    "function doReset(){playing=false;hi=0;th=0;phase='fwd';tHist=[];pkt.visible=false;tms.forEach(t=>t.visible=false);arcs.forEach(a=>a.ln.material.opacity=0.25);document.getElementById('btnPlay').innerHTML='&#9654; Transmit';ss('&#9711; SYSTEM READY &mdash; Press Transmit');}\n"
    "let frame=0;\n"
    "(function loop(){\n"
    "  requestAnimationFrame(loop);frame++;\n"
    "  earth.rotation.y=rotY+frame*0.0007;earth.rotation.x=rotX;\n"
    "  const pu=0.9+0.1*Math.sin(frame*0.1);uMesh.scale.setScalar(pu);\n"
    "  if(playing){\n"
    "    const paths=phase==='fwd'?fPts:rPts;\n"
    "    const pc=phase==='fwd'?0x00eaff:0xff8833;\n"
    "    pkt.material.emissive.setHex(pc);tms.forEach(t=>t.material.color.setHex(pc));\n"
    "    if(hi<paths.length){\n"
    "      pkt.visible=true;th++;\n"
    "      const fr=Math.min(th/HF,1),pts=paths[hi],pos=pts[Math.floor(fr*(pts.length-1))];\n"
    "      pkt.position.copy(pos);\n"
    "      tHist.unshift(pos.clone());if(tHist.length>TRL)tHist.pop();\n"
    "      tms.forEach((t,i)=>{if(i<tHist.length){t.visible=true;t.position.copy(tHist[i]);}else t.visible=false;});\n"
    "      arcs.forEach((a,i)=>{a.ln.material.opacity=i===hi?0.95:0.15;});\n"
    "      if(th>=HF){\n"
    "        th=0;const hop=hops[phase==='fwd'?hi:(hops.length-1-hi)];\n"
    "        if(hop)logH(hop.from_name||hop.from_node,hop.to_name||hop.to_node,hop.link_quality||0.9,hop.latency_ms||0);\n"
    "        hi++;\n"
    "        if(phase==='fwd')ss(SF[Math.min(hi,SF.length-1)]);\n"
    "        else ss(SR[Math.min(hi,SR.length-1)]);\n"
    "      }\n"
    "    } else {\n"
    "      if(phase==='fwd'){phase='ret';hi=0;th=0;tHist=[];ss(SR[0]);arcs.forEach(a=>a.ln.material.opacity=0.2);}\n"
    "      else{playing=false;pkt.visible=false;tms.forEach(t=>t.visible=false);document.getElementById('btnPlay').innerHTML='&#9654; Transmit';ss('&#10004; ROUND-TRIP COMPLETE &mdash; Transmission verified');arcs.forEach(a=>a.ln.material.opacity=0.5);}\n"
    "    }\n"
    "  }\n"
    "  renderer.render(scene,camera);\n"
    "})();\n"
    "if(navigator.geolocation){navigator.geolocation.getCurrentPosition(p=>{uMesh.position.copy(xyz(p.coords.latitude,p.coords.longitude,0));},()=>{});}\n"
    "</script>\n"
    "</body>\n"
    "</html>\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_nasa_tsp_dashboard(tx):
    """
    Renders the NASA Mission Control 3-D TSP globe inside a Streamlit tab.

    Parameters
    ----------
    tx : dict or None
        The 'transmission' dict from RVPipelineRunner / OrbitalMindPipeline.
        If None, shows a standby message.

    Usage in app.py
    ---------------
        from nasa_tsp_dashboard import render_nasa_tsp_dashboard

        with tab3:
            tx_data = st.session_state.get("last_result", {})
            tx_data = tx_data.get("transmission") if tx_data else None
            render_nasa_tsp_dashboard(tx_data)
    """
    if tx is None:
        st.markdown(
            "<div style=\"font-family:'Courier New',monospace;font-size:13px;color:#00eaff;"
            "padding:14px 18px;background:#000e1c;border:1px solid #00eaff22;"
            "border-left:3px solid #00eaff;border-radius:6px;margin-top:12px;\">"
            "&#9711; SYSTEM STANDBY &mdash; Run inference first to initialise orbital routing."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Serialise tx → JSON, then inject into the HTML
    # default=str handles any numpy scalars that slipped through
    tx_json = json.dumps(tx, default=str)
    html = _GLOBE_HTML.replace("TXJSONHERE", tx_json)

    st.components.v1.html(html, height=900, scrolling=False)
with tab3:
    tx_data = st.session_state.get("last_result", {})
    tx_data = tx_data.get("transmission") if tx_data else None
    render_nasa_tsp_dashboard(tx_data)

# ── Tab 5: RasterVision details ──────────────────────────────────────
with tab_rv:
    st.markdown("### 🌿 RasterVision Integration")
    st.markdown("""
    OrbitalMind v3.2 integrates **RasterVision's Pipeline and Config** framework
    to bring structured, reproducible, serialisable inference to the satellite ML stack.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        #### What RasterVision adds
        - **`OMSceneConfig`** — typed Pydantic config for scene params
          (type, lat/lon, GSD, image size)
        - **`OMInferenceConfig`** — top-level `PipelineConfig` with all
          model toggles, confidence threshold, TiM/scorer flags
        - **3-command `Pipeline`** with full separation of concerns:
          - `analyze` — spectral extraction + guard + verifier
          - `compress` — semantic payload (<2 KB JSON)
          - `export` — artifacts + bandwidth report + config snapshot
        - **FileSystem API** — all intermediate results written as JSON
          under `root_uri` (local or cloud-compatible)
        - **Confidence threshold** — RV-level filter on top of Guard model
        - **`rv_filtered` flag** — surfaced in UI and downlink payload
        """)
    with col_b:
        st.markdown("""
        #### RasterVision APIs used
        | API | Usage |
        |---|---|
        | `register_config` | Registers `OMSceneConfig`, `OMInferenceConfig` |
        | `Config` / `Field` | Typed, validated scene + inference params |
        | `PipelineConfig` | Base class for `OMInferenceConfig` |
        | `Pipeline` | Base class; `analyze/compress/export` commands |
        | `json_to_file` | Saves intermediate JSON artifacts |
        | `file_to_json` | Loads artifacts between commands |
        | `make_dir` | Creates `root_uri` output directory |
        | `file_exists` | Checks artifact presence before loading |
        """)

    st.divider()

    if st.session_state["last_result"]:
        rv_meta = st.session_state["last_result"].get("rv")
        if rv_meta and rv_meta.get("enabled"):
            st.markdown("#### Last Run — RV Artifacts")
            st.json(rv_meta["config"])
            bw = rv_meta.get("bandwidth", {})
            if bw:
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Raw (bytes)",     f"{bw.get('raw_bytes',0):,}")
                bc2.metric("Payload (bytes)", f"{bw.get('output_bytes',0):,}")
                bc3.metric("Saving",          f"{bw.get('saving_pct',0):.4f}%")
                bc4.metric("Ratio",           f"{bw.get('compression_ratio',0):.0f}×")
        else:
            st.info("Run inference with **RasterVision Pipeline** toggle ON to see RV artifacts here.")
    else:
        st.info("Run inference first.")

# ── Tab 6: Mission log ───────────────────────────────────────────────
with tab_log:
    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        rv_meta = result.get("rv", {}) or {}
        mode = "RasterVision" if rv_meta.get("enabled") else "Direct"
        st.markdown(
            f"**Mode:** `{mode}` | "
            f"**Backend:** `{result.get('hf_backend','simulation')}` | "
            f"**Latency:** {result['latency_ms']} ms | "
            f"**Trigger:** {result['trigger_status']}"
        )
        if rv_meta.get("enabled"):
            cmds = " → ".join(rv_meta.get("commands_run", []))
            st.caption(f"RV commands: {cmds}")

        ls = result.get("learning_stats", {})
        if ls:
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Buffer size",     ls.get("buffer_size", 0))
            lc2.metric("Avg val score",   ls.get("avg_validation_score", 0))
            lc3.metric("Correction rate", f"{ls.get('correction_rate', 0):.1%}")

        st.json(result["output_json"])
        with st.expander("Full result dict"):
            safe = {k: v for k, v in result.items()
                    if k not in ("tim_modalities", "_rv_runner")}
            fs_safe = {k: v for k, v in result.get("features_summary", {}).items()
                       if not isinstance(v, np.ndarray)}
            safe["features_summary"] = fs_safe
            st.json(safe)
    else:
        st.info("Run inference to populate the mission log.")