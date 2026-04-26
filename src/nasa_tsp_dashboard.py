"""
nasa_tsp_dashboard.py — OrbitalMind NASA Mission Control TSP Visualizer
════════════════════════════════════════════════════════════════════════
Drop-in replacement for the TSP tab in app.py.
Consumes only the `tx` dict produced by TransmissionSimulator.simulate().

Usage in app.py (inside tab3):
    from nasa_tsp_dashboard import render_nasa_tsp_dashboard
    with tab3:
        render_nasa_tsp_dashboard(tx)
"""

import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────
_SPACE_BG   = "#020810"
_CARD_BG    = "#050d1a"
_BORDER     = "#0d2340"

_NODE_COLORS = {
    "LEO Satellite":           "#00f5ff",   # cyan
    "MEO Relay":               "#3b82f6",   # blue
    "GEO Hub":                 "#a855f7",   # purple
    "Space Datacenter":        "#facc15",   # yellow
    "Ground Station":          "#f8fafc",   # white
    "Earth (Mission Control)": "#22c55e",   # green
}
_NODE_SIZES = {
    "LEO Satellite":           14,
    "MEO Relay":               16,
    "GEO Hub":                 20,
    "Space Datacenter":        22,
    "Ground Station":          18,
    "Earth (Mission Control)": 28,
}
_NODE_SYMBOLS = {
    "LEO Satellite":           "circle",
    "MEO Relay":               "diamond",
    "GEO Hub":                 "square",
    "Space Datacenter":        "star",
    "Ground Station":          "triangle-up",
    "Earth (Mission Control)": "circle",
}


def _link_color(quality: float) -> str:
    if quality >= 0.85:
        return "#22c55e"   # green
    elif quality >= 0.75:
        return "#f97316"   # orange
    else:
        return "#ef4444"   # red


def _link_width(quality: float) -> int:
    if quality >= 0.85:
        return 4
    elif quality >= 0.75:
        return 3
    else:
        return 2


# ─────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Share+Tech+Mono&display=swap');

.nasa-title {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 6px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #00f5ff, #3b82f6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 4px;
}
.nasa-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #3b82f6;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 16px;
}
.mission-metric {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 100%);
    border: 1px solid #0d2340;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.mission-metric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5ff, transparent);
}
.mission-metric .label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #4b7ab5;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.mission-metric .value {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #00f5ff;
    text-shadow: 0 0 20px rgba(0,245,255,0.5);
}
.mission-metric .unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #3b82f6;
    margin-top: 2px;
}
.hop-card {
    background: #050d1a;
    border: 1px solid #0d2340;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'Share Tech Mono', monospace;
    transition: all 0.3s;
}
.hop-card.active {
    border-color: #00f5ff;
    background: #071525;
    box-shadow: 0 0 16px rgba(0,245,255,0.2), inset 0 0 16px rgba(0,245,255,0.05);
}
.hop-card .hop-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.hop-card .hop-idx {
    font-size: 10px;
    color: #4b7ab5;
    letter-spacing: 2px;
}
.hop-card .hop-route {
    font-size: 12px;
    color: #e2e8f0;
}
.hop-card .hop-badge {
    font-size: 10px;
    padding: 1px 8px;
    border-radius: 12px;
    border: 1px solid;
}
.quality-bar-wrap { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
.quality-label { font-size: 10px; color: #4b7ab5; width: 76px; }
.quality-bar-bg { flex: 1; background: #0d2340; border-radius: 3px; height: 6px; }
.quality-bar-fill { height: 6px; border-radius: 3px; }
.quality-val { font-size: 10px; width: 40px; text-align: right; }

.status-banner {
    background: linear-gradient(135deg, #050d1a, #071525);
    border: 1px solid #0d2340;
    border-left: 3px solid #00f5ff;
    border-radius: 6px;
    padding: 10px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #00f5ff;
    letter-spacing: 1px;
    margin: 8px 0;
    box-shadow: 0 0 20px rgba(0,245,255,0.1);
}
.route-pill {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    padding: 3px 12px;
    border-radius: 12px;
    border: 1px solid;
    margin: 2px;
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #3b82f6;
    text-transform: uppercase;
    margin: 16px 0 8px;
    border-bottom: 1px solid #0d2340;
    padding-bottom: 6px;
}
.scanline-overlay {
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,245,255,0.01) 2px,
        rgba(0,245,255,0.01) 4px
    );
    pointer-events: none;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD FUNCTION
# ─────────────────────────────────────────────────────────────────────

def render_nasa_tsp_dashboard(tx: dict):
    """
    Renders the full NASA Mission Control–style TSP dashboard.
    Only consumes `tx` dict from TransmissionSimulator.simulate().
    Call inside the TSP tab in app.py.
    """
    if tx is None:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.markdown(
            '<div class="status-banner">⚡ SYSTEM STANDBY — Run inference to initialise orbital routing.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(_CSS, unsafe_allow_html=True)

    node_coords  = tx["node_coords"]
    hops         = tx["hops"]
    route_names  = tx["route_names"]
    route_types  = tx["route_types"]

    # ── TITLE ────────────────────────────────────────────────────────
    st.markdown('<div class="nasa-title">⬡ ORBITAL MIND — MISSION CONTROL</div>', unsafe_allow_html=True)
    st.markdown('<div class="nasa-subtitle">TSP-OPTIMISED SPACE DATA RELAY NETWORK</div>', unsafe_allow_html=True)

    # ── STATUS + ANIMATE BUTTON ──────────────────────────────────────
    col_status, col_btn = st.columns([4, 1])
    with col_status:
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<div class="status-banner">🛰️ ROUTE COMPUTED — Press <b>INITIATE</b> to simulate payload transmission.</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_anim = st.button("▶ INITIATE", type="primary", use_container_width=True)

    # ── TOP METRICS ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">■ MISSION TELEMETRY</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    _metric_card(mc1, "TOTAL HOPS",     str(tx["total_hops"]),              "segments")
    _metric_card(mc2, "END-TO-END",     f"{tx['total_latency_ms']:.1f}",    "ms latency")
    _metric_card(mc3, "RANGE",          f"{tx['total_distance_km']:,.0f}",  "km")
    _metric_card(mc4, "BANDWIDTH",      f"{tx['effective_bandwidth']}",     "Mbps")
    _metric_card(mc5, "TSP GAIN",       f"+{tx['tsp_improvement']:.1f}%",   "vs greedy")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP + ROUTE PANEL ─────────────────────────────────────────────
    map_col, route_col = st.columns([3, 1], gap="medium")

    with map_col:
        st.markdown('<div class="section-label">■ GLOBAL RELAY NETWORK — LIVE VIEW</div>', unsafe_allow_html=True)
        map_placeholder = st.empty()
        map_placeholder.plotly_chart(
            _build_static_map(node_coords, hops, active_hop=-1),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with route_col:
        st.markdown('<div class="section-label">■ ROUTE MANIFEST</div>', unsafe_allow_html=True)
        route_placeholder = st.empty()
        _render_route_cards(route_placeholder, hops, active_hop=-1, route_names=route_names, route_types=route_types)

    # ── LINK QUALITY CHART ───────────────────────────────────────────
    st.markdown('<div class="section-label">■ LINK QUALITY PROFILE</div>', unsafe_allow_html=True)
    lq_placeholder = st.empty()
    lq_placeholder.plotly_chart(
        _build_link_quality_chart(hops, active_hop=-1),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── ANIMATION LOOP ───────────────────────────────────────────────
    if run_anim:
        status_messages = [
            "🚀 INITIATING ORBITAL TRANSMISSION SEQUENCE...",
            "📡 UPLINK ESTABLISHED — PACKET HANDSHAKE CONFIRMED",
            "🛰️ ROUTING THROUGH LEO CONSTELLATION...",
            "⚡ INTER-SATELLITE OPTICAL ISL ACTIVE",
            "🔵 MEO RELAY HANDOFF IN PROGRESS...",
            "🟣 GEO HUB ACQUISITION — SIGNAL LOCKED",
            "🟡 ORBITAL DATACENTER — SEMANTIC COMPRESSION VERIFIED",
            "📶 DOWNLINK BEAM FORMING...",
            "🌍 GROUND STATION ACQUISITION",
            "✅ TRANSMISSION COMPLETE — PAYLOAD DELIVERED",
        ]

        for hop_idx, hop in enumerate(hops):
            msg_idx = min(hop_idx, len(status_messages) - 2)
            status_placeholder.markdown(
                f'<div class="status-banner">{status_messages[msg_idx]}</div>',
                unsafe_allow_html=True,
            )

            map_placeholder.plotly_chart(
                _build_static_map(node_coords, hops, active_hop=hop_idx),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            _render_route_cards(route_placeholder, hops, active_hop=hop_idx,
                                route_names=route_names, route_types=route_types)
            lq_placeholder.plotly_chart(
                _build_link_quality_chart(hops, active_hop=hop_idx),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            time.sleep(0.55)

        # Final state
        status_placeholder.markdown(
            '<div class="status-banner">✅ TRANSMISSION COMPLETE — PAYLOAD DELIVERED TO MISSION CONTROL</div>',
            unsafe_allow_html=True,
        )
        map_placeholder.plotly_chart(
            _build_static_map(node_coords, hops, active_hop=len(hops)),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        _render_route_cards(route_placeholder, hops, active_hop=len(hops),
                            route_names=route_names, route_types=route_types)
        lq_placeholder.plotly_chart(
            _build_link_quality_chart(hops, active_hop=len(hops)),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── HOP-BY-HOP LOG ───────────────────────────────────────────────
    st.markdown('<div class="section-label">■ HOP-BY-HOP TRANSMISSION LOG</div>', unsafe_allow_html=True)
    _render_hop_log(hops)


# ─────────────────────────────────────────────────────────────────────
# HELPER: METRIC CARD
# ─────────────────────────────────────────────────────────────────────

def _metric_card(col, label: str, value: str, unit: str):
    col.markdown(
        f"""<div class="mission-metric">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="unit">{unit}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# HELPER: STATIC + ACTIVE MAP
# ─────────────────────────────────────────────────────────────────────

def _build_static_map(node_coords: list, hops: list, active_hop: int) -> go.Figure:
    fig = go.Figure()

    # Draw all route lines (dimmed)
    for i, hop in enumerate(hops):
        if i >= len(node_coords) - 1:
            break
        src = node_coords[i]
        dst = node_coords[i + 1]
        q   = hop["link_quality"]
        color = _link_color(q)
        w     = _link_width(q)
        is_active   = (i == active_hop)
        is_done     = (i < active_hop)

        # Dim future hops
        opacity = 1.0 if (is_active or is_done) else 0.18
        line_color = color if is_done else ("#00f5ff" if is_active else "#1e3a5f")
        line_width = (w + 2) if is_active else w

        fig.add_trace(go.Scattermapbox(
            lat=[src["lat"], dst["lat"]],
            lon=[src["lon"], dst["lon"]],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            opacity=opacity,
            hoverinfo="none",
            showlegend=False,
        ))

    # Glow effect for active hop line
    if 0 <= active_hop < len(hops) and active_hop < len(node_coords) - 1:
        src = node_coords[active_hop]
        dst = node_coords[active_hop + 1]
        for glow_w, glow_opacity in [(12, 0.08), (8, 0.12), (5, 0.25)]:
            fig.add_trace(go.Scattermapbox(
                lat=[src["lat"], dst["lat"]],
                lon=[src["lon"], dst["lon"]],
                mode="lines",
                line=dict(width=glow_w, color="#00f5ff"),
                opacity=glow_opacity,
                hoverinfo="none",
                showlegend=False,
            ))

    # All nodes — dimmed if future
    for i, n in enumerate(node_coords):
        is_done   = i <= active_hop
        is_active = i == active_hop + 1
        color  = _NODE_COLORS.get(n["type"], "#ffffff")
        size   = _NODE_SIZES.get(n["type"],  14)
        symbol = _NODE_SYMBOLS.get(n["type"], "circle")
        opacity = 1.0 if (is_done or is_active) else 0.2
        final_size = (size + 8) if is_active else size

        fig.add_trace(go.Scattermapbox(
            lat=[n["lat"]],
            lon=[n["lon"]],
            mode="markers+text",
            marker=dict(
                size=final_size,
                color=color,
                opacity=opacity,
            ),
            text=[n["name"]],
            textposition="top right",
            textfont=dict(size=9, color=color if (is_done or is_active) else "#1e3a5f",
                          family="Share Tech Mono"),
            hovertext=[
                f"<b>{n['name']}</b><br>{n['type']}<br>"
                f"Region: {n['region']}<br>"
                f"Alt: {n['altitude']:.0f} km<br>"
                f"Uptime: {n['uptime']}%"
            ],
            hoverinfo="text",
            showlegend=False,
        ))

    # Active payload pulse marker
    if 0 <= active_hop < len(node_coords):
        pn = node_coords[active_hop]
        for pulse_size, pulse_op in [(50, 0.08), (35, 0.15), (22, 0.4)]:
            fig.add_trace(go.Scattermapbox(
                lat=[pn["lat"]],
                lon=[pn["lon"]],
                mode="markers",
                marker=dict(size=pulse_size, color="#00f5ff", opacity=pulse_op),
                hoverinfo="none",
                showlegend=False,
            ))
        fig.add_trace(go.Scattermapbox(
            lat=[pn["lat"]],
            lon=[pn["lon"]],
            mode="markers",
            marker=dict(size=14, color="#ffffff", opacity=1.0,
                        symbol="circle"),
            hovertext=[f"📦 PAYLOAD @ {pn['name']}"],
            hoverinfo="text",
            showlegend=False,
        ))

    # After full transmission highlight destination
    if active_hop >= len(hops) and len(node_coords) > 0:
        dn = node_coords[-1]
        for pulse_size, pulse_op in [(60, 0.08), (40, 0.18), (26, 0.45)]:
            fig.add_trace(go.Scattermapbox(
                lat=[dn["lat"]],
                lon=[dn["lon"]],
                mode="markers",
                marker=dict(size=pulse_size, color="#22c55e", opacity=pulse_op),
                hoverinfo="none",
                showlegend=False,
            ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=20, lon=60),
            zoom=1.4,
        ),
        paper_bgcolor="#020810",
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# HELPER: ROUTE CARDS
# ─────────────────────────────────────────────────────────────────────

def _render_route_cards(placeholder, hops: list, active_hop: int,
                        route_names: list, route_types: list):
    cards_html = ""
    for i, hop in enumerate(hops):
        is_active = (i == active_hop)
        is_done   = (i < active_hop)
        active_cls = "active" if is_active else ""
        q = hop["link_quality"]
        q_color = _link_color(q)
        q_pct = int(q * 100)

        if is_done:
            icon = "✅"
        elif is_active:
            icon = "⚡"
        else:
            icon = "○"

        name_from = hop["from_name"].split(" ")[-1][:12]
        name_to   = hop["to_name"].split(" ")[-1][:12]

        badge_bg  = q_color
        badge_txt = "#020810"

        cards_html += f"""
        <div class="hop-card {active_cls}">
            <div class="hop-header">
                <span class="hop-idx">{icon} HOP {i+1:02d}</span>
                <span class="hop-badge" style="border-color:{q_color};color:{q_color};background:transparent;">
                    {hop['hop_type'].upper()}
                </span>
            </div>
            <div class="hop-route" style="color:{'#00f5ff' if is_active else ('#94a3b8' if is_done else '#2d4a6b')}">
                {name_from} → {name_to}
            </div>
            <div class="quality-bar-wrap">
                <span class="quality-label">QUALITY</span>
                <div class="quality-bar-bg">
                    <div class="quality-bar-fill" style="width:{q_pct}%;background:{q_color};
                    {'box-shadow:0 0 6px ' + q_color if is_active else ''}"></div>
                </div>
                <span class="quality-val" style="color:{q_color}">{q:.2%}</span>
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#4b7ab5;margin-top:4px;">
                ⏱ {hop['latency_ms']:.1f} ms &nbsp;|&nbsp; 📏 {hop['distance_km']:,.0f} km
            </div>
        </div>
        """

    placeholder.markdown(cards_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HELPER: LINK QUALITY CHART
# ─────────────────────────────────────────────────────────────────────

def _build_link_quality_chart(hops: list, active_hop: int) -> go.Figure:
    labels   = [f"H{i+1}" for i in range(len(hops))]
    qualities = [h["link_quality"] for h in hops]
    colors   = []
    opacities = []
    for i, q in enumerate(qualities):
        base_color = _link_color(q)
        if i < active_hop:
            colors.append(base_color)
            opacities.append(0.85)
        elif i == active_hop:
            colors.append("#00f5ff")
            opacities.append(1.0)
        else:
            colors.append("#0d2340")
            opacities.append(1.0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=qualities,
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(color="#020810", width=1),
        ),
        text=[f"{q:.2%}" for q in qualities],
        textposition="outside",
        textfont=dict(size=9, color="#4b7ab5", family="Share Tech Mono"),
        hovertemplate="<b>Hop %{x}</b><br>Link Quality: %{y:.3f}<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(y=0.85, line=dict(color="#22c55e", width=1, dash="dot"), opacity=0.5)
    fig.add_hline(y=0.75, line=dict(color="#f97316", width=1, dash="dot"), opacity=0.5)

    fig.add_annotation(x=len(hops)-0.5, y=0.86, text="STRONG",
                       font=dict(size=8, color="#22c55e", family="Share Tech Mono"),
                       showarrow=False, xanchor="right")
    fig.add_annotation(x=len(hops)-0.5, y=0.76, text="NOMINAL",
                       font=dict(size=8, color="#f97316", family="Share Tech Mono"),
                       showarrow=False, xanchor="right")

    # Highlight active hop
    if 0 <= active_hop < len(hops):
        fig.add_vline(x=active_hop, line=dict(color="#00f5ff", width=2, dash="solid"),
                      opacity=0.4)

    fig.update_layout(
        paper_bgcolor="#020810",
        plot_bgcolor="#050d1a",
        height=160,
        margin=dict(l=30, r=20, t=10, b=30),
        xaxis=dict(
            gridcolor="#0d2340", zerolinecolor="#0d2340",
            tickfont=dict(color="#3b82f6", size=9, family="Share Tech Mono"),
            title=dict(text="HOP SEGMENT", font=dict(color="#3b82f6", size=9,
                                                       family="Share Tech Mono")),
        ),
        yaxis=dict(
            gridcolor="#0d2340", zerolinecolor="#0d2340",
            tickfont=dict(color="#3b82f6", size=9, family="Share Tech Mono"),
            range=[0, 1.12],
            title=dict(text="QUALITY", font=dict(color="#3b82f6", size=9,
                                                  family="Share Tech Mono")),
        ),
        bargap=0.15,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# HELPER: DETAILED HOP LOG
# ─────────────────────────────────────────────────────────────────────

def _render_hop_log(hops: list):
    hop_type_icons = {
        "inter-sat": "🛰️→🛰️",
        "uplink":    "📡→🛰️",
        "downlink":  "🛰️→📡",
        "ground":    "🌍→🌍",
    }

    for hop in hops:
        q       = hop["link_quality"]
        q_color = _link_color(q)
        q_pct   = int(q * 100)
        h_icon  = hop_type_icons.get(hop["hop_type"], "→")

        st.markdown(f"""
<div style="background:#050d1a;border:1px solid #0d2340;border-radius:10px;
            padding:14px 18px;margin:5px 0;font-family:'Share Tech Mono',monospace;
            border-left:3px solid {q_color};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="color:#4b7ab5;font-size:10px;letter-spacing:2px;">HOP {hop['hop_index']+1:02d}</span>
      &nbsp;
      <span style="color:#00f5ff;font-weight:600;">{hop['from_name']}</span>
      <span style="color:#3b82f6;"> {h_icon} </span>
      <span style="color:#22c55e;font-weight:600;">{hop['to_name']}</span>
      &nbsp;
      <span style="background:#071525;border:1px solid {q_color}33;border-radius:8px;
                   padding:1px 8px;color:{q_color};font-size:10px;">{hop['hop_type'].upper()}</span>
    </div>
    <div style="text-align:right;font-size:10px;color:#4b7ab5;">
      ⏱ {hop['latency_ms']:.1f} ms &nbsp;|&nbsp;
      📏 {hop['distance_km']:,.0f} km &nbsp;|&nbsp;
      Elapsed: {hop['elapsed_ms']:.0f} ms
    </div>
  </div>
  <div style="margin-top:8px;display:flex;gap:20px;font-size:10px;color:#4b7ab5;">
    <span>📶 <b style="color:#e2e8f0;">{hop['bandwidth_mbps']} Mbps</b></span>
    <span>🔗 <b style="color:#e2e8f0;">{hop['protocol']}</b></span>
    <span>⚡ BER 10<sup>{hop['ber']:.0f}</sup></span>
    <span>📦 {hop['bytes_tx']:,} bytes</span>
  </div>
  <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
    <span style="font-size:9px;color:#4b7ab5;width:76px;letter-spacing:1px;">LINK QUALITY</span>
    <div style="flex:1;background:#0d2340;border-radius:3px;height:7px;">
      <div style="width:{q_pct}%;background:{q_color};border-radius:3px;height:7px;
                  box-shadow:0 0 8px {q_color}88;"></div>
    </div>
    <span style="font-size:10px;color:{q_color};width:40px;text-align:right;">{q:.2%}</span>
  </div>
</div>""", unsafe_allow_html=True)