"""
space_tsp.py — OrbitalMind Space Data Transmission Simulator
═════════════════════════════════════════════════════════════
Simulates payload routing through a network of orbital and ground-based
data-relay nodes using an Advanced TSP (Travelling Salesman Problem) solver.

Architecture
────────────
  SpaceNode          — one relay node (satellite / ground-station / datacenter)
  SpaceNetwork       — full node graph with link-quality matrix
  TSPRouter          — solver: Nearest-Neighbour seed → 2-opt → 3-opt refinement
  TransmissionHop    — one hop along the chosen route (timing, BER, latency)
  TransmissionResult — complete end-to-end journey ready for UI rendering

Design decisions
────────────────
  • ALL routing logic lives here; app.py only imports TransmissionSimulator
    and calls simulate(payload_bytes, scene_lat, scene_lon).
  • Pure Python + NumPy — no extra dependencies.
  • TSP cost = weighted sum of distance, link quality, and congestion.
  • 2-opt + 3-opt post-optimisation guarantees the route improves
    monotonically from the greedy seed.
  • All timing values are physics-informed:
      - speed of light: 299_792 km/s
      - GEO link latency: ~240 ms one-way
      - LEO-LEO inter-sat link: ~5–25 ms
      - Ground-station uplink: ~30–80 ms
"""

from __future__ import annotations

import math
import time
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SPEED_OF_LIGHT_KM_S = 299_792.0        # km/s in vacuum
LEO_ALTITUDE_KM     = 550.0            # typical LEO (Starlink band)
MEO_ALTITUDE_KM     = 20_200.0         # GPS-band MEO
GEO_ALTITUDE_KM     = 35_786.0         # geostationary orbit

EARTH_RADIUS_KM     = 6_371.0
BANDWIDTH_GBPS      = 1.2              # inter-satellite optical link

# Node type enum strings
TYPE_LEO   = "LEO Satellite"
TYPE_MEO   = "MEO Relay"
TYPE_GEO   = "GEO Hub"
TYPE_GROUND= "Ground Station"
TYPE_DC    = "Space Datacenter"
TYPE_EARTH = "Earth (Mission Control)"


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpaceNode:
    """
    One relay node in the space data network.

    lat/lon are orbital ground-track position (for LEO/MEO/GEO)
    or actual position (ground stations / Earth).
    altitude_km: 0 for ground nodes.
    """
    node_id:      str
    name:         str
    node_type:    str           # TYPE_* constant
    lat:          float         # degrees, ground-track
    lon:          float         # degrees
    altitude_km:  float         # 0 for surface nodes
    uptime_pct:   float = 99.5  # link reliability %
    congestion:   float = 0.15  # 0=clear, 1=saturated
    region:       str   = ""    # e.g. "Asia-Pacific"

    # Computed at runtime
    tx_power_dbm: float = 30.0  # transmit power (dBm)
    antenna_gain: float = 35.0  # dBi (high-gain dish)

    @property
    def is_orbital(self) -> bool:
        return self.altitude_km > 0

    @property
    def pos_3d(self) -> np.ndarray:
        """Cartesian ECEF position in km (simplified spherical Earth)."""
        r   = EARTH_RADIUS_KM + self.altitude_km
        lat = math.radians(self.lat)
        lon = math.radians(self.lon)
        return np.array([
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ])


@dataclass
class TransmissionHop:
    """One segment in the route: from_node → to_node."""
    hop_index:      int
    from_node:      str         # node_id
    to_node:        str         # node_id
    from_name:      str
    to_name:        str
    distance_km:    float
    latency_ms:     float       # propagation + processing
    bandwidth_mbps: float
    ber:            float       # bit error rate (log10 scale)
    link_quality:   float       # 0–1 (1=perfect)
    bytes_tx:       int
    elapsed_ms:     float       # cumulative time from origin
    hop_type:       str         # "inter-sat", "uplink", "downlink", "ground"
    protocol:       str         # e.g. "CCSDS/Optical", "DTN/RF"


@dataclass
class TransmissionResult:
    """Complete end-to-end transmission journey."""
    route_nodes:         List[str]           # node_ids in order
    route_names:         List[str]           # display names
    hops:                List[TransmissionHop]
    total_distance_km:   float
    total_latency_ms:    float
    total_hops:          int
    payload_bytes:       int
    effective_bandwidth: float               # Mbps end-to-end
    path_efficiency:     float               # vs naive direct path
    tsp_cost:            float               # TSP objective value
    tsp_iterations:      int                 # 2-opt passes run
    tsp_improvement:     float               # % cost reduction vs greedy seed
    origin_node:         str
    destination_node:    str
    node_coords:         List[Dict]          # [{id,name,lat,lon,alt,type}] for map
    link_qualities:      List[float]         # per-hop quality
    scene_lat:           float
    scene_lon:           float
    timestamp:           float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# SPACE NETWORK — fixed topology of 14 nodes
# ─────────────────────────────────────────────────────────────────────────────

def _build_space_network() -> List[SpaceNode]:
    """
    14-node representative space relay network:
      - 4 LEO satellites (polar + equatorial)
      - 2 MEO relay nodes
      - 2 GEO hubs (Indian Ocean + Pacific)
      - 3 ground stations (strategic)
      - 2 space datacenters (hypothetical L2/L4 points)
      - 1 Earth destination (Mission Control)

    Positions are illustrative but geographically plausible.
    """
    return [
        # ── LEO constellation ────────────────────────────────────────────────
        SpaceNode("LEO-1", "StarRelay Alpha",   TYPE_LEO,    28.5,   77.1,  LEO_ALTITUDE_KM,
                  uptime_pct=99.2, congestion=0.10, region="South Asia",
                  tx_power_dbm=33, antenna_gain=28),

        SpaceNode("LEO-2", "StarRelay Beta",    TYPE_LEO,   -33.9,  151.2,  LEO_ALTITUDE_KM,
                  uptime_pct=98.9, congestion=0.18, region="Australia-Pacific",
                  tx_power_dbm=33, antenna_gain=28),

        SpaceNode("LEO-3", "StarRelay Gamma",   TYPE_LEO,    51.5,   -0.1,  LEO_ALTITUDE_KM,
                  uptime_pct=99.5, congestion=0.12, region="Northern Europe",
                  tx_power_dbm=33, antenna_gain=28),

        SpaceNode("LEO-4", "StarRelay Delta",   TYPE_LEO,   37.8, -122.4,  LEO_ALTITUDE_KM,
                  uptime_pct=98.7, congestion=0.22, region="North America West",
                  tx_power_dbm=33, antenna_gain=28),

        # ── MEO relay nodes ──────────────────────────────────────────────────
        SpaceNode("MEO-1", "OrbitBridge I",     TYPE_MEO,    0.0,   80.0,  MEO_ALTITUDE_KM,
                  uptime_pct=99.8, congestion=0.08, region="Indian Ocean MEO",
                  tx_power_dbm=40, antenna_gain=42),

        SpaceNode("MEO-2", "OrbitBridge II",    TYPE_MEO,    0.0, -140.0,  MEO_ALTITUDE_KM,
                  uptime_pct=99.7, congestion=0.09, region="Pacific MEO",
                  tx_power_dbm=40, antenna_gain=42),

        # ── GEO hubs ─────────────────────────────────────────────────────────
        SpaceNode("GEO-1", "GeoHub Indra",      TYPE_GEO,    0.0,   83.0,  GEO_ALTITUDE_KM,
                  uptime_pct=99.95, congestion=0.05, region="Indian Ocean GEO",
                  tx_power_dbm=46, antenna_gain=52),

        SpaceNode("GEO-2", "GeoHub Pacific",    TYPE_GEO,    0.0,  177.0,  GEO_ALTITUDE_KM,
                  uptime_pct=99.93, congestion=0.06, region="Pacific GEO",
                  tx_power_dbm=46, antenna_gain=52),

        # ── Space Datacenters (hypothetical L4 / cislunar) ───────────────────
        SpaceNode("SDC-1", "Orbital DC Mumbai", TYPE_DC,    19.07,   72.87, 800.0,
                  uptime_pct=99.99, congestion=0.03, region="South Asia LEO-DC",
                  tx_power_dbm=50, antenna_gain=55),

        SpaceNode("SDC-2", "Orbital DC SFO",    TYPE_DC,    37.8, -122.4,  750.0,
                  uptime_pct=99.99, congestion=0.04, region="North America LEO-DC",
                  tx_power_dbm=50, antenna_gain=55),

        # ── Ground stations ──────────────────────────────────────────────────
        SpaceNode("GS-1",  "ISRO Bangalore",    TYPE_GROUND, 12.97,  77.59,  0.0,
                  uptime_pct=99.1, congestion=0.20, region="India",
                  tx_power_dbm=55, antenna_gain=60),

        SpaceNode("GS-2",  "ESA Darmstadt",     TYPE_GROUND, 49.87,   8.65,  0.0,
                  uptime_pct=99.3, congestion=0.15, region="Europe",
                  tx_power_dbm=55, antenna_gain=60),

        SpaceNode("GS-3",  "NASA JPL Pasadena", TYPE_GROUND, 34.20,-118.17,  0.0,
                  uptime_pct=99.4, congestion=0.12, region="North America",
                  tx_power_dbm=55, antenna_gain=60),

        # ── Earth destination (always last) ──────────────────────────────────
        SpaceNode("EARTH", "Mission Control",   TYPE_EARTH,  28.61,   77.21,  0.0,
                  uptime_pct=100.0, congestion=0.0, region="Earth",
                  tx_power_dbm=60, antenna_gain=65),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE & LINK QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def _distance_km(a: SpaceNode, b: SpaceNode) -> float:
    """3-D Euclidean distance in km (ECEF)."""
    return float(np.linalg.norm(a.pos_3d - b.pos_3d))


def _link_quality(a: SpaceNode, b: SpaceNode, dist_km: float) -> float:
    """
    Link quality Q ∈ [0, 1] based on:
      - Free-space path loss (Friis equation proxy)
      - Node uptime product
      - Congestion average
    """
    # Friis-inspired: quality falls as log of distance
    # Normalised so 500 km link = 0.95, 80 000 km = 0.40
    path_factor  = 1.0 - 0.08 * math.log10(max(dist_km, 1.0))
    uptime_factor= (a.uptime_pct / 100.0) * (b.uptime_pct / 100.0)
    cong_factor  = 1.0 - (a.congestion + b.congestion) / 2.0

    # Gain bonus for high-power nodes (GEO, DC)
    gain_bonus   = min(0.05, (a.antenna_gain + b.antenna_gain - 56) / 1000.0)

    q = path_factor * uptime_factor * cong_factor + gain_bonus
    return float(np.clip(q, 0.05, 1.0))


def _propagation_latency_ms(dist_km: float) -> float:
    """One-way propagation latency (speed-of-light)."""
    return (dist_km / SPEED_OF_LIGHT_KM_S) * 1000.0   # ms


def _processing_latency_ms(node_type: str) -> float:
    """Node processing overhead (store-and-forward)."""
    return {"LEO Satellite": 8.0, "MEO Relay": 12.0, "GEO Hub": 18.0,
            "Space Datacenter": 5.0, "Ground Station": 25.0,
            "Earth (Mission Control)": 0.0}.get(node_type, 10.0)


def _ber(dist_km: float, link_quality: float) -> float:
    """
    Bit error rate (log10).  Returns the log10(BER) value.
    Good optical ISL: ~1e-12; degraded RF: ~1e-6
    """
    base = -12.0 + 6.0 * (1.0 - link_quality)
    return round(base, 2)


def _hop_type(a: SpaceNode, b: SpaceNode) -> Tuple[str, str]:
    """Return (hop_type_str, protocol_str)."""
    both_orbital = a.is_orbital and b.is_orbital
    if both_orbital:
        return "inter-sat", "CCSDS/Optical-ISL"
    elif a.is_orbital and not b.is_orbital:
        return "downlink", "CCSDS/Ka-Band"
    elif not a.is_orbital and b.is_orbital:
        return "uplink", "CCSDS/Ka-Band"
    else:
        return "ground", "TCP/IP-Fibre"


def _bandwidth_mbps(hop_type: str, link_quality: float) -> float:
    """Effective bandwidth in Mbps for this hop."""
    base = {"inter-sat": 1200.0, "downlink": 600.0,
            "uplink": 400.0, "ground": 10_000.0}.get(hop_type, 500.0)
    return round(base * link_quality, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TSP COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _tsp_edge_cost(a: SpaceNode, b: SpaceNode) -> float:
    """
    TSP edge cost — lower is better.

    Cost = w1 * normalised_distance
         + w2 * (1 - link_quality)   ← prefer high-quality links
         + w3 * congestion_mean       ← avoid congested nodes
         + w4 * latency_ms / 1000    ← penalise high latency

    Weights tuned so that quality matters more than raw distance
    (space links: high distance is unavoidable, quality is controllable).
    """
    dist  = _distance_km(a, b)
    q     = _link_quality(a, b, dist)
    cong  = (a.congestion + b.congestion) / 2.0
    lat   = _propagation_latency_ms(dist)

    w1, w2, w3, w4 = 0.25, 0.45, 0.15, 0.15

    norm_dist = dist / 80_000.0     # max ~80 000 km (GEO to opposite side)
    norm_lat  = lat  / 270.0        # max ~270 ms (GEO round-trip half)

    return w1 * norm_dist + w2 * (1.0 - q) + w3 * cong + w4 * norm_lat


# ─────────────────────────────────────────────────────────────────────────────
# TSP ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class TSPRouter:
    """
    Advanced TSP solver for the space relay network.

    Algorithm:
      1. Nearest-Neighbour greedy seed starting from origin node.
      2. 2-opt improvement: try all (i,j) edge swaps, accept improvements.
      3. 3-opt improvement: try all (i,j,k) triple reconnections.
      4. Stop when no improvement found or max_iters reached.

    The tour visits a SUBSET of nodes (not all 14) —
    specifically those that improve the route quality vs skipping them.
    Mandatory nodes: [origin, SDC-1 or nearest DC, ≥1 GEO, EARTH].
    """

    def __init__(self, nodes: List[SpaceNode], max_2opt: int = 50, max_3opt: int = 20):
        self.nodes     = nodes
        self.node_map  = {n.node_id: n for n in nodes}
        self.max_2opt  = max_2opt
        self.max_3opt  = max_3opt

        # Pre-compute full cost matrix
        n = len(nodes)
        self.cost_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.cost_matrix[i, j] = _tsp_edge_cost(nodes[i], nodes[j])

    # ── index helpers ──────────────────────────────────────────────────────────
    def _idx(self, node_id: str) -> int:
        return next(i for i, n in enumerate(self.nodes) if n.node_id == node_id)

    def _cost(self, i: int, j: int) -> float:
        return float(self.cost_matrix[i, j])

    def _route_cost(self, route: List[int]) -> float:
        return sum(self._cost(route[k], route[k+1]) for k in range(len(route)-1))

    # ── greedy nearest-neighbour seed ─────────────────────────────────────────
    def _nn_seed(self, start_idx: int, must_visit: List[int], end_idx: int) -> List[int]:
        """
        Nearest-neighbour greedy tour.
        Visits all must_visit nodes plus opportunistic stops,
        always ends at end_idx.
        """
        candidates = set(range(len(self.nodes))) - {end_idx}
        # Force mandatory stops
        mandatory  = set(must_visit) | {start_idx}

        route    = [start_idx]
        visited  = {start_idx}
        current  = start_idx

        while mandatory - visited:
            # Next step must be closest unvisited mandatory node
            remaining_mandatory = list(mandatory - visited)
            nxt = min(remaining_mandatory,
                      key=lambda j: self._cost(current, j))
            route.append(nxt)
            visited.add(nxt)
            current = nxt

        # Optionally add high-quality intermediate hops
        unvisited = candidates - visited
        for j in sorted(unvisited, key=lambda j: self._cost(current, j)):
            node = self.nodes[j]
            # Only add if it improves quality (DC or GEO node worth visiting)
            if node.node_type in (TYPE_DC, TYPE_GEO, TYPE_MEO):
                edge_gain = (self._cost(current, end_idx)
                             - self._cost(current, j)
                             - self._cost(j, end_idx))
                if edge_gain > 0.005:  # insertion worth it
                    route.append(j)
                    visited.add(j)
                    current = j

        route.append(end_idx)
        return route

    # ── 2-opt improvement ─────────────────────────────────────────────────────
    def _two_opt(self, route: List[int]) -> Tuple[List[int], int]:
        """
        Standard 2-opt: reverse sub-tours to eliminate crossings.
        Returns (improved_route, passes_made).
        Keeps start and end fixed.
        """
        best      = route[:]
        best_cost = self._route_cost(best)
        passes    = 0
        improved  = True

        while improved and passes < self.max_2opt:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    # Reverse segment best[i:j+1]
                    candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    c = self._route_cost(candidate)
                    if c < best_cost - 1e-9:
                        best      = candidate
                        best_cost = c
                        improved  = True
            passes += 1

        return best, passes

    # ── 3-opt improvement ─────────────────────────────────────────────────────
    def _three_opt(self, route: List[int]) -> Tuple[List[int], int]:
        """
        3-opt: considers all triple-edge reconnections.
        Only runs if route length ≥ 6 (otherwise 2-opt sufficient).
        """
        if len(route) < 6:
            return route, 0

        best      = route[:]
        best_cost = self._route_cost(best)
        passes    = 0
        improved  = True

        while improved and passes < self.max_3opt:
            improved = False
            n = len(best)
            for i in range(1, n - 4):
                for j in range(i + 1, n - 3):
                    for k in range(j + 1, n - 1):
                        # Try all 7 reconnections of 3 edges
                        segs = [
                            best[:i] + best[i:j] + best[j:k] + best[k:],           # orig
                            best[:i] + best[i:j] + best[j:k][::-1] + best[k:],     # 2opt k
                            best[:i] + best[i:j][::-1] + best[j:k] + best[k:],     # 2opt j
                            best[:i] + best[i:j][::-1] + best[j:k][::-1] + best[k:],# 2opt j+k
                            best[:i] + best[j:k] + best[i:j] + best[k:],           # 3opt A
                            best[:i] + best[j:k] + best[i:j][::-1] + best[k:],     # 3opt B
                            best[:i] + best[j:k][::-1] + best[i:j] + best[k:],     # 3opt C
                        ]
                        for seg in segs[1:]:
                            c = self._route_cost(seg)
                            if c < best_cost - 1e-9:
                                best      = seg
                                best_cost = c
                                improved  = True
            passes += 1

        return best, passes

    # ── public interface ───────────────────────────────────────────────────────
    def solve(
        self,
        origin_id:  str,
        dest_id:    str,
        must_visit: Optional[List[str]] = None,
    ) -> Tuple[List[SpaceNode], float, float, int]:
        """
        Solve TSP from origin_id to dest_id through must_visit nodes.

        Returns
        ───────
          (ordered_nodes, final_cost, improvement_pct, total_2opt_passes)
        """
        start_idx  = self._idx(origin_id)
        end_idx    = self._idx(dest_id)
        must_idxs  = [self._idx(nid) for nid in (must_visit or [])]

        # Seed
        seed_route = self._nn_seed(start_idx, must_idxs, end_idx)
        seed_cost  = self._route_cost(seed_route)

        # 2-opt
        route2, passes2 = self._two_opt(seed_route)

        # 3-opt
        route3, passes3 = self._three_opt(route2)

        final_cost = self._route_cost(route3)
        improvement = 100.0 * (seed_cost - final_cost) / max(seed_cost, 1e-9)

        ordered_nodes = [self.nodes[i] for i in route3]
        return ordered_nodes, final_cost, improvement, passes2 + passes3


# ─────────────────────────────────────────────────────────────────────────────
# HOP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_hops(
    route: List[SpaceNode],
    payload_bytes: int,
) -> List[TransmissionHop]:
    """
    Convert a solved TSP route into a list of TransmissionHop objects,
    each carrying full physics-informed link metrics.
    """
    hops     = []
    elapsed  = 0.0

    for i in range(len(route) - 1):
        a, b     = route[i], route[i+1]
        dist     = _distance_km(a, b)
        quality  = _link_quality(a, b, dist)
        prop_lat = _propagation_latency_ms(dist)
        proc_lat = _processing_latency_ms(b.node_type)
        total_lat= prop_lat + proc_lat
        elapsed += total_lat
        h_type, protocol = _hop_type(a, b)
        bw = _bandwidth_mbps(h_type, quality)

        hops.append(TransmissionHop(
            hop_index=i,
            from_node=a.node_id,
            to_node=b.node_id,
            from_name=a.name,
            to_name=b.name,
            distance_km=round(dist, 1),
            latency_ms=round(total_lat, 2),
            bandwidth_mbps=bw,
            ber=_ber(dist, quality),
            link_quality=round(quality, 4),
            bytes_tx=payload_bytes,
            elapsed_ms=round(elapsed, 2),
            hop_type=h_type,
            protocol=protocol,
        ))

    return hops


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class TransmissionSimulator:
    """
    Public interface imported by pipeline.py.

    Usage
    ─────
        sim    = TransmissionSimulator()
        result = sim.simulate(
            payload_bytes = 1200,
            scene_lat     = 19.07,
            scene_lon     = 72.87,
        )

    The result dict is JSON-serialisable (all numpy scalars converted).
    """

    _EARTH_ID   = "EARTH"
    _MUST_VISIT = ["SDC-1", "GEO-1"]   # always route through a DC and a GEO hub

    def __init__(self):
        self.nodes   = _build_space_network()
        self.network = {n.node_id: n for n in self.nodes}

    def _origin_node_id(self, scene_lat: float, scene_lon: float) -> str:
        """
        Find the nearest LEO satellite to the scene (origin of uplink).
        Uses great-circle angular distance on ground-track.
        """
        leo_nodes = [n for n in self.nodes if n.node_type == TYPE_LEO]
        best, best_dist = leo_nodes[0], float("inf")
        for n in leo_nodes:
            d = math.sqrt((n.lat - scene_lat)**2 + (n.lon - scene_lon)**2)
            if d < best_dist:
                best, best_dist = n, d
        return best.node_id

    def simulate(
        self,
        payload_bytes: int,
        scene_lat:     float,
        scene_lon:     float,
        seed:          Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full TSP-optimised transmission simulation.

        Parameters
        ──────────
          payload_bytes : size of the downlink JSON payload
          scene_lat/lon : satellite scene centre (finds nearest relay)
          seed          : optional RNG seed for reproducible congestion jitter

        Returns
        ───────
          Dict — fully JSON-serialisable transmission result.
          Key fields consumed by app.py:
            "route_names"        list[str]   ordered relay names
            "route_types"        list[str]   node type per relay
            "hops"               list[dict]  per-hop metrics
            "total_latency_ms"   float
            "total_distance_km"  float
            "total_hops"         int
            "tsp_improvement"    float       % over greedy seed
            "tsp_iterations"     int
            "path_efficiency"    float       ratio vs naive direct
            "effective_bandwidth"float       Mbps
            "node_coords"        list[dict]  for map rendering
            "link_qualities"     list[float]
        """
        if seed is None:
            seed = abs(hash(f"{scene_lat:.2f},{scene_lon:.2f}")) % 2**31
        rng = random.Random(seed)

        # Jitter congestion slightly for demo realism
        for n in self.nodes:
            n.congestion = float(np.clip(n.congestion + rng.gauss(0, 0.02), 0, 0.5))

        origin_id = self._origin_node_id(scene_lat, scene_lon)

        router = TSPRouter(self.nodes)
        ordered_nodes, tsp_cost, improvement, iters = router.solve(
            origin_id=origin_id,
            dest_id=self._EARTH_ID,
            must_visit=self._MUST_VISIT,
        )

        hops = _build_hops(ordered_nodes, payload_bytes)

        total_dist   = sum(h.distance_km for h in hops)
        total_lat    = sum(h.latency_ms  for h in hops)
        total_hops   = len(hops)
        min_bw       = min(h.bandwidth_mbps for h in hops)  # bottleneck link

        # Path efficiency: compare to naive direct (origin→Earth) latency
        origin_node  = self.network[origin_id]
        earth_node   = self.network[self._EARTH_ID]
        direct_dist  = _distance_km(origin_node, earth_node)
        direct_lat   = _propagation_latency_ms(direct_dist) + _processing_latency_ms(TYPE_EARTH)
        path_eff     = round(direct_lat / max(total_lat, 1.0), 4)  # >1 = we're faster!

        # Serialise hops
        hops_dict = []
        for h in hops:
            hops_dict.append({
                "hop_index":     h.hop_index,
                "from_node":     h.from_node,
                "to_node":       h.to_node,
                "from_name":     h.from_name,
                "to_name":       h.to_name,
                "distance_km":   h.distance_km,
                "latency_ms":    h.latency_ms,
                "bandwidth_mbps":h.bandwidth_mbps,
                "ber":           h.ber,
                "link_quality":  h.link_quality,
                "bytes_tx":      h.bytes_tx,
                "elapsed_ms":    h.elapsed_ms,
                "hop_type":      h.hop_type,
                "protocol":      h.protocol,
            })

        # Node coordinates for map + animation
        node_coords = []
        for n in ordered_nodes:
            node_coords.append({
                "id":       n.node_id,
                "name":     n.name,
                "type":     n.node_type,
                "lat":      n.lat,
                "lon":      n.lon,
                "altitude": n.altitude_km,
                "region":   n.region,
                "uptime":   n.uptime_pct,
                "congestion": round(n.congestion, 3),
            })

        return {
            "route_ids":           [n.node_id for n in ordered_nodes],
            "route_names":         [n.name    for n in ordered_nodes],
            "route_types":         [n.node_type for n in ordered_nodes],
            "hops":                hops_dict,
            "total_distance_km":   round(total_dist,  1),
            "total_latency_ms":    round(total_lat,   2),
            "total_hops":          total_hops,
            "payload_bytes":       payload_bytes,
            "effective_bandwidth": round(min_bw, 1),
            "path_efficiency":     path_eff,
            "tsp_cost":            round(tsp_cost, 6),
            "tsp_iterations":      iters,
            "tsp_improvement":     round(improvement, 2),
            "origin_node":         origin_id,
            "destination_node":    self._EARTH_ID,
            "node_coords":         node_coords,
            "link_qualities":      [h.link_quality for h in hops],
            "scene_lat":           scene_lat,
            "scene_lon":           scene_lon,
        }