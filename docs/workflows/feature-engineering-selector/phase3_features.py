"""Phase 3 feature-extraction validation script.

Computes the 16-feature selector-MLP feature set on the Phase 1/2 sample data
(/tmp/selector_sample_data). Not part of the C-AIMMS package -- this is a
one-off validation script whose numeric output feeds phase3_output.md.

Run with the `caimms` conda env (numpy only, no torch needed).
"""
from __future__ import annotations

import numpy as np

np.random.seed(42)

DATA_DIR = "/tmp/selector_sample_data"
N_EPISODES = 10

# ---------------------------------------------------------------------------
# Load sample data
# ---------------------------------------------------------------------------
vs = np.load(f"{DATA_DIR}/vs_embeddings.npy")  # (10, 384)
vc = np.load(f"{DATA_DIR}/vc_features.npy")  # (10, 7, 7, 512)
hg_adj = [np.load(f"{DATA_DIR}/hg_data/adjacency_{i}.npy") for i in range(N_EPISODES)]
hg_nf = [np.load(f"{DATA_DIR}/hg_data/node_features_{i}.npy") for i in range(N_EPISODES)]

VS_DIM = vs.shape[1]  # 384
VS_ITEMSIZE = vs.dtype.itemsize  # 4 bytes (float32)
VC_GRID = vc.shape[1] * vc.shape[2]  # 49 cells
VC_CHANNELS = vc.shape[3]  # 512
VC_ITEMSIZE = vc.dtype.itemsize  # 4 bytes

# Documented assumption: per-node / per-edge JSON serialization cost for the
# (not-yet-implemented) 3-layer hypergraph. These are placeholder estimates
# (typed node: id, content/summary, confidence, temporal/spatial tags,
# keywords ~150 bytes; typed edge: type, role, endpoints ~80 bytes) pending
# measurement against a real HyperMem-produced JSON payload.
HG_BYTES_PER_NODE = 150.0
HG_BYTES_PER_EDGE = 80.0

EPS = 1e-12


# ---------------------------------------------------------------------------
# Group 1: Geometry & Scale
# ---------------------------------------------------------------------------
def vs_spread(v: np.ndarray) -> float:
    """Participation ratio of squared components, rescaled to [0,1].

    PR = (sum v_i^2)^2 / (D * sum v_i^4) in (1/D, 1]. High = energy spread
    across many dims (isotropic); low = energy concentrated in a few dims.
    """
    d = v.shape[0]
    num = (np.sum(v**2)) ** 2
    den = d * np.sum(v**4) + EPS
    pr = num / den
    return float((pr - 1 / d) / (1 - 1 / d))


def hg_edge_density(adj: np.ndarray) -> float:
    """|nonzero directed edges| / (N*(N-1)), bounded [0,1]. (Diagnostic use only
    -- see Design Decisions: dropped from the final M, near-collinear with
    hg_relational_richness on this sample.)"""
    n = adj.shape[0]
    if n < 2:
        return 0.0
    nonzero = np.count_nonzero(adj)
    return float(nonzero / (n * (n - 1)))


def hg_hierarchy_depth(n_layers_present: int = 1, max_layers: int = 3) -> float:
    """Distinct abstraction levels present (Facts/Episodes/Topics) / max_layers.

    Future-only: the sample graph (and the current no-op stub) exposes a flat,
    untyped node set -- no per-node layer label to count. Fixed at 1/3
    (single flat layer) for every episode until typed 3-layer output exists.
    """
    return float(n_layers_present / max_layers)


def _degree_dist(adj: np.ndarray) -> np.ndarray:
    """Weighted total (in+out) degree per node, normalized to a distribution."""
    deg = adj.sum(axis=0) + adj.sum(axis=1)
    total = deg.sum()
    if total <= EPS:
        return np.full_like(deg, 1.0 / len(deg))
    return deg / total


def hg_spread(adj: np.ndarray) -> float:
    """Normalized Shannon entropy of the weighted degree distribution, in [0,1].

    High = relational importance spread evenly across nodes; low = a few hub
    nodes dominate the graph's connectivity.
    """
    p = _degree_dist(adj)
    n = len(p)
    if n < 2:
        return 0.0
    p_safe = p[p > 0]
    h = -np.sum(p_safe * np.log(p_safe))
    return float(h / np.log(n))


def vc_spatial_map(feat: np.ndarray) -> np.ndarray:
    """Channel-averaged spatial activation map, shape (7,7) -> flattened (49,)."""
    return feat.mean(axis=-1).reshape(-1)


def vc_spread(feat: np.ndarray) -> float:
    """Normalized Shannon entropy of the spatial activation map, in [0,1]."""
    s = vc_spatial_map(feat)
    s = np.clip(s, 0, None)
    total = s.sum()
    if total <= EPS:
        return 1.0
    p = s / total
    p_safe = p[p > 0]
    h = -np.sum(p_safe * np.log(p_safe))
    return float(h / np.log(len(s)))


def vc_layout_concentration(feat: np.ndarray) -> float:
    """1 - vc_spread. High = information concentrated in few regions (matches
    MemOCR's layout-guided-importance design intent)."""
    return 1.0 - vc_spread(feat)


# ---------------------------------------------------------------------------
# Group 2: Efficiency & Memory Trade-off
# ---------------------------------------------------------------------------
def vs_footprint_kb() -> float:
    return VS_DIM * VS_ITEMSIZE / 1024.0


def vc_footprint_kb() -> float:
    return VC_GRID * VC_CHANNELS * VC_ITEMSIZE / 1024.0


def hg_footprint_kb(adj: np.ndarray) -> float:
    n = adj.shape[0]
    e = np.count_nonzero(adj)
    return (n * HG_BYTES_PER_NODE + e * HG_BYTES_PER_EDGE) / 1024.0


def hg_cost_share(adj: np.ndarray) -> float:
    hg = hg_footprint_kb(adj)
    total = hg + vs_footprint_kb() + vc_footprint_kb()
    return float(hg / total)


def format_footprint_spread(adj: np.ndarray) -> float:
    """Coefficient of variation across the three per-episode footprints, clipped to [0,1].

    Diagnostic only -- see Design Decisions: saturates at 1.0 for every
    episode (vc_footprint_kb=98KB dominates vs=1.5KB/hg~1-6KB by 1-2 orders
    of magnitude), i.e. zero variance. Superseded by hg_cost_position below.
    """
    vals = np.array([vs_footprint_kb(), hg_footprint_kb(adj), vc_footprint_kb()])
    cv = vals.std() / (vals.mean() + EPS)
    return float(min(cv, 1.0))


def hg_cost_position(adj: np.ndarray) -> float:
    """Where HG's footprint sits between VS's (cheap) and VC's (expensive), in [0,1]+.

    0 = as cheap as VS; 1 = as expensive as VC. Replaces
    format_footprint_spread, which saturated at a constant 1.0 (see above).
    """
    hg = hg_footprint_kb(adj)
    vs_kb, vc_kb = vs_footprint_kb(), vc_footprint_kb()
    return float((hg - vs_kb) / (vc_kb - vs_kb))


# ---------------------------------------------------------------------------
# Group 3: Complementarity & Overlap (structural-agreement proxy)
# ---------------------------------------------------------------------------
def structural_agreement(spread_a: float, spread_b: float) -> float:
    return float(1.0 - abs(spread_a - spread_b))


# ---------------------------------------------------------------------------
# Group 4: Robustness & Stability (perturb-and-remeasure proxy)
# ---------------------------------------------------------------------------
def vs_embedding_stability(v: np.ndarray, sigma: float = 0.01, trials: int = 20) -> float:
    dists = []
    for _ in range(trials):
        noise = np.random.normal(0, sigma, size=v.shape)
        v_pert = v + noise
        v_pert = v_pert / (np.linalg.norm(v_pert) + EPS)
        v_norm = v / (np.linalg.norm(v) + EPS)
        dists.append(np.linalg.norm(v_norm - v_pert))
    mean_dist = float(np.mean(dists))
    return float(1.0 - min(mean_dist / 2.0, 1.0))


def hg_graph_stability(adj: np.ndarray, sigma: float = 0.02, trials: int = 20) -> float:
    """1 - mean absolute change in hg_relational_richness under edge-weight jitter.

    Uses richness (continuous, weight-sensitive), not hg_edge_density (a
    count-based statistic that only changes if noise flips an edge across
    the zero threshold -- verified to saturate at a constant 1.0 for
    sigma=0.02 against these edge weights, see Design Decisions).
    """
    base = hg_relational_richness(adj)
    deltas = []
    for _ in range(trials):
        noise = np.random.normal(0, sigma, size=adj.shape)
        adj_pert = np.clip(adj + noise * (adj > 0), 0, 1)  # jitter existing edge weights only
        deltas.append(abs(hg_relational_richness(adj_pert) - base))
    mean_delta = float(np.mean(deltas))
    return float(1.0 - min(mean_delta, 1.0))


def vc_layout_stability(feat: np.ndarray, sigma: float = 0.02, trials: int = 10) -> float:
    base = vc_layout_concentration(feat)
    deltas = []
    for _ in range(trials):
        noise = np.random.normal(0, sigma, size=feat.shape)
        feat_pert = np.clip(feat + noise, 0, 1)
        deltas.append(abs(vc_layout_concentration(feat_pert) - base))
    mean_delta = float(np.mean(deltas))
    return float(1.0 - min(mean_delta, 1.0))


# ---------------------------------------------------------------------------
# Group 5: Task-Relevant / Domain-fit
# ---------------------------------------------------------------------------
def vs_semantic_specificity(v: np.ndarray, corpus_mean: np.ndarray) -> float:
    cos = np.dot(v, corpus_mean) / (np.linalg.norm(v) * np.linalg.norm(corpus_mean) + EPS)
    return float(1.0 - max(cos, 0.0))


def hg_relational_richness(adj: np.ndarray) -> float:
    n = adj.shape[0]
    if n < 2:
        return 0.0
    return float(adj.sum() / (n * (n - 1)))


def hg_information_per_kb(adj: np.ndarray) -> float:
    """hg_relational_richness normalized by hg's own per-episode footprint.

    The only encoder-output ratio in this feature set where numerator and
    denominator are not both driven by the same single scalar (unlike a
    naive vs_efficiency = vs_signal / vs_footprint_kb, which collapses to a
    rescaling of vs_signal because vs_footprint_kb is architecture-constant
    -- see the sanity check at the bottom of this script).
    """
    fp = hg_footprint_kb(adj)
    return float(hg_relational_richness(adj) / (fp + EPS))


def vc_layout_prominence(feat: np.ndarray) -> float:
    s = vc_spatial_map(feat)
    peak, mean = s.max(), s.mean()
    return float((peak - mean) / (peak + mean + EPS))


# ---------------------------------------------------------------------------
# Compute all features for all 10 episodes
# ---------------------------------------------------------------------------
corpus_mean = vs.mean(axis=0)

rows = []
for i in range(N_EPISODES):
    v = vs[i]
    adj = hg_adj[i]
    feat = vc[i]

    vs_sp = vs_spread(v)
    hg_sp = hg_spread(adj)
    vc_sp = vc_spread(feat)

    row = {
        "episode": i,
        "n_hg_nodes": adj.shape[0],
        "n_hg_edges": int(np.count_nonzero(adj)),
        "_diag_hg_edge_density": hg_edge_density(adj),  # diagnostic only, not in final M
        "_diag_format_footprint_spread": format_footprint_spread(adj),  # diagnostic only
        # Group 1
        "vs_energy_concentration": vs_sp,
        "hg_hierarchy_depth": hg_hierarchy_depth(),
        "vc_layout_concentration": vc_layout_concentration(feat),
        "_diag_hg_memory_footprint_log_kb": float(np.log1p(hg_footprint_kb(adj))),  # diagnostic only
        "_diag_hg_cost_share": hg_cost_share(adj),  # diagnostic only, merged into hg_cost_position
        # Group 2 (final)
        "hg_cost_position": hg_cost_position(adj),
        "hg_information_per_kb": hg_information_per_kb(adj),
        # Group 3
        "vs_hg_structural_agreement": structural_agreement(vs_sp, hg_sp),
        "vs_vc_structural_agreement": structural_agreement(vs_sp, vc_sp),
        "hg_vc_structural_agreement": structural_agreement(hg_sp, vc_sp),
        # Group 4
        "vs_embedding_stability": vs_embedding_stability(v),
        "hg_graph_stability": hg_graph_stability(adj),
        "vc_layout_stability": vc_layout_stability(feat),
        # Group 5
        "vs_semantic_specificity": vs_semantic_specificity(v, corpus_mean),
        "hg_relational_richness": hg_relational_richness(adj),
        "vc_layout_prominence": vc_layout_prominence(feat),
    }
    row["complementarity_score"] = 1.0 - np.mean(
        [row["vs_hg_structural_agreement"], row["vs_vc_structural_agreement"], row["hg_vc_structural_agreement"]]
    )
    rows.append(row)

import pandas as pd

df = pd.DataFrame(rows).set_index("episode")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)

print("=== Fixed architecture constants ===")
print(f"vs_footprint_kb = {vs_footprint_kb():.4f} KB (constant, all episodes)")
print(f"vc_footprint_kb = {vc_footprint_kb():.4f} KB (constant, all episodes)")
print()

print("=== Per-episode feature table ===")
print(df.to_string())
print()

non_feature_cols = [c for c in df.columns if c.startswith("_diag_")] + ["n_hg_nodes", "n_hg_edges"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]
diag_cols = [c for c in df.columns if c.startswith("_diag_")]

print("=== Diagnostic-only columns (rejected/merged features, kept for redundancy discussion) ===")
print(df[diag_cols].describe().T[["min", "max", "std"]])
print()
print(f"Final feature count M = {len(feature_cols)}")
print(feature_cols)
print()
print("=== Summary statistics ===")
print(df[feature_cols].describe().T[["min", "max", "mean", "std"]])
print()

print("=== Zero/near-zero variance check (std < 1e-6) ===")
stds = df[feature_cols].std()
print(stds[stds < 1e-6])
print()

print("=== Correlation matrix (feature_cols) ===")
corr = df[feature_cols].corr()
print(corr.to_string())
print()

print("=== High correlation pairs (|r| > 0.9, excluding diagonal) ===")
for i, a in enumerate(feature_cols):
    for b in feature_cols[i + 1 :]:
        r = corr.loc[a, b]
        if abs(r) > 0.9:
            print(f"{a} <-> {b}: r={r:.4f}")

# Sanity check: does a naive "vs_memory_efficiency = vs_spread / vs_footprint_kb"
# correlate perfectly with vs_energy_concentration, confirming redundancy
# (since vs_footprint_kb is constant across episodes)?
naive_vs_eff = df["vs_energy_concentration"] / vs_footprint_kb()
r_check = np.corrcoef(naive_vs_eff, df["vs_energy_concentration"])[0, 1]
print()
print(f"Sanity check -- naive vs_memory_efficiency vs vs_energy_concentration: r={r_check:.6f}")
