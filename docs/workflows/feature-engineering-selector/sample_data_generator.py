"""
Sample Data Generator for Selector Feature Engineering

Generates synthetic/minimal examples of VS, HG, VC encoder outputs
for testing feature extraction functions.

Usage:
    python sample_data_generator.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EncoderOutputs:
    """Container for encoder outputs suitable for selector."""

    vs_embedding: np.ndarray  # (384,) - sentence-transformers embedding
    hg_adjacency: np.ndarray  # (N, N) - graph adjacency matrix
    hg_node_features: np.ndarray  # (N, D) - node feature vectors
    vc_features: np.ndarray  # (7, 7, 512) - CNN feature maps


def generate_vs_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    """
    Generate a synthetic Vector Semantic (VS) embedding.

    Properties:
    - Dense, high-dimensional vector (384-dim like sentence-transformers)
    - Cosine-normalized to unit L2 norm
    - Dense (no sparsity)

    Args:
        seed: Random seed for reproducibility
        dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)

    Returns:
        np.ndarray of shape (dim,) with float32 values, L2-normalized
    """
    np.random.seed(seed)
    embedding = np.random.randn(dim).astype(np.float32)
    # Normalize to unit L2 norm
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def generate_hg_graph(seed: int = 0, num_nodes: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic Hierarchical Graph (HG) representation.

    Properties:
    - Sparse adjacency matrix (sparse but not ultra-sparse)
    - Some hierarchy structure (nodes organized in levels)
    - Node features for semantic information

    Args:
        seed: Random seed for reproducibility
        num_nodes: Number of nodes (if None, random between 5-20)

    Returns:
        Tuple of:
        - adjacency: np.ndarray of shape (N, N), float32, sparse
        - node_features: np.ndarray of shape (N, 64), float32, feature vectors
    """
    np.random.seed(seed)

    if num_nodes is None:
        num_nodes = np.random.randint(5, 21)

    # Create sparse adjacency matrix with hierarchy structure
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Add edges to create hierarchy (parents → children)
    # Level 0: root (node 0)
    # Level 1: nodes 1-2 (children of 0)
    # Level 2: nodes 3-6 (children of 1-2)
    # Level 3: nodes 7+ (children of level 2)

    if num_nodes > 1:
        adjacency[0, 1:3] = 1.0  # Root → children
    if num_nodes > 3:
        adjacency[1, 3:5] = 1.0  # Level 1 → level 2
    if num_nodes > 5:
        adjacency[2, 5 : min(7, num_nodes)] = 1.0
    if num_nodes > 7:
        adjacency[3, 7 : min(10, num_nodes)] = 1.0

    # Add some random edges for realism (10% edge density)
    edge_density = 0.1
    num_random_edges = int(num_nodes**2 * edge_density)
    for _ in range(num_random_edges):
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        if src != tgt:
            adjacency[src, tgt] = np.random.uniform(0.1, 1.0)

    # Node features (semantic embeddings, 64-dim)
    node_features = np.random.randn(num_nodes, 64).astype(np.float32)
    node_features = node_features / (np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8)

    return adjacency, node_features


def generate_vc_features(seed: int = 0) -> np.ndarray:
    """
    Generate synthetic Visual/Conceptual (VC) features.

    Properties:
    - CNN feature maps from final layer (7x7x512 from ResNet-like architecture)
    - Spatial structure (nearby pixels have correlated features)
    - Sparsity (many near-zero activations)

    Args:
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of shape (7, 7, 512), float32
    """
    np.random.seed(seed)

    # Base: sparse random features (10% activation)
    vc_features = np.random.randn(7, 7, 512).astype(np.float32)

    # Add sparsity (ReLU-like, values < 0 become 0)
    vc_features = np.maximum(vc_features, 0)

    # Apply spatial smoothing (nearby pixels correlate)
    from scipy.ndimage import gaussian_filter

    for c in range(512):
        vc_features[:, :, c] = gaussian_filter(vc_features[:, :, c], sigma=0.5)

    # Normalize per-channel
    vc_features = vc_features / (np.max(vc_features, axis=(0, 1), keepdims=True) + 1e-8)

    return vc_features


def generate_sample_episodes(num_episodes: int = 10, seed: int = 42) -> dict[str, np.ndarray]:
    """
    Generate a batch of sample encoder outputs for multiple episodes.

    Args:
        num_episodes: Number of different episodes to generate
        seed: Random seed

    Returns:
        Dictionary with keys:
        - 'vs_embeddings': (num_episodes, 384)
        - 'hg_adjacencies': List of (N_i, N_i) matrices
        - 'hg_node_features': List of (N_i, 64) matrices
        - 'vc_features': (num_episodes, 7, 7, 512)
    """
    np.random.seed(seed)

    vs_embeddings = np.array(
        [generate_vs_embedding(seed=seed + i) for i in range(num_episodes)], dtype=np.float32
    )

    hg_adjacencies = [generate_hg_graph(seed=seed + i)[0] for i in range(num_episodes)]

    hg_node_features = [generate_hg_graph(seed=seed + i)[1] for i in range(num_episodes)]

    vc_features = np.array(
        [generate_vc_features(seed=seed + i) for i in range(num_episodes)], dtype=np.float32
    )

    return {
        "vs_embeddings": vs_embeddings,
        "hg_adjacencies": hg_adjacencies,
        "hg_node_features": hg_node_features,
        "vc_features": vc_features,
    }


def save_sample_data(output_dir: str = "/tmp/selector_sample_data"):
    """
    Generate and save sample data to disk.

    Args:
        output_dir: Directory to save numpy files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating sample data for {10} episodes...")
    samples = generate_sample_episodes(num_episodes=10)

    # Save VS embeddings
    np.save(f"{output_dir}/vs_embeddings.npy", samples["vs_embeddings"])
    print(f"✓ Saved VS embeddings: shape {samples['vs_embeddings'].shape}")

    # Save HG data (as separate files since variable size)
    hg_dir = f"{output_dir}/hg_data"
    os.makedirs(hg_dir, exist_ok=True)
    for i, (adj, feat) in enumerate(zip(samples["hg_adjacencies"], samples["hg_node_features"])):
        np.save(f"{hg_dir}/adjacency_{i}.npy", adj)
        np.save(f"{hg_dir}/node_features_{i}.npy", feat)
    print(f"✓ Saved HG data: {len(samples['hg_adjacencies'])} graphs")

    # Save VC features
    np.save(f"{output_dir}/vc_features.npy", samples["vc_features"])
    print(f"✓ Saved VC features: shape {samples['vc_features'].shape}")

    print(f"\nAll data saved to: {output_dir}")
    return output_dir


def demonstrate_features():
    """
    Print sample statistics about generated encoder outputs.
    """
    print("\n" + "=" * 60)
    print("SAMPLE ENCODER OUTPUT STATISTICS")
    print("=" * 60)

    # VS Embedding
    vs = generate_vs_embedding()
    print("\n✓ VS Embedding (Vector Semantic):")
    print(f"  Shape: {vs.shape}")
    print(f"  L2 Norm: {np.linalg.norm(vs):.4f} (should be ~1.0)")
    print(f"  Mean: {vs.mean():.4f}, Std: {vs.std():.4f}")
    print(f"  Min: {vs.min():.4f}, Max: {vs.max():.4f}")
    print(f"  Sparsity (|x| < 0.1): {(np.abs(vs) < 0.1).sum() / vs.size:.1%}")

    # HG Graph
    adj, feat = generate_hg_graph()
    print("\n✓ HG Graph (Hierarchical Graph):")
    print(f"  Adjacency shape: {adj.shape}")
    print(f"  Edge density: {(adj > 0).sum() / adj.size:.1%}")
    print(f"  Node features shape: {feat.shape}")
    print(f"  Node count: {adj.shape[0]}")
    print(f"  Max path length: ~{int(np.sqrt(adj.shape[0]))}")

    # VC Features
    vc = generate_vc_features()
    print("\n✓ VC Features (Visual/Conceptual):")
    print(f"  Shape: {vc.shape}")
    print(f"  Activation (max > 0.1): {(vc > 0.1).sum() / vc.size:.1%}")
    print(f"  Mean: {vc.mean():.4f}, Std: {vc.std():.4f}")
    print(f"  Min: {vc.min():.4f}, Max: {vc.max():.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Generate sample data
    demonstrate_features()
    save_sample_data()

    # Also generate a batch for feature testing
    print("\n" + "=" * 60)
    print("BATCH GENERATION (10 episodes)")
    print("=" * 60)
    samples = generate_sample_episodes(num_episodes=10)
    print(f"\n✓ VS embeddings batch: {samples['vs_embeddings'].shape}")
    print(f"✓ HG graphs: {len(samples['hg_adjacencies'])} instances")
    print(f"✓ VC features batch: {samples['vc_features'].shape}")
    print("\nReady for feature engineering!")
