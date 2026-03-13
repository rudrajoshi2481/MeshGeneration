"""
preprocessing.py — Mesh loading, normalization, geometric feature extraction,
query point sampling, and disk caching. No open3d dependency.
"""

import os
import hashlib
import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from typing import Tuple, Optional, Dict


MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio",
    "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(MODELNET40_CLASSES)}


# ---------------------------------------------------------------------------
# Geometric feature computation (scipy-based, no open3d)
# ---------------------------------------------------------------------------

def compute_normals_curvature(points: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-point normals and curvature using PCA of local neighborhoods.
    Returns:
        normals    [N, 3]  — unit normals (consistent orientation not guaranteed)
        curvature  [N, 1]  — planarity-based curvature in [0, 1]
    """
    N = len(points)
    normals = np.zeros((N, 3), dtype=np.float32)
    curvature = np.zeros(N, dtype=np.float32)

    tree = KDTree(points)
    _, idx = tree.query(points, k=k + 1)  # first neighbor is self

    for i in range(N):
        neighbors = points[idx[i, 1:]]        # [k, 3]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / k       # [3, 3]
        eigvals, eigvecs = np.linalg.eigh(cov)
        # smallest eigenvector = normal
        normals[i] = eigvecs[:, 0]
        # curvature = smallest / sum of eigenvalues
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-8)

    # Flip normals toward centroid (rough consistency)
    center = points.mean(axis=0)
    dirs = center - points                    # [N, 3]
    flip = (normals * dirs).sum(axis=1) < 0
    normals[flip] *= -1

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    return normals, curvature.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Mesh loading and normalization
# ---------------------------------------------------------------------------

def load_and_normalize_mesh(path: str) -> Optional[trimesh.Trimesh]:
    """Load .ply mesh and normalize to [-0.25, 0.25]^3."""
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "geometry"):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            else:
                return None
        if len(mesh.vertices) < 10 or len(mesh.faces) < 4:
            return None
        # Normalize
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.vertices -= center
        scale = np.max(mesh.bounds[1] - mesh.bounds[0])
        if scale < 1e-6:
            return None
        mesh.vertices /= scale    # [-0.5, 0.5]
        return mesh
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Surface point sampling
# ---------------------------------------------------------------------------

def sample_surface_points(mesh: trimesh.Trimesh, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample n points from mesh surface, compute normals + curvature.
    Returns:
        points     [n, 3]
        normals    [n, 3]
        curvature  [n, 1]
    """
    points, _ = trimesh.sample.sample_surface(mesh, n)
    points = points.astype(np.float32)
    normals, curvature = compute_normals_curvature(points, k=min(15, n - 1))
    return points, normals, curvature


# ---------------------------------------------------------------------------
# Query point sampling (Balanced Surface Sampling)
# ---------------------------------------------------------------------------

def compute_occupancy(mesh: trimesh.Trimesh, query: np.ndarray) -> np.ndarray:
    """
    Compute signed occupancy labels for query points using scipy KDTree.
    Works on both watertight and non-watertight meshes — no rtree needed.

    Method:
      For each query point find closest vertex, then use dot product of
      (query - vertex) with vertex normal to decide inside/outside.
      Points whose dot < 0 are "inside" (behind surface normal).
    Returns float32 array in {0, 1}.
    """
    try:
        if mesh.is_watertight:
            return mesh.contains(query).astype(np.float32)
    except Exception:
        pass

    # Pure scipy / numpy approach: nearest-vertex signed distance
    try:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)    # [V, 3]
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)  # [V, 3]

        tree = KDTree(vertices)
        _, nn_idx = tree.query(query, k=1)                        # [M]
        nn_idx = nn_idx.flatten()

        closest_v = vertices[nn_idx]                              # [M, 3]
        closest_n = vertex_normals[nn_idx]                        # [M, 3]

        dirs = query - closest_v                                   # [M, 3]
        dot = (dirs * closest_n).sum(axis=1)                      # [M]

        # inside = dot < 0 (query is behind the surface normal)
        occ = (dot < 0).astype(np.float32)
        return occ
    except Exception:
        # Ultimate fallback: all near-surface points are "occupied"
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        tree = KDTree(vertices)
        dists, _ = tree.query(query, k=1)
        return (dists.flatten() < 0.03).astype(np.float32)


def sample_query_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    sigma_near: float = 0.05,
    sigma_fine: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced surface sampling:
        50% near-surface (σ=0.05)
        25% fine-surface (σ=0.01)
        25% uniform in [-0.5, 0.5]^3
    Returns:
        query_pts  [M, 3]
        occupancy  [M]    float32 in {0, 1}
    """
    n_near = int(num_points * 0.50)
    n_fine = int(num_points * 0.25)
    n_unif = num_points - n_near - n_fine

    surf, _ = trimesh.sample.sample_surface(mesh, n_near + n_fine)
    near = surf[:n_near] + np.random.normal(0, sigma_near, (n_near, 3))
    fine = surf[n_near:] + np.random.normal(0, sigma_fine, (n_fine, 3))
    unif = np.random.uniform(-0.5, 0.5, (n_unif, 3))  # [-0.5, 0.5] matches mesh range

    query = np.concatenate([near, fine, unif], axis=0).astype(np.float32)
    occ = compute_occupancy(mesh, query)
    return query, occ


# ---------------------------------------------------------------------------
# Contrastive augmentation
# ---------------------------------------------------------------------------

def augment_points(
    points: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    scale_range: float = 0.1,
    jitter_sigma: float = 0.01,
    dropout: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random augmentation: rotation + scale + jitter + dropout."""
    N = len(points)

    # Random SO(3) rotation
    R = random_rotation()
    pts_aug = (points @ R.T).astype(np.float32)
    nrm_aug = (normals @ R.T).astype(np.float32)
    nrm_aug = nrm_aug / (np.linalg.norm(nrm_aug, axis=1, keepdims=True) + 1e-8)

    # Scale
    s = 1.0 + np.random.uniform(-scale_range, scale_range)
    pts_aug = pts_aug * s

    # Jitter
    pts_aug = pts_aug + np.random.normal(0, jitter_sigma, pts_aug.shape).astype(np.float32)

    # Dropout
    keep = np.random.rand(N) > dropout
    if keep.sum() < 10:
        keep[:] = True
    pts_aug = pts_aug[keep]
    nrm_aug = nrm_aug[keep]
    cur_aug = curvature[keep]

    # Resample to original size
    if len(pts_aug) < N:
        idx = np.random.choice(len(pts_aug), N, replace=True)
        pts_aug = pts_aug[idx]
        nrm_aug = nrm_aug[idx]
        cur_aug = cur_aug[idx]

    return pts_aug, nrm_aug, cur_aug


def random_rotation() -> np.ndarray:
    """Sample a uniformly random rotation matrix via QR decomposition."""
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def load_or_compute(
    mesh_path: str,
    cache_dir: str,
    num_surface: int = 2048,
    num_query: int = 2048,
) -> Optional[Dict]:
    """
    Returns a dict with keys:
        points, normals, curvature   — [N, 3/3/1] float32 np arrays
        query_pts, occupancy         — [M, 3] / [M] float32
        label                        — int
    Returns None if mesh loading fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(f"{mesh_path}_{num_surface}_{num_query}")
    cache_path = os.path.join(cache_dir, f"{key}.pt")

    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            return data
        except Exception:
            pass

    # Determine label from filename
    fname = os.path.basename(mesh_path)
    class_name = "_".join(fname.split("_")[:-1])
    label = CLASS_TO_IDX.get(class_name, -1)
    if label == -1:
        return None

    mesh = load_and_normalize_mesh(mesh_path)
    if mesh is None:
        return None

    points, normals, curvature = sample_surface_points(mesh, num_surface)
    query_pts, occupancy = sample_query_points(mesh, num_query)

    data = {
        "points": torch.from_numpy(points),
        "normals": torch.from_numpy(normals),
        "curvature": torch.from_numpy(curvature),
        "query_pts": torch.from_numpy(query_pts),
        "occupancy": torch.from_numpy(occupancy),
        "label": label,
    }

    try:
        torch.save(data, cache_path)
    except Exception:
        pass

    return data
