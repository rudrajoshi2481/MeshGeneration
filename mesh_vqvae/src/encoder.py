"""
encoder.py — PointNet patch encoder → scatter to 3D grid → optional 3D conv downsampler.
Input: [B, N, 7] (XYZ + normals + curvature)
Output: grid_feat [B, C, R, R, R], grid_mask [B, R, R, R]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EncoderConfig


def scatter_mean_pt(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Pure PyTorch scatter mean: src [N, D], idx [N] → [dim_size, D]."""
    D = src.shape[1] if src.dim() > 1 else 1
    out = torch.zeros(dim_size, D, device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    src_ = src.reshape(-1, D)
    out.index_add_(0, idx, src_)
    count.index_add_(0, idx, torch.ones(len(idx), 1, device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1)


class MiniPointNet(nn.Module):
    """Lightweight per-patch PointNet: [B, P, k, in_ch] → [B, P, out_dim]"""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, k, C]
        x = self.mlp(x)          # [B, P, k, D]
        x = x.max(dim=2).values  # [B, P, D]  max-pool over neighborhood
        return x


class FPS(nn.Module):
    """Farthest Point Sampling — returns indices [B, n_patches]"""

    @staticmethod
    def forward(points: torch.Tensor, n: int) -> torch.Tensor:
        B, N, _ = points.shape
        device = points.device
        centroids = torch.zeros(B, n, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), device=device)
        for i in range(n):
            centroids[:, i] = farthest
            centroid = points[torch.arange(B), farthest].unsqueeze(1)  # [B,1,3]
            dist = ((points - centroid) ** 2).sum(-1)                  # [B, N]
            distances = torch.minimum(distances, dist)
            farthest = distances.argmax(dim=1)
        return centroids


def ball_query_knn(points: torch.Tensor, centroids_idx: torch.Tensor, k: int) -> torch.Tensor:
    """Return [B, P, k] indices — k nearest neighbors around each centroid."""
    B, N, _ = points.shape
    P = centroids_idx.shape[1]
    anchors = points[torch.arange(B).unsqueeze(1), centroids_idx]  # [B,P,3]
    dists = torch.cdist(anchors, points)                            # [B,P,N]
    _, idx = dists.topk(k, dim=2, largest=False)                    # [B,P,k]
    return idx


class PointGridEncoder(nn.Module):
    """
    Encode a point cloud [B, N, 7] to a 3D feature grid [B, C, R, R, R].
    Steps:
        1. FPS to get P anchors
        2. k-NN grouping around each anchor
        3. MiniPointNet per patch → [B, P, D]
        4. scatter_mean patch features onto R^3 voxel grid
        5. Optional 3D conv downsampler
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        R = cfg.grid_res
        C = cfg.grid_channels

        self.patch_net = MiniPointNet(cfg.in_channels, cfg.patch_dim)

        # Project patch dim → grid channels
        self.proj = nn.Linear(cfg.patch_dim, C)

        # 3D conv downsampler: keeps resolution but deepens features
        layers = []
        for _ in range(cfg.conv_layers):
            layers += [
                nn.Conv3d(C, C, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(C),
                nn.GELU(),
            ]
        self.conv3d = nn.Sequential(*layers) if layers else nn.Identity()

        self.grid_res = R
        self.grid_channels = C

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N, 7]   surface points with geometric features
        Returns:
            grid_feat: [B, C, R, R, R]
            grid_mask: [B, R, R, R]   bool, True where occupied
        """
        B, N, _ = x.shape
        R = self.grid_res
        device = x.device

        xyz = x[..., :3]  # [B, N, 3]

        # 1. FPS anchors
        P = self.cfg.num_patches
        anchor_idx = FPS.forward(xyz, min(P, N))   # [B, P]

        # 2. k-NN grouping
        k = min(self.cfg.patch_size, N - 1)
        nn_idx = ball_query_knn(xyz, anchor_idx, k)  # [B, P, k]

        # 3. Gather neighbors and compute relative features
        #    features: [B, P, k, 7]  (absolute xyz already encoded)
        nn_idx_flat = nn_idx.reshape(B, -1)          # [B, P*k]
        pts_gathered = x[
            torch.arange(B, device=device).unsqueeze(1), nn_idx_flat
        ].reshape(B, P, k, -1)                        # [B, P, k, 7]

        # Subtract anchor xyz for relative position
        anchor_pts = xyz[torch.arange(B, device=device).unsqueeze(1), anchor_idx]  # [B,P,3]
        pts_gathered[..., :3] = pts_gathered[..., :3] - anchor_pts.unsqueeze(2)

        # 4. MiniPointNet per patch
        patch_feats = self.patch_net(pts_gathered)   # [B, P, patch_dim]
        patch_feats = self.proj(patch_feats)          # [B, P, C]

        # 5. Scatter onto voxel grid
        #    Map anchor positions in [-0.5, 0.5] to [0, R-1]
        vox_coords = ((anchor_pts + 0.5) / 1.0 * (R - 1)).long()
        vox_coords = vox_coords.clamp(0, R - 1)      # [B, P, 3]

        # Flatten spatial index: x*R^2 + y*R + z
        vox_flat = (vox_coords[..., 0] * R * R +
                    vox_coords[..., 1] * R +
                    vox_coords[..., 2])                # [B, P]

        # Per-sample offset so scatter treats each batch independently
        batch_offset = torch.arange(B, device=device).unsqueeze(1) * (R ** 3)
        vox_flat_global = (vox_flat + batch_offset).reshape(-1)  # [B*P]

        feats_flat = patch_feats.reshape(B * P, -1)   # [B*P, C]
        total_voxels = B * (R ** 3)
        grid_flat = scatter_mean_pt(feats_flat, vox_flat_global, dim_size=total_voxels)
        # [B*R^3, C]  → [B, R, R, R, C] → [B, C, R, R, R]
        grid = grid_flat.reshape(B, R, R, R, -1).permute(0, 4, 1, 2, 3).contiguous()

        # 6. Binary occupancy mask (which voxels received any points)
        ones = torch.ones(B * P, 1, device=device)
        grid_count = scatter_mean_pt(ones, vox_flat_global, dim_size=total_voxels)
        grid_mask = (grid_count.squeeze(1) > 0).reshape(B, R, R, R)

        # 7. 3D conv (deepens features in occupied cells)
        grid = self.conv3d(grid)                      # [B, C, R, R, R]

        return grid, grid_mask
