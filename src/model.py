"""
model.py — MaskedVQVAE3D: the full pipeline assembled.

Pipeline order (from plan):
    Encoder → Masker scores → VQ all positions → Classifier + Fingerprint heads
    → Masked selection → De-Masker → Decoder → Losses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional

from config import SmallModelConfig, LargeModelConfig
from encoder import PointGridEncoder
from masker import GeometricMasker3D
from quantizer import VQEmbeddingEMA
from demasker import VolumetricDemasker3D
from decoder import OccupancyDecoder
from heads import CategoryClassifierHead, GeometricFingerprintHead, info_nce_loss


class MaskedVQVAE3D(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        R = cfg.encoder.grid_res
        C = cfg.encoder.grid_channels
        N = R ** 3

        self.encoder = PointGridEncoder(cfg.encoder)
        self.masker = GeometricMasker3D(cfg.masker)
        self.quantizer = VQEmbeddingEMA(cfg.vq)
        self.demasker = VolumetricDemasker3D(cfg.demasker, grid_size=R)
        self.decoder = OccupancyDecoder(cfg.decoder, grid_res=R)
        self.classifier = CategoryClassifierHead(cfg.classifier)
        self.fingerprint = GeometricFingerprintHead(cfg.fingerprint)

        self.beta = cfg.train.vq_beta
        self.cls_weight_max = cfg.train.cls_weight_max
        self.cls_ramp = cfg.train.cls_ramp_steps
        self.fp_weight_max = cfg.train.fp_weight_max
        self.fp_ramp = cfg.train.fp_ramp_steps
        self._step = 0

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def encode_and_quantize(self, Xbd: torch.Tensor, normals: torch.Tensor,
                             curvature: torch.Tensor, labels: torch.Tensor = None):
        """
        Returns the masker output dict and quantization results.
        """
        encoder_input = torch.cat([Xbd, normals, curvature], dim=-1)  # [B, N, 7]
        grid_feat, grid_mask = self.encoder(encoder_input)

        masker_out = self.masker(grid_feat, grid_mask, labels=labels)

        # Single VQ pass on ALL positions
        all_feat = masker_out["all_features"]  # [B, N_vox, d_q]
        B, N_vox, d_q = all_feat.shape
        all_flat = all_feat.reshape(-1, d_q)

        quant_flat, vq_loss, idx_flat = self.quantizer(all_flat)

        quant_all = quant_flat.reshape(B, N_vox, d_q)
        code_all = idx_flat.reshape(B, N_vox)

        # Kept positions (from masker top-k selection)
        kept_idx = masker_out["sample_index"]  # [B, K]
        quant_kept = quant_all.gather(
            1, kept_idx.unsqueeze(-1).expand(-1, -1, d_q)
        )  # [B, K, d_q]

        return {
            "masker_out": masker_out,
            "quant_all": quant_all,
            "quant_kept": quant_kept,
            "code_all": code_all,
            "vq_loss": vq_loss,
            "grid_mask": grid_mask,
        }

    def forward(self, Xbd, normals, curvature, Xtg, labels=None):
        enc = self.encode_and_quantize(Xbd, normals, curvature, labels)

        masker_out = enc["masker_out"]
        quant_kept = enc["quant_kept"]
        code_all = enc["code_all"]
        vq_loss = enc["vq_loss"]

        # ---- Classifier head (ALL codes + spatial coords) ----
        # Use full voxel grid so classifier sees all spatial structure
        R = self.cfg.encoder.grid_res
        vox_lin = torch.linspace(-0.5, 0.5, R, device=code_all.device)
        gx, gy, gz = torch.meshgrid(vox_lin, vox_lin, vox_lin, indexing='ij')
        voxel_coords = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)  # [1, R³, 3]
        voxel_coords = voxel_coords.expand(code_all.shape[0], -1, -1)        # [B, R³, 3]
        class_logits = self.classifier(code_all, masker_out["score_map"], voxel_coords)

        # ---- Fingerprint head (kept codes) ----
        fp_vec, attn_w = self.fingerprint(quant_kept)

        # ---- De-Masker → Decoder ----
        R = self.cfg.encoder.grid_res
        N_vox = R ** 3
        full_feat = self.demasker(
            quant_kept,
            masker_out["sample_index"],
            masker_out["remain_index"],
            masker_out["grid_mask_flat"],
        )  # [B, N_vox, output_dim]

        recon_logits = self.decoder(full_feat, Xtg)  # [B, M]

        return {
            "logits": recon_logits,
            "vq_loss": vq_loss,
            "class_logits": class_logits,
            "fingerprint": fp_vec,
            "code_indices": code_all,
            "mask_scores": masker_out["score_map"],
        }

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _ramp(self, max_val: float, ramp_steps: int) -> float:
        return min(max_val, max_val * self._step / max(ramp_steps, 1))

    def compute_loss(self, out: Dict, batch: Dict) -> Dict:
        Ytg = batch["occupancy"]                          # [B, M]
        labels = batch["label"]

        pos_weight = torch.tensor(4.0, device=Ytg.device, dtype=Ytg.dtype)  # ~17% occupancy → 4x upweight
        recon_loss = F.binary_cross_entropy_with_logits(out["logits"], Ytg, pos_weight=pos_weight)

        cls_w = self._ramp(self.cls_weight_max, self.cls_ramp)
        cls_loss = F.cross_entropy(out["class_logits"], labels)

        fp_loss = torch.tensor(0.0, device=recon_loss.device)
        fp_w = self._ramp(self.fp_weight_max, self.fp_ramp)
        if "fingerprint_aug" in out and fp_w > 0:
            fp_loss = info_nce_loss(out["fingerprint"], out["fingerprint_aug"])

        total = recon_loss + self.beta * out["vq_loss"] + cls_w * cls_loss + fp_w * fp_loss

        return {
            "total": total,
            "recon": recon_loss,
            "vq": out["vq_loss"],
            "cls": cls_loss,
            "fp": fp_loss,
            "cls_w": cls_w,
            "fp_w": fp_w,
        }

    # ------------------------------------------------------------------
    # Training / Validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        self._step += 1

        out = self.forward(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )

        # Second pass for contrastive fingerprint (augmented view)
        if "points_aug" in batch and self._ramp(self.fp_weight_max, self.fp_ramp) > 0:
            enc_aug = self.encode_and_quantize(
                batch["points_aug"], batch["normals"], batch["curvature"], batch["label"]
            )
            fp_aug, _ = self.fingerprint(enc_aug["quant_kept"])
            out["fingerprint_aug"] = fp_aug

        losses = self.compute_loss(out, batch)

        # Codebook health metrics
        util = self.quantizer.codebook_utilization(out["code_indices"])
        perp = self.quantizer.perplexity(out["code_indices"])

        log = {
            "train/total": losses["total"],
            "train/recon": losses["recon"],
            "train/vq": losses["vq"],
            "train/cls": losses["cls"],
            "train/fp": losses["fp"],
            "train/cls_w": losses["cls_w"],
            "train/fp_w": losses["fp_w"],
            "train/codebook_util": util,
            "train/perplexity": perp,
        }
        self.log_dict(log, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        out = self.forward(
            batch["points"], batch["normals"], batch["curvature"],
            batch["query_pts"], batch["label"],
        )
        losses = self.compute_loss(out, batch)

        # IoU at threshold 0.5
        preds = (out["logits"].sigmoid() > 0.5).float()
        iou = self._batch_iou(preds, batch["occupancy"])

        # Classification accuracy
        acc = (out["class_logits"].argmax(dim=1) == batch["label"]).float().mean()

        util = self.quantizer.codebook_utilization(out["code_indices"])

        self.log_dict({
            "val/total":        losses["total"],
            "val/recon":        losses["recon"],
            "val/cls_loss":     losses["cls"],
            "val/iou":          iou,
            "val/cls_acc":      acc,
            "val/codebook_util": util,
        }, prog_bar=True, sync_dist=True)

    @staticmethod
    def _batch_iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = (preds * targets).sum(dim=1)
        union = ((preds + targets) > 0).float().sum(dim=1)
        return (intersection / (union + 1e-8)).mean()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        base_lr = self.cfg.train.lr
        classifier_lr = self.cfg.train.classifier_lr
        wd = self.cfg.train.weight_decay

        # Separate learning rates: classifier needs much higher LR
        classifier_params = list(self.classifier.parameters())
        classifier_ids = {id(p) for p in classifier_params}
        base_params = [p for p in self.parameters() if id(p) not in classifier_ids]

        param_groups = [
            {"params": base_params,       "lr": base_lr,        "name": "base"},
            {"params": classifier_params, "lr": classifier_lr,  "name": "classifier"},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=wd)

        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < self.cfg.train.warmup_steps:
                return step / max(1, self.cfg.train.warmup_steps)
            progress = (step - self.cfg.train.warmup_steps) / max(
                1, self.cfg.train.max_steps - self.cfg.train.warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
