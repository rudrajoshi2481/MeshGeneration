"""
train.py — Full training entry point using PyTorch Lightning + DDP on 8x A100.
Usage:
    python train.py --model small   # small model
    python train.py --model large   # large model
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SmallModelConfig, LargeModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset, build_dataloaders
from preprocessing import MODELNET40_CLASSES


class DashboardCallback(Callback):
    """Generates a full plot dashboard + reconstruction + eval after every epoch."""

    def __init__(self, cfg, run_dir, every_n_val=1):
        super().__init__()
        self.cfg       = cfg
        self.run_dir   = run_dir
        self.plot_dir  = os.path.join(run_dir, "plots")
        self.every_n   = every_n_val
        self._val_count = 0
        self._metric_history = {"epoch": [], "iou": [], "cls_acc": [],
                                 "recon": [], "vq": [], "cls_loss": [], "fp": [],
                                 "codebook_util": [], "perplexity": []}
        os.makedirs(self.plot_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        self._val_count += 1
        if self._val_count % self.every_n != 0:
            return
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        step  = trainer.global_step
        print(f"\n[Dashboard] Generating plots @ epoch={epoch} step={step}...")
        try:
            self._generate(pl_module, trainer, epoch, step)
        except Exception as e:
            import traceback
            print(f"[Dashboard] WARNING: {e}\n{traceback.format_exc()}")

    def _collect_samples(self, model, device, n=200, seed=42, epoch=0):
        """Run inference on n val samples, return collected arrays."""
        cfg = self.cfg
        ds = ModelNet40Dataset(
            cfg.train.data_dir, cfg.train.cache_dir, split="val",
            num_surface=cfg.train.num_surface_points,
            num_query=cfg.train.num_query_points,
            use_augmentation=False, use_contrastive=False,
        )
        # CRITICAL: Use completely separate RNG to avoid Lightning's global seed interference
        import random
        rng = np.random.RandomState(seed + epoch * 1000 + random.randint(0, 10000))
        idxs = rng.choice(len(ds), min(n, len(ds)), replace=False)
        K      = model.quantizer.num_embeddings

        ious, cls_ok, codes_list, fps_list, labels_list, recon_ex = [], [], [], [], [], []

        # CRITICAL: Force model into eval mode and disable gradients
        model.eval()
        with torch.no_grad():
            for i in idxs:
                item = ds[int(i)]
                b = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in item.items()}
                b["label"] = torch.tensor([item["label"]], device=device)
                try:
                    out = model(b["points"], b["normals"], b["curvature"],
                                b["query_pts"], b["label"])
                    occ   = b["occupancy"]
                    prob  = out["logits"].sigmoid()
                    pred  = (prob > 0.5).float()
                    inter = (pred * occ).sum(1)
                    union = ((pred + occ) > 0).float().sum(1)
                    iou   = (inter / (union + 1e-8)).mean().item()
                    ious.append(iou)
                    cls_ok.append(int(out["class_logits"].argmax(1).item() == item["label"]))
                    codes_list.append(out["code_indices"].cpu().squeeze(0))
                    fps_list.append(out["fingerprint"].cpu().squeeze(0))
                    labels_list.append(item["label"])
                    if len(recon_ex) < 8:
                        recon_ex.append({
                            "label": MODELNET40_CLASSES[item["label"]],
                            "pts":   b["query_pts"].cpu().squeeze(0).numpy(),
                            "gt":    occ.cpu().squeeze(0).numpy(),
                            "pred":  prob.cpu().squeeze(0).numpy(),
                            "iou":   iou,
                        })
                except Exception:
                    pass

        if not ious:
            return None

        ious       = np.array(ious)
        cls_arr    = np.array(cls_ok)
        codes_flat = torch.stack(codes_list).reshape(-1).numpy()
        counts     = np.bincount(codes_flat, minlength=K)
        util       = (counts > 0).sum() / K
        perp       = np.exp(-np.sum((counts / counts.sum() + 1e-8) *
                                    np.log(counts / counts.sum() + 1e-8)))
        labels_arr = np.array(labels_list)
        fps_arr    = torch.stack(fps_list).numpy()

        return dict(ious=ious, cls_arr=cls_arr, counts=counts, util=util,
                    perp=perp, labels_arr=labels_arr, fps_arr=fps_arr,
                    recon_ex=recon_ex, K=K)

    def _generate(self, model, trainer, epoch, step):
        model.eval()
        device = next(model.parameters()).device

        data = self._collect_samples(model, device, n=200, seed=42, epoch=epoch)
        if data is None:
            model.train()
            return

        ious       = data["ious"]
        cls_arr    = data["cls_arr"]
        counts     = data["counts"]
        util       = data["util"]
        perp       = data["perp"]
        labels_arr = data["labels_arr"]
        fps_arr    = data["fps_arr"]
        recon_ex   = data["recon_ex"]
        K          = data["K"]

        # Pull logged metrics from trainer for history
        logged = trainer.logged_metrics
        self._metric_history["epoch"].append(epoch)
        self._metric_history["iou"].append(ious.mean())
        self._metric_history["cls_acc"].append(cls_arr.mean())
        self._metric_history["codebook_util"].append(util)
        self._metric_history["perplexity"].append(perp)
        self._metric_history["recon"].append(float(logged.get("train/recon", 0)))
        self._metric_history["vq"].append(float(logged.get("train/vq", 0)))
        self._metric_history["cls_loss"].append(float(logged.get("train/cls", 0)))
        self._metric_history["fp"].append(float(logged.get("train/fp", 0)))

        tag      = f"ep{epoch:04d}"
        eval_dir = os.path.join(self.run_dir, "eval", tag)
        os.makedirs(eval_dir, exist_ok=True)

        # ── Fig 1: Main Dashboard (loss curves + metrics) ─────────────────
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f"Training Dashboard  |  epoch={epoch}  step={step}  "
                     f"IoU={ious.mean():.4f}  Cls={cls_arr.mean()*100:.1f}%  Util={util*100:.1f}%",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

        hist_e = self._metric_history["epoch"]

        # Row 0: Loss curves
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(hist_e, self._metric_history["recon"], label="recon", color="steelblue")
        ax.plot(hist_e, self._metric_history["vq"],    label="vq×10",
                color="orange", alpha=0.8)
        ax.set_title("Reconstruction & VQ Loss"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 2:])
        ax.plot(hist_e, self._metric_history["cls_loss"], label="cls", color="crimson")
        ax.plot(hist_e, self._metric_history["fp"],       label="fp",  color="purple", alpha=0.8)
        ax.set_title("Classifier & Fingerprint Loss"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Row 1: IoU + Cls Acc history, IoU hist, Codebook usage
        ax = fig.add_subplot(gs[1, :1])
        ax.plot(hist_e, self._metric_history["iou"], color="#2ecc71", linewidth=2)
        ax.axhline(0.38, color="red", linestyle="--", alpha=0.6, label="target 0.38")
        ax.set_title("Val IoU Over Time"); ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.0); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 1:2])
        ax.plot(hist_e, np.array(self._metric_history["cls_acc"]) * 100,
                color="#3498db", linewidth=2)
        ax.axhline(2.5, color="gray", linestyle=":", alpha=0.5, label="random 2.5%")
        ax.set_title("Cls Accuracy % Over Time"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 2:3])
        ax.hist(ious, bins=20, color="steelblue", edgecolor="white")
        ax.axvline(ious.mean(), color="red", linestyle="--", label=f"mean={ious.mean():.3f}")
        ax.axvline(0.38, color="green", linestyle=":", label="target")
        ax.set_title("IoU Distribution"); ax.set_xlabel("IoU")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 3:])
        dead = (counts == 0).sum()
        ax.bar(np.arange(K), np.log1p(counts), width=1.0, linewidth=0, color="coral")
        ax.set_title(f"Codebook Usage (log-scale)  —  {dead} dead / {K} codes")
        ax.set_xlabel("Code Index"); ax.grid(axis="y", alpha=0.3)

        # Row 2: Per-class IoU
        unique = sorted(np.unique(labels_arr))
        cls_iou_vals = [ious[labels_arr == l].mean() if (labels_arr == l).any() else 0 for l in unique]
        cls_names    = [MODELNET40_CLASSES[l][:8] for l in unique]
        order        = np.argsort(cls_iou_vals)[::-1]
        ax = fig.add_subplot(gs[2, :])
        colors = ["#2ecc71" if v > 0.4 else ("#f39c12" if v > 0.25 else "#e74c3c")
                  for v in [cls_iou_vals[i] for i in order]]
        ax.barh([cls_names[i] for i in order], [cls_iou_vals[i] for i in order], color=colors)
        ax.axvline(ious.mean(), color="navy", linestyle="--", label=f"mean={ious.mean():.3f}")
        ax.set_title(f"Per-Class IoU  (green>0.4 / orange>0.25 / red<0.25)")
        ax.set_xlabel("IoU"); ax.set_xlim(0, 1.0); ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

        dash_path = os.path.join(eval_dir, "dashboard.png")
        plt.savefig(dash_path, dpi=120, bbox_inches="tight")
        plt.close()

        # ── Fig 2: Reconstruction — 8 objects × 6 views ──────────────────
        n_show = len(recon_ex)
        if n_show > 0:
            fig2, axes = plt.subplots(n_show, 6, figsize=(20, 3.2 * n_show))
            if n_show == 1:
                axes = axes[None, :]
            # Calculate average threshold for subtitle
            avg_thresh = 0.3  # Fixed low threshold to ensure visibility
            fig2.suptitle(f"Reconstruction  |  epoch={epoch}  "
                          f"(columns: GT-XY, Pred-XY, GT-XZ, Pred-XZ, GT-YZ, Error-YZ)\n"
                          f"Prediction threshold: {avg_thresh:.2f} (fixed for visibility)",
                          fontsize=11, fontweight="bold")
            for row, d in enumerate(recon_ex):
                pts  = d["pts"]
                gt   = d["gt"]
                pred = d["pred"]
                # Fixed threshold for visibility
                pred_thresh = 0.3
                inside_gt = pts[gt > 0.5]
                inside_p  = pts[pred > pred_thresh]
                err = np.abs((pred > 0.5).astype(float) - gt)
                n_gt   = len(inside_gt)
                n_pred = len(inside_p)
                class_name = d['label'].upper()

                views = [
                    (inside_gt, 0, 1, f"{class_name}\nGT-XY  ({n_gt}pts)",   "blue"),
                    (inside_p,  0, 1, f"Pred-XY ({n_pred}pts)\nIoU={d['iou']:.3f}", "red"),
                    (inside_gt, 0, 2, f"{class_name}\nGT-XZ",   "blue"),
                    (inside_p,  0, 2, f"Pred-XZ",  "red"),
                    (inside_gt, 1, 2, f"{class_name}\nGT-YZ",   "blue"),
                ]
                for col, (pts_v, xi, yi, title, color) in enumerate(views):
                    ax = axes[row, col]
                    if len(pts_v):
                        ax.scatter(pts_v[:, xi], pts_v[:, yi], c=color, s=1.5, alpha=0.6)
                    ax.set_title(title, fontsize=6, pad=2)
                    ax.set_aspect("equal"); ax.axis("off")
                # Error heatmap (col 5)
                ax = axes[row, 5]
                sc = ax.scatter(pts[:, 1], pts[:, 2], c=err, cmap="RdYlGn_r",
                                s=1.5, alpha=0.7, vmin=0, vmax=1)
                ax.set_title(f"Error-YZ\nred=wrong", fontsize=6, pad=2)
                ax.set_aspect("equal"); ax.axis("off")
                if row == 0:
                    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            recon_path = os.path.join(eval_dir, "reconstruction.png")
            plt.savefig(recon_path, dpi=130, bbox_inches="tight")
            plt.close()

        # ── Fig 3: t-SNE of fingerprints ─────────────────────────────────
        try:
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, random_state=42,
                       perplexity=min(30, len(fps_arr)-1)).fit_transform(fps_arr)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            cmap = matplotlib.colormaps.get_cmap("tab20")
            for lbl in np.unique(labels_arr):
                m = labels_arr == lbl
                ax3.scatter(emb[m, 0], emb[m, 1], c=[cmap(lbl % 20)],
                            label=MODELNET40_CLASSES[lbl], s=15, alpha=0.8)
            ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=2)
            ax3.set_title(f"Fingerprint t-SNE  |  epoch={epoch}")
            fig3.tight_layout()
            plt.savefig(os.path.join(eval_dir, "tsne.png"), dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[Dashboard] t-SNE skipped: {e}")

        print(f"[Dashboard] IoU={ious.mean():.4f} util={util:.3f} "
              f"cls={cls_arr.mean():.3f} dead={dead} → {eval_dir}/")
        model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "small":
        cfg = SmallModelConfig()
    else:
        cfg = LargeModelConfig()

    cfg.train.num_gpus = args.gpus
    
    # Override output_dir if provided
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "logs"), exist_ok=True)
    os.makedirs(cfg.train.cache_dir, exist_ok=True)

    pl.seed_everything(cfg.train.seed)

    # Build data
    train_loader, val_loader = build_dataloaders(cfg.train, num_gpus=args.gpus)

    # Build model
    model = MaskedVQVAE3D(cfg)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] {args.model.upper()} — {num_params:.2f}M parameters")

    # Callbacks
    run_dir = cfg.train.output_dir
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename=f"{args.model}-{{epoch:04d}}-{{val/iou:.4f}}-{{val/cls_acc:.4f}}",
        monitor="val/recon",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    lr_cb   = LearningRateMonitor(logging_interval="step")
    dash_cb = DashboardCallback(cfg, run_dir, every_n_val=1)
    early_cb = EarlyStopping(
        monitor="val/recon",
        mode="min",
        patience=30,
        min_delta=0.001,
        verbose=True,
    )

    logger = CSVLogger(
        save_dir=cfg.train.output_dir,
        name=f"logs_{args.model}",
    )

    from pytorch_lightning.strategies import DDPStrategy

    # Trainer — 8x A100, bfloat16, DDP
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16",
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=min(cfg.train.log_every, 10),
        check_val_every_n_epoch=1,
        callbacks=[ckpt_cb, lr_cb, dash_cb, early_cb],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=cfg.train.output_dir,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    print("[DONE] Training complete.")
    print(f"[DONE] Checkpoints saved to: {os.path.join(cfg.train.output_dir, 'checkpoints')}")


if __name__ == "__main__":
    main()
