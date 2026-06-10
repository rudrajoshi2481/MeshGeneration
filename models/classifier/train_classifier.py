"""
train_classifier.py
-------------------
Train a classifier on generated SEDD tokens to verify if class conditioning works.

Usage:
    python train_classifier.py --tokens_path PATH --mode conditional
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))  # models/classifier/
_BASE = os.path.dirname(os.path.dirname(_HERE))      # MeshGeneration/
MESHVQVAE = os.path.join(_BASE, "mesh_vqvae", "src")
sys.path.insert(0, MESHVQVAE)
from preprocessing import MODELNET40_CLASSES

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = ["#5B8DB8", "#F4A35A", "#6DBF8A", "#D96B6B", "#A48CC4"]


class TokenDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens.long()
        self.labels = labels.long()
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return {"tokens": self.tokens[idx], "label": self.labels[idx]}


class TokenClassifier(pl.LightningModule):
    """
    Simple classifier: Token Embedding → Mean Pool → MLP → Class
    """
    def __init__(self, vocab_size=256, seq_len=4096, embed_dim=256, 
                 hidden_dim=512, num_classes=40, lr=1e-3, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for mask
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.train_acc = []
        self.val_acc = []
        self.train_losses = []
        self.val_losses = []
        self.val_preds = []
        self.val_labels = []
        # Temporary storage for per-batch metrics (averaged at epoch end)
        self._epoch_train_accs = []
        self._epoch_train_losses = []
        self._epoch_val_losses = []
    
    def forward(self, tokens):
        # tokens: [B, seq_len]
        embeds = self.embedding(tokens)  # [B, seq_len, embed_dim]
        pooled = embeds.mean(dim=1)      # [B, embed_dim]
        logits = self.mlp(pooled)        # [B, num_classes]
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch["tokens"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        
        # Store for epoch-level averaging
        self._epoch_train_accs.append(acc.item())
        self._epoch_train_losses.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        # Average and store per-epoch metrics
        if len(self._epoch_train_accs) > 0:
            self.train_acc.append(np.mean(self._epoch_train_accs))
            self.train_losses.append(np.mean(self._epoch_train_losses))
            self._epoch_train_accs.clear()
            self._epoch_train_losses.clear()
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["tokens"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        
        # Store for confusion matrix
        self.val_preds.append(logits.argmax(dim=1).cpu())
        self.val_labels.append(batch["label"].cpu())
        self._epoch_val_losses.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds)
            labels = torch.cat(self.val_labels)
            acc = (preds == labels).float().mean().item()
            self.val_acc.append(acc)
            # Store average val loss for this epoch
            if len(self._epoch_val_losses) > 0:
                self.val_losses.append(np.mean(self._epoch_val_losses))
                self._epoch_val_losses.clear()
            self.val_preds.clear()
            self.val_labels.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


def plot_training_curves(train_acc, val_acc, train_losses=None, val_losses=None, save_path="training_curves.png"):
    """Plot comprehensive training curves with both accuracy and loss."""
    has_loss = train_losses is not None and val_losses is not None and len(train_losses) > 0 and len(val_losses) > 0
    
    if has_loss:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax2 = None
    
    # Handle mismatched lengths (train/val may have different number of epochs recorded)
    n_epochs = min(len(train_acc), len(val_acc))
    if len(train_acc) != len(val_acc):
        print(f"[WARN] Mismatched lengths: train_acc={len(train_acc)}, val_acc={len(val_acc)}, using first {n_epochs} epochs")
    
    epochs = np.arange(n_epochs)
    train_acc_plot = train_acc[:n_epochs]
    val_acc_plot = val_acc[:n_epochs]
    
    # Accuracy plot
    ax1.plot(epochs, train_acc_plot, color=PALETTE[0], lw=2, label="train_acc", marker='o', markersize=4)
    ax1.plot(epochs, val_acc_plot, color=PALETTE[1], lw=2, label="val_acc", marker='s', markersize=4)
    ax1.set_xlabel("Epoch", labelpad=8)
    ax1.set_ylabel("Accuracy", labelpad=8)
    ax1.set_title("Token Classifier - Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Loss plot (if available)
    if ax2 is not None:
        n_epochs_loss = min(len(train_losses), len(val_losses))
        epochs_loss = np.arange(n_epochs_loss)
        ax2.plot(epochs_loss, train_losses[:n_epochs_loss], color=PALETTE[0], lw=2, label="train_loss", marker='o', markersize=4)
        ax2.plot(epochs_loss, val_losses[:n_epochs_loss], color=PALETTE[1], lw=2, label="val_loss", marker='s', markersize=4)
        ax2.set_xlabel("Epoch", labelpad=8)
        ax2.set_ylabel("Loss", labelpad=8)
        ax2.set_title("Token Classifier - Loss", fontsize=13, fontweight="bold", pad=10)
        ax2.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
        ax2.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved training curves to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # For 40 classes, show subset or use smaller font
    n_classes = len(class_names)
    figsize = (20, 18) if n_classes > 20 else (14, 12)
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    sns.heatmap(cm_norm, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
    
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold", pad=12)
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved confusion matrix to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path="per_class_accuracy.png"):
    """Plot per-class accuracy bar chart."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    # Sort by accuracy
    sorted_idx = np.argsort(per_class_acc)[::-1]
    sorted_acc = per_class_acc[sorted_idx]
    sorted_names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in sorted_idx]
    
    # Create color map based on accuracy
    colors = [PALETTE[2] if acc > 0.7 else PALETTE[0] if acc > 0.5 else PALETTE[3] for acc in sorted_acc]
    
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    bars = ax.barh(range(len(sorted_acc)), sorted_acc, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(acc + 0.01, i, f"{acc:.2f}", va='center', fontsize=8)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy", fontsize=12, labelpad=8)
    ax.set_ylabel("Class", fontsize=12, labelpad=8)
    ax.set_title("Per-Class Accuracy (Sorted)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim([0, 1.05])
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[2], label='High (>70%)'),
        Patch(facecolor=PALETTE[0], label='Medium (50-70%)'),
        Patch(facecolor=PALETTE[3], label='Low (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.9)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved per-class accuracy to {save_path}")


def generate_classification_report(y_true, y_pred, class_names, save_path="classification_report.txt"):
    """Save detailed classification report to text file."""
    # Get unique labels present in the data
    unique_labels = sorted(set(y_true) | set(y_pred))
    # Filter class_names to only include present labels
    present_class_names = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in unique_labels]
    
    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=present_class_names, digits=3)
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Classification Report - Token Classifier\n")
        f.write("=" * 60 + "\n")
        f.write(f"Classes present: {len(unique_labels)}/{len(class_names)}\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    print(f"[REPORT] Saved classification report to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["conditional", "unconditional", "conditional_no_class", "unconditional_no_class"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Classifier Training — {args.mode}")
    print(f"  Tokens: {args.tokens_path}")
    print(f"{'='*60}\n")
    
    # Load tokens (support both "tokens" and "codes" keys)
    data = torch.load(args.tokens_path, weights_only=False)
    tokens = data.get("tokens", data.get("codes"))
    labels = data["labels"]
    print(f"[INFO] Loaded {len(tokens)} samples, tokens shape: {tuple(tokens.shape)}")
    
    # Create dataset
    dataset = TokenDataset(tokens, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    # Model
    model = TokenClassifier(vocab_size=256, seq_len=4096, lr=args.lr)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"classifier_{args.mode}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_acc", mode="max", save_top_k=1
    )
    early_stop = EarlyStopping(monitor="val_acc", patience=10, mode="max", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Trainer
    strategy = "ddp_find_unused_parameters_false" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy=strategy,
        precision="bf16",
        callbacks=[ckpt_cb, early_stop, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=args.out_dir,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Plot and evaluate
    if trainer.global_rank == 0:
        print("\n" + "="*60)
        print("  Generating comprehensive plots and evaluation...")
        print("="*60)
        
        # 1. Training curves with loss and accuracy
        plot_training_curves(
            model.train_acc, model.val_acc,
            model.train_losses, model.val_losses,
            os.path.join(plot_dir, "training_curves.png")
        )
        
        # 2. Run final evaluation on full validation set for confusion matrix
        print("[INFO] Running final evaluation on validation set...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["tokens"].to(model.device))
                preds = logits.argmax(dim=1).cpu()
                labels = batch["label"]
                all_preds.append(preds)
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # 3. Confusion matrix
        plot_confusion_matrix(
            all_labels, all_preds,
            MODELNET40_CLASSES,
            os.path.join(plot_dir, "confusion_matrix.png")
        )
        
        # 4. Per-class accuracy
        plot_per_class_accuracy(
            all_labels, all_preds,
            MODELNET40_CLASSES,
            os.path.join(plot_dir, "per_class_accuracy.png")
        )
        
        # 5. Classification report (text)
        generate_classification_report(
            all_labels, all_preds,
            MODELNET40_CLASSES,
            os.path.join(args.out_dir, "classification_report.txt")
        )
        
        # 6. Save results
        final_acc = (all_preds == all_labels).mean()
        results = {
            "mode": args.mode,
            "final_val_acc": float(final_acc),
            "best_val_acc": float(max(model.val_acc)) if model.val_acc else 0.0,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "best_ckpt": ckpt_cb.best_model_path,
            "plots_dir": plot_dir,
            "plots_generated": [
                "training_curves.png",
                "confusion_matrix.png",
                "per_class_accuracy.png",
                "classification_report.txt"
            ]
        }
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[DONE] Final val_acc: {final_acc:.4f}")
        print(f"[DONE] Best val_acc: {results['best_val_acc']:.4f}")
        print(f"[DONE] Plots → {plot_dir}/")
        print(f"[DONE] Results → {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
