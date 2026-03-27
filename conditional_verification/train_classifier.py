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

# Add paths
MESHVQVAE = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
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
        self.val_preds = []
        self.val_labels = []
    
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["tokens"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        
        # Store for confusion matrix
        self.val_preds.append(logits.argmax(dim=1).cpu())
        self.val_labels.append(batch["label"].cpu())
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds)
            labels = torch.cat(self.val_labels)
            acc = (preds == labels).float().mean().item()
            self.val_acc.append(acc)
            self.val_preds.clear()
            self.val_labels.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


def plot_training_curves(train_acc, val_acc, save_path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    epochs = np.arange(len(val_acc))
    
    ax.plot(epochs, val_acc, color=PALETTE[0], lw=2, label="val_acc", marker='o', markersize=4)
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Accuracy", labelpad=8)
    ax.set_title("Token Classifier Training", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    
    # Load tokens
    data = torch.load(args.tokens_path, weights_only=False)
    tokens = data["tokens"]
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
    
    # Plot
    if trainer.global_rank == 0:
        plot_training_curves(model.train_acc, model.val_acc,
                            os.path.join(plot_dir, "training_curves.png"))
        
        # Save results
        results = {
            "mode": args.mode,
            "final_val_acc": float(model.val_acc[-1]) if model.val_acc else 0.0,
            "best_val_acc": float(max(model.val_acc)) if model.val_acc else 0.0,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "best_ckpt": ckpt_cb.best_model_path,
        }
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[DONE] Best val_acc: {results['best_val_acc']:.4f}")
        print(f"[DONE] Results → {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
