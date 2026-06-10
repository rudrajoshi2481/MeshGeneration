"""
train_dot_mesh.py
-----------------
Train a Decoder-Only Transformer (NanoGPT) on pre-extracted MeshVQVAE code sequences.
This is the DoT counterpart to diffusion_model/train_sedd.py.

Reads: trash/data/train_codes.pt, trash/data/val_codes.pt
       {"codes": [N, 4096] long, "labels": [N] long}

Outputs: trash/dot_runs/<run_id>/

Usage:
  python train_dot_mesh.py --mode small    # fast sanity check
  python train_dot_mesh.py --mode full     # full training
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))      # models/autoregressive/
_BASE     = os.path.dirname(os.path.dirname(_HERE))           # MeshGeneration/
_SEMENTIC = os.path.dirname(_BASE)                            # sementic_channel_project/
_TRASH    = os.path.join(_SEMENTIC, "trash")
MESHVQVAE = os.path.join(_BASE, "mesh_vqvae", "src")

sys.path.insert(0, MESHVQVAE)

from preprocessing import MODELNET40_CLASSES


# ── Inline NanoGPT (from MLopsThesis/Models/DecoderOnlyTransformers.py) ───────
# Copied here to avoid the `generative` (MONAI) dependency at the module level.

class _CausalSelfAttention(nn.Module):
    """Memory-efficient causal self-attention via F.scaled_dot_product_attention (flash-attn)."""
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        assert n_embd % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = n_embd // num_heads
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = dropout
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                          # each [B, T, H, D]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [B, H, T, D]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(out))


class _FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa   = _CausalSelfAttention(n_embd, n_head, dropout)
        self.ffwd = _FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class _TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, additional_vocab):
        super().__init__()
        self.vocab_size  = vocab_size
        self.block_size  = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size + additional_vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size + 2, n_embd)
        self.blocks = nn.Sequential(*[_Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size + additional_vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, self.vocab_size:] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# Plotting callback
# ─────────────────────────────────────────────────────────────────────────────

class DoTPlotCallback(Callback):
    """Logs training curves and generation samples every N epochs for DoT."""

    def __init__(self, plot_dir: str, val_dataset, vocab_size: int,
                 seq_len: int, plot_every: int = 5, n_gen: int = 8):
        self.plot_dir   = plot_dir
        self.val_ds     = val_dataset
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.plot_every = plot_every
        self.n_gen      = n_gen
        os.makedirs(plot_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses   = []
        self.epochs       = []

    def on_train_epoch_end(self, trainer, pl_module):
        ep = trainer.current_epoch
        tl = float(trainer.callback_metrics.get("train_loss", 0))
        vl = float(trainer.callback_metrics.get("val_loss", 0))
        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.epochs.append(ep)

        if ep % self.plot_every == 0:
            self._plot_curves(ep)
            self._plot_generation(pl_module, ep)
            self._plot_code_distribution(pl_module, ep)

    def _plot_curves(self, ep: int):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.epochs, self.train_losses, label="train_loss", color="#5B8DB8", linewidth=2)
        ax.plot(self.epochs, self.val_losses,   label="val_loss", color="#F4A35A", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11, labelpad=8)
        ax.set_ylabel("Loss", fontsize=11, labelpad=8)
        ax.set_title("DoT Training Curves", fontsize=13, fontweight='bold', pad=10)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"curves_ep{ep:04d}.png"), dpi=150, facecolor="white")
        plt.close()

    @torch.no_grad()
    def _plot_generation(self, model, ep: int):
        """Generate samples and plot code histograms."""
        device = next(model.parameters()).device
        n_cls  = min(10, NUM_CLASSES)
        # Generate one sample per class
        prompts = []
        for c in range(n_cls):
            cls_tok = torch.tensor([[BOS_TOKEN, BOS_TOKEN + c + 1]], dtype=torch.long, device=device)
            prompts.append(cls_tok)
        
        all_samples = []
        for prompt in prompts:
            gen = model.generate(prompt, max_new_tokens=self.seq_len)
            all_samples.append(gen[0, 2:].cpu())  # Remove BOS and class token
        
        samples = torch.stack(all_samples)  # [n_cls, seq_len]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()
        for i in range(n_cls):
            codes = samples[i].numpy()
            axes[i].hist(codes, bins=range(self.vocab_size + 1), color="#5B8DB8", alpha=0.85, edgecolor="white")
            axes[i].set_title(f"Class {i}", fontsize=10, fontweight='semibold')
            axes[i].set_xlabel("Code ID", fontsize=9, labelpad=6)
            axes[i].set_ylabel("Count", fontsize=9, labelpad=6)
            axes[i].grid(True, alpha=0.3)
        plt.suptitle(f"DoT Generated Code Histograms (epoch {ep})", fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"gen_hist_ep{ep:04d}.png"), dpi=150, facecolor="white")
        plt.close()

    @torch.no_grad()
    def _plot_code_distribution(self, model, ep: int):
        """Compare real vs generated code usage distribution."""
        device = next(model.parameters()).device

        # Real distribution from validation set
        real_codes = []
        for i in range(min(100, len(self.val_ds))):
            x, _ = self.val_ds[i]
            real_codes.append(x[2:].numpy())  # Skip BOS and class tokens
        real_codes = np.concatenate(real_codes)
        real_hist = np.bincount(real_codes, minlength=self.vocab_size).astype(float)
        real_hist /= real_hist.sum() + 1e-8

        # Generated distribution
        n_gen_samples = 50
        gen_codes = []
        for c in range(NUM_CLASSES):
            cls_tok = torch.tensor([[BOS_TOKEN, BOS_TOKEN + c + 1]], dtype=torch.long, device=device)
            for _ in range(n_gen_samples // NUM_CLASSES):
                gen = model.generate(cls_tok, max_new_tokens=self.seq_len)
                gen_codes.append(gen[0, 2:].cpu().numpy())
        gen_codes = np.concatenate(gen_codes)
        gen_hist = np.bincount(gen_codes, minlength=self.vocab_size).astype(float)
        gen_hist /= gen_hist.sum() + 1e-8

        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(self.vocab_size)
        ax.bar(x, real_hist, alpha=0.6, label="Real", color="#5B8DB8", width=1.0)
        ax.bar(x, gen_hist,  alpha=0.6, label="Generated", color="#F4A35A", width=1.0)
        ax.set_xlabel("Code ID", fontsize=11, labelpad=8)
        ax.set_ylabel("Frequency", fontsize=11, labelpad=8)
        ax.set_title(f"DoT: Real vs Generated Code Distribution (epoch {ep})", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"code_dist_ep{ep:04d}.png"), dpi=150, facecolor="white")
        plt.close()


class NanoGpt(pl.LightningModule):
    def __init__(self, vocab_size=256, n_embd=256, block_size=352, n_head=8, n_layer=8,
                 dropout=0.1, additional_vocab=11, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = _TransformerModel(vocab_size, n_embd, block_size, n_head, n_layer,
                                       dropout, additional_vocab)
        self.learning_rate  = learning_rate
        self.vocab_size     = vocab_size
        self.additional_vocab = additional_vocab
        self.block_size     = block_size

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                 betas=(0.9, 0.99), weight_decay=0.01)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        return self.model.generate(idx, max_new_tokens)

DATA_DIR = os.path.join(_TRASH, "data")
OUT_BASE  = os.path.join(_TRASH, "dot_runs")

VOCAB_SIZE   = 256
SEQ_LEN      = 4096
NUM_CLASSES  = 40
BOS_TOKEN    = VOCAB_SIZE          # index 256 = BOS
# Class tokens: 257..296  (BOS + class_id + 1)
ADDITIONAL_VOCAB = 1 + NUM_CLASSES  # BOS + 40 class tokens


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CodeSequenceDataset(Dataset):
    """Wraps pre-extracted MeshVQVAE code sequences for autoregressive training."""

    def __init__(self, pt_path: str, block_size: int = SEQ_LEN, n_samples: int = -1):
        data = torch.load(pt_path, weights_only=False)
        codes  = data.get("codes", data.get("tokens")).long()   # [N, seq_len]
        labels = data["labels"].long()
        if n_samples > 0:
            idx = torch.randperm(len(codes))[:n_samples]
            codes, labels = codes[idx], labels[idx]
        self.codes      = codes
        self.labels     = labels
        self.block_size = block_size
        print(f"[Dataset] {os.path.basename(pt_path)}: {len(self.codes)} samples")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        codes = self.codes[idx][:self.block_size]
        label = self.labels[idx]
        # Prepend [BOS, class_token] so the model knows the target class
        cls_tok = torch.tensor([BOS_TOKEN, BOS_TOKEN + label.item() + 1], dtype=torch.long)
        seq = torch.cat([cls_tok, codes])          # [2 + block_size]
        x   = seq[:-1]                             # input
        y   = seq[1:]                              # target (shifted right)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Config presets
# ─────────────────────────────────────────────────────────────────────────────

SMALL_CFG = dict(n_embd=128, n_head=4, n_layer=3, learning_rate=5e-4, epochs=5,  batch_size=8,  n_samples=200)
FULL_CFG  = dict(n_embd=512, n_head=8, n_layer=6, learning_rate=1e-4, epochs=30, batch_size=16, n_samples=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--out_base", type=str, default=OUT_BASE)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=None, help="Override n_train (-1=all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    cfg = dict(SMALL_CFG if args.mode == "small" else FULL_CFG)  # copy to allow overrides
    if args.n_train is not None:
        cfg["n_samples"] = args.n_train
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    run_id = datetime.now().strftime("dot_run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_base, run_id)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot frequency based on mode
    plot_every = 5 if args.mode == "small" else 10

    print(f"\n{'='*60}")
    print(f"  DoT Training — {args.mode.upper()} mode")
    print(f"  Output → {out_dir}")
    print(f"  Plots  → {plot_dir} (every {plot_every} epochs)")
    print(f"{'='*60}\n")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = CodeSequenceDataset(
        os.path.join(args.data_dir, "train_codes.pt"), n_samples=cfg["n_samples"]
    )
    val_ds = CodeSequenceDataset(
        os.path.join(args.data_dir, "val_codes.pt"), n_samples=cfg["n_samples"] // 5 if cfg["n_samples"] > 0 else -1
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = NanoGpt(
        vocab_size       = VOCAB_SIZE,
        block_size       = SEQ_LEN,
        n_embd           = cfg["n_embd"],
        n_head           = cfg["n_head"],
        n_layer          = cfg["n_layer"],
        additional_vocab = ADDITIONAL_VOCAB,
        learning_rate    = cfg["learning_rate"],
    )
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ── Callbacks ────────────────────────────────────────────────────────────
    plot_cb = DoTPlotCallback(
        plot_dir=plot_dir,
        val_dataset=val_ds,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        plot_every=plot_every,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="dot-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=2
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # ── Trainer ──────────────────────────────────────────────────────────────
    strategy = "ddp_find_unused_parameters_false" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs   = cfg["epochs"],
        accelerator  = "gpu" if torch.cuda.is_available() else "cpu",
        devices      = args.gpus,
        strategy     = strategy,
        precision    = "bf16-mixed",
        callbacks    = [plot_cb, ckpt_cb, early_stop],
        default_root_dir = out_dir,
        log_every_n_steps = 10,
    )

    trainer.fit(model, train_loader, val_loader)

    # ── Save final model ─────────────────────────────────────────────────────
    if trainer.global_rank == 0:
        final_path = os.path.join(out_dir, "dot_final.pt")
        torch.save(model.state_dict(), final_path)
        report = {
            "mode": args.mode,
            "best_ckpt": ckpt_cb.best_model_path,
            "final_model": final_path,
            "config": cfg,
        }
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[DONE] Best ckpt → {ckpt_cb.best_model_path}")
        print(f"[DONE] Report    → {out_dir}/report.json")


if __name__ == "__main__":
    main()
