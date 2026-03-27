"""
evaluate.py
-----------
Evaluate trained classifier with confusion matrix, per-class metrics, and t-SNE visualization.

Usage:
    python evaluate.py --ckpt PATH --tokens_path PATH --mode conditional --out_dir DIR
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add paths
MESHVQVAE = "/data/joshi/tmp/MeshGeneration/mesh_vqvae/src"
sys.path.insert(0, MESHVQVAE)
from preprocessing import MODELNET40_CLASSES

# Import classifier
sys.path.insert(0, os.path.dirname(__file__))
from train_classifier import TokenClassifier, TokenDataset

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


@torch.no_grad()
def extract_embeddings_and_predictions(model, loader, device):
    """Extract embeddings and predictions from classifier"""
    model.eval()
    all_embeds = []
    all_preds = []
    all_labels = []
    
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].to(device)
        
        # Get embeddings (mean pooled)
        embeds = model.embedding(tokens).mean(dim=1)
        logits = model.mlp(embeds)
        preds = logits.argmax(dim=1)
        
        all_embeds.append(embeds.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    return (np.concatenate(all_embeds), 
            np.concatenate(all_preds), 
            np.concatenate(all_labels))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(16, 14), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", interpolation="nearest")
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    
    ax.set_xlabel("Predicted Class", labelpad=10, fontsize=11)
    ax.set_ylabel("True Class", labelpad=10, fontsize=11)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold", pad=12)
    
    plt.colorbar(im, ax=ax, label="Frequency", shrink=0.8)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tsne(embeddings, labels, class_names, save_path, n_samples=2000):
    """Plot t-SNE visualization of embeddings"""
    # Subsample for speed
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    
    print(f"[INFO] Running t-SNE on {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    
    # Plot each class
    for i in range(min(40, len(class_names))):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      s=15, alpha=0.6, label=class_names[i])
    
    ax.set_xlabel("t-SNE Dimension 1", labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", labelpad=8)
    ax.set_title("t-SNE Visualization of Token Embeddings", 
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path):
    """Plot per-class accuracy bar chart"""
    per_class_acc = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(class_names))
    ax.bar(x, per_class_acc, color=PALETTE[0], alpha=0.85, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class", labelpad=8)
    ax.set_ylabel("Accuracy", labelpad=8)
    ax.set_title("Per-Class Classification Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax.axhline(y=0.025, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label="Chance (2.5%)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Classifier checkpoint")
    parser.add_argument("--tokens_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["conditional", "unconditional", "conditional_no_class", "unconditional_no_class"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  Evaluation — {args.mode}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*60}\n")
    
    # Load model
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        model = TokenClassifier.load_from_checkpoint(args.ckpt, map_location=device)
    finally:
        torch.load = _orig_load
    
    model.eval().to(device)
    print("[INFO] Model loaded")
    
    # Load tokens
    data = torch.load(args.tokens_path, weights_only=False)
    dataset = TokenDataset(data["tokens"], data["labels"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"[INFO] Evaluating on {len(dataset)} samples...")
    
    # Extract embeddings and predictions
    embeddings, preds, labels = extract_embeddings_and_predictions(model, loader, device)
    
    # Compute metrics
    acc = (preds == labels).mean()
    print(f"\n[RESULTS] Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification report
    report = classification_report(labels, preds, target_names=MODELNET40_CLASSES, 
                                   output_dict=True, zero_division=0)
    
    # Save metrics
    results = {
        "mode": args.mode,
        "overall_accuracy": float(acc),
        "per_class_metrics": report,
        "chance_level": 1.0 / 40,
    }
    
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Saved metrics → {args.out_dir}/eval_results.json")
    
    # Generate plots
    print("[INFO] Generating plots...")
    
    plot_confusion_matrix(labels, preds, MODELNET40_CLASSES,
                         os.path.join(plot_dir, "confusion_matrix.png"))
    print("  ✓ Confusion matrix")
    
    plot_per_class_accuracy(labels, preds, MODELNET40_CLASSES,
                           os.path.join(plot_dir, "per_class_accuracy.png"))
    print("  ✓ Per-class accuracy")
    
    plot_tsne(embeddings, labels, MODELNET40_CLASSES,
             os.path.join(plot_dir, "tsne_embeddings.png"))
    print("  ✓ t-SNE visualization")
    
    print(f"\n[DONE] All plots saved to {plot_dir}/")
    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"  - Accuracy: {acc*100:.2f}% (chance: 2.5%)")
    print(f"  - Interpretation: {'✓ Class conditioning WORKS' if acc > 0.5 else '✗ No class information'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
