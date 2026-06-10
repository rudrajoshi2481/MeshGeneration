"""
mesh_transmit_semantic_channel.py
----------------------------------
Semantic channel experiment for point cloud tokens.

Implements Steps 5-8 from the experiment plan:
  - SEDD fakes (full generation) → classify  (SEDD_full_accuracy)
  - DoT fakes (full generation)  → classify  (DoT_full_accuracy)
  - SEDD puncture+refill         → classify  (partial_SEDD_accuracy vs % missing)
  - DoT  puncture+refill         → classify  (partial_DoT_accuracy  vs % missing)

Puncture modes
  --sedd_mode interval  : mask every N tokens in a strided pattern (like 1d_transmit_sedd.py)
  --dot_mode  prefix    : keep only the first K tokens (like 1d_transmit_fakes.py)

Outputs saved to TRASH_DIR / semantic_channel_results/

Usage:
  python mesh_transmit_semantic_channel.py \\
      --classifier_ckpt  <path/to/classifier.ckpt> \\
      --sedd_ckpt        <path/to/sedd.ckpt> \\
      --dot_ckpt         <path/to/dot.ckpt> \\
      --real_codes_path  <path/to/val_codes.pt>
"""

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))                  # evaluation/semantic_channel/
_BASE  = os.path.dirname(os.path.dirname(_HERE))                       # MeshGeneration/
_TRASH = os.path.join(os.path.dirname(_BASE), "trash")

DIFFUSION = os.path.join(_BASE, "models", "diffusion")
MESHVQVAE = os.path.join(_BASE, "mesh_vqvae", "src")
CLASSIFIER = os.path.join(_BASE, "models", "classifier")

sys.path.insert(0, DIFFUSION)
sys.path.insert(0, MESHVQVAE)
sys.path.insert(0, CLASSIFIER)

from SEDD import DiscreteDiffusionTransformer
from train_classifier import TokenClassifier, TokenDataset

# ── constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 256
SEQ_LEN      = 4096
NUM_CLASSES  = 40
MASK_ID      = VOCAB_SIZE          # token used to mark corrupted positions

# Puncture levels to test
MASK_INTERVALS = [1, 2, 4, 8, 16, 32]    # for SEDD interval masking
CONTEXT_LENGTHS = [4096, 2048, 512, 128, 32]  # for DoT prefix (fewer = faster eval)

PLOT_STYLE = dict(
    figure_facecolor="white",
    axes_facecolor="white",
)


# ─────────────────────────────────────────────────────────────────────────────
# Masking utilities  (from MLopsThesis/Evaluation/1d_transmit_sedd.py)
# ─────────────────────────────────────────────────────────────────────────────

def create_masked_indices(indices: torch.Tensor, mask_interval: int, mask_id: int) -> torch.Tensor:
    """
    Strided masking: keep 1 token, mask `mask_interval` tokens, repeat.
    mask_interval=1 → 50% masked, mask_interval=32 → ~97% masked.
    """
    masked = indices.clone()
    stride = mask_interval + 1
    for pos in range(indices.shape[1]):
        if pos % stride != 0:
            masked[:, pos] = mask_id
    return masked


def pct_missing(mask_interval: int) -> float:
    stride = mask_interval + 1
    return mask_interval / stride * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# SEDD reverse diffusion  (from MLopsThesis/Evaluation/1d_transmit_sedd.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reverse_diffusion_sedd(sedd_model, masked_indices: torch.Tensor,
                           class_labels: torch.Tensor = None,
                           num_steps: int = 50, temperature: float = 1.0,
                           device: str = "cuda") -> torch.Tensor:
    """Recover masked tokens using SEDD's reverse diffusion with proper timestep schedule."""
    sedd_model.eval()
    x = masked_indices.to(device)
    batch_size = x.shape[0]
    if class_labels is not None:
        class_labels = class_labels.to(device)

    # Use the same timestep schedule as SEDD's sample() method
    num_timesteps = sedd_model.noise_schedule.num_timesteps  # typically 1000
    stride = max(1, num_timesteps // num_steps)
    timesteps = list(range(num_timesteps - 1, -1, -stride))

    for t in timesteps:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        logits = sedd_model(x, t_tensor, class_labels)
        if temperature > 0:
            logits = logits / temperature
        mask = (x == MASK_ID)
        if not mask.any():
            break
        masked_logits = logits[mask]
        masked_probs = F.softmax(masked_logits, dim=-1)
        masked_samples = torch.multinomial(masked_probs, 1).squeeze(-1)
        x_new = x.clone()
        x_new[mask] = masked_samples
        x = x_new
    return x


# ─────────────────────────────────────────────────────────────────────────────
# DoT (NanoGPT) prefix completion
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dot_prefix_completion(dot_model, indices: torch.Tensor, labels: torch.Tensor,
                           context_len: int, bos_token: int,
                           device: str = "cuda") -> torch.Tensor:
    """Keep first context_len tokens, autoregressively generate the rest.
    Prompt format: [BOS, class_token, codes[:context_len]] matching training."""
    dot_model.eval()
    batch_size = indices.shape[0]
    bos = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
    cls_tokens = (bos_token + labels + 1).unsqueeze(1).to(device)  # [B, 1]
    prefix = indices[:, :context_len].to(device)
    prompt = torch.cat([bos, cls_tokens, prefix], dim=1)  # [B, 2 + context_len]
    rem = SEQ_LEN - context_len
    out = dot_model.generate(prompt, rem)  # [B, 2 + context_len + rem] = [B, 2 + SEQ_LEN]
    return out[:, 2:]  # strip BOS + class_token → [B, SEQ_LEN]


# ─────────────────────────────────────────────────────────────────────────────
# Classification evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_tokens(classifier, tokens: torch.Tensor, labels: torch.Tensor,
                    batch_size: int = 64, device: str = "cuda") -> float:
    """Returns accuracy (0-100)."""
    classifier.eval()
    correct = 0
    total = 0
    for i in range(0, len(tokens), batch_size):
        t = tokens[i:i + batch_size].long().to(device)
        l = labels[i:i + batch_size].long().to(device)
        logits = classifier(t)
        correct += (logits.argmax(dim=1) == l).sum().item()
        total += l.shape[0]
    return correct / total * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# SEDD full-generation fakes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sedd_fakes(sedd_model, n_per_class: int = 10, num_steps: int = 20,
                        device: str = "cuda") -> tuple:
    """Generate SEDD tokens from scratch (conditioned on class label)."""
    sedd_model.eval()
    all_tokens, all_labels = [], []
    print("[SEDD fakes] Generating conditional samples ...")
    for cls_id in tqdm(range(NUM_CLASSES)):
        class_labels = torch.full((n_per_class,), cls_id, dtype=torch.long, device=device)
        tokens = sedd_model.generate(
            batch_size=n_per_class, seq_len=SEQ_LEN,
            class_labels=class_labels, temperature=1.0, num_steps=num_steps
        )
        all_tokens.append(tokens.cpu())
        all_labels.append(class_labels.cpu())
    return torch.cat(all_tokens), torch.cat(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# DoT full-generation fakes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_dot_fakes(dot_model, bos_token: int, n_per_class: int = 50,
                       device: str = "cuda") -> tuple:
    """Generate DoT tokens from scratch using class-conditioned [BOS, class_token] prompt."""
    dot_model.eval()
    all_tokens, all_labels = [], []
    print("[DoT fakes] Generating conditional samples ...")
    for cls_id in tqdm(range(NUM_CLASSES)):
        # Prompt = [BOS, class_token] matching training format in train_dot_mesh.py
        bos_col = torch.full((n_per_class, 1), bos_token, dtype=torch.long, device=device)
        cls_col = torch.full((n_per_class, 1), bos_token + cls_id + 1,
                             dtype=torch.long, device=device)
        prompt = torch.cat([bos_col, cls_col], dim=1)  # [n, 2]
        out = dot_model.generate(prompt, SEQ_LEN)      # [n, 2 + SEQ_LEN]
        all_tokens.append(out[:, 2:].cpu())            # strip BOS + class_token
        all_labels.append(torch.full((n_per_class,), cls_id, dtype=torch.long))
    return torch.cat(all_tokens), torch.cat(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (paper-style accuracy vs % missing)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_missing(results: dict, save_path: str):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    COLORS = {
        "SEDD (interval mask)": "#5B8DB8",
        "DoT (prefix)":         "#F4A35A",
        "Clean":                "#6DBF8A",
        "SEDD full":            "#A48CC4",
        "DoT full":             "#D96B6B",
    }

    if "clean_accuracy" in results:
        ax.axhline(results["clean_accuracy"], color=COLORS["Clean"],
                   linestyle="--", linewidth=1.5, label=f"Clean ({results['clean_accuracy']:.1f}%)")

    if "sedd_full_accuracy" in results:
        ax.axhline(results["sedd_full_accuracy"], color=COLORS["SEDD full"],
                   linestyle=":", linewidth=1.5,
                   label=f"SEDD full gen ({results['sedd_full_accuracy']:.1f}%)")

    if "dot_full_accuracy" in results:
        ax.axhline(results["dot_full_accuracy"], color=COLORS["DoT full"],
                   linestyle=":", linewidth=1.5,
                   label=f"DoT full gen ({results['dot_full_accuracy']:.1f}%)")

    if "sedd_partial" in results:
        xs = [r["pct_missing"] for r in results["sedd_partial"]]
        ys = [r["accuracy"] for r in results["sedd_partial"]]
        ax.plot(xs, ys, "o-", color=COLORS["SEDD (interval mask)"],
                linewidth=2, markersize=6, label="SEDD (interval mask)")

    if "dot_partial" in results:
        xs = [r["pct_missing"] for r in results["dot_partial"]]
        ys = [r["accuracy"] for r in results["dot_partial"]]
        ax.plot(xs, ys, "s-", color=COLORS["DoT (prefix)"],
                linewidth=2, markersize=6, label="DoT (prefix)")

    ax.set_xlabel("% Tokens Missing", labelpad=8, fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", labelpad=8, fontsize=12)
    ax.set_title("Semantic Channel: Accuracy vs Token Loss\n(Point Cloud / ModelNet40)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Path to trained TokenClassifier .ckpt")
    parser.add_argument("--sedd_ckpt", type=str, default=None,
                        help="Path to trained SEDD .ckpt")
    parser.add_argument("--dot_ckpt", type=str, default=None,
                        help="Path to trained NanoGPT/Monai DoT .ckpt (optional)")
    parser.add_argument("--real_codes_path", type=str,
                        default=os.path.join(_TRASH, "data", "val_codes.pt"),
                        help="Path to val_codes.pt with real token sequences")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_TRASH, "semantic_channel_results"),
                        help="Output directory for results and plots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_per_class", type=int, default=10,
                        help="Samples per class for full-generation evaluation")
    parser.add_argument("--sedd_steps", type=int, default=20,
                        help="Number of SEDD diffusion steps (fewer = faster)")
    parser.add_argument("--skip_dot_full_gen", action="store_true",
                        help="Skip slow DoT full-generation accuracy step")
    parser.add_argument("--dot_n_eval", type=int, default=200,
                        help="Number of val samples to use for DoT prefix eval (fewer = faster)")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*65}")
    print(f"  Mesh Semantic Channel Experiment")
    print(f"  Output → {args.out_dir}")
    print(f"{'='*65}\n")

    results = {}

    # ── Load real codes ──────────────────────────────────────────────────────
    print(f"[INFO] Loading real codes from {args.real_codes_path} ...")
    raw = torch.load(args.real_codes_path, weights_only=False)
    real_tokens = raw.get("codes", raw.get("tokens")).long()
    real_labels = raw["labels"].long()
    print(f"[INFO] Real codes: {tuple(real_tokens.shape)}, labels: {tuple(real_labels.shape)}")

    # ── Load classifier ──────────────────────────────────────────────────────
    print(f"\n[INFO] Loading classifier from {args.classifier_ckpt} ...")
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        classifier = TokenClassifier.load_from_checkpoint(
            args.classifier_ckpt, map_location=device
        )
    finally:
        torch.load = _orig
    classifier.eval().to(device)

    # ── Step 1: Clean accuracy ───────────────────────────────────────────────
    print("\n[STEP] Computing clean accuracy (real codes) ...")
    clean_acc = classify_tokens(classifier, real_tokens, real_labels,
                                batch_size=args.batch_size, device=str(device))
    results["clean_accuracy"] = clean_acc
    print(f"  ✓ Clean accuracy: {clean_acc:.2f}%")

    # ── Load SEDD ────────────────────────────────────────────────────────────
    sedd_model = None
    if args.sedd_ckpt is not None:
        print(f"\n[INFO] Loading SEDD from {args.sedd_ckpt} ...")
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
        try:
            sedd_model = DiscreteDiffusionTransformer.load_from_checkpoint(
                args.sedd_ckpt, map_location=device
            )
        finally:
            torch.load = _orig
        sedd_model.eval().to(device)
        print(f"  ✓ SEDD loaded: {sum(p.numel() for p in sedd_model.parameters())/1e6:.2f}M params")

        # ── Step 5: SEDD full generation accuracy ────────────────────────────
        print("\n[STEP 5] SEDD full generation accuracy ...")
        sedd_fake_tokens, sedd_fake_labels = generate_sedd_fakes(
            sedd_model, n_per_class=args.n_per_class, num_steps=args.sedd_steps, device=str(device)
        )
        sedd_full_acc = classify_tokens(classifier, sedd_fake_tokens, sedd_fake_labels,
                                        batch_size=args.batch_size, device=str(device))
        results["sedd_full_accuracy"] = sedd_full_acc
        torch.save({"tokens": sedd_fake_tokens, "labels": sedd_fake_labels},
                   os.path.join(args.out_dir, "sedd_full_fakes.pt"))
        print(f"  ✓ SEDD full accuracy: {sedd_full_acc:.2f}%")

        # ── Step 7a: SEDD puncture + refill ──────────────────────────────────
        print("\n[STEP 7a] SEDD semantic channel (interval masking) ...")
        sedd_partial = []
        for interval in MASK_INTERVALS:
            pct = pct_missing(interval)
            print(f"  mask_interval={interval} ({pct:.1f}% missing) ...")
            all_recovered, all_labels_list = [], []
            for i in tqdm(range(0, len(real_tokens), args.batch_size),
                          desc=f"  interval={interval}", leave=False):
                batch_tok = real_tokens[i:i + args.batch_size]
                batch_lbl = real_labels[i:i + args.batch_size]
                masked = create_masked_indices(batch_tok, interval, MASK_ID)
                recovered = reverse_diffusion_sedd(
                    sedd_model, masked, class_labels=batch_lbl,
                    num_steps=args.sedd_steps, device=str(device)
                )
                all_recovered.append(recovered.cpu())
                all_labels_list.append(batch_lbl)
            recovered_tokens = torch.cat(all_recovered)
            recovered_labels = torch.cat(all_labels_list)
            acc = classify_tokens(classifier, recovered_tokens, recovered_labels,
                                  batch_size=args.batch_size, device=str(device))
            sedd_partial.append({"mask_interval": interval, "pct_missing": pct, "accuracy": acc})
            print(f"    → accuracy: {acc:.2f}%")
            np.save(os.path.join(args.out_dir, f"sedd_recovered_interval{interval}.npy"),
                    recovered_tokens.numpy())
        results["sedd_partial"] = sedd_partial

    # ── Load DoT ─────────────────────────────────────────────────────────────
    dot_model = None
    if args.dot_ckpt is not None:
        print(f"\n[INFO] Loading DoT model from {args.dot_ckpt} ...")
        try:
            sys.path.insert(0, _HERE)  # semantic_channel/ — NanoGpt is inlined in train_dot_mesh
            from train_dot_mesh import NanoGpt
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            try:
                dot_model = NanoGpt.load_from_checkpoint(args.dot_ckpt, map_location=device)
            finally:
                torch.load = _orig
            dot_model.eval().to(device)
            BOS_TOKEN = VOCAB_SIZE
            print(f"  ✓ DoT loaded: {sum(p.numel() for p in dot_model.parameters())/1e6:.2f}M params")
        except Exception as e:
            print(f"  ✗ DoT load failed: {e}")
            dot_model = None

        if dot_model is not None:
            # ── Step 6: DoT full generation accuracy ─────────────────────────
            if not args.skip_dot_full_gen:
                print("\n[STEP 6] DoT full generation accuracy ...")
                dot_fake_tokens, dot_fake_labels = generate_dot_fakes(
                    dot_model, BOS_TOKEN, n_per_class=args.n_per_class, device=str(device)
                )
                dot_full_acc = classify_tokens(classifier, dot_fake_tokens, dot_fake_labels,
                                               batch_size=args.batch_size, device=str(device))
                results["dot_full_accuracy"] = dot_full_acc
                torch.save({"tokens": dot_fake_tokens, "labels": dot_fake_labels},
                           os.path.join(args.out_dir, "dot_full_fakes.pt"))
                print(f"  ✓ DoT full accuracy: {dot_full_acc:.2f}%")
            else:
                print("\n[STEP 6] Skipping DoT full generation (--skip_dot_full_gen)")

            # ── Step 7b: DoT prefix truncation + completion ───────────────────
            print("\n[STEP 7b] DoT semantic channel (prefix truncation) ...")
            # Subsample val set to keep autoregressive generation tractable
            n_dot = min(args.dot_n_eval, len(real_tokens))
            dot_eval_tokens = real_tokens[:n_dot]
            dot_eval_labels = real_labels[:n_dot]
            print(f"  Using {n_dot} val samples for DoT prefix eval")
            dot_partial = []
            for ctx_len in CONTEXT_LENGTHS:
                pct = (1.0 - ctx_len / SEQ_LEN) * 100.0
                print(f"  context_len={ctx_len} ({pct:.1f}% missing) ...")
                all_completed, all_labels_list = [], []
                for i in tqdm(range(0, len(dot_eval_tokens), args.batch_size),
                              desc=f"  ctx={ctx_len}", leave=False):
                    batch_tok = dot_eval_tokens[i:i + args.batch_size]
                    batch_lbl = dot_eval_labels[i:i + args.batch_size]
                    completed = dot_prefix_completion(
                        dot_model, batch_tok, batch_lbl, ctx_len, BOS_TOKEN, device=str(device)
                    )
                    all_completed.append(completed.cpu())
                    all_labels_list.append(batch_lbl)
                completed_tokens = torch.cat(all_completed)
                completed_labels = torch.cat(all_labels_list)
                acc = classify_tokens(classifier, completed_tokens, completed_labels,
                                      batch_size=args.batch_size, device=str(device))
                dot_partial.append({"context_len": ctx_len, "pct_missing": pct, "accuracy": acc})
                print(f"    → accuracy: {acc:.2f}%")
                np.save(os.path.join(args.out_dir, f"dot_completed_ctx{ctx_len}.npy"),
                        completed_tokens.numpy())
            results["dot_partial"] = dot_partial

    # ── Save results JSON ────────────────────────────────────────────────────
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results → {results_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(plot_dir, "accuracy_vs_missing_tokens.png")
    plot_accuracy_vs_missing(results, plot_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Clean accuracy          : {results.get('clean_accuracy', 'N/A'):.2f}%")
    if "sedd_full_accuracy" in results:
        print(f"  SEDD full gen accuracy  : {results['sedd_full_accuracy']:.2f}%")
    if "dot_full_accuracy" in results:
        print(f"  DoT  full gen accuracy  : {results['dot_full_accuracy']:.2f}%")
    if "sedd_partial" in results:
        sedd_summary = ["{:.0f}%->{:.1f}%".format(r["pct_missing"], r["accuracy"]) for r in results["sedd_partial"]]
        print("  SEDD partial (intervals): {}".format(sedd_summary))
    if "dot_partial" in results:
        dot_summary = ["{:.0f}%->{:.1f}%".format(r["pct_missing"], r["accuracy"]) for r in results["dot_partial"]]
        print("  DoT  partial (prefix)   : {}".format(dot_summary))
    print(f"\n  Plots → {plot_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
