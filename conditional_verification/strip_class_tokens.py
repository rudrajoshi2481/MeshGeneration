"""
strip_class_tokens.py
---------------------
Remove class tokens from conditional SEDD generated sequences to test if
class information is encoded in the token patterns themselves.

Usage:
    python strip_class_tokens.py --tokens_path PATH --out_path PATH
"""

import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Stripping Class Tokens")
    print(f"  Input: {args.tokens_path}")
    print(f"{'='*60}\n")
    
    # Load tokens
    data = torch.load(args.tokens_path, weights_only=False)
    tokens = data["tokens"]
    labels = data["labels"]
    
    print(f"[INFO] Loaded tokens shape: {tuple(tokens.shape)}")
    print(f"[INFO] Original sequence length: {tokens.shape[1]}")
    
    # Check if first token position contains class information
    # If conditional SEDD prepends class token, it would be at position 0
    first_tokens = tokens[:, 0]
    unique_first = first_tokens.unique()
    
    print(f"[INFO] Unique values in first position: {len(unique_first)}")
    print(f"[INFO] Labels unique values: {len(labels.unique())}")
    
    # Check correlation between first token and label
    matches = (first_tokens == labels).sum().item()
    total = len(tokens)
    correlation = matches / total
    
    print(f"[INFO] First token matches label: {matches}/{total} ({correlation*100:.1f}%)")
    
    if correlation > 0.9:
        print(f"[WARNING] High correlation detected! First token likely IS the class token.")
        print(f"[ACTION] Removing first token position...")
        tokens_stripped = tokens[:, 1:]  # Remove position 0
        print(f"[INFO] New sequence length: {tokens_stripped.shape[1]}")
    else:
        print(f"[INFO] Low correlation. First token doesn't appear to be class token.")
        print(f"[ACTION] Keeping all tokens (no stripping needed)...")
        tokens_stripped = tokens
    
    # Save
    torch.save({"tokens": tokens_stripped, "labels": labels}, args.out_path)
    
    print(f"\n[DONE] Saved {len(tokens_stripped)} samples → {args.out_path}")
    print(f"       tokens shape: {tuple(tokens_stripped.shape)}")
    print(f"       labels shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
