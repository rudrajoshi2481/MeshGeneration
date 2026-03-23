"""
dataset.py
-----------
Dataset for class-as-first-token conditional SEDD training.

Loads pre-extracted codes and prepends the class token to each sequence:
    Original: [code_0, code_1, ..., code_4095]          shape: [4096]
    New:      [class_token, code_0, ..., code_4095]      shape: [4097]

Class token ids: 257 + class_index  (so class 0 → 257, class 39 → 296)
"""

import torch
from torch.utils.data import Dataset

CLASS_TOKEN_OFFSET = 257


class ClassPrefixedCodeDataset(Dataset):
    """
    Wraps pre-extracted MeshGPT code files and prepends the class token.

    Args:
        pt_path:   path to .pt file with {"codes": [N,4096], "labels": [N]}
        n_samples: if > 0, randomly subsample this many samples
    """

    def __init__(self, pt_path: str, n_samples: int = -1):
        data   = torch.load(pt_path, weights_only=False)
        codes  = data["codes"].long()    # [N, 4096]  values 0-255
        labels = data["labels"].long()   # [N]        values 0-39

        if n_samples > 0:
            idx    = torch.randperm(len(codes))[:n_samples]
            codes  = codes[idx]
            labels = labels[idx]

        # Prepend class token: class c → token id (257 + c)
        class_tokens = (labels + CLASS_TOKEN_OFFSET).unsqueeze(1)  # [N, 1]
        self.sequences = torch.cat([class_tokens, codes], dim=1)   # [N, 4097]
        self.labels    = labels                                      # [N]

        print(f"[Dataset] {pt_path.split('/')[-1]}: {len(self.sequences)} samples, "
              f"seq_len={self.sequences.shape[1]}, "
              f"class_token range=[{int(class_tokens.min())}, {int(class_tokens.max())}]")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": self.sequences[idx],   # [4097]  class_token + 4096 mesh codes
            "label":     self.labels[idx],       # scalar  original class index (0-39)
        }
