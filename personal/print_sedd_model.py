"""
Print SEDD (DiscreteDiffusionTransformer) model summary using torchinfo.
Output: personal/sedd_summary.log
"""
import sys
import os

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_BASE, 'models', 'diffusion'))
sys.path.insert(0, os.path.join(_BASE, 'mesh_vqvae', 'src'))

import torch
from SEDD import DiscreteDiffusionTransformer
from torchinfo import summary

# Small config (what you trained)
SMALL_CFG = {
    "vocab_size": 256,
    "max_seq_len": 4096,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "mask_id": 256,
    "num_classes": 40,
    "num_timesteps": 1000,
    "schedule_type": "linear",
    "learning_rate": 1e-4,
}

# Create model
model = DiscreteDiffusionTransformer(**SMALL_CFG)

# Create dummy inputs
batch_size = 2
seq_len = 4096
tokens = torch.randint(0, 257, (batch_size, seq_len))  # [B, L] with mask
timesteps = torch.randint(0, 1000, (batch_size,))       # [B]
class_labels = torch.randint(0, 40, (batch_size,))      # [B]

# Generate summary
print("="*80)
print("SEDD (DiscreteDiffusionTransformer) Model Summary - Small Config")
print("="*80)
print(f"Config: d_model={SMALL_CFG['d_model']}, nhead={SMALL_CFG['nhead']}, layers={SMALL_CFG['num_layers']}")
print("="*80)
summary(
    model,
    input_data=[tokens, timesteps, class_labels],
    col_names=["output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=1,
    depth=4
)
