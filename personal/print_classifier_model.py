"""
Print TokenClassifier model summary using torchinfo.
Output: personal/classifier_summary.log
"""
import sys
import os

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_BASE, 'models', 'classifier'))
sys.path.insert(0, os.path.join(_BASE, 'mesh_vqvae', 'src'))

import torch
from train_classifier import TokenClassifier
from torchinfo import summary

# Create model
model = TokenClassifier(
    vocab_size=256,
    seq_len=4096,
    embed_dim=256,
    hidden_dim=512,
    num_classes=40,
    dropout=0.2
)

# Create dummy input (batch of token sequences)
batch_size = 2
inputs = torch.randint(0, 257, (batch_size, 4096))  # [B, seq_len] with mask token

# Generate summary
print("="*80)
print("TokenClassifier Model Summary")
print("="*80)
summary(
    model,
    input_data=inputs,
    col_names=["output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=1,
    depth=4
)
