"""
Print DoT (NanoGPT) model summary using torchinfo.
Output: personal/dot_summary.log
"""
import sys
import os

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_BASE, 'models', 'autoregressive'))
sys.path.insert(0, os.path.join(_BASE, 'mesh_vqvae', 'src'))

import torch
from train_dot_mesh import NanoGpt
from torchinfo import summary

# Small config
VOCAB_SIZE = 256
SEQ_LEN = 4096
NUM_CLASSES = 40
BOS_TOKEN = 256
ADDITIONAL_VOCAB = 1 + NUM_CLASSES  # BOS + 40 class tokens

# Create model
model = NanoGpt(
    vocab_size=VOCAB_SIZE,
    n_embd=128,
    block_size=352,  # Context window
    n_head=4,
    n_layer=3,
    dropout=0.1,
    additional_vocab=ADDITIONAL_VOCAB,
    learning_rate=1e-4
)

# Create dummy input (with BOS and class token)
batch_size = 2
seq_len = 352  # block_size
inputs = torch.randint(0, VOCAB_SIZE + ADDITIONAL_VOCAB, (batch_size, seq_len))

# Generate summary
print("="*80)
print("DoT (NanoGPT) Model Summary - Small Config")
print("="*80)
print(f"Config: n_embd=128, n_head=4, n_layer=3, block_size=352")
print(f"Vocab: {VOCAB_SIZE} base + {ADDITIONAL_VOCAB} additional (BOS + {NUM_CLASSES} classes)")
print("="*80)
summary(
    model,
    input_data=inputs,
    col_names=["output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=1,
    depth=4
)
