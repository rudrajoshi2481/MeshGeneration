# Semantic Channel Model Documentation

Generated: June 9, 2026

## Overview

This document provides a complete reference of all models in the semantic channel pipeline for 3D point cloud token sequence generation and evaluation.

---

## 1. MeshVQVAE (Vector Quantized Variational Autoencoder)

**Location**: `mesh_vqvae/src/`

### Purpose
Converts 3D point cloud meshes into discrete token sequences (4096 tokens per mesh).

### Key Components
- **Model**: `MaskedVQVAE3D`
- **Config**: `SmallModelConfig`
- **Input**: Point clouds (2048 surface points + 2048 query points)
- **Output**: 4096 codebook indices (tokens)

### Architecture
```
Encoder: Point Cloud → Latent → VQ (Vector Quantization) → 4096 tokens
Decoder: 4096 tokens → Occupancy prediction → 3D mesh
```

### Files
- `model.py` - VQVAE architecture
- `config.py` - Model configurations (small/medium/full)
- `dataset.py` - ModelNet40 dataset loader
- `preprocessing.py` - Mesh loading and feature extraction

### Usage
```bash
# Training (done)
python train_vqvae.py --mode small

# Code extraction (from models/diffusion/)
python extract_fresh_codes.py --ckpt <checkpoint> --out_dir ../../trash/data
```

---

## 2. TokenClassifier

**Location**: `models/classifier/`

### Purpose
Evaluates the quality of generated/recovered token sequences by classifying them into 40 ModelNet40 categories.

### Architecture
```
Token Embedding (256+1 vocab) → Mean Pooling → MLP (512→256→40) → Class Logits
```

### Key Parameters
- `vocab_size`: 256
- `seq_len`: 4096
- `embed_dim`: 256
- `hidden_dim`: 512
- `num_classes`: 40
- `dropout`: 0.2

### Files
- `train_classifier.py` - Training script
- `evaluate.py` - Evaluation with t-SNE, confusion matrix

### Usage
```bash
python train_classifier.py --tokens_path trash/data/train_codes.pt --mode conditional
python evaluate.py --ckpt <classifier.ckpt> --tokens_path trash/data/val_codes.pt
```

### Performance
- Clean tokens (real VQVAE): ~60% accuracy

---

## 3. SEDD (Score-based Edit Diffusion Model)

**Location**: `diffusion_model/`

### Purpose
Discrete diffusion model for token sequence generation and recovery. Trained to denoise masked tokens.

### Architecture
```
Token Embedding + Positional Encoding + Time Embedding + [Class Embedding]
↓
Transformer Encoder (N layers)
↓
Output Projection → Token Logits
```

### Key Parameters (Small Config)
- `vocab_size`: 256
- `max_seq_len`: 4096
- `d_model`: 128
- `nhead`: 4
- `num_layers`: 3
- `mask_id`: 256
- `num_classes`: 40 (for class conditioning)
- `num_timesteps`: 1000

### Forward Diffusion (Training)
```python
x_noisy = q_sample(x_start, t)  # Mask tokens based on timestep
logits = model(x_noisy, t, class_labels)  # Predict original tokens
loss = cross_entropy(logits, x_start)
```

### Reverse Diffusion (Generation)
```python
x = all_masked_tokens
for t in reversed(timesteps):
    logits = model(x, t, class_labels)
    x = sample_from_logits(logits, mask_positions)
return x  # Generated tokens
```

### Files
- `SEDD.py` - Model definition with DiscreteDiffusionTransformer
- `train_sedd.py` - Training script

### Usage
```bash
# Training
python train_sedd.py --mode small --n_train -1 --epochs 200

# Generation (in semantic channel eval)
# Uses reverse_diffusion_sedd() function
```

### Performance
- Full generation (from class ID): ~32% accuracy
- Puncture refill (50% missing): ~55% accuracy

---

## 4. DoT (Decoder-only Transformer / NanoGPT)

**Location**: `models/autoregressive/`

### Purpose
Autoregressive model for token sequence generation. Predicts next token given previous tokens.

### Architecture
```
Input: [BOS, class_token, token_1, token_2, ...]
↓
Token Embedding + Position Embedding
↓
Causal Self-Attention (Flash Attention)
↓
Feed Forward
↓
LayerNorm
↓
Repeat N times
↓
Linear Projection → Next Token Logits
```

### Key Parameters (Small Config)
- `vocab_size`: 256 (base tokens)
- `additional_vocab`: 41 (1 BOS + 40 class tokens)
- `n_embd`: 128
- `n_head`: 4
- `n_layer`: 3
- `block_size`: 352 (context window)
- `dropout`: 0.1

### Training Format
```python
# Input sequence
[BOS, class_token, code_0, code_1, ..., code_4095]

# Targets (shifted by 1)
[class_token, code_0, code_1, ..., code_4095, <padding>]
```

### Generation
```python
prompt = [BOS, class_token, prefix_tokens...]
for i in range(remaining_tokens):
    logits = model(prompt)
    next_token = sample(logits[-1])
    prompt.append(next_token)
```

### Files
- `train_dot_mesh.py` - Complete NanoGPT implementation with training

### Usage
```bash
python train_dot_mesh.py --mode small --epochs 100 --batch_size 16
```

### Performance
- Prefix completion (50% context): ~69.5% accuracy
- Prefix completion (87% missing): ~11% accuracy

---

## 5. Semantic Channel Evaluation

**Location**: `evaluation/semantic_channel/`

### Purpose
Tests both SEDD and DoT on semantic channel scenarios: puncture sequences and recover missing tokens.

### Experiments

#### 5.1 SEDD Puncture + Refill
```python
# Create punctured sequence
masked = create_masked_indices(clean_tokens, mask_interval=1)  # 50% masked

# Recover using SEDD reverse diffusion
recovered = reverse_diffusion_sedd(sedd_model, masked, class_labels)

# Evaluate
accuracy = classify_tokens(classifier, recovered, labels)
```

#### 5.2 DoT Prefix Completion
```python
# Keep only first K tokens as context
prefix = clean_tokens[:, :context_len]

# Complete using DoT autoregressive generation
completed = dot_prefix_completion(dot_model, prefix, labels, context_len)

# Evaluate
accuracy = classify_tokens(classifier, completed, labels)
```

### Files
- `mesh_transmit_semantic_channel.py` - Main evaluation script
- `run_all.py` - Unified pipeline runner

### Usage
```bash
# Full pipeline
python run_all.py

# Individual evaluation
python mesh_transmit_semantic_channel.py \
    --classifier_ckpt <path> \
    --sedd_ckpt <path> \
    --dot_ckpt <path> \
    --real_codes_path trash/data/val_codes.pt
```

---

## Model Comparison

| Model | Type | Conditioning | Generation | Best Use Case |
|-------|------|--------------|------------|---------------|
| **MeshVQVAE** | Autoencoder | None | Encoding/Decoding | Token extraction from meshes |
| **SEDD** | Diffusion | Class label | Parallel denoising | Puncture recovery (moderate missing %) |
| **DoT** | Autoregressive | BOS + Class prefix | Sequential | Prefix completion (high context %) |

---

## Token Flow

```
3D Point Cloud (ModelNet40)
    ↓
MeshVQVAE.encode()
    ↓
4096 Token Sequence
    ↓
┌──────────────────────────────────────┐
│  Train Classifier (baseline ~60%)    │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Train SEDD (denoising)              │
│  Train DoT (next-token prediction)     │
└──────────────────────────────────────┘
    ↓
Semantic Channel Evaluation
    ↓
Puncture → Recover → Classify → Report Accuracy
```

---

## Key Constants

```python
VOCAB_SIZE = 256          # Codebook size
SEQ_LEN = 4096            # Tokens per mesh
NUM_CLASSES = 40          # ModelNet40 categories
MASK_ID = 256             # Special mask token
BOS_TOKEN = 256           # Beginning of sequence (DoT)
```

---

## File Structure (Organized)

```
MeshGeneration/
├── models/
│   ├── autoregressive/
│   │   └── train_dot_mesh.py      # DoT training
│   ├── classifier/
│   │   ├── train_classifier.py    # Classifier training
│   │   └── evaluate.py            # Classifier evaluation
│   └── diffusion_model/
│       ├── SEDD.py                # SEDD model
│       └── train_sedd.py          # SEDD training
├── evaluation/
│   ├── generate_tokens.py         # Token generation
│   ├── generate_tokens_parallel.py # Parallel generation
│   └── semantic_channel/
│       ├── mesh_transmit_semantic_channel.py  # Semantic channel eval
│       └── run_all.py             # Unified pipeline
├── utils/
│   └── strip_class_tokens.py      # Utility scripts
└── mesh_vqvae/
    └── src/                       # VQVAE code
```

---

## Quick Reference

### Train Everything
```bash
cd evaluation/semantic_channel
python run_all.py
```

### Train Individual Models
```bash
# 1. Extract codes (if needed)
python models/diffusion/extract_fresh_codes.py --out_dir trash/data

# 2. Train classifier
python models/classifier/train_classifier.py \
    --tokens_path trash/data/train_codes.pt --mode conditional

# 3. Train SEDD
python models/diffusion/train_sedd.py --mode small --epochs 200

# 4. Train DoT
python models/autoregressive/train_dot_mesh.py --mode small --epochs 100

# 5. Evaluate semantic channel
python evaluation/semantic_channel/mesh_transmit_semantic_channel.py \
    --classifier_ckpt <path> --sedd_ckpt <path> --dot_ckpt <path> \
    --real_codes_path trash/data/val_codes.pt
```

---

*End of Documentation*
