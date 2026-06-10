# Semantic Channel for 3D Point Cloud Tokens

End-to-end pipeline for semantic channel evaluation using MeshVQVAE, SEDD, and DoT.

## Quick Start

```bash
# Run entire pipeline (from scratch)
cd evaluation/semantic_channel
python run_all.py

# Or run individual components - see READMEs below
```

## Model READMEs

| Component | Path | Description |
|-----------|------|-------------|
| **VQ-VAE** | [mesh_vqvae/README.md](mesh_vqvae/README.md) | 3D mesh to token encoder |
| **Classifier** | [models/classifier/README.md](models/classifier/README.md) | Token sequence classifier |
| **SEDD** | [models/diffusion/README.md](models/diffusion/README.md) | Diffusion model for tokens |
| **DoT** | [models/autoregressive/README.md](models/autoregressive/README.md) | Autoregressive model |
| **Evaluation** | [evaluation/README.md](evaluation/README.md) | Testing & semantic channel |

## Pipeline Overview

```
3D Mesh → VQ-VAE → Tokens → [Classifier | SEDD | DoT] → Evaluation
```

## Key Specs

- **Sequence length**: 4096 tokens per mesh
- **Codebook**: 256 entries (0-255)
- **Classes**: 40 (ModelNet40)

## Branches

- `main` - Stable version (original structure)
- `experimental` - Reorganized structure (testing)

See individual READMEs for detailed training commands.
