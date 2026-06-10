# Unified Test Script

## Quick Test (1 Epoch Each)

```bash
# Test everything (needs train_codes.pt and val_codes.pt in trash/data/)
python test_all_models.py --data_dir trash/data --out_dir trash/test_runs

# Test specific components only
python test_all_models.py --skip_sedd --skip_dot  # Classifier only
python test_all_models.py --skip_classifier --skip_dot  # SEDD only
```

## What It Tests

| Test | Purpose | Time |
|------|---------|------|
| **Token Shapes** | Verifies [N, 4096] shape, vocab [0-255] | <1 sec |
| **Puncture Logic** | Unit tests SEDD/DoT masking | <1 sec |
| **Classifier** | 1 epoch on 128 samples | ~1 min |
| **SEDD** | 1 epoch on 32 samples | ~2 min |
| **DoT** | 1 epoch on 32 samples | ~2 min |

## Requirements

Data files in `trash/data/`:
- `train_codes.pt` - Training tokens [N, 4096]
- `val_codes.pt` - Validation tokens [M, 4096]

## Output

```
trash/test_runs/
├── test_classifier/
│   ├── training.log        # Full training output
│   └── checkpoints/        # Model checkpoints
├── test_sedd/
│   ├── training.log        # Full training output
│   └── checkpoints/
├── test_dot/
│   ├── training.log        # Full training output
│   └── checkpoints/
├── all_tests.log           # Combined log (all models)
└── test_report.json        # Summary report
```

## Log Files

Each model test generates detailed logs:

**Individual logs** (in each test_*/ folder):
- Full command executed
- Timestamp
- Complete stdout/stderr output
- Return code

**Combined log** (`all_tests.log`):
- All individual logs concatenated
- Useful for debugging the entire pipeline

## Report Format

```json
{
  "timestamp": "2025-...",
  "results": {
    "token_shapes": true,
    "puncture_logic": true,
    "classifier": true,
    "sedd": true,
    "dot": true
  },
  "passed": 5,
  "total": 5,
  "success_rate": 1.0,
  "log_files": {
    "classifier": "test_classifier/training.log",
    "sedd": "test_sedd/training.log",
    "dot": "test_dot/training.log",
    "combined": "all_tests.log"
  }
}
```

## Verification

All models use:
- **Sequence length**: 4096 tokens
- **Vocab size**: 256 (0-255) + 1 mask token (256)
- **Classes**: 40 (ModelNet40)

Puncture refill tested with:
- SEDD: interval masking [1, 2, 4, 8, 16, 32]
- DoT: prefix completion [4096, 2048, 512, 128, 32]
