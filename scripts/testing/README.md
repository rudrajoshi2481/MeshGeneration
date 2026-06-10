# Testing Scripts

## Overview

Quick test scripts to verify everything works before running full training.

## Scripts

| Script | Purpose | Time | Usage |
|--------|---------|------|-------|
| `quick_test.py` | Test all models (1 epoch each) | ~5 min | `python quick_test.py` |
| `test_plots.py` | Verify plot generation | ~3 min | `python test_plots.py` |
| `verify_setup.py` | Check data and dependencies | <1 min | `python verify_setup.py` |

## When to Use

### Before Full Training
```bash
# Quick sanity check (5 min)
python scripts/testing/quick_test.py

# If it passes, run full training (hours)
python scripts/training/train_all.py
```

### Debug Plot Issues
```bash
# Test plot generation specifically
python scripts/testing/test_plots.py
```

## Test Coverage

- **quick_test.py**: Trains each model for 1 epoch on small data subset
- **test_plots.py**: Verifies all 3 model types generate expected plot files
- **verify_setup.py**: Checks data files exist, imports work, GPU available
