# Mamba-Quaternion-Lite

Prototype PyTorch implementation of the *Mamba-Quaternion-Lite* architecture
described in the accompanying paper notes. The model keeps the continuous-time
SSM dynamics scalar (matching Mamba-2) while using quaternion-valued injections
(`B_t`) and projections (`C_t`) to add geometric expressivity without altering
the parallel scan assumptions.

## What's included

* Quaternion helpers (`mambaqclite/quaternion.py`) with multiplication,
  normalization, and isotropic initialization.
* A minimal Mamba-Quaternion-Lite block implementing scalar bilinear
  discretization with quaternion gates (`mambaqclite/mql.py`).
* A compact language model wrapper tailored to TinyStories-scale experiments
  with a GPT-2 style vocabulary size of 2048.

## Quickstart

Install dependencies (PyTorch required) and run a small shape check:

```bash
python - <<'PY'
import torch
from mambaqclite import MQLConfig, MambaQuaternionLiteModel

config = MQLConfig(d_model=256, n_layers=2, state_dim=8, max_seq_len=32)
model = MambaQuaternionLiteModel(config)
input_ids = torch.randint(0, config.vocab_size, (2, 16))
logits = model(input_ids)
print(logits.shape)
PY
```

This prints `torch.Size([2, 16, 2048])`, confirming the forward pass works with
random data.

## Next steps

* The block already uses a vectorized scalar SSM recurrence to avoid Python
  loops; swapping in the official Mamba-2 scan or fused kernels could bring
  further speedups.
* Add a TinyStories dataloader and training harness using the provided
  GPT-2-style tokenizer (vocab size 2048).
* Experiment with quaternion initializations and regularization strategies for
  stability.
