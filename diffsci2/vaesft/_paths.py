"""Default paths for the SFT subpackage's external data (regressor checkpoint,
regressor cached chunks/targets, raw stone volumes).

These are environment-specific and will change if you move the data. The SFT
classes accept these as constructor arguments — these are just the defaults
that match the current notebook layout on `/home/ubuntu/repos/DiffSci2/`.

If `poreregressor` ever ships as an installable package, these defaults
should move to `poreregressor.data_constants` and be imported from there.
For now they live here so the SFT engine is self-contained.
"""
from __future__ import annotations

import os
import sys


# --- poreregressor bootstrap ---------------------------------------------
# The reward regressor lives under
# `notebooks/exploratory/dfnai/scripts/poreregressor/`. Add the parent
# directory to sys.path so `import poreregressor` works regardless of
# where the calling script lives.

_POREREGRESSOR_PARENT = (
    "/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfnai/scripts"
)
if _POREREGRESSOR_PARENT not in sys.path:
    sys.path.insert(0, _POREREGRESSOR_PARENT)


# --- Default paths -------------------------------------------------------

DEFAULT_REGRESSOR_CKPT = os.path.join(
    _POREREGRESSOR_PARENT,
    "poreregressor/checkpoints/run02/epoch023-r20.9801.ckpt",
)

DEFAULT_REGRESSOR_TRAIN_CHUNKS = os.path.join(
    _POREREGRESSOR_PARENT, "poreregressor/cache/chunks_train.npz"
)
DEFAULT_REGRESSOR_TRAIN_TARGETS = os.path.join(
    _POREREGRESSOR_PARENT, "poreregressor/cache/targets_train.npy"
)
DEFAULT_REGRESSOR_TEST_CHUNKS = os.path.join(
    _POREREGRESSOR_PARENT, "poreregressor/cache/chunks_test.npz"
)
DEFAULT_REGRESSOR_TEST_TARGETS = os.path.join(
    _POREREGRESSOR_PARENT, "poreregressor/cache/targets_test.npy"
)
