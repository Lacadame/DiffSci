"""Host-specific paths used by the `pipelines/vae/*.py` scripts.

Defaults match the layout on Danilo's lab box; override with CLI flags or
edit here if you're running elsewhere. These intentionally do *not* live
inside `diffsci2/` because they encode a particular machine's data
layout (especially the regressor cache, the raw stone volumes, and the
notebook-side metrics CSV).
"""
from __future__ import annotations

import os

# Where training-run checkpoints + logs go (overridable via CLI).
CKPT_DIR = (
    "/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfnai/scripts/"
    "vaeporesft/checkpoints"
)
LOG_DIR = (
    "/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfnai/scripts/"
    "vaeporesft/logs"
)
PLOT_DIR = (
    "/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfnai/scripts/"
    "vaeporesft/plots"
)

# Where the 256^3 reference CSVs (from the 0003 notebook) live.
EXISTING_CSV_DIR = (
    "/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfnai/0003-data"
)
