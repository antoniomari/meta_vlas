"""Result directory roots."""

import os
from pathlib import Path


def results_root() -> Path:
    return Path(os.getenv("META_LIBERO_RESULTS_DIR", "meta_libero/results"))
