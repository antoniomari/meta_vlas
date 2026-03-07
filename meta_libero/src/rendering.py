"""Rendering utilities re-exported from `meta_libero.rendering`."""

from meta_libero import rendering as _rendering
from meta_libero.rendering import *  # noqa: F403

# `import *` excludes underscore-prefixed symbols; re-export this legacy helper.
_draw_step_on_frame = _rendering._draw_step_on_frame

