# -*- coding: utf-8 -*-
"""
Makes `robust_kernels` importable during tests without requiring
`pip install -e .` first, by adding `src/` to sys.path.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
