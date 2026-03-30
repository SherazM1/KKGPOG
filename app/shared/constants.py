from __future__ import annotations

import re

LAST5_RE = re.compile(r"\b(\d{5})\b")
DIGITS_RE = re.compile(r"\D+")
N_COLS = 3
NAVY_RGB = (0.10, 0.16, 0.33)
DISPLAY_STANDARD = "Standard Flat Display"
DISPLAY_FULL_PALLET = "Full Pallet / Multi-Zone Display"
IMAGE_ANCHOR_ROW_0BASED = 5
