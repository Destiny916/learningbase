#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_CODES = ROOT / "inference_codes"
if str(INFERENCE_CODES) not in sys.path:
    sys.path.insert(0, str(INFERENCE_CODES))

from hand_scalar import main


if __name__ == "__main__":
    main()
