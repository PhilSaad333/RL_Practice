"""
Resolve dataset cache / artifact locations without hard-coding Google Drive.
"""

import os
from pathlib import Path

BASE = Path(os.environ.get("DATA_ROOT", "./datasets")).expanduser()
BASE.mkdir(parents=True, exist_ok=True)
