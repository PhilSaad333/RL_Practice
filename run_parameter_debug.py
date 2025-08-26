#!/usr/bin/env python3
"""Quick runner for parameter debugging."""

import subprocess
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Set environment to reduce verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Run the debug script
subprocess.run([sys.executable, "entropy_experiments/debug_parameters.py"])