#!/usr/bin/env python
"""
Deprecated — use ``python main.py cube`` instead.

This script redirects to the new experiment launcher.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Rewrite argv to look like: python main.py cube [original args...]
sys.argv = ["main.py", "cube"] + sys.argv[1:]

from main import main  # noqa: E402
main()
