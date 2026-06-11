"""
Central path configuration for all figure-generation and analysis scripts.

Override defaults via environment variables:

    PMEM_FIGDIR    — where to write output figures (default: ./figures)
    PMEM_CACHEDIR  — where to cache intermediate results  (default: ./cache)
    PMEM_DATADIR   — root data directory (default: ./data)
    MNE_DATA       — MNE-Python data directory (default: ~/mne_data)

    PMEM_SEDATION_DATA — path to Chennu propofol sedation dataset
                         (available upon request from the original authors)
                         default: <PMEM_DATADIR>/Sedation-RestingState

Example (bash):
    export PMEM_FIGDIR=/path/to/output/figures
    export PMEM_SEDATION_DATA=/path/to/chennu_data
    python sedation_analysis.py
"""
import os
from pathlib import Path

# Repository root (directory containing this file)
ROOT = Path(__file__).parent

FIGDIR   = Path(os.environ.get("PMEM_FIGDIR",   ROOT / "figures"))
CACHEDIR = Path(os.environ.get("PMEM_CACHEDIR", ROOT / "cache"))
DATADIR  = Path(os.environ.get("PMEM_DATADIR",  ROOT / "data"))

# MNE auto-download location (PhysioNet data is fetched automatically by MNE)
MNE_DATA = Path(os.environ.get("MNE_DATA", Path.home() / "mne_data"))

# Chennu propofol sedation dataset (restricted; available on request)
SEDATION_DATA = Path(os.environ.get("PMEM_SEDATION_DATA", DATADIR / "Sedation-RestingState"))

FIGDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR.mkdir(parents=True, exist_ok=True)
