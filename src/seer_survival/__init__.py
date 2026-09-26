"""Survival analysis and 5-year mortality prediction on the SEER breast cancer cohort."""
import os

# joblib/loky runs `wmic` to count physical cores, and newer Windows builds no longer
# include it. Giving loky a value below the logical core count (about the physical
# count on hyper-threaded CPUs) skips that lookup and its warning.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

__version__ = "1.0.0"
