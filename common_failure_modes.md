# Common Failure Modes Encountered So Far
**Date:** April 28, 2026

1. **Missing local dependencies**
   Standard packages (`numpy`, `scikit-learn`, etc.) were unavailable at first

2. **Execution-time overhead from plotting/import side effects**
   Non-essential imports (e.g., `matplotlib` in evaluation utilities) add cache/setup delays.

3. **Metric-goal mismatch risk**
   Iterations optimize for AUC while assignment expectations may also depend on recall/F1 behavior.

4. **Limited iteration diversity**
   Only a few model families are tested in each cycle, which can miss better-performing regions.
