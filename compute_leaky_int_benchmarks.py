"""
Leaky-integrator benchmark — drop-in alternative to the Kalman-filter pipeline.

Computes predictive distributions (mu, sigma) at every trial using exponentially-weighted
running mean and std of past observations. Outputs match the KF schema so downstream code
works unchanged.

Design choices:
  - Deviants are masked (zero weight in running average, preserving timeline and tau interpretation)
  - Tau fit by online MLE on standards-only history (O(T²·n_tau) cost per run)
  - No deviant context (single-stream LI, so _dev columns are NaN)

Output files use FILE_PREFIX ('leaky_predictions') with KF-like naming.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm





# Single source of truth for the output filename stem, used both by
# compute_leaky_benchmarks_at_deviants (default) and by the __main__ skip-check, so the
# two can never drift apart. See the aggregate_data caveat in the module docstring.
FILE_PREFIX = "leaky_predictions"


from leaky_integrator import *


if __name__ == "__main__":

    subs = ['04', '05', '06']

    # Output directory for the leaky-integrator predictions. Files are named with
    # FILE_PREFIX; point `compare_RT_3subj_4sess.py`'s `preds_liks_path` at this folder
    # (and see the aggregate_data prefix caveat in the module docstring).
    results_save_path = os.path.join(os.path.dirname(__file__), FILE_PREFIX)

    for sub in tqdm(subs[::-1], desc="Subject"):

        n_sessions = len(os.listdir(os.path.join(os.path.dirname(__file__), f"triallists/sub-{sub}/")))

        for sess in tqdm(range(0, n_sessions), desc="Session", leave=False):

            trials_path = glob.glob(os.path.join(
                os.path.dirname(__file__), f"triallists/sub-{sub}/sub-{sub}_ses-{sess+1}*.csv"))

            pred_file = os.path.join(results_save_path, f'{FILE_PREFIX}_sub-{sub}_ses-{sess+1}.csv')
            likelihood_file = os.path.join(
                results_save_path, f'{FILE_PREFIX}_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv')

            if os.path.exists(pred_file) and os.path.exists(likelihood_file):
                print(f'Leaky results already exist for sub-{sub} ses-{sess+1}, skipping.')
                continue

            compute_leaky_benchmarks_at_deviants(trials_path[0], sub, sess+1, results_save_path)
