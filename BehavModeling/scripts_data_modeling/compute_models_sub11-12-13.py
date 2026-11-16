"""Compute KF + leaky-integrator predictions/likelihoods for the subjects 11-12-13 cohort.

This is the modeling half of the `compare_RT_3sub_4sess_noise_levels_indiv.py` analysis, run on
subjects 11/12/13 in `BehavModeling/logfiles2clem/`. The RT-vs-likelihood figure is deferred:
those trial-design CSVs carry NO behavioural responses, and the only logfiles bearing these
subject numbers (`RNN_MRI/logfiles/`) are a different paradigm (~1500 Hz vs ~620 Hz tones,
deviant positions {4,5,6} vs {2..6}, non-aligning trial counts) -- so no reaction times match
these designs yet. When matching RTs arrive, the outputs written here feed straight into the
existing RT-comparison code.

Two structural differences from the 04/05/06 cohort, handled by `_combine_runs` below:

  1. ONE FILE PER RUN, not one triallist per session. The four `sub-XX_ses-1_run-{1..4}_trials.csv`
     are concatenated in run order into a single session triallist.
  2. `run_n` is 0 in every file and `trial_n` restarts at 0 each run. The KF/leaky code groups by
     `run_n` (and would otherwise fold all four runs into one 1920-row "run", never resetting the
     filter). We therefore reassign run_n = 0..3 and make trial_n global 0..239, matching the
     existing `PreProParadigm/triallists` convention exactly, so the combined file is a drop-in.

Observation noise: every session here is sigma_r = 0.02 (a STANDARD DEVIATION, read from the
`sigma_r` column). The KF is fixed at that true value -- squared into R by
`compute_likelihoods_at_deviants` -- and the leaky integrator floors its predictive std at the
same 0.02 (read from the column), so neither model has an information advantage. This mirrors the
"each session's true observation noise" (si_r) logic of the reference script.

Cost: the KF is O(T**2), ~65 s per run of T=480, so ~4.3 min per subject of 4 runs. Three
subjects on 3 workers is ~4-5 min; the leaky pass is comparatively negligible.
"""

import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

# Pin BLAS to one thread per worker BEFORE numpy is imported (below, via the model modules).
# We parallelize over subjects, so each worker only needs one core; left unset, OpenBLAS spawns
# its own threads per worker and spin-waits on the contended ones, burning CPU without doing work.
# (Same rationale as compute_kf_predictions.py.)
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import compute_likelihoods_at_deviants, _sigma_r_suffix
from compute_leaky_int_benchmarks import compute_leaky_benchmarks_at_deviants

# --- Paths (this cohort is kept self-contained under BehavModeling) ---
HERE = os.path.dirname(os.path.abspath(__file__))
LOGFILES2CLEM_DIR = os.path.join(HERE, 'logfiles2clem')
COMBINED_TRIALS_DIR = os.path.join(HERE, 'triallists_logfiles2clem')  # combined per-session triallists
KF_OUT_DIR = os.path.join(HERE, 'kalman_pred')                        # same folder as the 04/05/06 KF outputs
LEAKY_OUT_DIR = os.path.join(HERE, 'leaky_pred')

SUBS = ['11', '12', '13']
SESS = 1                # single session per participant
N_RUNS = 4
SIGMA_R = 0.02          # true observation-noise STD for every session in this cohort
TONES_PER_TRIAL = 8


def _combine_runs(sub, logfiles_dir=LOGFILES2CLEM_DIR, out_dir=COMBINED_TRIALS_DIR):
    """Concatenate the four per-run CSVs for `sub` into one session triallist and save it.

    Reassigns run_n = 0..3 and trial_n = global 0..(60*N_RUNS-1), matching the existing
    `PreProParadigm/triallists` convention (per-run run_n, session-global trial_n) so the combined
    file is a drop-in for the KF/leaky code. Returns the written path.
    """
    frames = []
    trial_offset = 0
    for run_idx in range(N_RUNS):
        matches = glob.glob(os.path.join(logfiles_dir, f'sub-{sub}_ses-*_run-{run_idx + 1}_trials.csv'))
        if not matches:
            raise FileNotFoundError(f'No trial file for sub-{sub} run-{run_idx + 1} in {logfiles_dir}')
        run = pd.read_csv(sorted(matches)[0])

        n_tones = len(run)
        if n_tones % TONES_PER_TRIAL != 0:
            raise ValueError(f'sub-{sub} run-{run_idx + 1}: {n_tones} rows not divisible by {TONES_PER_TRIAL}')
        n_trials = n_tones // TONES_PER_TRIAL

        # Each trial must be TONES_PER_TRIAL consecutive rows sharing one trial_n (relied on by
        # prepare_trials_data, which derives relative_pos = tile(arange(8), n_trials) from row order).
        if not (run['trial_n'].values == np.repeat(np.arange(n_trials), TONES_PER_TRIAL)).all():
            raise ValueError(f'sub-{sub} run-{run_idx + 1}: rows are not ordered as consecutive 8-tone trials')

        run = run.copy()
        run['run_n'] = run_idx                                    # 0..3 (was 0 in every file)
        run['trial_n'] = run['trial_n'] + trial_offset           # make trial_n session-global
        trial_offset += n_trials
        frames.append(run)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'sub-{sub}_ses-{SESS}_trials.csv')
    combined.to_csv(out_path, index=False)
    return out_path


def _run_subject(sub):
    """Build the combined triallist for one subject, then compute KF and leaky outputs."""
    warnings.filterwarnings('ignore')  # pykalman is noisy about masked arrays / EM convergence
    try:
        trials_path = _combine_runs(sub)

        # Kalman filter fixed at the session's true observation noise (sigma_r squared into R).
        compute_likelihoods_at_deviants(
            trials_path, sub, SESS, results_save_path=KF_OUT_DIR, sigma_r=SIGMA_R)

        # Leaky integrator; s_floor defaults to each run's own sigma_r column (0.02 here).
        compute_leaky_benchmarks_at_deviants(
            trials_path, sub, SESS, results_save_path=LEAKY_OUT_DIR)
    except Exception as exc:  # keep one bad subject from killing the batch
        return f'FAIL  sub-{sub}: {type(exc).__name__}: {exc}'
    return f'OK    sub-{sub}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subs', nargs='+', default=SUBS)
    parser.add_argument('--jobs', type=int, default=min(len(SUBS), os.cpu_count() or 1))
    args = parser.parse_args()

    for d in (COMBINED_TRIALS_DIR, KF_OUT_DIR, LEAKY_OUT_DIR):
        os.makedirs(d, exist_ok=True)

    suffix = _sigma_r_suffix(SIGMA_R)
    print(f'{len(args.subs)} subjects x 1 session, sigma_r={SIGMA_R} (KF suffix "{suffix}") on {args.jobs} workers')
    print(f'  combined triallists -> {COMBINED_TRIALS_DIR}')
    print(f'  KF predictions      -> {KF_OUT_DIR}')
    print(f'  leaky predictions   -> {LEAKY_OUT_DIR}')

    with Pool(args.jobs) as pool:
        for i, status in enumerate(pool.imap_unordered(_run_subject, args.subs), 1):
            print(f'[{i}/{len(args.subs)}] {status}', flush=True)

    print('done')


if __name__ == '__main__':
    main()
