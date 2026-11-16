"""Compute Kalman filter (KF) predictions for all subjects and sessions, saved in `BehavModeling/kalman_pred/`.

Runs `PreProParadigm.model_RTs.compute_likelihoods_at_deviants` for each (subject, session,
observation noise setting), in parallel over these combinations.

Observation noise settings (`sigma_r`, a standard deviation, squared into the covariance R):
  None    R estimated by EM          ->  kalman_predictions_sub-{sub}_ses-{sess}.csv
  'true'  R fixed at the session's   ->  kalman_predictions_sub-{sub}_ses-{sess}_sigma_r_{si_r}.csv
          own true si_r (read from its triallist)
  float   R fixed at this value      ->  same file name, with the given value as {si_r}
With --sigma-r, the fixed values replace 'true' and the EM variant is added (unless --no-em-fitted).

Outputs, per (subject, session, setting):
  - kalman_predictions_*.csv: predictions, plus the KF parameters at each timestep:
    A_{ctx}_{ij}, Q_{ctx}_{ij}, H_{ctx}_{j} and R_{ctx}, for ctx in {std, dev}
    (each context has its own filter). R_{ctx} is a variance: compare it to the true si_r squared.
  - kalman_predictions_and_likelihoods_at_deviants_*.csv: likelihoods at deviants.
All saved `sigma_*` columns are standard deviations.

Existing outputs are recomputed by default, so that a rerun always reflects the current code
(unlike `PreProParadigm/compute_predictions_and_likelihoods.py`, which skips existing files and
can keep outputs that are no longer valid). Use --skip-existing to resume an interrupted run.

Cost: the filter is O(T**2) (EM is refitted from scratch at every timestep). About 65 s per run of
T=480, so ~4.3 min per (sub, sess, sigma_r) task of 4 runs. 3 subs x 4 sessions x 2 settings
(24 tasks) take ~7-9 min on 16 cores.

Notes:
  - EM-fitted R is worse than fixed R, but not because the model is misspecified: the 2-state,
    H=[1,1] model can represent the process exactly (one state carries the level with a unit-root
    eigenvalue, the other the mean-reverting AR(1) part), and the true parameters give a much
    higher log-likelihood on real data than the EM fit. The cause is optimization: EM runs only
    n_iter=5 (see model_RTs.kalman_online_fit_predict_multicontext), starting from pykalman's
    default R=1.0, ~1e4 times the true R (~1e-4). The Kalman gain then starts near 0, the filter
    barely tracks, residuals stay large and the M-step keeps R large. EM lands at R ~0.06 (~570x
    the truth). With ~20+ iterations, R gets within a few-fold of the truth; initializing R near the
    data scale (e.g. 0.5 * var of the differenced std series) reaches ~6x at the same n_iter=5.
    The EM variant is left as is, since it is compared against the fixed-R variant.
  - Before `observation_covariance` was added to the EM variables, R was neither supplied nor
    estimated, and stayed at pykalman's default of 1.0.
  - The output also has `tau_hat_{ctx}`, the time constant implied by each context's A (from its
    non-unit eigenvalue, see model_RTs._tau_from_A), to compare with the triallists' tau_std /
    tau_dev. It is weakly identified: across the 48 runs of this cohort, Spearman rho ~0.27 with
    the true value, and the dev values stay near 1.7 for true tau_dev between 2 and 30.
    There is no `lim_hat`: `lim` lives in the state, not in A/Q/H/R.
  - DEFAULT_SIGMA_R_LEVELS currently contains only 'true' (no EM variant by default), which
    differs from the earlier default of EM + true.
"""

import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool

# Pin BLAS to one thread per worker, BEFORE numpy is imported (below, via model_RTs).
# We parallelize over tasks, so each worker only needs one core. Left unset, OpenBLAS spawns
# its own threads per worker (15 workers x 2 threads on 16 cores) and spin-waits on the
# contended ones -- which burns CPU without doing work. Measured: ~5x slower per task, to the
# point that no task finished in 21 min that takes 4.2 min standalone.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

# Reuse the KF pipeline's writer rather than reimplementing it.
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import compute_likelihoods_at_deviants, _sigma_r_suffix
import pandas as pd  # after the BLAS pinning above (imports numpy transitively)

TRIALLISTS_DIR = os.path.join(WORKSPACE, 'PreProParadigm', 'triallists')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'kalman_pred')

SUBS = ['04', '05', '06']
SESSIONS = [1, 2, 3, 4]

# The noise settings run per (sub, session) by default. Two entries, on purpose -- we compare
# the EM-estimated R against R pinned at the truth, not a testing over of si_r values:
#   None    -> EM estimates R itself (free to drift from the truth)
#   'true'  -> R fixed at THIS session's own true si_r, resolved per task from its triallist
# (see _run_task). Pass --sigma-r to fix R at explicit values instead.
DEFAULT_SIGMA_R_LEVELS = ['true']


def _triallist_path(sub, sess):
    """The triallist CSV for one (sub, sess), or None if none matches."""
    matches = glob.glob(os.path.join(TRIALLISTS_DIR, f'sub-{sub}', f'sub-{sub}_ses-{sess}*.csv'))
    return matches[0] if matches else None


def _true_sigma_r(trials_path):
    """This session's TRUE observation-noise std, read from the triallists' `sigma_r` column.

    That column is a single constant per session for the 04/05/06 cohort; this asserts it and
    returns a plain float, so the value both drives the KF (squared into R) and names the output
    `_sigma_r_{si_r}` -- identical to what the old fixed-sigma_r sweep wrote for that level, so
    downstream readers keyed on the session's true si_r keep finding these files.
    """
    vals = pd.read_csv(trials_path, usecols=['sigma_r'])['sigma_r'].dropna().unique()
    if len(vals) != 1:
        raise ValueError(f"{os.path.basename(trials_path)}: expected a single sigma_r, got {sorted(vals.tolist())}")
    return float(vals[0])


def _outputs(results_dir, sub, sess, sigma_r):
    """The (predictions, likelihoods) CSV pair a task writes."""
    suffix = _sigma_r_suffix(sigma_r)
    return (
        os.path.join(results_dir, f'kalman_predictions_sub-{sub}_ses-{sess}{suffix}.csv'),
        os.path.join(results_dir, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}{suffix}.csv'),
    )


def _run_task(task):
    """Compute one (sub, sess, sigma_r) cell. Returns a short status string for the log.

    sigma_r may be the sentinel 'true', meaning "use this session's own true si_r"; it is
    resolved here from the triallist, so the fixed-R fit lands at the session's real observation
    noise and its output file is named `_sigma_r_{that si_r}`.
    """
    sub, sess, sigma_r, results_dir, skip_existing = task

    # Glob first: resolving the 'true' sentinel and building output names both need the triallist.
    trials_path = _triallist_path(sub, sess)
    if trials_path is None:
        return f'MISS  sub-{sub} ses-{sess} sigma_r={sigma_r} (no triallist)'

    if sigma_r == 'true':
        try:
            sigma_r = _true_sigma_r(trials_path)
        except Exception as exc:
            return f'FAIL  sub-{sub} ses-{sess} sigma_r=true: {type(exc).__name__}: {exc}'

    label = f'sub-{sub} ses-{sess} sigma_r={sigma_r}'

    if skip_existing and all(os.path.exists(p) for p in _outputs(results_dir, sub, sess, sigma_r)):
        return f'SKIP  {label} (outputs exist)'

    # pykalman is noisy about masked arrays and EM convergence; the signal is in the outputs.
    warnings.filterwarnings('ignore')
    try:
        compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=results_dir, sigma_r=sigma_r)
    except Exception as exc:  # keep one bad cell from killing the whole grid
        return f'FAIL  {label}: {type(exc).__name__}: {exc}'
    return f'OK    {label}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subs', nargs='+', default=SUBS)
    parser.add_argument('--sessions', nargs='+', type=int, default=SESSIONS)
    parser.add_argument('--sigma-r', nargs='+', type=float, default=None,
                        help="Fixed observation-noise STDs to use instead of the default. Omit to "
                             "run the default per session: EM-fitted R + that session's OWN true si_r.")
    parser.add_argument('--no-em-fitted', action='store_true',
                        help="Skip the EM-fitted-R variant, leaving only the fixed-R fit(s) "
                             "(the session's true si_r by default, or the --sigma-r values).")
    parser.add_argument('--results-dir', default=RESULTS_DIR)
    parser.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument('--skip-existing', action='store_true',
                        help="Resume: leave already-written outputs alone. Off by default so a "
                             "rerun always reflects the current code.")
    args = parser.parse_args()

    if args.sigma_r is None:
        # Default: EM-fitted R + each session's OWN true si_r (DEFAULT_SIGMA_R_LEVELS), dropping
        # the EM-fitted (None) variant when --no-em-fitted is set.
        levels = [lvl for lvl in DEFAULT_SIGMA_R_LEVELS if lvl is not None or not args.no_em_fitted]
    else:
        levels = ([] if args.no_em_fitted else [None]) + list(args.sigma_r)

    os.makedirs(args.results_dir, exist_ok=True)
    tasks = [(sub, sess, sigma_r, args.results_dir, args.skip_existing)
             for sub in args.subs for sess in args.sessions for sigma_r in levels]

    print(f'{len(tasks)} tasks ({len(args.subs)} subs x {len(args.sessions)} sessions x '
          f'{len(levels)} noise settings) on {args.jobs} workers -> {args.results_dir}')
    print(f"noise settings per (sub,ses): {levels}  "
          f"(None = EM-fitted R; 'true' = the session's own si_r; floats are STDs, squared into R)")

    with Pool(args.jobs) as pool:
        for i, status in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            print(f'[{i}/{len(tasks)}] {status}', flush=True)

    print('done')


if __name__ == '__main__':
    main()
