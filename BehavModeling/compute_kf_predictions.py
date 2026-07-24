"""Generate Kalman-filter predictions into `BehavModeling/kalman_pred/`.

Driver for `PreProParadigm.model_RTs.compute_likelihoods_at_deviants`, run across
subjects x sessions x observation-noise settings and parallelized over those tasks.

Two kinds of output, distinguished by filename suffix:

  EM-fitted R   sigma_r=None  ->  kalman_predictions_sub-{sub}_ses-{sess}.csv
      The KF estimates the observation noise itself. Note this only does anything now that
      `observation_covariance` is back in the EM variable list: previously R was neither
      supplied nor estimated and silently sat at pykalman's default of 1.0, against a true
      R of ~1e-4. Even now, EM lands at ~0.06 (~570x the true 1e-4) and this variant stays
      worse than the fixed-sigma_r ones -- but NOT because the model is misspecified. The
      2-state / H=[1,1] model CAN represent the OU exactly (one state carries the level with
      a unit-root eigenvalue, the other the mean-reverting AR(1) part; H sums them). Scored
      on real data, the true parameters give a far higher loglik than the EM fit. The real
      cause is optimization: EM runs only n_iter=5 (see model_RTs.kalman_online_fit_predict_
      multicontext) starting from pykalman's default R=1.0, which is ~1e4x too big. From that
      init the Kalman gain starts near 0, the filter barely tracks, smoothing residuals stay
      large, and the M-step keeps R large; 5 iterations do not undo it. Given ~20+ iterations
      R falls to within a few-x of the truth. Cheaper still: initializing R near the data
      scale (e.g. 0.5*var(diff of the std series)) reaches within ~6x of the truth at the
      SAME n_iter=5. So this variant is fixable (more iterations or a better R init), not
      fundamentally limited. It is left as-is here because we compare against the fixed-
      sigma_r variant, which sidesteps the R estimation entirely.

  Fixed R       sigma_r=X     ->  kalman_predictions_sub-{sub}_ses-{sess}_sigma_r_{X}.csv
      sigma_r is a STANDARD DEVIATION on the scale of the triallists' `sigma_r` column; it is
      squared into R by compute_likelihoods_at_deviants. By default X is each session's OWN true
      si_r, read from its triallist (0.005 or 0.01 here), so R is pinned at the truth rather than
      swept -- pass --sigma-r to fix it at other values instead.

Every saved `sigma_*` column is a standard deviation.

Both CSVs also carry the per-timestep fitted KF parameters as extra columns: A_{ctx}_{ij},
Q_{ctx}_{ij}, H_{ctx}_{j} and R_{ctx} for ctx in {std, dev} (each context has its own filter).
R_{ctx} is the fitted observation-noise VARIANCE -- compare it against the true si_r SQUARED:
pinned at it in the fixed-R variant, or EM-estimated and free to drift in the EM-fitted one.

Unlike `PreProParadigm/compute_predictions_and_likelihoods.py`, this recomputes by default --
that script's skip-on-exists silently preserves stale files across code changes. Pass
--skip-existing if you actually want to resume an interrupted run.

Cost: the filter is O(T**2) (EM refits from scratch at every timestep), ~65s per run of T=480,
so ~4.3 min per (sub, sess, sigma_r) task of 4 runs. The default 3x4x2 grid (EM-fitted + true
si_r per session, 24 tasks) is ~7-9 min on 16 cores.
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
# the EM-estimated R against R pinned at the truth, not a sweep of si_r values:
#   None    -> EM estimates R itself (free to drift from the truth)
#   'true'  -> R fixed at THIS session's own true si_r, resolved per task from its triallist
# (see _run_task). Pass --sigma-r to fix R at explicit values instead.
DEFAULT_SIGMA_R_LEVELS = [None, 'true']


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
