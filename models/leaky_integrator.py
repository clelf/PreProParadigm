import os
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

# Reuse the exact same data preparation as the KF pipeline.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_RTs import prepare_trials_data


# Minimum number of standards in the history before we trust a prediction, mirroring the
# KF's MIN_OBS_FOR_EM warm-up so the two models stay aligned (early predictions are NaN).
MIN_OBS_FOR_LI = 3

FILE_PREFIX = "leaky_predictions"



def leaky_average_masked(y, is_std, taus, s_floor=0.0):
    """Masked leaky integrator: predictive mean / std / log-lik for every tau, every t.

    For each t, the prediction uses observations j < t with recency weight
    exp(-(t-j)/tau), with deviant timesteps (is_std[j] == False) given zero weight and the
    remaining weights renormalized. Weights that reference the current tau grid are computed
    for all taus at once (vectorized over the tau axis); the O(t) history loop makes this
    O(T**2 * n_tau) overall.

    Parameters
    ----------
    y : array (T,)
        Observation stream for one run.
    is_std : bool array (T,)
        True where the observation is a standard (included in the running estimate),
        False at deviants (masked out).
    taus : array (n_tau,)
        Strictly-positive candidate integration time constants.
    s_floor : float
        Lower bound on the predictive std, applied as s = max(s_slid, s_floor) BEFORE logp.
        Default 0.0 (no floor) -- KEEP IT THAT WAY unless you have re-measured the effects
        below; this used to default to sigma_r and that was a mistake. See the `s_floor`
        note in the module docstring for the measurements.

    Returns
    -------
    z, s, logp : arrays (T, n_tau)
        Predictive mean, predictive std (post-floor), and predictive log-likelihood of y[t]
        under N(z[t], s[t]**2). Entries with no usable history are NaN.
    """
    y = np.asarray(y, dtype=float)
    is_std = np.asarray(is_std, dtype=bool)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    T, m = len(y), len(taus)
    mask = is_std.astype(float)  # 1 for standards, 0 for deviants

    z = np.full((T, m), np.nan)
    s = np.full((T, m), np.nan)

    inv_tau = 1.0 / taus
    for t in range(1, T):
        j = np.arange(t)                                  # history indices 0..t-1
        dist = (t - j).astype(float)                      # recency distance >= 1
        # (t, m) weights: exp(-dist/tau) with deviants zeroed out
        w = np.exp(-np.outer(dist, inv_tau)) * mask[j][:, None]
        wsum = w.sum(0)                                   # (m,)
        valid = wsum > 0
        if not np.any(valid):
            continue
        wn = w[:, valid] / wsum[valid]                    # (t, n_valid), renormalized
        yv = y[j][:, None]                                # (t, 1)
        zt = (wn * yv).sum(0)                             # (n_valid,)
        st = np.sqrt((wn * (yv - zt[None, :]) ** 2).sum(0))
        z[t, valid] = zt
        s[t, valid] = st

    # Floor the predictive std at the known observation noise (NaNs are preserved by max).
    if s_floor > 0:
        s = np.maximum(s, s_floor)

    # Predictive log-likelihood of y[t] under N(z, s**2); s==0 (degenerate) -> NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        s_safe = np.where(s > 0, s, np.nan)
        logp = -0.5 * np.log(2 * np.pi) - np.log(s_safe) - 0.5 * ((y[:, None] - z) / s_safe) ** 2
    return z, s, logp


def leaky_online_fit_predict(y, is_std, taus=None, min_obs=MIN_OBS_FOR_LI, s_floor=0.0):
    """Online-MLE leaky integrator: recursively fit tau at each t and predict.

    At each t we select the tau maximizing the cumulative predictive log-likelihood over
    the standards strictly before t (no look-ahead), then read off that tau's prediction
    for t. Predictions are produced only once at least `min_obs` standards precede t.

    Parameters
    ----------
    y : array (T,)
        Observation stream for one run.
    is_std : bool array (T,)
        True at standards, False at deviants.
    taus : array (n_tau,), optional
        Candidate tau grid. Defaults to a log-spaced grid in [1, T//2].
    min_obs : int
        Minimum number of preceding standards before a prediction is emitted.
    s_floor : float
        Lower bound on the predictive std (see leaky_average_masked). Default 0.0 (no floor).
        NOTE: any nonzero value is applied to `logp` BEFORE tau is selected below, so it does
        not merely adjust the reported uncertainty -- it changes which tau wins. See the
        `s_floor` note in the module docstring.

    Returns
    -------
    z_hat, s_hat, logp_hat, tau_hat : arrays (T,)
        Chosen predictive mean, std, log-likelihood of y[t], and selected tau. Warm-up /
        unpredictable timesteps are NaN.
    """
    y = np.asarray(y, dtype=float)
    is_std = np.asarray(is_std, dtype=bool)
    T = len(y)

    if taus is None:
        taus = np.unique(np.clip(np.round(np.geomspace(1, max(2, T // 2), 250)).astype(int), 1, max(1, T - 1)))
    taus = np.atleast_1d(np.asarray(taus, dtype=float))

    z, s, logp = leaky_average_masked(y, is_std, taus, s_floor=s_floor)

    # Count of standards strictly before each t.
    std_count = np.cumsum(is_std.astype(int))
    std_count_before = np.concatenate([[0], std_count[:-1]])
    predictable = std_count_before >= min_obs

    # Fitting evidence: predictive log-lik at standard, predictable timesteps only.
    # NaNs/inf (warm-up, degenerate std) contribute nothing to the cumulative sum.
    contrib = np.where(is_std[:, None] & predictable[:, None], logp, 0.0)
    contrib = np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    # Cumulative evidence over j < t (exclude current t), shape (T, n_tau).
    cum_before = np.cumsum(contrib, axis=0) - contrib

    z_hat = np.full(T, np.nan)
    s_hat = np.full(T, np.nan)
    logp_hat = np.full(T, np.nan)
    tau_hat = np.full(T, np.nan)

    idx = np.where(predictable)[0]
    if idx.size:
        best = np.argmax(cum_before[idx], axis=1) # chosen tau index per predictable t
        z_hat[idx] = z[idx, best]
        s_hat[idx] = s[idx, best]
        logp_hat[idx] = logp[idx, best]
        tau_hat[idx] = taus[best]

    return z_hat, s_hat, logp_hat, tau_hat


def compute_leaky_benchmarks_at_deviants(trials_path, sub, sess, results_save_path=None,
                                         taus=None, file_prefix=FILE_PREFIX, s_floor=0.0):
    """Leaky-integrator analogue of `model_RTs.compute_likelihoods_at_deviants`.

    Produces the same two CSVs (same columns, same filenames by default) as the KF pipeline
    so downstream code reads them unchanged. `_dev`-context columns are NaN (see module doc).

    Args:
        trials_path: Path to the run's trials CSV (glob-resolved by the caller).
        sub, sess: Subject id / 1-indexed session number (for filenames).
        results_save_path: Output directory (created if needed). Defaults to script dir.
        taus: Optional tau grid passed to the online fit; default is per-run log-spaced.
        file_prefix: Filename stem (defaults to the module-level FILE_PREFIX). Pass the same
            value as `aggregate_data`'s `file_prefix` to read these files back.
        s_floor: Lower bound on the predictive std. Default 0.0 = no floor, which is what you
            want (see the `s_floor` note in the module docstring). Pass None to reproduce the
            superseded behaviour of flooring at each run's own sigma_r -- that is how the
            `leaky_predictions_sfloor_*` files were generated, kept only for comparison.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    else:
        os.makedirs(results_save_path, exist_ok=True)

    trials = prepare_trials_data(trials_path)

    # Full-sequence predictions, mirroring the KF results schema exactly.
    results = {
        'run_id': [], 'trial_no': [],
        'mu_pred': [], 'sigma_pred': [],
        'mu_pred_std': [], 'sigma_pred_std': [],
        'mu_pred_dev': [], 'sigma_pred_dev': [],
    }
    # Predictions & likelihoods evaluated at deviant positions.
    results_devpos = {
        'mu_pred_std_at_dev': [], 'sigma_pred_std_at_dev': [],
        'mu_pred_dev_at_dev': [], 'sigma_pred_dev_at_dev': [],
        'likelihood_obs_std_at_dev': [], 'likelihood_obs_dev_at_dev': [],
        'dpos': [], 'trial_no': [],
    }

    for run_id in tqdm(trials['run_n'].unique(), desc="Run", leave=False):
        run = trials[trials['run_n'] == run_id]

        observations = run['observation'].to_numpy(dtype=float)
        contexts = run['contexts'].to_numpy(dtype=bool)   # True at deviant positions
        is_std = ~contexts                                # standards feed the running estimate

        # Floor the predictive std at this run's observation noise sigma_r unless overridden.
        run_s_floor = float(run['sigma_r'].iloc[0]) if s_floor is None else s_floor

        # Online-MLE leaky integrator on the standards-only stream.
        z_hat, s_hat, logp_hat, _tau_hat = leaky_online_fit_predict(
            observations, is_std, taus=taus, s_floor=run_s_floor)

        # Full-sequence predictions. Single (standard) context: mu_pred == mu_pred_std,
        # sigma columns hold standard deviations; the `_dev` context does not exist.
        results['run_id'].append(np.full(len(z_hat), run_id))
        results['trial_no'].append(run['trial_no'].to_numpy())
        results['mu_pred'].append(z_hat)
        results['sigma_pred'].append(s_hat)
        results['mu_pred_std'].append(z_hat)
        results['sigma_pred_std'].append(s_hat)
        results['mu_pred_dev'].append(np.full(len(z_hat), np.nan))
        results['sigma_pred_dev'].append(np.full(len(z_hat), np.nan))

        # At deviant positions: standard-context prediction, and likelihood of the (deviant)
        # observation under that standard predictive Gaussian = exp(predictive log-lik).
        mu_pred_std_at_dev = z_hat[contexts]
        sigma_pred_std_at_dev = s_hat[contexts]
        likelihood_obs_std_at_dev = np.exp(logp_hat[contexts])
        n_dev = int(contexts.sum())

        results_devpos['trial_no'].append(run['trial_no'][contexts].values)
        results_devpos['dpos'].append(run['dpos'][contexts].values)
        results_devpos['mu_pred_std_at_dev'].append(mu_pred_std_at_dev)
        results_devpos['sigma_pred_std_at_dev'].append(sigma_pred_std_at_dev)
        results_devpos['mu_pred_dev_at_dev'].append(np.full(n_dev, np.nan))
        results_devpos['sigma_pred_dev_at_dev'].append(np.full(n_dev, np.nan))
        results_devpos['likelihood_obs_std_at_dev'].append(likelihood_obs_std_at_dev)
        results_devpos['likelihood_obs_dev_at_dev'].append(np.full(n_dev, np.nan))

    results = pd.DataFrame({k: np.concatenate(v) for k, v in results.items()})
    results.to_csv(os.path.join(results_save_path, f'{file_prefix}_sub-{sub}_ses-{sess}.csv'), index=False)

    results_devpos = pd.DataFrame({k: np.concatenate(v) for k, v in results_devpos.items()})
    results_devpos.to_csv(
        os.path.join(results_save_path, f'{file_prefix}_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}.csv'),
        index=False,
    )