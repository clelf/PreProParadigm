import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from scipy import stats as ss
from matplotlib.axes import Axes
import re
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from models.baseline_models import (
    prior_tone_is_dev_given_prev_rule_and_prev_stds,
    prior_tone_is_dev_given_prev_rule_cue_and_prev_stds,
    prior_tone_is_dev_given_prev_stds,
    prior_tone_is_dev_given_rule_and_prev_stds,
    prob_null_trial,
)

# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman_fit_em import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext
    from Kalman.kalman_core import likelihood_observation, contexts_to_probabilities
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman_fit_em import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext
    from Kalman.kalman_core import likelihood_observation, contexts_to_probabilities
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")


def prepare_trials_data(trials_path, exclude_null=False):
    # Load generated sequences
    trials = pd.read_csv(trials_path)
    
    # Standardize trial indexing: rename 'trial_n' to 'trial_no' if it exists
    if 'trial_n' in trials.columns and 'trial_no' not in trials.columns:
        trials.rename(columns={'trial_n': 'trial_no'}, inplace=True)
    
    trials['dpos'] = trials['dpos'].fillna(0).astype(int) 
    n_trials = len(trials)/8
    # If trial_no doesn't exist yet, create it; otherwise keep the loaded one
    if 'trial_no' not in trials.columns:
        trials['trial_no'] = np.repeat(np.arange(int(n_trials)), 8) # 0 to n_trials-1 for each run
    
    trials['relative_pos'] = np.tile(np.arange(8), int(n_trials)) # 0 to 7 for each run, to match dpos indexing
    if exclude_null:
        trials = trials[trials['dpos'] != 0].reset_index(drop=True)
    trials['contexts'] = trials['relative_pos'] == trials['dpos'] # 0: standard, 1: deviant
    # dpos = 0 is actually not a deviant but an omission, so convert it to standard
    if not exclude_null:
        # if null rule trials have been kept, still flag their position as standard
        trials.loc[trials.dpos == 0, 'contexts'] = False # TODO: but was I even using the contexts column later on?
    return trials



def _sigma_r_suffix(sigma_r):
    """Filename suffix of the KF prediction files computed at a fixed observation-noise std
    `sigma_r` (empty when R is fitted by EM)."""
    return "" if sigma_r is None else f"_sigma_r_{sigma_r}"


# Context labels in the names of the fitted KF parameter columns (one set of parameters per context)
_KF_CTX_NAMES = ('std', 'dev')


def _tau_from_A(A_stack):
    """Time constant tau implied by one context's fitted transition matrices, at every timestep.

    `A_stack` has shape (T, 2, 2). The generative process is x[t] = a * x[t-1] + b with
    a = 1 - 1/tau, so tau = 1 / (1 - a). The fitted A is only defined up to a change of basis, so
    `a` is read from its eigenvalues: one is ~1 (the state component carrying the constant `lim`),
    the other is `a`.

    Returns an array of length T, NaN where tau is undefined (context not fitted yet, complex or
    non-decaying eigenvalue).
    """
    tau = np.full(len(A_stack), np.nan)
    usable = np.isfinite(A_stack).all(axis=(1, 2))
    if not usable.any():
        return tau

    eigenvalues = np.linalg.eigvals(A_stack[usable]).astype(complex)  # (n_usable, 2)

    # Keep the eigenvalue furthest from 1 (the one closest to 1 belongs to `lim`)
    order = np.argsort(np.abs(eigenvalues - 1.0), axis=1)
    a = np.take_along_axis(eigenvalues, order[:, 1:2], axis=1)[:, 0]

    decaying = (np.abs(a.imag) < 1e-12) & (a.real < 1.0)
    resolved = np.full(len(a), np.nan)
    resolved[decaying] = 1.0 / (1.0 - a.real[decaying])
    tau[usable] = resolved
    return tau


def _kf_param_columns(kf_params):
    """Fitted KF parameters of every context (A, Q, H, R and the implied tau), as named columns
    of length T.

    `kf_params` is the dict returned by kalman_online_fit_predict_multicontext(...,
    return_params=True). Names carry the context and matrix entry, e.g. A_std_01, R_dev,
    tau_hat_std. R is a variance (compare it with sigma_r**2). Values are NaN before a context
    is first fitted.
    """
    A, Q, H, R = kf_params['A'], kf_params['Q'], kf_params['H'], kf_params['R']
    n_ctx = A.shape[1]
    cols = {}
    for c in range(n_ctx):
        ctx = _KF_CTX_NAMES[c] if c < len(_KF_CTX_NAMES) else str(c)
        for i in range(A.shape[2]):
            for j in range(A.shape[3]):
                cols[f'A_{ctx}_{i}{j}'] = A[:, c, i, j]
                cols[f'Q_{ctx}_{i}{j}'] = Q[:, c, i, j]
        for j in range(H.shape[3]):
            cols[f'H_{ctx}_{j}'] = H[:, c, 0, j]
        cols[f'R_{ctx}'] = R[:, c, 0, 0]
        cols[f'tau_hat_{ctx}'] = _tau_from_A(A[:, c])
    return cols


def compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=None, sigma_r=None):
    """Run the multi-context KF on one session; save its predictions, and its predictions and
    likelihoods at the deviant positions.

    The key output is `likelihood_obs_std_at_dev`, the likelihood of each deviant under the
    standard context's prediction. Writes two CSVs (all tones / deviant positions only) and
    returns the deviant-position DataFrame.

    sigma_r: fixed observation-noise std (the KF's R is its square); None lets EM fit R. Every
        saved `sigma_*` column is a std.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    else:
        os.makedirs(results_save_path, exist_ok=True)

    # The KF takes R as a variance
    observation_noise = None if sigma_r is None else float(sigma_r) ** 2
    
    # Prepare data
    trials = prepare_trials_data(trials_path)

    # Prepare storage for predictions across runs
    results = {
        'run_id': [],
        'mu_pred': [],
        'sigma_pred': [],
        'mu_pred_std': [],
        'sigma_pred_std': [],
        'mu_pred_dev': [],
        'sigma_pred_dev': [],
        'trial_no': []
    }

    # Prepare storage for predictions and likelihoods at deviant position across runs
    results_devpos = {
        'mu_pred_std_at_dev': [], # TODO: check that called like that
        'sigma_pred_std_at_dev': [],
        'mu_pred_dev_at_dev': [],
        'sigma_pred_dev_at_dev': [],
        'likelihood_obs_std_at_dev': [],
        'likelihood_obs_dev_at_dev': [],
        'dpos': [],
        'trial_no': []
    }

    # Separate into runs
    for run_id in tqdm(trials['run_n'].unique(), desc="Run", leave=False):
        run = trials[trials['run_n'] == run_id]

        # Get variables
        observations = run['observation']
        states_std = run['state_std']
        states_dev = run['state_dev']
        dpos = run['dpos']
        contexts = run['contexts']
        rules = run['rule']

        # Convert context labels to prior context probabilites for multicontext KF
        ctx_probabilities = contexts_to_probabilities(contexts, n_ctx=2) # shape (T, n_ctx)

        # Run the multi-context KF. sigma_pred is a std, per_ctx_var holds variances.
        mu_pred, sigma_pred, kfs_fitted, per_ctx_mu_pred, per_ctx_var, kf_params = kalman_online_fit_predict_multicontext(
            observations, ctx_probabilities, n_iter=5, return_per_ctx=True, return_params=True, observation_noise=observation_noise
        )
        per_ctx_var_std = per_ctx_var[:, 0]
        per_ctx_var_dev = per_ctx_var[:, 1]

        # Save predictions (every saved `sigma_*` column is a std)
        results['run_id'].append(np.full(len(mu_pred), run_id))
        results['trial_no'].append(run['trial_no'])
        results['mu_pred'].append(mu_pred)
        results['sigma_pred'].append(sigma_pred)
        results['mu_pred_std'].append(per_ctx_mu_pred[:, 0])
        results['sigma_pred_std'].append(np.sqrt(per_ctx_var_std))
        results['mu_pred_dev'].append(per_ctx_mu_pred[:, 1])
        results['sigma_pred_dev'].append(np.sqrt(per_ctx_var_dev))

        # Fitted KF parameters at every timestep (e.g. to compare R with sigma_r**2)
        run_param_cols = _kf_param_columns(kf_params)
        for col, values in run_param_cols.items():
            results.setdefault(col, []).append(values)

        # Predictions of each context at the deviant positions (NaN until each context's KF is
        # first fitted, after MIN_OBS_FOR_EM observations)
        mu_pred_std_at_dev = per_ctx_mu_pred[contexts, 0]
        var_std_at_dev = per_ctx_var_std[contexts]
        mu_pred_dev_at_dev = per_ctx_mu_pred[contexts, 1]
        var_dev_at_dev = per_ctx_var_dev[contexts]

        # Likelihood of each deviant under each context's prediction (likelihood_observation
        # takes a variance)
        observations_at_dev = observations[contexts]
        likelihood_obs_std_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=mu_pred_std_at_dev,
            sigma=var_std_at_dev
        )

        likelihood_obs_dev_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=mu_pred_dev_at_dev,
            sigma=var_dev_at_dev
        )

        # Save predictions and likelihoods at the deviant positions
        results_devpos['trial_no'].append(run['trial_no'][contexts].values) # TODO: verify if correct
        results_devpos['mu_pred_std_at_dev'].append(mu_pred_std_at_dev)
        results_devpos['sigma_pred_std_at_dev'].append(np.sqrt(var_std_at_dev))
        results_devpos['mu_pred_dev_at_dev'].append(mu_pred_dev_at_dev)
        results_devpos['sigma_pred_dev_at_dev'].append(np.sqrt(var_dev_at_dev))
        results_devpos['likelihood_obs_std_at_dev'].append(likelihood_obs_std_at_dev)
        results_devpos['likelihood_obs_dev_at_dev'].append(likelihood_obs_dev_at_dev)
        results_devpos['dpos'].append(run['dpos'][contexts].values)

        # Fitted KF parameters at the deviant positions
        for col, values in run_param_cols.items():
            results_devpos.setdefault(f'{col}_at_dev', []).append(values[contexts.values])

    ## Concatenate results across runs
    # Predictions over entire sequences
    results = pd.DataFrame({
        k: np.concatenate(v) for k, v in results.items()
    })

    noise_suffix = _sigma_r_suffix(sigma_r)

    results.to_csv(os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess}{noise_suffix}.csv'), index=False)

    # Predictions and likelihoods at deviant locations
    results_devpos = pd.DataFrame({
        k: np.concatenate(v) for k, v in results_devpos.items()
    })
    results_devpos.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}{noise_suffix}.csv'), index=False)

    return results_devpos



def compute_dev_likelihoods_over_dpos(trials_path, results_df, sub, sess, results_save_path=None,
                                      inplace=False, p_no_dev=0.0):
    """Add the global deviant-position prior to the deviant-position results, and the likelihood
    weighted by it (`likelihood_obs_std_at_dev_over_dpos`).

    The prior is the probability that a position holds a standard, given that the previous tones
    of the trial were standards.
    p_no_dev: probability that a trial has no deviant: prob_null_trial(pi_rule) for cohorts
        with a null rule, 0.0 otherwise.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)

    # Load exp data
    trials = prepare_trials_data(trials_path)

    # Prior that each position holds the deviant (or a standard), given that the previous tones
    # were standards
    trials['prior_dpos_dev'] = trials['relative_pos'].apply(
        lambda x: prior_tone_is_dev_given_prev_stds(x, p_no_dev=p_no_dev))
    trials['prior_dpos_std'] = 1 - trials['prior_dpos_dev']

    # Priors at the deviant positions
    results_df['prior_dpos_std'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_std'].reset_index(drop=True)
    results_df['prior_dpos_dev'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_dev'].reset_index(drop=True)

    # Likelihood weighted by the prior
    results_df['likelihood_obs_std_at_dev_over_dpos'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_std']

    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}.csv'), index=False)
    return results_df



def compute_dev_likelihoods_over_rules(trials_path, results_df, sub, sess, pi_rule, results_save_path=None, inplace=False, cue_rule_lik=None):
    """Add the rule-based deviant-position prior to the deviant-position results, and the
    likelihood weighted by it (`likelihood_obs_std_at_dev_over_rules`).

    The prior depends on the previous trial's rule through the transition matrix `pi_rule`. When
    the trials have a `cue` column, the cue also informs the rule belief; the prior without the
    cue is then kept for comparison in the `_nocue` columns.
    cue_rule_lik: {cue -> {rule -> P(cue | rule)}}, only used with a `cue` column.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    
    # Load exp data
    trials = prepare_trials_data(trials_path)
    if "prev_rule" not in trials.columns or "proba_dpos" not in trials.columns:
        # Rule of the previous trial (looked up by trial number, not by previous row)
        trials['prev_rule'] = (trials['trial_no'] - 1).map(dict(zip(trials['trial_no'], trials['rule']))).astype('Int64')

        # Prior that each position holds the deviant, given the previous rule, and the cue if
        # the trials have one
        if 'cue' in trials.columns:
            trials['prior_dpos_rules_dev'] = trials.apply(
                lambda row: prior_tone_is_dev_given_prev_rule_cue_and_prev_stds(
                    row['relative_pos'], pi_rule, row['prev_rule'], row['cue'], cue_rule_lik)
                if pd.notna(row['prev_rule']) else np.nan, axis=1)
            # Same prior without the cue, to isolate the effect of the cue
            trials['prior_dpos_rules_dev_nocue'] = trials.apply(
                lambda row: prior_tone_is_dev_given_prev_rule_and_prev_stds(row['relative_pos'], pi_rule, row['prev_rule'])
                if pd.notna(row['prev_rule']) else np.nan, axis=1)
        else:
            trials['prior_dpos_rules_dev'] = trials.apply(
                lambda row: prior_tone_is_dev_given_prev_rule_and_prev_stds(row['relative_pos'], pi_rule, row['prev_rule'])
                if pd.notna(row['prev_rule']) else np.nan, axis=1)

        # Prior given each rule
        trials['prior_dpos_rule0'] = trials.apply(lambda row: prior_tone_is_dev_given_rule_and_prev_stds(row['relative_pos'], r=0), axis=1)
        trials['prior_dpos_rule1'] = trials.apply(lambda row: prior_tone_is_dev_given_rule_and_prev_stds(row['relative_pos'], r=1), axis=1)

        # Prior that each position holds a standard
        trials['prior_dpos_rules_std'] = 1 - trials['prior_dpos_rules_dev']
        if 'prior_dpos_rules_dev_nocue' in trials.columns:
            trials['prior_dpos_rules_std_nocue'] = 1 - trials['prior_dpos_rules_dev_nocue']
    
    # Rules and priors at the deviant positions
    results_df['rule'] = trials[trials['contexts']==True].drop_duplicates()['rule'].reset_index(drop=True)
    results_df['prev_rule'] = trials[trials['contexts']==True].drop_duplicates()['prev_rule'].reset_index(drop=True)
    results_df['prior_dpos_rule0'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rule0'].reset_index(drop=True)
    results_df['prior_dpos_rule1'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rule1'].reset_index(drop=True)
    # results_df['p_rule']=results_df.apply(lambda x: pi_rule[int(x['prev_rule']), int(x['rule'])] if pd.notna(x['prev_rule']) else np.nan, axis=1)
    results_df['prior_dpos_rules_std'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_std'].reset_index(drop=True)
    results_df['prior_dpos_rules_dev'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_dev'].reset_index(drop=True)

    # Likelihood weighted by the prior
    results_df['likelihood_obs_std_at_dev_over_rules'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_rules_std']

    # Same without the cue (cohorts with a cue only)
    if 'prior_dpos_rules_std_nocue' in trials.columns:
        results_df['prior_dpos_rules_std_nocue'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_std_nocue'].reset_index(drop=True)
        results_df['likelihood_obs_std_at_dev_over_rules_nocue'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_rules_std_nocue']

    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}.csv'), index=False)
    return results_df


# Above this value, `response_time_dev` is an absolute timestamp (~1e9 s) rather than an RT
_ABS_GETSECS_THRESHOLD = 1e6

# Time between the onsets of two consecutive tones (tone duration 0.1 s + ISI 0.65 s)
_TONE_SOA = 0.1 + 0.65

# Deviant position named by each response key, for the two keyboard layouts used (letters,
# numbers)
_KEY_TO_DPOS = {'v': 2, 'y': 3, 'u': 4, 'i': 5, 'l': 6,
                '9': 2, '1': 3, '2': 4, '3': 5, '4': 6}


def _first_float(x):
    """First float in x, whether x is already a number or a string like '[1781096969.46]'."""
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(x))
    return float(m.group()) if m else np.nan


def _concat_run_logs(logfiles_path, sub, sess, n_run=4):
    """Logfiles of all runs of one subject and session, concatenated in run order.

    Handles both naming conventions: behavioural logs (`sub-XX-ses-N*runR*.tsv`) and fMRI event
    logs (`sub-XX_ses-N*run-R*.tsv`).
    """
    logy = []
    for r in range(0, n_run):
        matches = (glob.glob(f"{logfiles_path}/sub-{sub}-ses-{sess}*run{r+1}*.tsv")
                   or glob.glob(f"{logfiles_path}/sub-{sub}_ses-{sess}*run-{r+1}*.tsv"))
        if not matches:
            raise FileNotFoundError(
                f"No logfile for sub-{sub} ses-{sess} run {r+1} in {logfiles_path}")
        logy.append(pd.read_csv(sorted(matches)[0], sep="\t"))
    return pd.concat(logy, ignore_index=True)


def _deviant_rt(logs):
    """Reaction time to the deviant, for every logfile row.

    Behavioural logs store it as `rt_getsecs_dev` (or `rt_getsecs`). fMRI logs store
    `response_time_dev`, either as an absolute timestamp (sub-11; the run's trigger time is
    subtracted) or already relative to the deviant (sub-12/13); the two are told apart by
    magnitude.
    """
    if 'response_time_dev' in logs.columns:
        rtd = logs['response_time_dev']
        trig = logs['trigger_times'].map(_first_float)
        is_absolute = rtd.abs() > _ABS_GETSECS_THRESHOLD
        return rtd.where(~is_absolute, rtd - trig)
    for col in ('rt_getsecs_dev', 'rt_getsecs'):
        if col in logs.columns:
            return logs[col]
    raise ValueError("No reaction-time column found "
                     "(expected 'response_time_dev', 'rt_getsecs_dev', or 'rt_getsecs')")


def _identified_dpos(key_pressed):
    """Deviant position named by each response key (NaN for no or unknown key); see
    `_KEY_TO_DPOS`."""
    def _pos(k):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            return np.nan
        s = str(k).strip()
        if s.endswith('.0'):  # numeric key read as e.g. 9.0
            s = s[:-2]
        return _KEY_TO_DPOS.get(s, np.nan)
    return key_pressed.map(_pos)


def load_log_RTs(logfiles_path, sub, sess, n_run=4):
    """Per-trial reaction-time-to-deviant (one value per trial) from a subject/session's logfiles."""
    logs = _concat_run_logs(logfiles_path, sub, sess, n_run)
    logs = logs.assign(rt=_deviant_rt(logs))
    # Remove trials without deviant (as in the likelihoods), then keep one row per trial
    logs = logs[logs['dpos'] != 0]
    return logs['rt'][::8].reset_index(drop=True)


def load_logs(logfiles_path, sub, sess, filter=False, n_run=4):
    logs = _concat_run_logs(logfiles_path, sub, sess, n_run)
    logs = logs.assign(rt=_deviant_rt(logs))

    # Remove trials without deviant (as in the likelihoods)
    logs = logs[logs['dpos'] != 0]

    # RT to the tone named by the key: rt + (dpos - named position) * _TONE_SOA (NaN if no valid
    # key)
    if 'key_pressed' in logs.columns:
        identified = _identified_dpos(logs['key_pressed'])
        logs = logs.assign(pressed_tone_rt=logs['rt'] + (logs['dpos'] - identified) * _TONE_SOA)
    else:
        logs = logs.assign(pressed_tone_rt=np.nan)

    # Trial number over the whole session, when the logs lack one (fMRI logs)
    if 'trial_no' not in logs.columns:
        logs = logs.reset_index(drop=True)
        logs['trial_no'] = logs.index // 8

    if filter:
        # Remove missing and negative RTs
        logs = logs.dropna(subset=['rt'])
        logs = logs[logs['rt'] >= 0]

    # Keep the columns used in the analyses, one row per trial
    relevant_columns = ['trial_no', 'dpos', 'rule', 'runs', 'correct', 'confidence', 'rt', 'pressed_tone_rt', 'key_pressed']
    logs = logs[[col for col in relevant_columns if col in logs.columns]].drop_duplicates().reset_index(drop=True)

    return logs


def _correct_dpos_mask(df):
    """Boolean mask of the trials where the subject identified the deviant's position correctly.

    The logfiles' `correct` column is coded differently across cohorts (numbers, words, or
    numbers read as strings in the fMRI logs), hence the test on both the number 1 and the word
    'correct'. Follows the paradigm's own scoring: right key pressed outside the response window
    ('too slow') counts as incorrect.
    """
    if 'correct' not in df.columns:
        raise ValueError(
            "correcttrials=True needs the logfiles' `correct` column, but it is absent from the "
            "aggregated df -- check that `load_logs` kept it for this cohort.")
    col = df['correct']
    return (pd.to_numeric(col, errors='coerce') == 1) | \
           (col.astype(str).str.strip().str.lower() == 'correct')


def compare_likelihoods_with_RTs(results_df, logfiles_path, sub, sess, RT_results_path=None, n_run=4):
    """Scatter of RT vs KF likelihood at the deviants, for one subject and session.

    Early version, superseded by the multi-subject figures below: reads behavioural logs only and
    ignores the global deviant-position prior.
    """
    if RT_results_path is None:
        RT_results_path = os.path.dirname(__file__)
    else:
        os.makedirs(RT_results_path, exist_ok=True)
    
    logy = []

    for r in range(0, n_run):
        logfile_path = glob.glob(f"{logfiles_path}/sub-{sub}-ses-{sess}*run{r+1}*.tsv")
        log = pd.read_csv(logfile_path[0], sep="\t")
        logy.append(log)

    if not logy:
        raise ValueError("No logfiles were loaded. Check logfiles_path and file patterns.")
    
    logs = pd.concat(logy, ignore_index=True)
    
    # Remove trials without deviant (as in the likelihoods)
    logs = logs[logs['dpos'] != 0]

    # RT, one value per trial
    if 'rt_getsecs_dev' in logs.columns:
        rt = logs['rt_getsecs_dev'][::8].reset_index(drop=True)
    elif 'rt_getsecs' in logs.columns:
        rt = logs['rt_getsecs'][::8].reset_index(drop=True)
    else:
        raise ValueError("No reaction time column found (expected 'rt_getsecs_dev' or 'rt_getsecs')")


    # Trials with a valid likelihood and RT
    losad = np.array(results_df['likelihood_obs_std_at_dev'])
    rt = np.array(rt)
    mask = ~np.isnan(losad) & ~np.isnan(rt) & (rt>=0)
    losad_clean = losad[mask]
    rt_clean = rt[mask]

    # RT vs likelihood_obs_std_at_dev
    plt.figure()
    plt.scatter(losad_clean, rt_clean)
    slope, intercept = np.polyfit(losad_clean, rt_clean, 1)
    plt.plot(losad_clean, slope * losad_clean + intercept, color='red', linewidth=2)  # regression line
    plt.xlabel("likelihood_obs_std_at_dev")
    plt.ylabel("RT")
    plt.title(f"Correlation coef: {ss.pearsonr(losad_clean, rt_clean)[0]:.2f}, p-value: {ss.pearsonr(losad_clean, rt_clean)[1]:.2e}")
    plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_std_sub-{sub}_ses-{sess}.png'))
    plt.close()

    # RT vs likelihood_obs_std_at_dev_over_rules
    if 'likelihood_obs_std_at_dev_over_rules' in results_df.columns:
        lor = np.array(results_df['likelihood_obs_std_at_dev_over_rules'])
        mask = ~np.isnan(lor) & ~np.isnan(rt) & (rt>=0)
        lor_clean = lor[mask]
        rt_clean = rt[mask]

        plt.figure()
        plt.scatter(lor_clean, rt_clean)
        slope, intercept = np.polyfit(lor_clean, rt_clean, 1)
        plt.plot(lor_clean, slope * lor_clean + intercept, color='red', linewidth=2)  # regression line
        plt.xlabel("likelihood_obs_std_at_dev_over_rules")
        plt.ylabel("RT")
        plt.title(f"Correlation coef: {ss.pearsonr(lor_clean, rt_clean)[0]:.2f}, p-value: {ss.pearsonr(lor_clean, rt_clean)[1]:.2e}")
        plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_std_rules_sub-{sub}_ses-{sess}.png'))
        plt.close()
        
    
    print(f"RT comparison plots saved to: {RT_results_path}")


# Response-locked mode (allvalidtrials): columns read by the RT figures, and the columns at the
# tone named by the key that replace them
_AT_PRESSED_ALIASES = {
    'rt': 'pressed_tone_rt',
    'likelihood_obs_std_at_dev': 'likelihood_obs_std_at_pressed',
    'prior_dpos_std': 'prior_dpos_std_at_pressed',
    'prior_dpos_rules_std': 'prior_dpos_rules_std_at_pressed',
    'likelihood_obs_std_at_dev_over_dpos': 'likelihood_obs_std_at_pressed_over_dpos',
    'likelihood_obs_std_at_dev_over_rules': 'likelihood_obs_std_at_pressed_over_rules',
}


def _response_locked_columns(df_all, df_trials_preds, pi_rule):
    """Likelihood and priors at the tone the subject's key named, instead of at the true deviant.

    Response-locked counterparts of the deviant-locked columns (see `_AT_PRESSED_ALIASES`),
    computed with the same functions, so both agree on trials where the named tone is the
    deviant. NaN where no valid key was pressed.
    """
    pressed = _identified_dpos(df_all['key_pressed']).to_numpy(dtype=float)

    # Observation and standard-context prediction at the named tone (NaN if no valid key)
    key = pd.DataFrame({'trial_no': df_all['trial_no'].to_numpy(), 'relative_pos': pressed})
    at = key.merge(
        df_trials_preds[['trial_no', 'relative_pos', 'observation', 'mu_pred_std', 'sigma_pred_std']],
        on=['trial_no', 'relative_pos'], how='left')

    # likelihood_observation takes a variance; sigma_pred_std is a std
    lik = likelihood_observation(
        y=at['observation'].to_numpy(dtype=float),
        mu=at['mu_pred_std'].to_numpy(dtype=float),
        sigma=at['sigma_pred_std'].to_numpy(dtype=float) ** 2)

    # Priors at the named position, computed as in compute_dev_likelihoods_over_dpos / _over_rules
    p_no_dev = prob_null_trial(pi_rule)
    prev_rule = df_all['prev_rule']
    cue = df_all['cue'] if 'cue' in df_all.columns else None
    prior_std = np.full(len(df_all), np.nan)
    prior_rules_std = np.full(len(df_all), np.nan)
    for i, d in enumerate(pressed):
        if not np.isfinite(d):
            continue
        d = int(d)
        prior_std[i] = 1.0 - prior_tone_is_dev_given_prev_stds(d, p_no_dev=p_no_dev)
        r_prev = prev_rule.iloc[i]
        if pd.notna(r_prev):
            if cue is not None:
                dev = prior_tone_is_dev_given_prev_rule_cue_and_prev_stds(d, pi_rule, r_prev, cue.iloc[i])
            else:
                dev = prior_tone_is_dev_given_prev_rule_and_prev_stds(d, pi_rule, r_prev)
            prior_rules_std[i] = 1.0 - dev

    out = pd.DataFrame({
        'likelihood_obs_std_at_pressed': lik,
        'prior_dpos_std_at_pressed': prior_std,
        'prior_dpos_rules_std_at_pressed': prior_rules_std,
    }, index=df_all.index)
    out['likelihood_obs_std_at_pressed_over_dpos'] = (
        out['likelihood_obs_std_at_pressed'] * out['prior_dpos_std_at_pressed'])
    out['likelihood_obs_std_at_pressed_over_rules'] = (
        out['likelihood_obs_std_at_pressed'] * out['prior_dpos_rules_std_at_pressed'])
    return out


def aggregate_data(trials_path, logfiles_path, subject, session_num, likelihoods_path=None, pi_rule=None, file_prefix='kalman_predictions', sigma_r=None, allvalidtrials=False):
    """Trial information, model predictions, likelihoods and priors, and behaviour (logfiles) of
    one subject and session, merged into one DataFrame with one row per deviant.

    This is the input of every RT figure below.
    file_prefix: stem of the prediction files: 'kalman_predictions' (KF) or 'leaky_predictions'
        (LI).
    sigma_r: observation-noise std at which the KF predictions were computed (selects the file
        suffixed `_sigma_r_{sigma_r}`); None for the KF with R fitted by EM, and for the LI.
    allvalidtrials: response-locked mode: `rt` and the model columns are taken at the tone the
        subject's key named (see `_response_locked_columns`).
    """
    suffix = _sigma_r_suffix(sigma_r)

    # Trials file of either cohort, in a per-subject folder or not
    trials_file = (glob.glob(f"{trials_path}/sub-{subject}/sub-{subject}_ses-{session_num}*trials.csv")
                   or glob.glob(f"{trials_path}/sub-{subject}_ses-{session_num}*trials.csv"))[0]
    trials = prepare_trials_data(trials_file)

    if likelihoods_path is not None and pi_rule is not None:
        preds = pd.read_csv(
            os.path.join(likelihoods_path, f'{file_prefix}_sub-{subject}_ses-{session_num}{suffix}.csv'),
            index_col=False
        )
        preds_liks_dev = pd.read_csv(
            os.path.join(likelihoods_path, f'{file_prefix}_and_likelihoods_at_deviants_sub-{subject}_ses-{session_num}{suffix}.csv'),
            index_col=False
        )
        # Add the priors (p_no_dev is 0 for cohorts without a null rule)
        preds_liks_dev = compute_dev_likelihoods_over_dpos(
            trials_file, preds_liks_dev, subject, session_num,
            p_no_dev=prob_null_trial(pi_rule))
        preds_liks_dev = compute_dev_likelihoods_over_rules(trials_file, preds_liks_dev, subject, session_num, pi_rule, results_save_path=None, inplace=False)

        # Keep a single `trial_no` column (the one of `trials`), so the merges below don't fail
        if 'trial_no' in preds.columns:
            preds = preds.drop(columns=['trial_no'])
        if 'trial_no' in preds_liks_dev.columns:
            preds_liks_dev = preds_liks_dev.drop(columns=['trial_no'])

    logs = load_logs(logfiles_path, subject, session_num, filter=False, n_run=4)

    # Trials and predictions at all tones; then, at the deviant positions only, the log columns
    # not already present and the deviant-position results (rows aligned by index)
    df_trials_preds = pd.merge(trials, preds, left_index=True, right_index=True)
    cols = logs.columns.difference(df_trials_preds.columns)
    df_all = pd.merge(df_trials_preds[df_trials_preds['contexts']==True].reset_index(drop=True), logs[cols], left_index=True, right_index=True)
    df_all = pd.merge(df_all, preds_liks_dev.drop(columns=['dpos', 'rule']), left_index=True, right_index=True)

    # Response-locked mode: replace the analysis columns by their values at the tone named by the
    # key (unchanged on trials where that tone is the deviant)
    if allvalidtrials and likelihoods_path is not None and pi_rule is not None:
        df_all = pd.concat([df_all, _response_locked_columns(df_all, df_trials_preds, pi_rule)], axis=1)
        for canonical, pressed in _AT_PRESSED_ALIASES.items():
            if canonical in df_all.columns and pressed in df_all.columns:
                df_all[canonical] = df_all[pressed]

    return df_all


def get_session_types(trials_path):
    """Session types of the 04/05/06 cohort, read from the trials filenames.

    Returns the sorted list of (d, si_stat, si_r) configurations, and a dict mapping each
    configuration to {subject: session number}.
    """

    # Find unique session types (configurations of d, si_stat, si_r)
    all_trial_files = glob.glob(f"{trials_path}/sub-*/sub-*_ses-*-*trials.csv")
    session_types = set()
    session_type_to_params = {}  # Map from (d, si_stat, si_r) to dict of {subject: session_number}
    
    for trial_file in all_trial_files:
        filename = os.path.basename(trial_file)
        
        # Session type parameters
        param_match = re.search(r'-d(\d+)-si_stat([\d.]+)-si_r([\d.]+)', filename)
        # Subject and session number
        sub_match = re.search(r'sub-(\d+)_ses-(\d+)', filename)
        
        if param_match and sub_match:
            d = int(param_match.group(1))
            si_stat = float(param_match.group(2))
            si_r = float(param_match.group(3))
            session_type = (d, si_stat, si_r)
            
            sub = sub_match.group(1)
            sess_num = int(sub_match.group(2))  # 1-indexed session number from filename
            
            session_types.add(session_type)
            if session_type not in session_type_to_params:
                session_type_to_params[session_type] = {}
            session_type_to_params[session_type][sub] = sess_num
    
    session_types = sorted(list(session_types))
    
    return session_types, session_type_to_params


# For later retrieval:
# def _fit_label(x, y):
    # """Build a regplot legend label summarising the fit of likelihood x vs RT y.

    # Reports Pearson rho and its p-value, R^2 (= Pearson rho squared, the fraction of
    # RT variance the linear fit explains), and Spearman rho. Spearman is rank-based and
    # therefore invariant to any monotonic transform of x or y, so computing it here on the
    # (possibly log-transformed) columns gives the exact same value as raw, pre-log RT.
    # """
    # r, p = ss.pearsonr(x, y)
    # rs, _ = ss.spearmanr(x, y)
    # return rf"$\rho$: {r:.2f}, $R^2$: {r**2:.2f}, $\rho_s$: {rs:.2f}, p: {p:.1g}"


def _p_stars(p):
    """Significance stars of a p-value (GraphPad convention, from 'ns' to '****')."""
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


# Correlations available for the rho annotated on the RT figures (`corr`). Every R2 is computed
# with Pearson, whatever `corr` is.
CORR_FUNCS = {'pearson': ss.pearsonr, 'spearman': ss.spearmanr}


def _corr_stats(x, y, corr='pearson'):
    """(rho, p-value) between x and y, Pearson or Spearman according to `corr`.

    Only the Pearson rho squared is an R2.
    """
    key = str(corr).lower()
    if key not in CORR_FUNCS:
        raise ValueError(f"corr must be one of {sorted(CORR_FUNCS)}, got {corr!r}")
    r, p = CORR_FUNCS[key](x, y)
    return float(r), float(p)


def _corr_symbol(corr='pearson'):
    """Math symbol for the chosen rho: plain for Pearson, subscript-s for Spearman."""
    return r"$\rho_s$" if str(corr).lower() == 'spearman' else r"$\rho$"


def _corr_suffix(corr='pearson'):
    """Figure-name suffix of the annotated correlation: '_spearman', or empty for Pearson."""
    return '_spearman' if str(corr).lower() == 'spearman' else ''


def _fit_label(x, y, corr='pearson'):
    """Legend label of a likelihood vs RT regression line: rho (per `corr`), the OLS R2 and
    significance stars."""
    r, p = _corr_stats(x, y, corr)
    r_ols, _ = _corr_stats(x, y, 'pearson')
    return rf"{_corr_symbol(corr)}: {r:.2f}, $R^2$: {r_ols**2:.2f} {_p_stars(p)}"


def compare_likelihoods_with_RTs_global(subjects, trials_path, preds_liks_path, logfiles_path,
                                        comparison_save_path, pi_rule, take_dpos=True, take_rules=False,
                                        loglik=False, logrt=False, file_prefix='kalman_predictions',
                                        leaky_preds_path=None, leaky_file_prefix='leaky_predictions', correcttrials=False,
                                        indiv_kf_sir=False):
    """Subjects x session-types grid of RT vs likelihood scatters (each session pooling its runs),
    overlaying the KF and optionally the LI fits. Laid out for 3 subjects and 4 session types.

    take_dpos: use the likelihood weighted by the global deviant-position prior.
    take_rules: also overlay the likelihood weighted by the rule-based prior.
    indiv_kf_sir: use the KF computed at each session's true observation-noise std (si_r);
        otherwise the KF with R fitted by EM.
    leaky_preds_path: if given, overlay the leaky integrator (LI) fits.
    correcttrials: keep only trials where the deviant's position was correctly identified.
    """

    # Find unique session types (configurations of d, si_stat, si_r)
    session_types, session_type_to_params = get_session_types(trials_path)

    fig, axs = plt.subplots(len(subjects), len(session_types), figsize=(12, 10), sharex=True, sharey=True)
    plt.style.context('seaborn-poster')
    sessions = []  # List to store session type parameters for column headers

    for j, session_type in tqdm(enumerate(session_types), desc="Session Type"):
        d, si_stat, si_r = session_type
        sessions.append([d, si_stat, si_r])

        # KF observation-noise std: the session's own (indiv_kf_sir), or None for R fitted by EM
        kf_sigma_r = si_r if indiv_kf_sir else None

        for i, sub in tqdm(enumerate(subjects), desc="Subject", leave=False):
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue

            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number

            # Trials with a valid likelihood and RT
            df_all = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path, pi_rule, file_prefix=file_prefix, sigma_r=kf_sigma_r)
            lik_col = 'likelihood_obs_std_at_dev_over_dpos' if take_dpos else 'likelihood_obs_std_at_dev'
            mask = ~np.isnan(df_all[lik_col]) & ~np.isnan(df_all['rt']) & (df_all['rt'] >= 0)
            if correcttrials:
                mask &= _correct_dpos_mask(df_all)
            df_filtered = df_all[mask].reset_index(drop=True)

            # Log-transform before the regression, removing the values that become infinite
            if loglik:
                df_filtered[lik_col] = np.log(df_filtered[lik_col])
                df_filtered['likelihood_obs_std_at_dev_over_rules'] = np.log(df_filtered['likelihood_obs_std_at_dev_over_rules'])
                df_filtered = df_filtered[np.isfinite(df_filtered[lik_col])].reset_index(drop=True)
            if logrt:
                df_filtered['rt'] = np.log(df_filtered['rt'])
                df_filtered = df_filtered[np.isfinite(df_filtered['rt'])].reset_index(drop=True)

            # Same trial selection and transforms for the leaky integrator (LI). Its files have
            # no noise suffix: the LI takes its observation noise from the trials.
            df_leaky = None
            if leaky_preds_path is not None:
                df_leaky = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path, pi_rule, file_prefix=leaky_file_prefix)
                mask_leaky = ~np.isnan(df_leaky[lik_col]) & ~np.isnan(df_leaky['rt']) & (df_leaky['rt'] >= 0)
                if correcttrials:
                    mask_leaky &= _correct_dpos_mask(df_leaky)
                df_leaky = df_leaky[mask_leaky].reset_index(drop=True)
                if loglik:
                    df_leaky[lik_col] = np.log(df_leaky[lik_col])
                    if take_rules:
                        df_leaky['likelihood_obs_std_at_dev_over_rules'] = np.log(df_leaky['likelihood_obs_std_at_dev_over_rules'])
                    df_leaky = df_leaky[np.isfinite(df_leaky[lik_col])].reset_index(drop=True)
                if logrt:
                    df_leaky['rt'] = np.log(df_leaky['rt'])
                    df_leaky = df_leaky[np.isfinite(df_leaky['rt'])].reset_index(drop=True)

            if not (take_rules and not take_dpos):
                # KF, global prior
                sns.regplot(df_filtered, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[0],
                            line_kws={'color': sns.color_palette("Paired")[1], 'linewidth': 2,
                                    'label': _fit_label(df_filtered[lik_col], df_filtered['rt'])})

                # LI, global prior
                if df_leaky is not None:
                    sns.regplot(df_leaky, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[4],
                                line_kws={'color': sns.color_palette("Paired")[5], 'linewidth': 2,
                                          'label': _fit_label(df_leaky[lik_col], df_leaky['rt'])})

            if take_rules:
                # KF, rule prior
                df_filtered_rules = df_filtered[np.isfinite(df_filtered['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                sns.regplot(df_filtered_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[2],
                            line_kws={'color': sns.color_palette("Paired")[3], 'linewidth': 2,
                                      'label': _fit_label(df_filtered_rules['likelihood_obs_std_at_dev_over_rules'], df_filtered_rules['rt'])})

                # LI, rule prior
                if df_leaky is not None:
                    df_leaky_rules = df_leaky[np.isfinite(df_leaky['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                    sns.regplot(df_leaky_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[6],
                                line_kws={'color': sns.color_palette("Paired")[7], 'linewidth': 2,
                                          'label': _fit_label(df_leaky_rules['likelihood_obs_std_at_dev_over_rules'], df_leaky_rules['rt'])})
            
            axs[i, j].set_xlabel("log likelihood" if loglik else "likelihood", fontsize=12)
            # RT label (the first column's is set below)
            if j > 0:
                axs[i, j].set_ylabel("log RT" if logrt else "RT (s)", fontsize=12)
            leg = axs[i, j].legend(loc='upper right', fontsize=10)
            # Colour each legend label like its regression line
            leg_handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
            for text, handle in zip(leg.get_texts(), leg_handles):
                text.set_color(handle.get_color())
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)

    # Session parameters as column titles
    for ax, session_params in zip(axs[0], sessions):
        obs_note = f"\n(KF $\\sigma_r$: {session_params[2]})" if indiv_kf_sir else ""
        ax.set_title(f"d: {session_params[0]}, $\\sigma$_stat: {session_params[1]},\n$\\sigma$_r: {session_params[2]}{obs_note}", fontsize=16 if not indiv_kf_sir else 14, pad=20)

    # Subject labels left of each row, and RT label of the first column
    # subject_labels = [f"Subject {subj}" for subj in subjects]
    subject_labels = [f"Subject {subj}" for subj in ['1', '2', '3']] # NOTE: this is for publication

    for k, ax in enumerate(axs[:, 0]):
        ax.set_ylabel("log RT" if logrt else "RT (s)", fontsize=12, rotation=90, labelpad=6, va='center')
        fig.text(0.042, ax.get_position().y0 + ax.get_position().height/2, subject_labels[k], 
                fontsize=16, rotation=90, va='center', ha='right', weight='normal')

    # Figure legend of the plotted lines (KF / LI x global / rule prior)
    plot_dpos = not (take_rules and not take_dpos)
    leaky = leaky_preds_path is not None
    custom_lines, custom_labels = [], []
    if leaky:
        if plot_dpos:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[1], lw=2)]
            custom_labels += ['KF, global prior']
        if take_rules:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[3], lw=2)]
            custom_labels += ['KF, rule prior']
        if plot_dpos:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[5], lw=2)]
            custom_labels += ['LI, global prior']
        if take_rules:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[7], lw=2)]
            custom_labels += ['LI, rule prior']
    elif take_rules and plot_dpos:
        custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[1], lw=2),
                         Line2D([0], [0], color=sns.color_palette("Paired")[3], lw=2)]
        custom_labels += ['without rules', 'with rules']
    if custom_lines:
        fig.legend(custom_lines, custom_labels, loc='lower center', ncol=len(custom_lines),
                bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    save_name = f"{comparison_save_path}/RT_3sub_4sess{'_with_rules' if take_rules else ''}{'_wout_dpos' if not take_dpos else ''}{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{'_leaky' if leaky_preds_path is not None else ''}{'_indivkfsir' if indiv_kf_sir else ''}{'_correcttrials' if correcttrials else ''}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")


def compare_likelihoods_with_RTs_global_indiv(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, take_dpos=True, take_rules=False, loglik=False, logrt=False, file_prefix='kalman_predictions', leaky_preds_path=None, leaky_file_prefix='leaky_predictions', correcttrials=False):
    """Same figure as `compare_likelihoods_with_RTs_global`, with each session's KF likelihoods
    computed at that session's own true observation-noise std (si_r)."""

    # Find unique session types (configurations of d, si_stat, si_r)
    session_types, session_type_to_params = get_session_types(trials_path)

    fig, axs = plt.subplots(len(subjects), len(session_types), figsize=(12, 10), sharex=True, sharey=True)
    plt.style.context('seaborn-poster')
    sessions = []  # List to store session type parameters for column headers

    for j, session_type in tqdm(enumerate(session_types), desc="Session Type"):
        d, si_stat, si_r = session_type
        sessions.append([d, si_stat, si_r])

        # The session's observation-noise std is si_r (si_stat is the state noise)

        for i, sub in tqdm(enumerate(subjects), desc="Subject", leave=False):
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue

            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number

            # Trials with a valid likelihood and RT
            df_all = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path, pi_rule, file_prefix=file_prefix, sigma_r=si_r)
            lik_col = 'likelihood_obs_std_at_dev_over_dpos' if take_dpos else 'likelihood_obs_std_at_dev'
            mask = ~np.isnan(df_all[lik_col]) & ~np.isnan(df_all['rt']) & (df_all['rt'] >= 0)
            if correcttrials:
                mask &= _correct_dpos_mask(df_all)
            df_filtered = df_all[mask].reset_index(drop=True)

            # Log-transform before the regression, removing the values that become infinite
            if loglik:
                df_filtered[lik_col] = np.log(df_filtered[lik_col])
                df_filtered['likelihood_obs_std_at_dev_over_rules'] = np.log(df_filtered['likelihood_obs_std_at_dev_over_rules'])
                df_filtered = df_filtered[np.isfinite(df_filtered[lik_col])].reset_index(drop=True)
            if logrt:
                df_filtered['rt'] = np.log(df_filtered['rt'])
                df_filtered = df_filtered[np.isfinite(df_filtered['rt'])].reset_index(drop=True)

            # Same trial selection and transforms for the leaky integrator (LI). Its files have
            # no noise suffix: the LI takes its observation noise from the trials.
            df_leaky = None
            if leaky_preds_path is not None:
                df_leaky = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path, pi_rule, file_prefix=leaky_file_prefix)
                mask_leaky = ~np.isnan(df_leaky[lik_col]) & ~np.isnan(df_leaky['rt']) & (df_leaky['rt'] >= 0)
                if correcttrials:
                    mask_leaky &= _correct_dpos_mask(df_leaky)
                df_leaky = df_leaky[mask_leaky].reset_index(drop=True)
                if loglik:
                    df_leaky[lik_col] = np.log(df_leaky[lik_col])
                    if take_rules:
                        df_leaky['likelihood_obs_std_at_dev_over_rules'] = np.log(df_leaky['likelihood_obs_std_at_dev_over_rules'])
                    df_leaky = df_leaky[np.isfinite(df_leaky[lik_col])].reset_index(drop=True)
                if logrt:
                    df_leaky['rt'] = np.log(df_leaky['rt'])
                    df_leaky = df_leaky[np.isfinite(df_leaky['rt'])].reset_index(drop=True)

            if not (take_rules and not take_dpos):
                # KF, global prior
                sns.regplot(df_filtered, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[0],
                            line_kws={'color': sns.color_palette("Paired")[1], 'linewidth': 2,
                                    'label': _fit_label(df_filtered[lik_col], df_filtered['rt'])})

                # LI, global prior
                if df_leaky is not None:
                    sns.regplot(df_leaky, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[4],
                                line_kws={'color': sns.color_palette("Paired")[5], 'linewidth': 2,
                                          'label': _fit_label(df_leaky[lik_col], df_leaky['rt'])})

            if take_rules:
                # KF, rule prior
                df_filtered_rules = df_filtered[np.isfinite(df_filtered['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                sns.regplot(df_filtered_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[2],
                            line_kws={'color': sns.color_palette("Paired")[3], 'linewidth': 2,
                                      'label': _fit_label(df_filtered_rules['likelihood_obs_std_at_dev_over_rules'], df_filtered_rules['rt'])})

                # LI, rule prior
                if df_leaky is not None:
                    df_leaky_rules = df_leaky[np.isfinite(df_leaky['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                    sns.regplot(df_leaky_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[6],
                                line_kws={'color': sns.color_palette("Paired")[7], 'linewidth': 2,
                                          'label': _fit_label(df_leaky_rules['likelihood_obs_std_at_dev_over_rules'], df_leaky_rules['rt'])})

            axs[i, j].set_xlabel("log likelihood" if loglik else "likelihood", fontsize=12)
            # RT label (the first column's is set below)
            if j > 0:
                axs[i, j].set_ylabel("log RT" if logrt else "RT (s)", fontsize=12)
            leg = axs[i, j].legend(loc='upper right', fontsize=10)
            # Colour each legend label like its regression line
            leg_handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
            for text, handle in zip(leg.get_texts(), leg_handles):
                text.set_color(handle.get_color())
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)

    # Session parameters as column titles
    for ax, session_params in zip(axs[0], sessions):
        ax.set_title(f"d: {session_params[0]}, $\\sigma$_stat: {session_params[1]},\n$\\sigma$_r: {session_params[2]}\n(KF obs noise: {session_params[2]})", fontsize=14, pad=20)

    # Subject labels left of each row, and RT label of the first column
    # subject_labels = [f"Subject {subj}" for subj in subjects]
    subject_labels = [f"Subject {subj}" for subj in ['1', '2', '3']] # NOTE: this is for publication

    for k, ax in enumerate(axs[:, 0]):
        ax.set_ylabel("log RT" if logrt else "RT (s)", fontsize=12, rotation=90, labelpad=6, va='center')
        fig.text(0.042, ax.get_position().y0 + ax.get_position().height/2, subject_labels[k],
                fontsize=16, rotation=90, va='center', ha='right', weight='normal')

    # Figure legend of the plotted lines (KF / LI x global / rule prior)
    plot_dpos = not (take_rules and not take_dpos)
    leaky = leaky_preds_path is not None
    custom_lines, custom_labels = [], []
    if leaky:
        if plot_dpos:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[1], lw=2)]
            custom_labels += ['KF, global prior']
        if take_rules:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[3], lw=2)]
            custom_labels += ['KF, rule prior']
        if plot_dpos:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[5], lw=2)]
            custom_labels += ['LI, global prior']
        if take_rules:
            custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[7], lw=2)]
            custom_labels += ['LI, rule prior']
    elif take_rules and plot_dpos:
        custom_lines += [Line2D([0], [0], color=sns.color_palette("Paired")[1], lw=2),
                         Line2D([0], [0], color=sns.color_palette("Paired")[3], lw=2)]
        custom_labels += ['without rules', 'with rules']
    if custom_lines:
        fig.legend(custom_lines, custom_labels, loc='lower center', ncol=len(custom_lines),
                bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    save_name = f"{comparison_save_path}/RT_3sub_4sess{'_with_rules' if take_rules else ''}{'_wout_dpos' if not take_dpos else ''}{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{'_leaky' if leaky_preds_path is not None else ''}_trueobsnoise{'_correcttrials' if correcttrials else ''}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")


# Models compared in the RT figures below. `src`: predictions the column is read from ('kf' or
# 'li'); the priors do not depend on the model and are read from the KF predictions.
PER_RUN_MODELS = [
    {'name': 'KF',     'src': 'kf', 'col': 'likelihood_obs_std_at_dev'},
    {'name': 'LI',     'src': 'li', 'col': 'likelihood_obs_std_at_dev'},
    {'name': 'global prior', 'src': 'kf', 'col': 'prior_dpos_std'},
    {'name': 'rule prior',   'src': 'kf', 'col': 'prior_dpos_rules_std'},
    {'name': 'KF + global',  'src': 'kf', 'col': 'likelihood_obs_std_at_dev_over_dpos'},
    {'name': 'LI + global',  'src': 'li', 'col': 'likelihood_obs_std_at_dev_over_dpos'},
    {'name': 'KF + rule',    'src': 'kf', 'col': 'likelihood_obs_std_at_dev_over_rules'},
    {'name': 'LI + rule',    'src': 'li', 'col': 'likelihood_obs_std_at_dev_over_rules'},
]

# One colour per model, the same in every figure
_TAB10 = sns.color_palette('tab10', 10)
MODEL_COLORS = {
    'KF':           _TAB10[4],  # purple
    'LI':           _TAB10[7],  # gray
    'global prior': _TAB10[3],  # red
    'rule prior':   _TAB10[0],  # blue
    'KF + global':  _TAB10[6],  # pink
    'LI + global':  _TAB10[5],  # brown
    'KF + rule':    _TAB10[1],  # orange
    'LI + rule':    _TAB10[2],  # green
}


def _model_colors(order=None):
    """Colours of the models named in `order` (default: PER_RUN_MODELS order), from
    MODEL_COLORS."""
    if order is None:
        order = [m['name'] for m in PER_RUN_MODELS]
    fallback = sns.color_palette('tab10', max(len(order), 10))
    return [MODEL_COLORS.get(name, fallback[i]) for i, name in enumerate(order)]


def _rho_stars_annotation(ax, r, p, fontsize=9, color='black', corr='pearson'):
    """Write rho and its significance stars in the top-left corner of `ax`."""
    val = TextArea(rf"{_corr_symbol(corr)}{r:.2f}", textprops=dict(color=color, size=fontsize))
    stars = TextArea(_p_stars(p), textprops=dict(color=color, size=fontsize, weight='bold'))
    box = HPacker(children=[val, stars], align="baseline", pad=0, sep=1)
    ax.add_artist(AnchoredOffsetbox(loc='upper left', child=box, frameon=False,
                                    bbox_to_anchor=(0.02, 0.99), bbox_transform=ax.transAxes,
                                    borderpad=0.0, pad=0.0))


def _select_rt_trials(df, logrt=True, correcttrials=False):
    """Trials analysed by the RT figures (valid RT, optionally a correctly identified deviant
    position), with RT optionally log-transformed."""
    keep = ~np.isnan(df['rt']) & (df['rt'] >= 0)
    if correcttrials:
        keep &= _correct_dpos_mask(df)
    df = df[keep].reset_index(drop=True)
    if logrt:
        df = df.copy()
        df['rt'] = np.log(df['rt'])
        df = df[np.isfinite(df['rt'])].reset_index(drop=True)
    return df


def _sorted_subjects(subjects):
    """Subjects in natural ASCENDING index order (numeric where possible, else lexical)."""
    try:
        return sorted(subjects, key=lambda s: int(s))
    except (TypeError, ValueError):
        return sorted(subjects)


def _subject_prefix(subjects):
    """Cohort filename prefix naming the subjects, e.g. ['06', '04', '05'] -> 'sub04-05-06_'."""
    return "sub" + "-".join(str(s) for s in _sorted_subjects(subjects)) + "_"


def _subject_display_map(subjects):
    """Map each subject id to its publication label 'Subject N', N being its 1-based rank in
    ascending index order (so 04/05/06 -> Subject 1/2/3, and 11/12/13 -> Subject 1/2/3)."""
    return {s: f"Subject {i}" for i, s in enumerate(_sorted_subjects(subjects), start=1)}


def _sub_label(subject_labels, sub):
    """Publication label for `sub`: from `subject_labels` if given, else the raw 'Subject {sub}'."""
    return (subject_labels or {}).get(sub) or f"Subject {sub}"


def _per_run_model_grid(groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                        pi_rule, comparison_save_path, save_stem, save_suffix='', n_run=4,
                        logrt=True, loglik=False, leaky_file_prefix='leaky_predictions',
                        figsize=(35, 12), correcttrials=False, subject_labels=None,
                        save_prefix='', allvalidtrials=False, corr='pearson'):
    """Grid of RT vs likelihood scatters: one row per model (PER_RUN_MODELS), one column per run.

    `groups` lists the (subject, session) blocks in column order, each a dict with `sub`,
    `sess_num`, `sigma_r` (observation-noise std of the KF file) and `band` (header
    naming the session's parameters). Within a block, runs are ordered by (tau_std, d).
    corr: correlation annotated in each cell (the regression line is OLS either way).
    """
    n_cols = len(groups) * n_run
    n_rows = len(PER_RUN_MODELS)
    # Shared RT axis; each subplot keeps its own likelihood axis
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False)
    colors = _model_colors()
    # Left margin and row-label position, set in inches
    left_frac = 1.12 / figsize[0]
    label_x = 0.49 / figsize[0]

    col_meta = []  # information on each column, to place the headers
    col = 0
    all_y = []  # all plotted RTs, to set the shared RT limit

    for g in tqdm(groups, desc="Subject/session"):
        sub, sess_num = g['sub'], g['sess_num']

        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, allvalidtrials=allvalidtrials, sigma_r=g['sigma_r'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix, allvalidtrials=allvalidtrials)

        # Runs ordered by (tau_std, d), read before the trial selection so every run keeps its
        # column
        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")
        d_varies = run_keys['d'].nunique() > 1  # show d in the titles only when it varies

        df_kf = _select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials)
        df_li = _select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials)

        for run_idx, (run_n, prow) in enumerate(run_keys.iterrows()):
            df_kf_r = df_kf[df_kf['run_n'] == run_n]
            df_li_r = df_li[df_li['run_n'] == run_n]
            col_meta.append({'col': col, 'sub': sub, 'band': g['band'],
                             'tau': int(prow['tau_std']), 'd': int(prow['d']),
                             'show_d': bool(d_varies), 'run_idx': run_idx})

            for row, m in enumerate(PER_RUN_MODELS):
                ax = axs[row, col]
                d = df_kf_r if m['src'] == 'kf' else df_li_r
                x = d[m['col']].to_numpy(dtype=float)
                y = d['rt'].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if loglik:
                    pos = x > 0  # log(0) -> -inf; drop non-positive likelihoods/priors
                    x, y = np.log(x[pos]), y[pos]

                if len(x):
                    all_y.append(y)
                if len(x) >= 2:
                    sns.regplot(x=x, y=y, ax=ax, color=colors[row],
                                scatter_kws={'s': 6, 'alpha': 0.25},
                                line_kws={'linewidth': 1.2})
                    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                        r, p = _corr_stats(x, y, corr)
                        _rho_stars_annotation(ax, r, p, fontsize=9, corr=corr)
                ax.tick_params(labelsize=7, length=2)
                # Few x ticks, so the labels of neighbouring columns don't overlap
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            col += 1

    assert col == n_cols, f"Filled {col} columns, expected {n_cols}"

    # Shared RT limit, ignoring outlier RTs
    if all_y:
        y_all = np.concatenate(all_y)
        ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
        if yhi > ylo:
            axs[0, 0].set_ylim(ylo, yhi)

    # Margins, set before placing the headers; more room at the top for two-line titles (tau, d)
    two_line_titles = any(m['show_d'] for m in col_meta)
    plt.subplots_adjust(left=left_frac, right=0.995, top=0.88 if two_line_titles else 0.9,
                        bottom=0.06, wspace=0.4, hspace=0.5)

    # Model name left of each row, and RT label on the first column
    for row, m in enumerate(PER_RUN_MODELS):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, m['name'], fontsize=10, rotation=90,
                 va='center', ha='center', color=colors[row])
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    def _band_centre(group):
        p0 = axs[0, group[0]['col']].get_position()
        p1 = axs[0, group[-1]['col']].get_position()
        return (p0.x0 + p1.x1) / 2

    # Column titles: tau (and d when it varies across runs)
    for meta in col_meta:
        tau_txt = f"$\\tau$={meta['tau']}"
        if meta['show_d']:
            tau_txt += f"\nd={meta['d']}"
        axs[0, meta['col']].set_title(tau_txt, fontsize=9, pad=3)

    # Session parameters above each session's columns
    for start in range(0, n_cols, n_run):
        group = col_meta[start:start + n_run]
        fig.text(_band_centre(group), 0.915, group[0]['band'], fontsize=12,
                 va='bottom', ha='center')

    # Dotted lines between sessions, reaching higher where they also separate subjects
    session_sep_top = 0.905
    subject_sep_top = 0.955
    sep_bottom = axs[-1, 0].get_position().y0
    for start in range(n_run, n_cols, n_run):
        x = (axs[0, start - 1].get_position().x1 + axs[0, start].get_position().x0) / 2
        crosses_subject = col_meta[start]['sub'] != col_meta[start - 1]['sub']
        sep_top = subject_sep_top if crosses_subject else session_sep_top
        fig.add_artist(Line2D([x, x], [sep_bottom, sep_top], transform=fig.transFigure,
                              color='gray', linestyle='--', linewidth=1, zorder=0))

    # Subject label above each subject's columns
    start = 0
    while start < n_cols:
        end = start
        while end < n_cols and col_meta[end]['sub'] == col_meta[start]['sub']:
            end += 1
        group = col_meta[start:end]
        fig.text(_band_centre(group), 0.957, _sub_label(subject_labels, group[0]['sub']),
                 fontsize=13, va='bottom', ha='center')
        start = end

    fig.supxlabel("log likelihood" if loglik else "likelihood", fontsize=16, y=0.02)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}{'_allvalidtrials' if allvalidtrials else ''}"
                 f"{_corr_suffix(corr)}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def _per_subject_model_grid(subject_specs, trials_path, logfiles_path, preds_liks_path,
                            leaky_preds_path, pi_rule, comparison_save_path, save_stem,
                            save_suffix='', logrt=True, loglik=False,
                            leaky_file_prefix='leaky_predictions', figsize=None,
                            correcttrials=False, subject_labels=None, save_prefix='',
                            allvalidtrials=False, corr='pearson'):
    """Grid of RT vs likelihood scatters pooling all trials of each subject: one row per subject,
    one column per model.

    `subject_specs` lists the subjects in row order, each a dict with `sub` and `sessions` (a list
    of {'sess_num', 'sigma_r'}, so each session's KF file is selected before pooling).
    """
    n_rows = len(subject_specs)
    n_cols = len(PER_RUN_MODELS)
    if figsize is None:
        # Inches per column and per row, plus room for the margins
        figsize = (1.4 * n_cols + 1., 1.15 * n_rows + 1.35)

    # Shared RT axis; each subplot keeps its own likelihood axis
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False,
                            squeeze=False)
    colors = _model_colors()
    # Left margin and right edge of the subject labels, set in inches
    left_frac = 1.55 / figsize[0]
    label_x = 1.0 / figsize[0]

    row_meta = []  # information on each row, to place the subject labels
    all_y = []  # all plotted RTs, to set the shared RT limit

    for row, spec in enumerate(tqdm(subject_specs, desc="Subject")):
        sub = spec['sub']

        # Pool all sessions of the subject, each with its own KF file and trial selection
        kf_parts, li_parts = [], []
        for s in spec['sessions']:
            df_kf = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], preds_liks_path,
                                   pi_rule, allvalidtrials=allvalidtrials, sigma_r=s['sigma_r'])
            df_li = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], leaky_preds_path,
                                   pi_rule, file_prefix=leaky_file_prefix, allvalidtrials=allvalidtrials)
            kf_parts.append(_select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials))
            li_parts.append(_select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials))
        df_kf = pd.concat(kf_parts, ignore_index=True)
        df_li = pd.concat(li_parts, ignore_index=True)
        row_meta.append({'sub': sub, 'n_sess': len(spec['sessions'])})

        for col, m in enumerate(PER_RUN_MODELS):
            ax = axs[row, col]
            d = df_kf if m['src'] == 'kf' else df_li
            x = d[m['col']].to_numpy(dtype=float)
            y = d['rt'].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if loglik:
                pos = x > 0  # log(0) -> -inf; drop non-positive likelihoods/priors
                x, y = np.log(x[pos]), y[pos]

            if len(x):
                all_y.append(y)
            if len(x) >= 2:
                sns.regplot(x=x, y=y, ax=ax, color=colors[col],
                            scatter_kws={'s': 5, 'alpha': 0.2},
                            line_kws={'linewidth': 1.4})
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r, p = _corr_stats(x, y, corr)
                    _rho_stars_annotation(ax, r, p, fontsize=10, corr=corr)
                # Number of pooled trials (not shown):
                # ax.text(0.97, 0.03, f"n={len(x)}", transform=ax.transAxes, fontsize=6.5,
                #         va='bottom', ha='right', color='dimgray')
            ax.tick_params(labelsize=7, length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Shared RT limit, ignoring outlier RTs
    if all_y:
        y_all = np.concatenate(all_y)
        ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
        if yhi > ylo:
            axs[0, 0].set_ylim(ylo, yhi)

    # Margins, set before placing the labels (bottom room for the x-axis label)
    plt.subplots_adjust(left=left_frac, right=0.985, top=0.92, bottom=0.12, wspace=0.28, hspace=0.4)

    # Subject label left of each row, and RT label on the first column
    for row, meta in enumerate(row_meta):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, _sub_label(subject_labels, meta['sub']), fontsize=12,
                 va='center', ha='right')
        # axs[row, 0].set_title(_sub_label(subject_labels, meta['sub']), loc='left', fontsize=12, pad=8)
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    # Model names as column titles, in the model's colour
    for col, m in enumerate(PER_RUN_MODELS):
        axs[0, col].set_title(m['name'], fontsize=10, pad=8, color=colors[col])

    fig.supxlabel("log-likelihood" if loglik else "likelihood", fontsize=12, y=0.02)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}{'_allvalidtrials' if allvalidtrials else ''}"
                 f"{_corr_suffix(corr)}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def _run_level_rt_means(d, xcol, loglik=False, err='sem'):
    """Mean likelihood and mean RT of every run, with their SEM (err='sem') or SD (err='sd').

    Runs are identified by ('_pool_sess', 'run_n'). Likelihoods are log-transformed (loglik)
    before averaging. Returns run-aligned arrays (mx, my, ex, ey, n).
    """
    x = d[xcol].to_numpy(dtype=float)
    y = d['rt'].to_numpy(dtype=float)
    sess = d['_pool_sess'].to_numpy()
    run = d['run_n'].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if loglik:
        mask &= x > 0  # no log of 0
    x, y, sess, run = x[mask], y[mask], sess[mask], run[mask]
    if loglik:
        x = np.log(x)
    if len(x) == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty
    g = pd.DataFrame({'sess': sess, 'run': run, 'x': x, 'y': y}).groupby(['sess', 'run'], sort=False)
    agg = g.agg(mx=('x', 'mean'), my=('y', 'mean'),
                sdx=('x', 'std'), sdy=('y', 'std'), n=('x', 'size'))
    nrun = agg['n'].to_numpy(dtype=float)
    denom = np.sqrt(nrun) if err == 'sem' else np.ones_like(nrun)  # SEM = SD / sqrt(n)
    sdx = agg['sdx'].fillna(0.0).to_numpy()  # zero error for a run with a single trial
    sdy = agg['sdy'].fillna(0.0).to_numpy()
    return agg['mx'].to_numpy(), agg['my'].to_numpy(), sdx / denom, sdy / denom, nrun


def _plot_run_means_cell(ax, df_kf, df_li, m, color, loglik, err='sem', both_axes=True,
                       corr='pearson'):
    """Plot one model's run means in `ax`: one dot per run with error bars, linked in likelihood
    order, with a dashed OLS fit and the rho across runs.

    both_axes: error bars on x as well as y. Returns the y extent of the error bars (or None),
    for the shared y-limit.
    """
    d = df_kf if m['src'] == 'kf' else df_li
    mx, my, ex, ey, _ = _run_level_rt_means(d, m['col'], loglik=loglik, err=err)

    if len(mx):
        order = np.argsort(mx)
        # Line linking the run means in likelihood order
        ax.plot(mx[order], my[order], '-', color=color, linewidth=0.9, alpha=0.6, zorder=1)
        # Run means with their error bars
        ax.errorbar(mx, my, xerr=ex if both_axes else None, yerr=ey, fmt='o', ms=3.5, color=color,
                    ecolor=color, elinewidth=0.8, capsize=2, capthick=0.8, alpha=0.9,
                    linestyle='none', zorder=2)
        # Dashed OLS fit with its CI
        if len(mx) >= 2 and np.std(mx) > 0:
            sns.regplot(x=mx, y=my, ax=ax, color=color, scatter=False,
                        line_kws={'linewidth': 1.4, 'linestyle': '--'})
            if len(mx) >= 3 and np.std(my) > 0:
                r, p = _corr_stats(mx, my, corr)
                _rho_stars_annotation(ax, r, p, fontsize=9, corr=corr)
        # Number of runs
        ax.text(0.97, 0.03, f"n={len(mx)}", transform=ax.transAxes, fontsize=6.5,
                va='bottom', ha='right', color='dimgray')

    ax.tick_params(labelsize=7, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return np.concatenate([my - ey, my + ey]) if len(mx) else None


def _per_subject_runmean_grid(subject_specs, trials_path, logfiles_path, preds_liks_path,
                              leaky_preds_path, pi_rule, comparison_save_path, save_stem,
                              save_suffix='', logrt=True, loglik=False,
                              leaky_file_prefix='leaky_predictions', figsize=None,
                              correcttrials=False, err='sem', both_axes=True, subject_labels=None,
                              save_prefix='', allvalidtrials=False, corr='pearson'):
    """Grid of run-mean RT vs likelihood: one row per model, one column per subject, one dot per
    run (see `_plot_run_means_cell`).

    `subject_specs` as in `_per_subject_model_grid`.
    """
    n_cols = len(subject_specs)
    n_rows = len(PER_RUN_MODELS)
    if figsize is None:
        figsize = (2.6 * n_cols + 1.8, 1.55 * n_rows + 1.4)

    # Shared RT axis; each subplot keeps its own likelihood axis
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False,
                            squeeze=False)
    colors = _model_colors()
    left_frac = 1.12 / figsize[0]
    label_x = 0.49 / figsize[0]

    col_meta = []  # information on each column, to place the subject labels
    all_y = []  # extent of all plotted error bars, to set the shared RT limit
    for col, spec in enumerate(tqdm(subject_specs, desc="Subject (run means)")):
        sub = spec['sub']

        # Pool all sessions of the subject, marking the session ('_pool_sess') so that each run
        # stays identified after pooling
        kf_parts, li_parts = [], []
        for s in spec['sessions']:
            df_kf = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], preds_liks_path,
                                   pi_rule, allvalidtrials=allvalidtrials, sigma_r=s['sigma_r'])
            df_li = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], leaky_preds_path,
                                   pi_rule, file_prefix=leaky_file_prefix,
                                   allvalidtrials=allvalidtrials)
            kf = _select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials)
            li = _select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials)
            kf['_pool_sess'] = s['sess_num']
            li['_pool_sess'] = s['sess_num']
            kf_parts.append(kf)
            li_parts.append(li)
        df_kf = pd.concat(kf_parts, ignore_index=True)
        df_li = pd.concat(li_parts, ignore_index=True)
        col_meta.append({'sub': sub, 'n_sess': len(spec['sessions'])})

        for row, m in enumerate(PER_RUN_MODELS):
            yp = _plot_run_means_cell(axs[row, col], df_kf, df_li, m, colors[row], loglik,
                                    err=err, both_axes=both_axes, corr=corr)
            if yp is not None:
                all_y.append(yp)

    # Shared RT limit, ignoring outliers
    if all_y:
        y_all = np.concatenate(all_y)
        y_all = y_all[np.isfinite(y_all)]
        if len(y_all):
            ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
            if yhi > ylo:
                axs[0, 0].set_ylim(ylo, yhi)

    plt.subplots_adjust(left=left_frac, right=0.985, top=0.92, bottom=0.06, wspace=0.28, hspace=0.4)

    # Model name left of each row, and RT label on the first column
    for row, m in enumerate(PER_RUN_MODELS):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, m['name'], fontsize=10, rotation=90,
                 va='center', ha='center', color=colors[row])
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    # Subject label above each column
    for col, meta in enumerate(col_meta):
        p = axs[0, col].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.945, _sub_label(subject_labels, meta['sub']), fontsize=13,
                 va='bottom', ha='center')

    fig.supxlabel("log likelihood" if loglik else "likelihood", fontsize=14, y=0.02)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_sd' if err == 'sd' else ''}{'_yerr' if not both_axes else ''}"
                 f"{'_correcttrials' if correcttrials else ''}"
                 f"{'_allvalidtrials' if allvalidtrials else ''}{_corr_suffix(corr)}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def compare_likelihoods_with_RTs_per_subject_runmeans_models(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, logrt=True, loglik=False, figsize=None, correcttrials=False, err='sem', both_axes=True, allvalidtrials=False, corr='pearson'):
    """Run-mean RT vs likelihood for the 04/05/06 cohort: one row per model, one column per
    subject, one dot per run (16 per subject).

    Each session uses the KF computed at its own true observation noise. err: 'sem' or 'sd'
    error bars.
    """
    session_types, session_type_to_params = get_session_types(trials_path)

    subject_specs = []
    for sub in subjects:
        sessions = []
        for session_type in session_types:
            d_par, si_stat, si_r = session_type
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            sessions.append({
                'sess_num': session_type_to_params[session_type][sub],  # 1-indexed
                'sigma_r': si_r,
            })
        subject_specs.append({'sub': sub, 'sessions': sessions})

    return _per_subject_runmean_grid(
        subject_specs, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_persubject_perrun_linrel', save_suffix='_trueobsnoise',
        logrt=logrt, loglik=loglik, figsize=figsize, correcttrials=correcttrials,
        allvalidtrials=allvalidtrials, err=err, corr=corr,
        both_axes=both_axes, subject_labels=_subject_display_map(subjects),
        save_prefix=_subject_prefix(subjects))


def compare_likelihoods_with_RTs_per_subject_models(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, logrt=True, loglik=False, figsize=None, correcttrials=False, allvalidtrials=False, corr='pearson'):
    """RT vs likelihood scatters for the 04/05/06 cohort, pooling all trials of each subject: one
    row per subject, one column per model.

    Each session uses the KF computed at its own true observation-noise std (si_r).
    """
    session_types, session_type_to_params = get_session_types(trials_path)

    subject_specs = []
    for sub in subjects:
        sessions = []
        for session_type in session_types:
            d_par, si_stat, si_r = session_type
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            sessions.append({
                'sess_num': session_type_to_params[session_type][sub],  # 1-indexed
                'sigma_r': si_r,
            })
        subject_specs.append({'sub': sub, 'sessions': sessions})

    return _per_subject_model_grid(
        subject_specs, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_persubject_linrel', save_suffix='_trueobsnoise',
        logrt=logrt, loglik=loglik, figsize=figsize, correcttrials=correcttrials,
        allvalidtrials=allvalidtrials, corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


def compare_likelihoods_with_RTs_per_run_models(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, n_run=4, logrt=True, loglik=False, figsize=(35, 12), correcttrials=False, allvalidtrials=False, corr='pearson'):
    """RT vs likelihood scatters for the 04/05/06 cohort: one row per model, one column per
    (subject, session, run), i.e. 8 x 48.

    Each session uses the KF computed at its own true observation-noise std (si_r).
    """
    session_types, session_type_to_params = get_session_types(trials_path)

    groups = []
    for sub in subjects:
        for session_type in session_types:
            d_par, si_stat, si_r = session_type
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            groups.append({
                'sub': sub,
                'sess_num': session_type_to_params[session_type][sub],  # 1-indexed
                'sigma_r': si_r,
                'band': f"d={d_par}, $\\sigma_{{stat}}$={si_stat}\n$\\sigma_r$={si_r}",
            })

    return _per_run_model_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_permodel_perrun', save_suffix='_trueobsnoise',
        n_run=n_run, logrt=logrt, loglik=loglik, figsize=figsize, correcttrials=correcttrials,
        allvalidtrials=allvalidtrials, corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


def _plot_rt_vs_likelihood_cell(ax, df_kf, df_li, m, color, loglik, scatter_kws, line_kws,
                             show_n=False, corr='pearson'):
    """Plot one model's RT vs likelihood scatter in `ax`, with its regression line and rho.

    Returns the plotted RTs (or None), for the shared y-limit.
    """
    d = df_kf if m['src'] == 'kf' else df_li
    x = d[m['col']].to_numpy(dtype=float)
    y = d['rt'].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if loglik:
        pos = x > 0  # log(0) -> -inf; drop non-positive likelihoods/priors
        x, y = np.log(x[pos]), y[pos]

    y_plotted = y if len(x) else None
    if len(x) >= 2:
        sns.regplot(x=x, y=y, ax=ax, color=color, scatter_kws=scatter_kws, line_kws=line_kws)
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r, p = _corr_stats(x, y, corr)
            _rho_stars_annotation(ax, r, p, fontsize=9, corr=corr)
        if show_n:
            ax.text(0.97, 0.03, f"n={len(x)}", transform=ax.transAxes, fontsize=6.5,
                    va='bottom', ha='right', color='dimgray')
    ax.tick_params(labelsize=7, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return y_plotted


def _per_run_and_subject_model_grid(subject_specs, groups, trials_path, logfiles_path,
                                    preds_liks_path, leaky_preds_path, pi_rule,
                                    comparison_save_path, save_stem, save_suffix='', n_run=4,
                                    logrt=True, loglik=False,
                                    leaky_file_prefix='leaky_predictions', figsize=None,
                                    correcttrials=False, subject_labels=None, save_prefix='',
                                    allvalidtrials=False, corr='pearson'):
    """Grid of RT vs likelihood scatters with one row per model: one pooled column per subject,
    followed by the per-run columns of `_per_run_model_grid`.

    `subject_specs` and `groups` are the layouts of `_per_subject_model_grid` and
    `_per_run_model_grid`.
    """
    n_pooled = len(subject_specs)
    n_run_cols = len(groups) * n_run
    n_cols = n_pooled + n_run_cols
    n_rows = len(PER_RUN_MODELS)
    if figsize is None:
        # Same column width as the per-run figure (35 in for 48 columns)
        figsize = (0.73 * n_cols + 1.0, 12)

    # Shared RT axis; each subplot keeps its own likelihood axis
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False,
                            squeeze=False)
    colors = _model_colors()
    left_frac = 1.12 / figsize[0]
    label_x = 0.49 / figsize[0]

    # Scatter style of the pooled and per-run columns
    pooled_scatter_kws, pooled_line_kws = {'s': 5, 'alpha': 0.2}, {'linewidth': 1.4}
    run_scatter_kws, run_line_kws = {'s': 6, 'alpha': 0.25}, {'linewidth': 1.2}

    col_meta = []  # information on each column; 'kind' is 'pooled' or 'run'
    all_y = []  # all plotted RTs, to set the shared RT limit
    col = 0

    # Pooled columns, one per subject
    for spec in tqdm(subject_specs, desc="Subject (pooled)"):
        sub = spec['sub']
        # Pool all sessions of the subject, each with its own KF file and trial selection
        kf_parts, li_parts = [], []
        for s in spec['sessions']:
            df_kf = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], preds_liks_path,
                                   pi_rule, allvalidtrials=allvalidtrials, sigma_r=s['sigma_r'])
            df_li = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], leaky_preds_path,
                                   pi_rule, file_prefix=leaky_file_prefix, allvalidtrials=allvalidtrials)
            kf_parts.append(_select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials))
            li_parts.append(_select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials))
        df_kf = pd.concat(kf_parts, ignore_index=True)
        df_li = pd.concat(li_parts, ignore_index=True)
        col_meta.append({'col': col, 'kind': 'pooled', 'sub': sub})
        for row, m in enumerate(PER_RUN_MODELS):
            yp = _plot_rt_vs_likelihood_cell(axs[row, col], df_kf, df_li, m, colors[row], loglik,
                                          pooled_scatter_kws, pooled_line_kws, show_n=True,
                                          corr=corr)
            if yp is not None:
                all_y.append(yp)
        col += 1

    # Per-run columns
    for g in tqdm(groups, desc="Subject/session (per-run)"):
        sub, sess_num = g['sub'], g['sess_num']
        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, allvalidtrials=allvalidtrials, sigma_r=g['sigma_r'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix, allvalidtrials=allvalidtrials)
        # Runs ordered by (tau_std, d), read before the trial selection
        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")
        d_varies = run_keys['d'].nunique() > 1
        df_kf = _select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials)
        df_li = _select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials)
        for run_idx, (run_n, prow) in enumerate(run_keys.iterrows()):
            df_kf_r = df_kf[df_kf['run_n'] == run_n]
            df_li_r = df_li[df_li['run_n'] == run_n]
            col_meta.append({'col': col, 'kind': 'run', 'sub': sub, 'band': g['band'],
                             'tau': int(prow['tau_std']), 'd': int(prow['d']),
                             'show_d': bool(d_varies), 'run_idx': run_idx})
            for row, m in enumerate(PER_RUN_MODELS):
                yp = _plot_rt_vs_likelihood_cell(axs[row, col], df_kf_r, df_li_r, m, colors[row],
                                              loglik, run_scatter_kws, run_line_kws, show_n=False,
                                              corr=corr)
                if yp is not None:
                    all_y.append(yp)
            col += 1

    assert col == n_cols, f"Filled {col} columns, expected {n_cols}"

    # Shared RT limit, ignoring outlier RTs
    if all_y:
        y_all = np.concatenate(all_y)
        ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
        if yhi > ylo:
            axs[0, 0].set_ylim(ylo, yhi)

    run_meta = [m for m in col_meta if m['kind'] == 'run']
    two_line_titles = any(m['show_d'] for m in run_meta)
    plt.subplots_adjust(left=left_frac, right=0.995, top=0.88 if two_line_titles else 0.9,
                        bottom=0.06, wspace=0.4, hspace=0.5)

    # Model name left of each row, and RT label on the first column
    for row, m in enumerate(PER_RUN_MODELS):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, m['name'], fontsize=10, rotation=90,
                 va='center', ha='center', color=colors[row])
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    def _band_centre(cols):
        p0 = axs[0, cols[0]].get_position()
        p1 = axs[0, cols[-1]].get_position()
        return (p0.x0 + p1.x1) / 2

    # Column titles: "pooled", or tau (and d when it varies across runs)
    for meta in col_meta:
        if meta['kind'] == 'pooled':
            axs[0, meta['col']].set_title("pooled", fontsize=9, pad=3)
        else:
            tau_txt = f"$\\tau$={meta['tau']}"
            if meta['show_d']:
                tau_txt += f"\nd={meta['d']}"
            axs[0, meta['col']].set_title(tau_txt, fontsize=9, pad=3)

    # Session parameters above each session's per-run columns
    for start in range(0, len(run_meta), n_run):
        group = run_meta[start:start + n_run]
        fig.text(_band_centre([g['col'] for g in group]), 0.915, group[0]['band'], fontsize=12,
                 va='bottom', ha='center')

    # Separators: a solid line between the pooled and per-run columns, dotted lines between
    # sessions (reaching higher where they also separate subjects)
    session_sep_top = 0.905
    subject_sep_top = 0.955
    sep_bottom = axs[-1, 0].get_position().y0
    if n_pooled and n_run_cols:
        x = (axs[0, n_pooled - 1].get_position().x1 + axs[0, n_pooled].get_position().x0) / 2
        fig.add_artist(Line2D([x, x], [sep_bottom, subject_sep_top], transform=fig.transFigure,
                              color='gray', linestyle='-', linewidth=2, zorder=0))
    for start in range(n_pooled + n_run, n_cols, n_run):
        x = (axs[0, start - 1].get_position().x1 + axs[0, start].get_position().x0) / 2
        crosses_subject = col_meta[start]['sub'] != col_meta[start - 1]['sub']
        sep_top = subject_sep_top if crosses_subject else session_sep_top
        fig.add_artist(Line2D([x, x], [sep_bottom, sep_top], transform=fig.transFigure,
                              color='gray', linestyle='--', linewidth=1, zorder=0))

    # Subject label above each subject's pooled column and above its per-run columns (smaller
    # font above the narrow pooled columns)
    start = 0
    while start < n_cols:
        end = start
        while (end < n_cols and col_meta[end]['sub'] == col_meta[start]['sub']
               and col_meta[end]['kind'] == col_meta[start]['kind']):
            end += 1
        centre = _band_centre([col_meta[c]['col'] for c in range(start, end)])
        label = _sub_label(subject_labels, col_meta[start]['sub'])
        if col_meta[start]['kind'] == 'pooled':
            fig.text(centre, 0.957, label, fontsize=11, va='bottom', ha='center')
        else:
            fig.text(centre, 0.957, label, fontsize=13, va='bottom', ha='center')
        start = end

    fig.supxlabel("log likelihood" if loglik else "likelihood", fontsize=16, y=0.02)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}{'_allvalidtrials' if allvalidtrials else ''}"
                 f"{_corr_suffix(corr)}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def compare_likelihoods_with_RTs_per_run_and_subject_models(subjects, trials_path, preds_liks_path,
                                                            logfiles_path, comparison_save_path,
                                                            pi_rule, leaky_preds_path, n_run=4,
                                                            logrt=True, loglik=False,
                                                            figsize=None,
                                                            correcttrials=False,
                                                            allvalidtrials=False,
                                                            corr='pearson'):
    """RT vs likelihood scatters for the 04/05/06 cohort, pooled per subject and per run in one
    figure (see `_per_run_and_subject_model_grid`)."""
    session_types, session_type_to_params = get_session_types(trials_path)

    # Layouts of the pooled and per-run columns, in the same subject and session order
    subject_specs = []
    groups = []
    for sub in subjects:
        sessions = []
        for session_type in session_types:
            d_par, si_stat, si_r = session_type
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            sess_num = session_type_to_params[session_type][sub]  # 1-indexed
            sessions.append({'sess_num': sess_num, 'sigma_r': si_r})
            groups.append({
                'sub': sub,
                'sess_num': sess_num,
                'sigma_r': si_r,
                'band': f"d={d_par}, $\\sigma_{{stat}}$={si_stat}\n$\\sigma_r$={si_r}",
            })
        subject_specs.append({'sub': sub, 'sessions': sessions})

    return _per_run_and_subject_model_grid(
        subject_specs, groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
        pi_rule, comparison_save_path, save_stem='RT_perrun_persubject',
        save_suffix='_trueobsnoise', n_run=n_run, logrt=logrt, loglik=loglik, figsize=figsize,
        correcttrials=correcttrials, allvalidtrials=allvalidtrials, corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


def _single_session_layout(subjects, sess_num=1, sigma_r=0.02):
    """Layouts (`subject_specs`, `groups`) of the grid functions for the 11/12/13 cohort: one
    session per subject, all at observation noise `sigma_r`.

    Replaces `get_session_types`, which only reads the 04/05/06 filenames.
    """
    subject_specs = [{'sub': sub,
                      'sessions': [{'sess_num': sess_num, 'sigma_r': sigma_r}]}
                     for sub in subjects]
    groups = [{'sub': sub,
               'sess_num': sess_num,
               'sigma_r': sigma_r,
               'band': f"$\\sigma_r$={sigma_r}"} for sub in subjects]
    return subject_specs, groups


def compare_likelihoods_with_RTs_per_subject_models_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, sess_num=1, sigma_r=0.02, logrt=True, loglik=False,
        leaky_file_prefix='leaky_predictions', figsize=None, correcttrials=False,
        allvalidtrials=False, corr='pearson'):
    """11/12/13-cohort version of `compare_likelihoods_with_RTs_per_subject_models`."""
    subject_specs, _ = _single_session_layout(subjects, sess_num=sess_num, sigma_r=sigma_r)
    return _per_subject_model_grid(
        subject_specs, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_persubject_linrel', save_suffix='_trueobsnoise',
        logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix, figsize=figsize,
        correcttrials=correcttrials, allvalidtrials=allvalidtrials, corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


def compare_likelihoods_with_RTs_per_subject_runmeans_models_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, sess_num=1, sigma_r=0.02, logrt=True, loglik=False,
        leaky_file_prefix='leaky_predictions', figsize=None, correcttrials=False,
        allvalidtrials=False, err='sem', both_axes=True, corr='pearson'):
    """11/12/13-cohort version of `compare_likelihoods_with_RTs_per_subject_runmeans_models`.

    With only 4 runs per subject, the rho across runs is fragile.
    """
    subject_specs, _ = _single_session_layout(subjects, sess_num=sess_num, sigma_r=sigma_r)
    return _per_subject_runmean_grid(
        subject_specs, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_persubject_perrun_linrel', save_suffix='_trueobsnoise',
        logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix, figsize=figsize,
        correcttrials=correcttrials, allvalidtrials=allvalidtrials, err=err, both_axes=both_axes,
        corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


def compare_likelihoods_with_RTs_per_run_and_subject_models_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, sess_num=1, sigma_r=0.02, n_run=4, logrt=True, loglik=False,
        leaky_file_prefix='leaky_predictions', figsize=None, correcttrials=False,
        allvalidtrials=False, corr='pearson'):
    """11/12/13-cohort version of `compare_likelihoods_with_RTs_per_run_and_subject_models`."""
    subject_specs, groups = _single_session_layout(subjects, sess_num=sess_num, sigma_r=sigma_r)
    return _per_run_and_subject_model_grid(
        subject_specs, groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
        pi_rule, comparison_save_path, save_stem='RT_perrun_persubject',
        save_suffix='_trueobsnoise', n_run=n_run, logrt=logrt, loglik=loglik,
        leaky_file_prefix=leaky_file_prefix, figsize=figsize, correcttrials=correcttrials,
        allvalidtrials=allvalidtrials, corr=corr, subject_labels=_subject_display_map(subjects),
        save_prefix=_subject_prefix(subjects))


def compare_likelihoods_with_RTs_per_run_models_single_session(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, sess_num=1, sigma_r=0.02, n_run=4, logrt=True, loglik=False, leaky_file_prefix='leaky_predictions', save_stem='RT_permodel_perrun', save_suffix='_trueobsnoise', figsize=(13, 12), correcttrials=False, allvalidtrials=False, corr='pearson'):
    """11/12/13-cohort version of `compare_likelihoods_with_RTs_per_run_models` (8 x 12).

    In this cohort `d` varies across runs and tau_std is not unique within a subject, so a column
    position does not mean the same parameters across subjects: read each column's header. The
    rule-prior models use the cue-aware prior.
    """
    _, groups = _single_session_layout(subjects, sess_num=sess_num, sigma_r=sigma_r)

    return _per_run_model_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem=save_stem, save_suffix=save_suffix, n_run=n_run,
        logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix, figsize=figsize,
        correcttrials=correcttrials, allvalidtrials=allvalidtrials, corr=corr,
        subject_labels=_subject_display_map(subjects), save_prefix=_subject_prefix(subjects))


# Models of the likelihood histograms (top and bottom half of the figure). `row_color` indexes
# `_model_colors()`, so each model has the same colour as in the RT figures.
LIK_DIST_MODELS = [
    {'name': 'KF alone', 'src': 'kf', 'row_color': 0},
    {'name': 'LI alone', 'src': 'li', 'row_color': 1},
]

# Below this value, a likelihood looks like 0 on the linear x-axis of the RT figures
_LIK_ZERO_THRESHOLD = 1e-3


def _likelihood_dist_grid(groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                          pi_rule, comparison_save_path, save_stem, save_suffix='', n_run=4,
                          lik_col='likelihood_obs_std_at_dev', n_cols=12, n_bins=45, loglik=True,
                          leaky_file_prefix='leaky_predictions', figsize=None,
                          correcttrials=False):
    """Histograms of the KF and LI likelihoods of every run, i.e. what the x-axis of the per-run
    RT figures contains.

    Same data and trials as `_per_run_model_grid`. Runs are wrapped over `n_cols` columns, KF
    histograms in the top half of the figure, LI ones in the bottom half. Each panel reports the
    median likelihood, the median |z| of the deviant under the standard prediction (bare
    likelihood only) and the share of trials below `_LIK_ZERO_THRESHOLD`.
    lik_col: likelihood column to describe.
    loglik: log10 bins (default). The likelihood spans many orders of magnitude, so on the raw
        scale almost every trial falls in the first bin.
    """
    bare_lik = (lik_col == 'likelihood_obs_std_at_dev')

    def _abs_z(df):
        """|z| of each deviant under the standard context's prediction (NaN if the file lacks
        the prediction columns)."""
        needed = {'observation', 'mu_pred_std_at_dev', 'sigma_pred_std_at_dev'}
        if not needed <= set(df.columns):
            return np.full(len(df), np.nan)
        return np.abs((df['observation'] - df['mu_pred_std_at_dev'])
                      / df['sigma_pred_std_at_dev']).to_numpy(dtype=float)

    # Likelihoods of each run, with the same data, run order and trials as `_per_run_model_grid`
    records = []  # one entry per run, in figure order
    for g in tqdm(groups, desc="Subject/session"):
        sub, sess_num = g['sub'], g['sess_num']
        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, sigma_r=g['sigma_r'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix)

        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")
        d_varies = run_keys['d'].nunique() > 1

        # logrt=True only to select the same trials as the log-RT figures (RT is not plotted)
        df_kf = _select_rt_trials(df_kf, logrt=True, correcttrials=correcttrials)
        df_li = _select_rt_trials(df_li, logrt=True, correcttrials=correcttrials)

        for run_n, prow in run_keys.iterrows():
            rec = {'sub': sub, 'sess': sess_num, 'tau': int(prow['tau_std']),
                   'd': int(prow['d']), 'show_d': bool(d_varies)}
            for m in LIK_DIST_MODELS:
                d = (df_kf if m['src'] == 'kf' else df_li)
                d = d[d['run_n'] == run_n]
                x = d[lik_col].to_numpy(dtype=float)
                keep = np.isfinite(x)
                rec[m['src']] = x[keep]
                rec[m['src'] + '_z'] = _abs_z(d)[keep]
            records.append(rec)

    # Bins shared by all panels. On the log scale, likelihoods equal to 0 are not binned but
    # counted in each panel's annotation.
    pooled = np.concatenate([r[m['src']] for r in records for m in LIK_DIST_MODELS])
    if loglik:
        pos = pooled[pooled > 0]
        assert pos.size, f"No positive values in '{lik_col}' -- nothing to plot on a log scale"
        lo, hi = np.floor(np.log10(pos.min())), np.ceil(np.log10(pos.max()))
    else:
        lo, hi = 0.0, float(pooled.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    def _scaled(v):
        """Map a likelihood onto the figure's x-axis (log10 or raw)."""
        return np.log10(v) if loglik else v

    # Runs over n_cols columns, KF in the top half, LI in the bottom half; shared axes
    n_runs = len(records)
    half = int(np.ceil(n_runs / n_cols))
    n_rows = half * len(LIK_DIST_MODELS)
    if figsize is None:
        figsize = (1.55 * n_cols + 1.0, 1.45 * n_rows + 1.4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True,
                            squeeze=False)
    colors = _model_colors()

    # Title colour per subject, since a subject's runs can span two rows
    subs_in_order = list(dict.fromkeys(r['sub'] for r in records))
    sub_colors = dict(zip(subs_in_order, sns.color_palette('Dark2', max(len(subs_in_order), 3))))

    for mi, m in enumerate(LIK_DIST_MODELS):
        for k, rec in enumerate(records):
            ax = axs[mi * half + k // n_cols, k % n_cols]
            x = rec[m['src']]
            n_zero = int(np.sum(x <= 0))  # no log of 0
            binnable = x[x > 0] if loglik else x
            if binnable.size:
                ax.hist(_scaled(binnable), bins=bins, color=colors[m['row_color']], alpha=0.8)
                med = np.median(x)
                frac = 100.0 * np.mean(x < _LIK_ZERO_THRESHOLD)
                if med <= 0:
                    txt = "med 0"
                else:
                    txt = f"med $10^{{{np.log10(med):.1f}}}$" if loglik else f"med {med:.3g}"
                if bare_lik:
                    z = rec[m['src'] + '_z']
                    if np.isfinite(z).any():
                        txt += f"\n|z|={np.nanmedian(z):.1f}"
                txt += f"\n{frac:.0f}% <1e-3"
                if n_zero and loglik:
                    txt += f"\n{n_zero} =0"
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
                        va='top', ha='left', color='dimgray')
                # Median of the run
                if med > 0 or not loglik:
                    ax.axvline(_scaled(med), color=colors[m['row_color']], linestyle='-',
                               linewidth=1.1)
            # Threshold below which a likelihood looks like 0 on a linear axis
            if loglik:
                ax.axvline(_scaled(_LIK_ZERO_THRESHOLD), color='gray', linestyle=':',
                           linewidth=0.9)

            title = f"S{rec['sub']}·s{rec['sess']}  $\\tau$={rec['tau']}"
            if rec['show_d']:
                title += f", d={rec['d']}"
            ax.set_title(title, fontsize=7.5, pad=2, color=sub_colors[rec['sub']])
            ax.tick_params(labelsize=6.5, length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide the unused panels of the last row
        for k in range(n_runs, half * n_cols):
            axs[mi * half + k // n_cols, k % n_cols].axis('off')

    # Room above the bars for the annotations
    axs[0, 0].set_ylim(top=axs[0, 0].get_ylim()[1] * 1.35)

    plt.subplots_adjust(left=0.045, right=0.995, top=0.885, bottom=0.075, wspace=0.25, hspace=0.62)

    # Axis labels on the last row and first column only
    for col in range(n_cols):
        if axs[-1, col].axison:
            axs[-1, col].set_xlabel("$\\log_{10}$ likelihood" if loglik else "likelihood",
                                    fontsize=7)
    for row in range(n_rows):
        axs[row, 0].set_ylabel("trials", fontsize=7)

    trial_sel = "trials with a valid RT" + (" and a correct deviant position" if correcttrials
                                            else "")
    scale_note = (f"shared log10 bins; solid line = run median, dotted line = "
                  f"{_LIK_ZERO_THRESHOLD:g} "
                  f"(below it, a likelihood is indistinguishable from 0 on a linear axis)"
                  if loglik else
                  "shared RAW-likelihood bins, i.e. the scale the RT scatters use; "
                  "solid line = run median")
    fig.suptitle(f"Per-run distribution of `{lik_col}`  --  {trial_sel}\n{scale_note}",
                 fontsize=11, y=0.995)

    # Model name above each half, and a dashed line between the halves
    for mi, m in enumerate(LIK_DIST_MODELS):
        p_top = axs[mi * half, 0].get_position()
        if mi:
            prev_bottom = axs[mi * half - 1, 0].get_position().y0
            sep_y = (prev_bottom + p_top.y1) / 2 + 0.2 * (prev_bottom - p_top.y1)
            fig.add_artist(Line2D([0.02, 0.98], [sep_y, sep_y], transform=fig.transFigure,
                                  color='gray', linestyle='--', linewidth=1, zorder=0))
            band_y = (sep_y + p_top.y1) / 2
        else:
            band_y = p_top.y1 + 0.4 * (1 - p_top.y1)
        fig.text(0.5, band_y, m['name'], fontsize=14, ha='center', va='center',
                 color=colors[m['row_color']])

    # No '_logrt' in the name: RT is not plotted
    save_name = (f"{comparison_save_path}/{save_stem}_{n_runs}runs"
                 f"{'_loglik' if loglik else '_rawlik'}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def plot_likelihood_distributions_per_run(subjects, trials_path, preds_liks_path, logfiles_path,
                                          comparison_save_path, pi_rule, leaky_preds_path,
                                          lik_col='likelihood_obs_std_at_dev', n_run=4,
                                          n_cols=12, loglik=True, 
                                          figsize=None, correcttrials=False):
    """Per-run KF and LI likelihood histograms for the 04/05/06 cohort, ordered by session type
    so that each figure row is one session type (with n_cols=12)."""
    session_types, session_type_to_params = get_session_types(trials_path)

    groups = []
    for session_type in session_types:
        d_par, si_stat, si_r = session_type
        for sub in subjects:
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            groups.append({
                'sub': sub,
                'sess_num': session_type_to_params[session_type][sub],  # 1-indexed
                'sigma_r': si_r,
            })

    return _likelihood_dist_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='likdist_perrun_KF_LI', save_suffix='_trueobsnoise',
        lik_col=lik_col, n_run=n_run, n_cols=n_cols, loglik=loglik, figsize=figsize,
        correcttrials=correcttrials)


def plot_likelihood_distributions_per_run_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, sess_num=1, sigma_r=0.02, lik_col='likelihood_obs_std_at_dev',
        n_run=4, n_cols=6, loglik=True, leaky_file_prefix='leaky_predictions',
        save_stem='likdist_perrun_KF_LI_sub11-12-13', save_suffix=None, figsize=None,
        correcttrials=False):
    """Per-run KF and LI likelihood histograms for the 11/12/13 cohort.

    save_suffix: defaults to `_sigma_r_{sigma_r}`.
    """
    if save_suffix is None:
        save_suffix = _sigma_r_suffix(sigma_r)
    groups = [{
        'sub': sub,
        'sess_num': sess_num,
        'sigma_r': sigma_r,
    } for sub in subjects]

    return _likelihood_dist_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem=save_stem, save_suffix=save_suffix, lik_col=lik_col,
        n_run=n_run, n_cols=n_cols, loglik=loglik, leaky_file_prefix=leaky_file_prefix,
        figsize=figsize, correcttrials=correcttrials)


def _get_perrun_models_R2(groups, trials_path, logfiles_path, preds_liks_path,
                              leaky_preds_path, pi_rule, n_run=4, logrt=True, loglik=False,
                              leaky_file_prefix='leaky_predictions', correcttrials=False,
                              allvalidtrials=False):
    """Table of the R2 of every (run, model) cell of the per-run RT figure, used by the violin
    and Cohen's d figures.

    Same data, run ordering and trials as `_per_run_model_grid`; `groups` as there, with an
    optional `meta` dict copied into every row. R2 is always the OLS one (Pearson rho squared).
    Cells with fewer than 3 points are skipped.
    Returns one row per cell: model, R2, rho, p, corr, sub, sess_num, run_n, run_idx, tau, d,
    n_points (+ meta).
    """
    records = []
    for g in tqdm(groups, desc="Subject/session (R2)"):
        sub, sess_num = g['sub'], g['sess_num']

        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, allvalidtrials=allvalidtrials, sigma_r=g['sigma_r'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix, allvalidtrials=allvalidtrials)

        # Runs ordered as in `_per_run_model_grid`
        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")

        df_kf = _select_rt_trials(df_kf, logrt=logrt, correcttrials=correcttrials)
        df_li = _select_rt_trials(df_li, logrt=logrt, correcttrials=correcttrials)

        for run_idx, (run_n, prow) in enumerate(run_keys.iterrows()):
            df_kf_r = df_kf[df_kf['run_n'] == run_n]
            df_li_r = df_li[df_li['run_n'] == run_n]

            for m in PER_RUN_MODELS:
                d = df_kf_r if m['src'] == 'kf' else df_li_r
                x = d[m['col']].to_numpy(dtype=float)
                y = d['rt'].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if loglik:
                    pos = x > 0
                    x, y = np.log(x[pos]), y[pos]

                # Same condition as the rho annotation of the per-run figure; R2 always from Pearson
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r, p = _corr_stats(x, y, 'pearson')
                    records.append({
                        'model': m['name'], 'R2': r ** 2, 'rho': r, 'p': p,
                        'corr': 'pearson',
                        'sub': sub, 'sess_num': sess_num,
                        'run_n': int(run_n), 'run_idx': run_idx,
                        'tau': int(prow['tau_std']), 'd': int(prow['d']),
                        **g.get('meta', {}),
                        'n_points': int(len(x)),
                    })

    return pd.DataFrame.from_records(records)


def get_perrun_models_R2(subjects, trials_path, preds_liks_path, logfiles_path, pi_rule,
                             leaky_preds_path, n_run=4, logrt=True, loglik=False,
                             correcttrials=False, allvalidtrials=False):
    """Per-run R2 table for the 04/05/06 cohort (see `_get_perrun_models_R2`), with the
    session-type parameters added as d_par, si_stat and si_r."""
    session_types, session_type_to_params = get_session_types(trials_path)

    groups = []
    for sub in subjects:
        for session_type in session_types:
            d_par, si_stat, si_r = session_type
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            groups.append({
                'sub': sub,
                'sess_num': session_type_to_params[session_type][sub],  # 1-indexed
                'sigma_r': si_r,
                'meta': {'d_par': d_par, 'si_stat': si_stat, 'si_r': si_r},
            })

    return _get_perrun_models_R2(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        n_run=n_run, logrt=logrt, loglik=loglik, correcttrials=correcttrials,
        allvalidtrials=allvalidtrials)


def get_perrun_models_R2_single_session(subjects, trials_path, preds_liks_path, logfiles_path,
                                            pi_rule, leaky_preds_path, sess_num=1, sigma_r=0.02,
                                            n_run=4, logrt=True, loglik=False,
                                            leaky_file_prefix='leaky_predictions',
                                            correcttrials=False, allvalidtrials=False):
    """Per-run R2 table for the 11/12/13 cohort (see `_get_perrun_models_R2`)."""
    groups = [{
        'sub': sub,
        'sess_num': sess_num,
        'sigma_r': sigma_r,
        'meta': {'si_r': sigma_r},
    } for sub in subjects]

    return _get_perrun_models_R2(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        n_run=n_run, logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix,
        correcttrials=correcttrials, allvalidtrials=allvalidtrials)


def _plot_perrun_models_R2_violins_on_ax(ax, df, logrt=True, loglik=False,
                                         order=None, palette=None, show_xticklabels=True):
    """Plot one violin per model of its per-run R2 in `ax`, with the individual values and the
    mean.

    `order` and `palette` let several panels share the model order and colours. Returns the
    number of runs per model.
    """
    if order is None:
        order = [m['name'] for m in PER_RUN_MODELS]
    if palette is None:
        palette = dict(zip(order, _model_colors(order)))

    # One coloured violin per model, kept within the observed R2 range (cut=0)
    sns.violinplot(data=df, x='model', y='R2', order=order, hue='model', hue_order=order,
                   palette=palette, legend=False, cut=0, inner='quartile', ax=ax, saturation=1)
    # Individual per-run R2 values
    sns.stripplot(data=df, x='model', y='R2', order=order, color='0.15', size=3.5,
                  alpha=0.5, jitter=0.15, ax=ax)

    # Mean of each model (white diamond)
    counts = []
    for i, name in enumerate(order):
        vals = df.loc[df['model'] == name, 'R2']
        counts.append(len(vals))
        if len(vals):
            ax.scatter([i], [vals.mean()], marker='D', s=32, color='white',
                       edgecolor='black', zorder=5)

    ax.set_xlabel('')
    ylab = r"R$^2$" # (per-run RT vs. ' + ('log-' if loglik else '') + 'likelihood)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_ylim(bottom=0)
    if show_xticklabels:
        # Model names under every panel, also when the x-axis is shared
        ax.tick_params(labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    else:
        ax.tick_params(labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return counts


def _plot_perrun_models_R2_violins(df, comparison_save_path, save_stem, save_suffix='',
                                   logrt=True, loglik=False):
    """Plot and save the per-run R2 violins, all subjects pooled; `df` comes from
    `_get_perrun_models_R2`."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_perrun_models_R2_violins_on_ax(ax, df, logrt=logrt, loglik=loglik)
    plt.tight_layout()

    save_name = (f"{comparison_save_path}/{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def _check_R2_table(df, subjects=None, R2_function='get_perrun_models_R2'):
    """Raise a readable error if an R2 table is empty, lacks a subject, or was not computed with
    Pearson.

    An empty table usually means the R2 function does not match the cohort (11/12/13 needs
    `get_perrun_models_R2_single_session`). The Pearson check rejects tables from older
    versions of this module, where R2 could be a squared Spearman rho.
    """
    if len(df) == 0:
        raise ValueError(
            f"{R2_function} returned an empty R2 table"
            + (f" for subjects {list(subjects)}" if subjects is not None else "")
            + ". No (subject, session) was matched: check that the paths "
              "(trials/logfiles/kalman/leaky) and the R2 function match the cohort. Subjects "
              "11/12/13 need `get_perrun_models_R2_single_session`.")
    missing = [s for s in (subjects or []) if s not in set(df['sub'])]
    if missing:
        raise ValueError(f"R2 table has no rows for subject(s) {missing} "
                         f"(present: {sorted(set(df['sub']))}).")
    # Older tables have no `corr` column
    if 'corr' in df.columns:
        found = sorted(set(df['corr'].dropna()))
        if found != ['pearson']:
            raise ValueError(
                f"R2 table was computed with corr={found}; these figures need the R2 from "
                f"Pearson. Recompute the table with {R2_function}(...).")


def _plot_perrun_models_R2_violins_per_subject(df, comparison_save_path, save_stem, save_suffix='',
                                               logrt=True, loglik=False, subjects=None):
    """Plot and save the per-run R2 violins with one panel per subject, stacked vertically."""
    _check_R2_table(df, subjects)
    order = [m['name'] for m in PER_RUN_MODELS]
    palette = dict(zip(order, _model_colors(order)))

    if subjects is None:
        subjects = sorted(df['sub'].unique())
    n = len(subjects)
    subject_labels = _subject_display_map(subjects)
    save_prefix = _subject_prefix(subjects)

    # Shared R2 axis
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.8 * n), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[:, 0]

    for sub, ax in zip(subjects, axes):
        sub_df = df[df['sub'] == sub]
        _plot_perrun_models_R2_violins_on_ax(
            ax, sub_df, logrt=logrt, loglik=loglik, order=order, palette=palette,
            show_xticklabels=True)
        ax.set_title(_sub_label(subject_labels, sub), fontsize=11, loc='center')

    plt.tight_layout()

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def plot_perrun_models_R2_violins(subjects, trials_path, preds_liks_path, logfiles_path,
                                  comparison_save_path, pi_rule, leaky_preds_path, n_run=4,
                                  logrt=True, loglik=False, 
                                  correcttrials=False):
    """Violins of each model's per-run R2 for the 04/05/06 cohort: which model's likelihood best
    explains RT, and how much this varies across runs."""
    df = get_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik,
        correcttrials=correcttrials,
    )
    save_name = _plot_perrun_models_R2_violins(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_violins',
        save_suffix='_trueobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)
    return save_name, df


def plot_perrun_models_R2_violins_per_subject(subjects, trials_path, preds_liks_path,
                                              logfiles_path, comparison_save_path, pi_rule,
                                              leaky_preds_path, n_run=4, logrt=True, loglik=False,
                                              correcttrials=False,
                                              allvalidtrials=False, df=None):
    """Per-subject version of `plot_perrun_models_R2_violins`.

    df: an R2 table already computed (e.g. for the 11/12/13 cohort), which skips its computation.
    """
    if df is None:
        df = get_perrun_models_R2(
            subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
            n_run=n_run, logrt=logrt, loglik=loglik,
            correcttrials=correcttrials, allvalidtrials=allvalidtrials,
        )
    save_name = _plot_perrun_models_R2_violins_per_subject(
        df, comparison_save_path, save_stem='RT_persubject_R2',
        save_suffix='_trueobsnoise' + ('_correcttrials' if correcttrials else '')
                    + ('_allvalidtrials' if allvalidtrials else ''),
        logrt=logrt, loglik=loglik, subjects=subjects)
    return save_name, df


def plot_perrun_models_R2_violins_single_session(subjects, trials_path, preds_liks_path,
                                                 logfiles_path, comparison_save_path, pi_rule,
                                                 leaky_preds_path, sess_num=1, sigma_r=0.02,
                                                 n_run=4, logrt=True, loglik=False,
                                                 leaky_file_prefix='leaky_predictions',
                                                 save_stem='RT_perrun_8models_R2_violins_sub11-12-13',
                                                 save_suffix='_cueprior', correcttrials=False):
    """11/12/13-cohort version of `plot_perrun_models_R2_violins` (12 runs per violin)."""
    df = get_perrun_models_R2_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        sess_num=sess_num, sigma_r=sigma_r, n_run=n_run, logrt=logrt, loglik=loglik,
        leaky_file_prefix=leaky_file_prefix, correcttrials=correcttrials,
    )
    save_name = _plot_perrun_models_R2_violins(
        df, comparison_save_path, save_stem=save_stem,
        save_suffix=save_suffix + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)
    return save_name, df


def _cohens_d_paired(a, b):
    """Cohen's d between two models' per-run R2: (mean(a) - mean(b)) / pooled SD.

    This compares the two distributions; the run pairing is only used by the Wilcoxon test
    reported alongside.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    s_pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if s_pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / s_pooled


def _model_R2_cohensd_matrices(df, order):
    """Cohen's d (D, 'row minus column'), Wilcoxon signed-rank p-value (P) and number of paired
    runs (N) for every pair of models, their R2 being paired by run.

    The diagonal and pairs with fewer than 2 shared runs are NaN.
    """
    # R2 of each model, paired by run (identified by run_n, since tau_std is not unique in every
    # cohort)
    mat = df.pivot_table(index=['sub', 'sess_num', 'run_n'], columns='model', values='R2')
    mat = mat.reindex(columns=order)

    n = len(order)
    D = np.full((n, n), np.nan)  # Cohen's d
    P = np.full((n, n), np.nan)  # Wilcoxon signed-rank p-value
    N = np.zeros((n, n), dtype=int)  # number of paired runs
    for i in range(n):
        for j in range(i + 1, n):
            pair = mat[[order[i], order[j]]].dropna()  # runs with an R2 for both models
            a = pair[order[i]].to_numpy()
            b = pair[order[j]].to_numpy()
            N[i, j] = N[j, i] = len(a)
            if len(a) >= 2:
                D[i, j] = _cohens_d_paired(a, b)
                try:
                    _, P[i, j] = ss.wilcoxon(a, b)
                except ValueError:
                    P[i, j] = np.nan  # e.g. all paired differences are zero
                # Lower triangle: d changes sign, p stays the same
                D[j, i] = -D[i, j]
                P[j, i] = P[i, j]
    return D, P, N


def _plot_model_R2_cohensd_heatmap_on_ax(ax, D, P, order, vmax, show_yticklabels=True):
    """Plot the Cohen's d matrix as a heatmap in `ax` (red: the row model explains RT better),
    with d and the Wilcoxon stars in each cell.

    `vmax` sets the symmetric colour scale, so several panels can share it. Returns the image,
    for the colorbar.
    """
    n = len(order)
    Dm = np.ma.masked_invalid(D)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white')  # empty diagonal, and pairs with too few shared runs

    im = ax.imshow(Dm, cmap=cmap, vmin=-vmax, vmax=vmax)

    # d value in each cell, with the significance stars below
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(D[i, j]):
                continue
            frac = abs(D[i, j]) / vmax if vmax else 0.0
            txt_color = 'white' if frac > 0.6 else 'black'
            ax.text(j, i - 0.16, f"{D[i, j]:+.2f}", ha='center', va='center',
                    fontsize=9, color=txt_color)
            ax.text(j, i + 0.2, _p_stars(P[i, j]) if np.isfinite(P[i, j]) else 'ns',
                    ha='center', va='center', fontsize=10, color=txt_color, weight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=12)
    if show_yticklabels:
        ax.set_yticklabels(order, fontsize=9)
    else:
        ax.set_yticklabels([])
    # White borders between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _plot_model_R2_cohensd_heatmap(df, comparison_save_path, save_stem, save_suffix='',
                                   logrt=True, loglik=False):
    """Plot and save the Cohen's d heatmap, all subjects pooled; `df` comes from
    `_get_perrun_models_R2`. Returns (save_name, D, P)."""
    order = [m['name'] for m in PER_RUN_MODELS]
    D, P, N = _model_R2_cohensd_matrices(df, order)

    # Colour scale centred on 0
    vmax = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1.0

    fig, ax = plt.subplots(figsize=(12, 8))
    im = _plot_model_R2_cohensd_heatmap_on_ax(ax, D, P, order, vmax)

    cbar = fig.colorbar(im, ax=ax,  pad=0.04) # fraction=0.046,
    cbar.set_label(r"Cohen's $d$", fontsize=10) #  (row − column):  >0 row model fits RT better
    cbar.outline.set_visible(False)

    n_runs = int(np.nanmax(N)) if N.any() else 0
    # ax.set_title(
    #     "Pairwise separation of models' per-run $R^2$ (Cohen's d)\n"
    #     f"Wilcoxon signed-rank on {n_runs} paired runs ",
    #     fontsize=12,
    # )
    plt.tight_layout()

    save_name = (f"{comparison_save_path}/{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name, D, P


def _plot_model_R2_cohensd_heatmap_per_subject(df, comparison_save_path, save_stem, save_suffix='',
                                               logrt=True, loglik=False, subjects=None,
                                               scale=1.0):
    """Plot and save one Cohen's d heatmap per subject, stacked vertically, on a shared colour
    scale.

    scale: multiplies the figure size without changing font sizes, giving the cell labels more
    room. Returns (save_name, D_by_sub, P_by_sub).
    """
    _check_R2_table(df, subjects)
    order = [m['name'] for m in PER_RUN_MODELS]

    if subjects is None:
        subjects = sorted(df['sub'].unique())
    n_sub = len(subjects)
    subject_labels = _subject_display_map(subjects)
    save_prefix = _subject_prefix(subjects)

    # Matrices of every subject first, to share the colour scale across panels
    D_by_sub, P_by_sub = {}, {}
    for sub in subjects:
        D, P, _ = _model_R2_cohensd_matrices(df[df['sub'] == sub], order)
        D_by_sub[sub], P_by_sub[sub] = D, P
    finite_maxes = [np.nanmax(np.abs(D)) for D in D_by_sub.values() if np.isfinite(D).any()]
    vmax = max(finite_maxes) if finite_maxes else 1.0

    fig, axes = plt.subplots(n_sub, 1, figsize=(6 * scale, 3.5 * n_sub * scale), squeeze=False,
                             constrained_layout=True)
    axes = axes[:, 0]

    for sub, ax in zip(subjects, axes):
        im = _plot_model_R2_cohensd_heatmap_on_ax(
            ax, D_by_sub[sub], P_by_sub[sub], order, vmax, show_yticklabels=True)
        ax.set_title("", fontsize=12) # _sub_label(subject_labels, sub)
        # One colorbar per panel, all on the same scale
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Cohen's $d$", fontsize=12) #  (row − column):  >0 row model fits RT better
        cbar.outline.set_visible(False)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")
    return save_name, D_by_sub, P_by_sub


def _plot_model_R2_cohensd_heatmap_and_violins_per_subject(
        df, comparison_save_path, save_stem, save_suffix='', logrt=True, loglik=False,
        subjects=None, figsize=None, width_ratios=(5.4, 4.6), row_height=3.5, scale=1.0):
    """Plot and save the per-subject R2 violins (left) and Cohen's d heatmaps (right) in one
    figure, one row per subject.

    width_ratios and row_height set the layout; scale multiplies the derived figure size
    (ignored if figsize is given). Returns (save_name, D_by_sub, P_by_sub).
    """
    _check_R2_table(df, subjects)
    order = [m['name'] for m in PER_RUN_MODELS]
    palette = dict(zip(order, _model_colors(order)))

    if subjects is None:
        subjects = sorted(df['sub'].unique())
    n_sub = len(subjects)
    subject_labels = _subject_display_map(subjects)
    save_prefix = _subject_prefix(subjects)

    # Matrices of every subject first, to share the colour scale across panels
    D_by_sub, P_by_sub = {}, {}
    for sub in subjects:
        D, P, _ = _model_R2_cohensd_matrices(df[df['sub'] == sub], order)
        D_by_sub[sub], P_by_sub[sub] = D, P
    finite_maxes = [np.nanmax(np.abs(D)) for D in D_by_sub.values() if np.isfinite(D).any()]
    vmax = max(finite_maxes) if finite_maxes else 1.0

    if figsize is None:
        figsize = (sum(width_ratios) * scale, row_height * n_sub * scale)
    fig, axes = plt.subplots(n_sub, 2, figsize=figsize, squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(width_ratios=list(width_ratios)))

    for k, sub in enumerate(subjects):
        ax_v, ax_h = axes[k, 0], axes[k, 1]
        sub_df = df[df['sub'] == sub]

        im = _plot_model_R2_cohensd_heatmap_on_ax(
            ax_h, D_by_sub[sub], P_by_sub[sub], order, vmax, show_yticklabels=True)
        # One colorbar per panel, all on the same scale
        cbar = fig.colorbar(im, ax=ax_h)
        cbar.set_label(r"Cohen's $d$", fontsize=12)
        cbar.outline.set_visible(False)

        _plot_perrun_models_R2_violins_on_ax(
            ax_v, sub_df, logrt=logrt, loglik=loglik, order=order, palette=palette,
            show_xticklabels=True)
        ax_v.set_title(_sub_label(subject_labels, sub), fontsize=11, loc='center')

    # Same R2 range for all subjects
    top = max(ax.get_ylim()[1] for ax in axes[:, 0])
    for ax in axes[:, 0]:
        ax.set_ylim(0, top)

    save_name = (f"{comparison_save_path}/{save_prefix}{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")
    return save_name, D_by_sub, P_by_sub


def plot_model_R2_cohensd_heatmap(subjects, trials_path, preds_liks_path, logfiles_path,
                                  comparison_save_path, pi_rule, leaky_preds_path, n_run=4,
                                  logrt=True, loglik=False, 
                                  correcttrials=False):
    """Heatmap of Cohen's d between every pair of models' per-run R2, with Wilcoxon signed-rank
    stars (04/05/06 cohort): which model's likelihood explains RT better than which."""
    df = get_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik,
        correcttrials=correcttrials,
    )
    return _plot_model_R2_cohensd_heatmap(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_cohensd_heatmap',
        save_suffix='_trueobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)


def plot_model_R2_cohensd_heatmap_per_subject(subjects, trials_path, preds_liks_path,
                                              logfiles_path, comparison_save_path, pi_rule,
                                              leaky_preds_path, n_run=4, logrt=True, loglik=False,
                                              correcttrials=False,
                                              allvalidtrials=False, df=None, scale=1.0):
    """Per-subject version of `plot_model_R2_cohensd_heatmap`.

    df: an R2 table already computed (e.g. for the 11/12/13 cohort), which skips its computation.
    scale: see `_plot_model_R2_cohensd_heatmap_per_subject`.
    """
    if df is None:
        df = get_perrun_models_R2(
            subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
            n_run=n_run, logrt=logrt, loglik=loglik,
            correcttrials=correcttrials, allvalidtrials=allvalidtrials,
        )
    return _plot_model_R2_cohensd_heatmap_per_subject(
        df, comparison_save_path, save_stem='RT_persubject_R2_cohensd_heatmap',
        save_suffix='_trueobsnoise' + ('_correcttrials' if correcttrials else '')
                    + ('_allvalidtrials' if allvalidtrials else ''),
        logrt=logrt, loglik=loglik, subjects=subjects, scale=scale)


def plot_model_R2_cohensd_heatmap_and_violins_per_subject(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, n_run=4, logrt=True, loglik=False, 
        correcttrials=False, allvalidtrials=False, df=None, figsize=None,
        width_ratios=(5.4, 4.6), row_height=3.5, scale=1.0):
    """Per-subject R2 violins and Cohen's d heatmaps in one figure (see
    `_plot_model_R2_cohensd_heatmap_and_violins_per_subject`). Returns (save_name, df).

    df: an R2 table already computed, which skips its computation.
    """
    if df is None:
        df = get_perrun_models_R2(
            subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
            n_run=n_run, logrt=logrt, loglik=loglik,
            correcttrials=correcttrials, allvalidtrials=allvalidtrials,
        )
    save_name, _, _ = _plot_model_R2_cohensd_heatmap_and_violins_per_subject(
        df, comparison_save_path, save_stem='RT_persubject_R2_cohensd_heatmap_violins',
        save_suffix='_trueobsnoise' + ('_correcttrials' if correcttrials else '')
                    + ('_allvalidtrials' if allvalidtrials else ''),
        logrt=logrt, loglik=loglik, subjects=subjects, figsize=figsize,
        width_ratios=width_ratios, row_height=row_height, scale=scale)
    return save_name, df


def plot_model_R2_cohensd_heatmap_single_session(subjects, trials_path, preds_liks_path,
                                                 logfiles_path, comparison_save_path, pi_rule,
                                                 leaky_preds_path, sess_num=1, sigma_r=0.02,
                                                 n_run=4, logrt=True, loglik=False,
                                                 leaky_file_prefix='leaky_predictions',
                                                 save_stem='RT_perrun_8models_R2_cohensd_heatmap_sub11-12-13',
                                                 save_suffix='_cueprior', correcttrials=False):
    """11/12/13-cohort version of `plot_model_R2_cohensd_heatmap`.

    With only 12 paired runs, the Wilcoxon test has little power: read d first.
    """
    df = get_perrun_models_R2_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        sess_num=sess_num, sigma_r=sigma_r, n_run=n_run, logrt=logrt, loglik=loglik,
        leaky_file_prefix=leaky_file_prefix, correcttrials=correcttrials,
    )
    return _plot_model_R2_cohensd_heatmap(
        df, comparison_save_path, save_stem=save_stem,
        save_suffix=save_suffix + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)


def run_KF_sub_sess(subject, session_num, trials_path, kf_preds_save_path):
    trials = prepare_trials_data(trials_path)
    compute_likelihoods_at_deviants(trials_path, subject, session_num, results_save_path=kf_preds_save_path)


def run_KF_all(subjects, trials_path, kf_preds_save_path):
    session_types, session_type_to_params = get_session_types(trials_path)

    for sub in tqdm(subjects, desc="Subject"):
        for session_type in tqdm(session_types, desc="Session Type"):
            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number
            trials_file = glob.glob(f"{trials_path}/sub-{sub}/sub-{sub}_ses-{sess_num}-*trials.csv")[0]
            run_KF_sub_sess(sub, sess_num, trials_file, kf_preds_save_path)




if __name__ == "__main__":
    print("in model_RTs.py main")
    pass
    