import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from scipy import stats as ss # to get pearson: ss.pearsonr(x, y)
from matplotlib.axes import Axes
import re
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# from audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM


# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext, likelihood_observation, contexts_to_probabilities
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext, likelihood_observation, contexts_to_probabilities
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
    
    trials['relative_pos'] = np.tile(np.arange(8), int(n_trials)) # 1 to 8 for each run, to match dpos indexing
    if exclude_null:
        trials = trials[trials['dpos'] != 0].reset_index(drop=True)
    trials['contexts'] = trials['relative_pos'] == trials['dpos'] # 0: standard, 1: deviant
    # dpos = 0 is actually not a deviant but an omission, so convert it to standard
    if not exclude_null:
        # if null rule trials have been kept, still flag their position as standard
        trials.loc[trials.dpos == 0, 'contexts'] = False # TODO: but was I even using the contexts column later on?
    return trials



def _sigma_r_suffix(sigma_r):
    """Filename suffix for a KF prediction file fixed at observation-noise std `sigma_r`.

    Empty when sigma_r is None (EM-fitted R), which keeps those filenames the drop-in default.
    Single source of truth, shared by the writer (compute_likelihoods_at_deviants) and the
    reader (aggregate_data), so the two can never drift apart.
    """
    return "" if sigma_r is None else f"_sigma_r_{sigma_r}"


# Context index -> label used in the fitted-parameter column names. The KF runs a separate
# per-context filter (see kalman_online_fit_predict_multicontext), so each context carries its
# own fitted A, Q, H, R.
_KF_CTX_NAMES = ('std', 'dev')


def _kf_param_columns(kf_params):
    """Flatten the per-timestep, per-context fitted KF parameters into named 1D columns.

    `kf_params` is the dict returned by
    ``kalman_online_fit_predict_multicontext(..., return_params=True)``: arrays indexed
    ``[t, c, ...]`` with A, Q -> (T, C, 2, 2); H -> (T, C, 1, 2); R -> (T, C, 1, 1).

    Returns a dict ``{column_name: array of length T}`` in a stable order. Column names carry
    the context label and the matrix entry, e.g. ``A_std_01``, ``Q_dev_11``, ``H_std_0``,
    ``R_dev``. NOTE ``R`` is the fitted observation-noise VARIANCE, so compare it against the
    true ``sigma_r`` SQUARED (not sigma_r itself). Entries are NaN before a context warms up.
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
    return cols


def compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=None, sigma_r=None):
    """
    Compute Kalman filter predictions, and predictions & likelihoods at deviant positions.

    Args:
        trials_path: Path to load trials from (glob patterns supported)
        sub: Subject identifier
        sess: Session number (1-indexed)
        results_save_path: Path to save results. If None, saves in script directory.
        sigma_r: Optional fixed observation noise, as a STANDARD DEVIATION on the same scale as
            the `sigma_r` column of the triallists (e.g. 0.01). Squared here into the variance R
            that the KF actually takes. If None, EM estimates R.

    Units: every saved `sigma_*` column is a STANDARD DEVIATION.

    Filenames are suffixed `_sigma_r_{sigma_r}` (not `_obs_noise_{...}`) so these files cannot be
    confused with the older `kalman_predictions_noise_comparison*/` outputs, whose `_obs_noise_X`
    suffix recorded X as a *variance* -- the same numbers meaning different things.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    else:
        os.makedirs(results_save_path, exist_ok=True)

    # The KF fixes R = observation_covariance, a variance; the caller supplies a std.
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

        # Run multirule-multicontext KF
        # So far just run multi-context KF:
        # NOTE on units: sigma_pred is a STANDARD DEVIATION (_aggregate_contexts sqrts it), but
        # per_ctx_var holds VARIANCES. Keep the two straight -- they are named accordingly here.
        mu_pred, sigma_pred, kfs_fitted, per_ctx_mu_pred, per_ctx_var, kf_params = kalman_online_fit_predict_multicontext(
            observations, ctx_probabilities, n_iter=5, return_per_ctx=True, return_params=True, observation_noise=observation_noise
        )
        per_ctx_var_std = per_ctx_var[:, 0]
        per_ctx_var_dev = per_ctx_var[:, 1]

        # Save predictions. Every saved `sigma_*` column is a STANDARD DEVIATION, so the
        # per-context variances are sqrt'd here to match sigma_pred and the column names.
        results['run_id'].append(np.full(len(mu_pred), run_id))
        results['trial_no'].append(run['trial_no'])
        results['mu_pred'].append(mu_pred)
        results['sigma_pred'].append(sigma_pred)
        results['mu_pred_std'].append(per_ctx_mu_pred[:, 0])
        results['sigma_pred_std'].append(np.sqrt(per_ctx_var_std))
        results['mu_pred_dev'].append(per_ctx_mu_pred[:, 1])
        results['sigma_pred_dev'].append(np.sqrt(per_ctx_var_dev))

        # Fitted KF parameters (A, Q, H, R) at every timestep, one set per context. These let
        # us check R against the true observation noise: it is held at sigma_r**2 when sigma_r
        # is supplied, and EM-estimated (free to drift) when it is None. R is a VARIANCE; the
        # other params are dimensionless / state-scale. See _kf_param_columns for the naming.
        run_param_cols = _kf_param_columns(kf_params)
        for col, values in run_param_cols.items():
            results.setdefault(col, []).append(values)

        # Note:
        # - First MIN_OBS_FOR_EM + 1 predictions are np.nan
        # - So per_ctx_mu_pred[:MIN_OBS_FOR_EM + 1, 0] and per_ctx_mu_pred[:8*(MIN_OBS_FOR_EM)+(dpos[0]+1), 1] are np.nan
        
        # Evaluate per_ctx_mu_pred[ctx=std] at positions of deviant
        # per_ctx_mu_pred has dimensions (n_timesteps, n_ctx), so need to slice for timestep = when contexts is 1
        mu_pred_std_at_dev = per_ctx_mu_pred[contexts, 0] # should have len=sum(contexts==False)= #standards in run
        var_std_at_dev = per_ctx_var_std[contexts]
        mu_pred_dev_at_dev = per_ctx_mu_pred[contexts, 1] # should have len=sum(contexts==True)= #deviants in run
        var_dev_at_dev = per_ctx_var_dev[contexts]


        # Evaluate likelihood for observations at positions of deviant if they were generated by the predicted distribution for the standard context:
        # likelihood_observation is VARIANCE-parameterized, so it takes var_*, not the sqrt'd
        # values written to the `sigma_*` columns below.
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

        # Store likelihood at t = dpos results for this run
        results_devpos['trial_no'].append(run['trial_no'][contexts].values) # TODO: verify if correct
        results_devpos['mu_pred_std_at_dev'].append(mu_pred_std_at_dev)
        results_devpos['sigma_pred_std_at_dev'].append(np.sqrt(var_std_at_dev))
        results_devpos['mu_pred_dev_at_dev'].append(mu_pred_dev_at_dev)
        results_devpos['sigma_pred_dev_at_dev'].append(np.sqrt(var_dev_at_dev))
        results_devpos['likelihood_obs_std_at_dev'].append(likelihood_obs_std_at_dev)
        results_devpos['likelihood_obs_dev_at_dev'].append(likelihood_obs_dev_at_dev)
        results_devpos['dpos'].append(run['dpos'][contexts].values)

        # The same fitted parameters, restricted to the deviant timesteps this file is about
        # (suffixed `_at_dev`, like the prediction/likelihood columns above).
        for col, values in run_param_cols.items():
            results_devpos.setdefault(f'{col}_at_dev', []).append(values[contexts.values])

    ## Concatenate results across runs
    # Predictions over entire sequences
    results = pd.DataFrame({
        k: np.concatenate(v) for k, v in results.items()
    })
    
    # Generate filename suffix based on the fixed observation noise std (empty when EM-fitted)
    noise_suffix = _sigma_r_suffix(sigma_r)

    results.to_csv(os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess}{noise_suffix}.csv'), index=False)

    # Predictions and likelihoods at deviant locations
    results_devpos = pd.DataFrame({
        k: np.concatenate(v) for k, v in results_devpos.items()
    })
    results_devpos.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}{noise_suffix}.csv'), index=False)

    return results_devpos


def get_valid_positions_per_rule():
    """
    Returns the set of valid deviant positions for each rule.
    
    Rule 1: Deviant can appear at positions {2, 3, 4}
    Rule 2: Deviant can appear at positions {4, 5, 6}
    Rule 0 (or unknown): No valid positions
    """
    return {
        0: [2, 3, 4],
        1: [4, 5, 6]
    }


def prior_dpos_given_prev_rule(d, r, valid_positions=None):
    """
    Prior probability of deviant occurring at position d, given rule r.

    Mathematical formulation:
    P(d = t | r) = { 1 / |D_r|  if t ∈ D_r
                   { 0          otherwise

    where:
      - d: deviant position
      - r: rule
      - D_r: set of valid deviant positions under rule r
      - |D_r|: cardinality (number of valid positions for rule r)

    Interpretation: Within each rule, all valid deviant positions are equally probable
    (uniform distribution over D_r).

    Parameters
    ----------
    valid_positions : dict, optional
        Mapping {rule -> list of valid deviant positions}. If None, falls back to
        get_valid_positions_per_rule() (the original hard-coded PreProParadigm map).
        Pass a dict derived from the data config's rules_dpos_set to reuse this
        hazard for a different GM (e.g. the RNN HierarchicalGM, where rule 0 = [3,4,5]).
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()

    # rules absent from valid_positions (e.g. a null rule) get zero mass
    if r not in valid_positions:
        return 0

    if d in valid_positions[r]:
        return 1.0 / (max(valid_positions[r])-d+1) # uniform over valid positions left
    else:
        return 0


def p_no_dev_before(d, r, valid_positions=None):
    """P(S_{<d} | r): probability that NO deviant has occurred before position d, under rule r.

    S_{<d} is the event "all tones at positions 0..d-1 were standards". Since the deviant is
    uniform over D_r, this is just the fraction of D_r still at or ahead of d:

        P(S_{<d} | r) = #{x in D_r : x >= d} / |D_r|

    (the survival function of the deviant position). A rule with NO valid deviant positions --
    the no-deviant rule 2 of subjects 04/05/06 -- never produces a deviant, so it always reaches
    position d: P(S_{<d} | r) = 1 for every d.

    This is the factor that makes "no deviant so far" evidence about the RULE, not just about the
    position: it is 0 for any rule whose deviant slots are all behind d, which excludes that rule
    outright. See BehavModeling/deviant_prior_maths.md.
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    D = valid_positions.get(r, [])
    if len(D) == 0:
        return 1.0
    return sum(1 for x in D if x >= d) / len(D)


def _hazard_without_rule_posterior(d, valid_positions):
    """Hazard at a position whose rule posterior is undefined (every rule with mass excluded).

    Every hazard here is sum_r h(d, r) * w(r), with w = unnorm / Z. When each rule carrying mass
    has P(S_{<d} | r) = 0 the weights are 0/0: S_{<d} is impossible, so there is no posterior over
    rules to average h against. Two different situations hide behind that 0/0:

      d is NOT a deviant slot of ANY rule -- position 7 of this design, exactly like positions 0
        and 1. Then h(d, r) = 0 for every rule, so the sum is 0 under EVERY choice of weights: the
        0/0 sits only in weights multiplying an all-zero vector, and the hazard is well defined at
        0. Sequences are read timestep by timestep and each timestep must return a usable
        probability -- positions 0 and 1 already report 0 for this reason (there Z simply happens
        to be > 0), and a NaN would propagate into the per-timestep mixture likelihood downstream.

      d IS a slot of some rule that merely carries zero mass -- only reachable with a degenerate
        pi_rule (say all mass on rule 0, asking about d = 5). The answer then swings between 0 and
        h(d, r) depending on which excluded rule you perturb toward, so it is genuinely undefined
        and stays NaN; zero-filling it would invent a prediction.
    """
    if any(d in D for D in valid_positions.values()):
        return np.nan
    return 0.0


def null_rule_marginal(pi_rule, valid_positions=None):
    """Stationary probability that a trial is a NO-DEVIANT trial, implied by `pi_rule`.

    "No-deviant rules" are those the transition matrix knows about but that own no deviant slots
    (absent from `valid_positions`, or mapped to an empty list) -- rule 2 of subjects 04/05/06.
    Their share of the chain's stationary distribution is the long-run fraction of trials with no
    deviant at all, which is what the rule-agnostic prior must reserve.

    Uses the NOMINAL matrix, not observed frequencies: the trial lists were sampled from `pi_rule`,
    so realized proportions scatter slightly around it (04/05/06 observe 0.0872 against the nominal
    1/11 = 0.0909 -- 0.70 binomial SE over 2880 trials, i.e. sampling noise). The generative value
    is what an ideal observer converges to.

    Returns 0.0 when every rule owns deviant slots (e.g. the 2x2 of subjects 11/12/13).
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    pi_rule = np.asarray(pi_rule, dtype=float)
    null_rules = [r for r in range(pi_rule.shape[0]) if len(valid_positions.get(r, [])) == 0]
    if not null_rules:
        return 0.0

    # Stationary distribution: the left eigenvector of pi_rule for eigenvalue 1, normalized.
    w, v = np.linalg.eig(pi_rule.T)
    stat = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    stat = stat / stat.sum()
    return float(sum(stat[r] for r in null_rules))


def prior_dpos_given_prev_stds(d, p_no_dev=0.0, valid_positions=None):
    """Deviant probability at position d given only that everything before it was a standard.

    The "global" prior: it MARGINALIZES OVER the rules rather than ignoring them, so it inherits
    the position imbalance the rules create, but it is blind to their dynamics (no transition
    matrix, no previous rule) and to which rule owns which slot. Hence the hazard is that of a
    deviant drawn uniformly from the pooled MULTISET of slots, {2,3,4} U {4,5,6} = {2,3,4,4,5,6},
    where position 4 appears twice because both rules can place a deviant there:

        h(d) = #{slots == d} / #{slots >= d}
             = 1/6, 1/5, 2/4, 1/2, 1/1   at d = 2..6      (the historical hardcoded table)

    Since rule structure IS represented here, a no-deviant rule belongs in it too: participants can
    learn that some trials carry no deviant at all, so surviving to the last slot must NOT make a
    deviant certain. `p_no_dev` adds that rule, with the remaining mass split evenly over the
    deviant-bearing rules (exact for this design, whose stationary distribution is symmetric
    between rules 0 and 1). For 04/05/06, p_no_dev = 1/11 gives

        h(d) = 0.152, 0.179, 0.435, 0.385, 0.625   at d = 2..6

    -- uniformly lower, and never certain. See BehavModeling/deviant_prior_maths.md.

    Parameters
    ----------
    p_no_dev : float
        Stationary probability that the trial has NO deviant. 0.0 (the default) reproduces the
        historical table exactly and is correct for cohorts without a null rule (11/12/13).
        Derive it from a transition matrix with `null_rule_marginal(pi_rule)`.
    valid_positions : dict, optional
        {rule -> valid deviant positions}; defaults to get_valid_positions_per_rule().

    Returns 0 at a position no rule owns (position 7), even when the conditioning event itself has
    probability 0 there because a deviant is guaranteed by position 6: no rule can place a deviant
    at 7, so the hazard is 0 whatever the rule weights -- as at positions 0 and 1. See
    `_hazard_without_rule_posterior` for the one case that stays NaN.
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    dev_rules = [r for r, D in valid_positions.items() if len(D) > 0]

    # Rule marginal: p_no_dev on the no-deviant rule, the rest split evenly over the others.
    p_each = (1.0 - p_no_dev) / len(dev_rules)
    unnorm = {r: p_each * p_no_dev_before(d, r, valid_positions) for r in dev_rules}
    Z = sum(unnorm.values()) + p_no_dev  # the no-deviant rule always survives to any position
    if Z == 0:
        return _hazard_without_rule_posterior(d, valid_positions)
    return sum(prior_dpos_given_prev_rule(d, r, valid_positions) * unnorm[r] / Z for r in dev_rules)

def pior_dpos_given_prev_rule_and_stds(d, pi_rule, r_prev, valid_positions=None,
                                       legacy_rule_weights=False):
    """
    Posterior probability of deviant at position d, given previous rule, transitions,
    AND that all positions before d are standard.
    [POSTERIOR: Conditional on observed data S_{<d}]

    Mathematical formulation:
    P(d | r_prev, π, S_{<d}) = Σ_r P(d | r, S_{<d}) · P(r | r_prev, S_{<d})

    where:
      - d: deviant position (what we want to know about)
      - r_prev: previous rule
      - π: rule transition matrix
      - S_{<d}: condition that all positions s < d are standard (OBSERVED DATA)
      - P(d | r, S_{<d}): prior_dpos_given_prev_rule(d, r) — position hazard within rule r
      - P(r | r_prev, S_{<d}) ∝ π[r_prev, r] · P(S_{<d} | r): the rule POSTERIOR, i.e. the
        transition prior REWEIGHTED by how likely each rule was to get this far without a deviant

    Interpretation: given what we've observed (that earlier positions were standard), where do we
    think the deviant is? S_{<d} is evidence about TWO things at once -- where the deviant is
    within a rule, and WHICH RULE is active -- and both must be applied. Reaching position 5 with
    no deviant is logically impossible under rule 0 (whose slots are {2,3,4}), so rule 0 must drop
    out of the mixture entirely; the transition prior alone would keep weighting it by π[r_prev,0].

    Until 2026-07 this function used the bare transition prior π[r_prev, r] as the weight, applying
    S_{<d} within each rule but not across rules. That understated the late-position deviant prior
    by up to 5x (17x in the cue-aware variant) and left the hazards internally inconsistent: the
    implied P(no deviant in trial) = Π_d (1 - h(d)) came out at 0.063/0.070/0.078 for r_prev =
    0/1/2 instead of the required π[r_prev, 2] = 0.10/0.10/0.00. Full derivation, numbers and
    consistency check in BehavModeling/deviant_prior_maths.md.

    Parameters
    ----------
    valid_positions : dict, optional
        Mapping {rule -> list of valid deviant positions}. If None, falls back to
        get_valid_positions_per_rule(). Pass a dict from the data config's
        rules_dpos_set to reuse this marginalization for another GM.
        Rules present in `pi_rule` but ABSENT here (or mapped to an empty list) are no-deviant
        rules: they contribute nothing to the numerator but real mass to the normalizer, which is
        what makes Π_d (1 - h(d)) come out at exactly π[r_prev, no-deviant rule].
    legacy_rule_weights : bool
        Reproduce the pre-2026-07 (incorrect) behaviour: weight by π[r_prev, r] without the
        P(S_{<d} | r) update and without renormalizing. Only for regenerating old figures.
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    r_prev = int(r_prev)

    if legacy_rule_weights:
        return sum(
            prior_dpos_given_prev_rule(d, r, valid_positions) * pi_rule[r_prev, r]
            for r in valid_positions
        )

    # Sum over EVERY rule the transition matrix knows about, not just those with deviant slots:
    # a no-deviant rule carries mass that belongs in the normalizer.
    rules = range(np.shape(pi_rule)[0])
    unnorm = {r: pi_rule[r_prev, r] * p_no_dev_before(d, r, valid_positions) for r in rules}
    Z = sum(unnorm.values())
    if Z == 0:
        # Every rule excluded: this position is unreachable given S_{<d}. 0 when no rule could
        # ever place a deviant here anyway (position 7), NaN when the answer is ambiguous.
        return _hazard_without_rule_posterior(d, valid_positions)
    return sum(prior_dpos_given_prev_rule(d, r, valid_positions) * unnorm[r] / Z for r in rules)


# Cue -> P(cue | rule) for the logfiles2clem cohort's known cue association [[0.8, 0.2], [0.2, 0.8]]
# (cue_1 favours rule 1, cue_2 favours rule 0). The design is balanced -- uniform rule AND cue
# marginals -- so P(cue | rule) == P(rule | cue) numerically, and the association matrix can be used
# directly as the cue LIKELIHOOD over rules. Keyed by the `cue` column value, then by rule id.
DEFAULT_CUE_RULE_LIK = {
    'cue_1': {0: 0.2, 1: 0.8},
    'cue_2': {0: 0.8, 1: 0.2},
}


def pior_dpos_given_prev_rule_cue_and_stds(d, pi_rule, r_prev, cue, cue_rule_lik=None,
                                           valid_positions=None, legacy_rule_weights=False):
    """Cue-aware version of `pior_dpos_given_prev_rule_and_stds`.

    Same marginalization over rules, but each rule is weighted by the cue-informed rule POSTERIOR
    rather than the bare transition prior. The cue and "no deviant so far" are TWO independent
    observations of the same latent rule, so both enter by Bayes' rule as likelihood factors on
    the transition prior:

        w(r) = P(r_t = r | r_prev, cue, S_{<d})
             = P(cue | r_t = r, r_prev) * P(r_t = r | r_prev) / P(cue | r_prev)   [cue alone]
             = P(r_t = r | r_prev) * P(cue | r_t = r) / P(cue | r_prev) (reorganizing, and dropping cue | r_prev since cue is independent of r_prev)
             = pi_rule[r_prev, r] * P(cue | rule=r) * P(S_{<d} | rule=r) / Z
        Z    = sum_{r' in rules} pi_rule[r_prev, r'] * P(cue | rule=r') * P(S_{<d} | rule=r')

        P(d | r_prev, cue, S_{<d}) = sum_r P(d | r, S_{<d}) * w(r)

    The third factor P(S_{<d} | r) (`p_no_dev_before`) was missing until 2026-07: the cue was
    applied to the rule weights but the standards observed so far were not. For this cohort the
    error is severe precisely because there is no no-deviant rule -- a deviant is CERTAIN by
    position 6, so h(6) must be 1.0, and the old code returned as little as 0.059. See
    BehavModeling/deviant_prior_maths.md.

    The DENOMINATOR Z is the marginal likelihood of the cue given the previous rule, P(cue | r_prev),
    summed over EXACTLY the rules in `valid_positions` (here {0, 1}: these subjects have no null rule).
    Restricting the sum to those rules also renormalizes away any null-rule mass that a 3x3 pi_rule
    would carry, so only the transition RATIO between rules 0 and 1 survives -- the correct behaviour
    for a no-null design.

    Parameters
    ----------
    cue : hashable
        The trial's cue value (a key of `cue_rule_lik`), e.g. 'cue_1'.
    cue_rule_lik : dict, optional
        {cue_value -> {rule -> P(cue | rule)}}. Defaults to DEFAULT_CUE_RULE_LIK (the balanced
        0.8/0.2 association). For this balanced design P(cue | rule) == P(rule | cue), so the
        association matrix may be passed directly; for an UNBALANCED design convert to a likelihood
        first (P(c|r) proportional to P(r|c) * P(c) / P(r)) to avoid double-counting a rule prior.
    valid_positions : dict, optional
        {rule -> valid deviant positions}; defaults to get_valid_positions_per_rule().
    legacy_rule_weights : bool
        Reproduce the pre-2026-07 (incorrect) behaviour: omit the P(S_{<d} | r) factor, so the
        standards observed so far never reweight the rules. Only for regenerating old figures.
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    if cue_rule_lik is None:
        cue_rule_lik = DEFAULT_CUE_RULE_LIK
    # Unlike the cue-less variant, the sum stays over `valid_positions` and renormalizes there:
    # this cohort has no no-deviant rule, so there is no extra rule whose mass belongs in Z.
    rules = list(valid_positions.keys())
    r_prev = int(r_prev)

    # Cue- and standards-informed, renormalized rule posterior:
    # transition prior x cue likelihood x P(no deviant before d | rule) / evidence.
    unnorm = {
        r: pi_rule[r_prev, r] * cue_rule_lik[cue][r]
           * (1.0 if legacy_rule_weights else p_no_dev_before(d, r, valid_positions))
        for r in rules
    }
    Z = sum(unnorm.values())
    if Z == 0:
        # Every rule excluded: this position is unreachable given S_{<d} (and the cue). 0 when no
        # rule could ever place a deviant here anyway (position 7), NaN when ambiguous.
        return _hazard_without_rule_posterior(d, valid_positions)
    w = {r: u / Z for r, u in unnorm.items()}

    return sum(prior_dpos_given_prev_rule(d, r, valid_positions) * w[r] for r in rules)


def compute_dev_likelihoods_over_dpos(trials_path, results_df, sub, sess, results_save_path=None,
                                      inplace=False, p_no_dev=0.0):
    """Get likelihood of observations at deviant positions, accounting for prior probability of deviant location given that previous observations must have been standards.

    p_no_dev : float
        Stationary probability that a trial carries NO deviant, passed through to
        `prior_dpos_given_prev_stds`. 0.0 (default) is correct for cohorts without a null rule;
        callers holding a transition matrix should pass `null_rule_marginal(pi_rule)`.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)

    # Load exp data
    trials = prepare_trials_data(trials_path)

    # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule)
    trials['prior_dpos_dev'] = trials['relative_pos'].apply(
        lambda x: prior_dpos_given_prev_stds(x, p_no_dev=p_no_dev))
    # trials[['relative_dpos', 'dpos','prior_dpos_dev']]
    
    # For the same positions, compute prior of standard as 1 - prior of deviant
    trials['prior_dpos_std'] = 1 - trials['prior_dpos_dev']
    
    # Also save priors in results_df for later analysis
    results_df['prior_dpos_std'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_std'].reset_index(drop=True)
    results_df['prior_dpos_dev'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_dev'].reset_index(drop=True)

    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    results_df['likelihood_obs_std_at_dev_over_dpos'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_std']

    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}.csv'), index=False)
    return results_df



def compute_dev_likelihoods_over_rules(trials_path, results_df, sub, sess, pi_rule, results_save_path=None, inplace=False, cue_rule_lik=None):
    """
    Compute likelihoods accounting for rule transitions.

    Args:
        trials_path: Path to load trials from (glob patterns supported)
        sub: Subject identifier
        sess: Session number (1-indexed)
        pi_rule: Rule transition matrix
        results_save_path: Path to read/save likelihood results. If None, uses script directory.
        cue_rule_lik: {cue_value -> {rule -> P(cue | rule)}}. Only used when the trials carry a
            `cue` column, in which case the cue-aware rule prior
            (pior_dpos_given_prev_rule_cue_and_stds) REPLACES the transition-only prior. Defaults
            to DEFAULT_CUE_RULE_LIK.

    Returns:
        results_df: DataFrame with likelihood_obs_dev_at_dev_over_rules column
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    
    # Load exp data
    trials = prepare_trials_data(trials_path)
    if "prev_rule" not in trials.columns or "proba_dpos" not in trials.columns:
        # Identify previous rule for each trial (not just read rule from previous row but rule associated with previous n_trial row)
        trials['prev_rule'] = (trials['trial_no'] - 1).map(dict(zip(trials['trial_no'], trials['rule']))).astype('Int64')

        # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule).
        # When the trials carry a `cue` column, the current cue is a second, independent observation
        # of the latent rule and MUST enter the rule prior (Bayes: transition-prior x cue-likelihood,
        # renormalized -- see pior_dpos_given_prev_rule_cue_and_stds). The cue-aware prior then
        # REPLACES the transition-only one directly. Cue-less cohorts keep the original behaviour.
        if 'cue' in trials.columns:
            trials['prior_dpos_rules_dev'] = trials.apply(
                lambda row: pior_dpos_given_prev_rule_cue_and_stds(
                    row['relative_pos'], pi_rule, row['prev_rule'], row['cue'], cue_rule_lik)
                if pd.notna(row['prev_rule']) else np.nan, axis=1)
            # QC counterpart: the transition-ONLY prior, recomputed here from the SAME pi_rule so
            # cue-vs-nocue is a like-for-like comparison. Any `_nocue` column already sitting in a
            # stored predictions CSV is overwritten -- those were written with whatever matrix was
            # current then (the old 3x3), which is NOT comparable to a 2x2 no-null-rule pi_rule.
            trials['prior_dpos_rules_dev_nocue'] = trials.apply(
                lambda row: pior_dpos_given_prev_rule_and_stds(row['relative_pos'], pi_rule, row['prev_rule'])
                if pd.notna(row['prev_rule']) else np.nan, axis=1)
        else:
            trials['prior_dpos_rules_dev'] = trials.apply(
                lambda row: pior_dpos_given_prev_rule_and_stds(row['relative_pos'], pi_rule, row['prev_rule'])
                if pd.notna(row['prev_rule']) else np.nan, axis=1)

        # Also get rule specific priors
        trials['prior_dpos_rule0'] = trials.apply(lambda row: prior_dpos_given_prev_rule(row['relative_pos'], r=0), axis=1)
        trials['prior_dpos_rule1'] = trials.apply(lambda row: prior_dpos_given_prev_rule(row['relative_pos'], r=1), axis=1)

        # For the same positions, compute prior of standard as 1 - prior of deviant
        trials['prior_dpos_rules_std'] = 1 - trials['prior_dpos_rules_dev']
        if 'prior_dpos_rules_dev_nocue' in trials.columns:
            trials['prior_dpos_rules_std_nocue'] = 1 - trials['prior_dpos_rules_dev_nocue']
    
    # Save current rule and previous rule for later analysis
    results_df['rule'] = trials[trials['contexts']==True].drop_duplicates()['rule'].reset_index(drop=True)
    results_df['prev_rule'] = trials[trials['contexts']==True].drop_duplicates()['prev_rule'].reset_index(drop=True)
    
    # Save rule specific priors
    results_df['prior_dpos_rule0'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rule0'].reset_index(drop=True)
    results_df['prior_dpos_rule1'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rule1'].reset_index(drop=True)
    # results_df['p_rule']=results_df.apply(lambda x: pi_rule[int(x['prev_rule']), int(x['rule'])] if pd.notna(x['prev_rule']) else np.nan, axis=1)

    # Also save priors in results_df for later analysis
    results_df['prior_dpos_rules_std'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_std'].reset_index(drop=True)
    results_df['prior_dpos_rules_dev'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_dev'].reset_index(drop=True)


    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    # results_df['likelihood_obs_dev_at_dev_over_rules'] = (1 - results_df['likelihood_obs_std_at_dev']) * trials[trials['dpos'] != 0][['trial_no', 'dpos', 'prior_dpos_dev']].drop_duplicates()['prior_dpos_dev'].reset_index(drop=True) # This was probably inaccurate
    results_df['likelihood_obs_std_at_dev_over_rules'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_rules_std']

    # Refresh the transition-only QC column from the same pi_rule (cue cohorts only), so it is
    # never left stale from an earlier matrix. See the note where the prior is computed above.
    if 'prior_dpos_rules_std_nocue' in trials.columns:
        results_df['prior_dpos_rules_std_nocue'] = trials[trials['contexts']==True].drop_duplicates()['prior_dpos_rules_std_nocue'].reset_index(drop=True)
        results_df['likelihood_obs_std_at_dev_over_rules_nocue'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_rules_std_nocue']

    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess}.csv'), index=False)
    return results_df


# Above this magnitude a `response_time_dev` value is an absolute getsecs timestamp (~1e9) rather
# than an already-relative RT (~seconds); see _deviant_rt. Well clear of both regimes.
_ABS_GETSECS_THRESHOLD = 1e6


def _first_float(x):
    """First float in x, whether x is already a number or a string like '[1781096969.46]'."""
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(x))
    return float(m.group()) if m else np.nan


def _concat_run_logs(logfiles_path, sub, sess, n_run=4):
    """Load and row-concatenate a subject/session's per-run event logfiles, in run order.

    Matches both logfile-naming generations, so the same loader serves every cohort:
      - PreProParadigm behavioural logs: `sub-{sub}-ses-{sess}*run{r}*.tsv`
      - fMRI (AuditPreProFmri) event logs: `sub-{sub}_ses-{sess}*run-{r}*.tsv`
    Original columns are preserved untouched.
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
    """Reaction-time-to-deviant as a Series, spanning both logfile generations.

    - PreProParadigm logs store it directly as `rt_getsecs_dev` (fallback `rt_getsecs`).
    - fMRI event logs store `response_time_dev` in one of TWO conventions across subjects:
        * as an ABSOLUTE getsecs timestamp of the deviant response (~1e9, same scale as the run
          trigger; e.g. sub-11) -- the RT relative to the deviant tone is that minus the run's
          trigger time (`trigger_times`, constant within a run, carried on every row); or
        * already as the RT relative to the deviant (small, ~seconds; e.g. sub-12/13) -- used
          as-is. The two are told apart per row by magnitude (`_ABS_GETSECS_THRESHOLD`), so a
          subject exported either way loads correctly. NaN (missed/no-response) is preserved.
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


def load_log_RTs(logfiles_path, sub, sess, n_run=4):
    """Per-trial reaction-time-to-deviant (one value per trial) from a subject/session's logfiles."""
    logs = _concat_run_logs(logfiles_path, sub, sess, n_run)
    logs = logs.assign(rt=_deviant_rt(logs))
    # get rid of dpos = 0 to match the likelihoods, then one row per 8-tone trial
    logs = logs[logs['dpos'] != 0]
    return logs['rt'][::8].reset_index(drop=True)


def load_logs(logfiles_path, sub, sess, filter=False, n_run=4):
    logs = _concat_run_logs(logfiles_path, sub, sess, n_run)
    logs = logs.assign(rt=_deviant_rt(logs))

    # get rid of dpos = 0 to match the likelihoods
    logs = logs[logs['dpos'] != 0]

    # Session-global per-trial id, so the per-trial de-duplication below is unambiguous even when
    # the logfile carries only a per-run trial index (fMRI logs) or none. Old logs already have a
    # unique trial_no, so this branch leaves them untouched.
    if 'trial_no' not in logs.columns:
        logs = logs.reset_index(drop=True)
        logs['trial_no'] = logs.index // 8

    if filter:
        # Get rid of NaN and null RT values
        logs = logs.dropna(subset=['rt'])
        logs = logs[logs['rt'] >= 0]

    # Only keep relevant columns
    relevant_columns = ['trial_no', 'dpos', 'rule', 'runs', 'correct', 'confidence', 'rt']  # Add other relevant columns as needed
    logs = logs[[col for col in relevant_columns if col in logs.columns]].drop_duplicates().reset_index(drop=True)

    return logs


def _correct_dpos_mask(df):
    """Boolean mask over trials: did the subject identify the DEVIANT'S POSITION correctly?

    Reads the logfile's own `correct` field, whose coding is NOT uniform across cohorts and must
    be normalised before it can be compared:

      - 04/05/06 and sub-11 code it numerically -- 1 = right key, 0 = wrong key, 3 = missed;
      - sub-12/13 code it as words -- 'correct' / 'incorrect' / 'miss';
      - the fMRI logs mix numbers with the words 'too slow' / 'too fast' / 'miss' in one column,
        so pandas reads it as OBJECT dtype and the correct trials arrive as the STRING '1', not
        the integer 1. A plain `df['correct'] == 1` therefore matches nothing for sub-11 -- it
        silently returns an empty selection rather than raising.

    Hence the two-pronged test below (numeric 1, or the word 'correct'). Verified against
    `key_pressed`: on every subject of both cohorts, this mask is exactly the set of trials whose
    key matched `dpos`.

    Caveat -- this is the paradigm's own definition, i.e. right key AND within the response
    window. The ~2% of 'too slow' (04/05/06: code 3) trials whose key DID match the deviant
    position are excluded here, as the paradigm scores them. Accept them by adding the relevant
    labels to the test below.
    """
    if 'correct' not in df.columns:
        raise ValueError(
            "correcttrials=True needs the logfiles' `correct` column, but it is absent from the "
            "aggregated frame -- check that `load_logs` kept it for this cohort.")
    col = df['correct']
    return (pd.to_numeric(col, errors='coerce') == 1) | \
           (col.astype(str).str.strip().str.lower() == 'correct')


def compare_likelihoods_with_RTs(results_df, logfiles_path, sub, sess, RT_results_path=None, n_run=4):
    """
    Compare Kalman likelihoods with reaction times from logfiles for ONE subject, ONE session
    NOTE: not up-to-date with need for considering prior for deviant location, as in function below
    Args:
        results_df: DataFrame with likelihood columns
        logfiles_path: Path to read logfiles from. Can use glob patterns with * or {}
        sub: Subject identifier
        sess: Session number (1-indexed)
        RT_results_path: Path to save RT comparison results. If None, uses script directory.
        n_run: Number of runs to load
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
    
    # get rid of dpos = 0 to match the likelihoods
    logs = logs[logs['dpos'] != 0]

    # get RT (and remove repeating lines so that we have one value per trial)
    if 'rt_getsecs_dev' in logs.columns:
        rt = logs['rt_getsecs_dev'][::8].reset_index(drop=True)
    elif 'rt_getsecs' in logs.columns:
        rt = logs['rt_getsecs'][::8].reset_index(drop=True)
    else:
        raise ValueError("No reaction time column found (expected 'rt_getsecs_dev' or 'rt_getsecs')")


    # transform to array and mask out nans, also mask negative RTs
    losad = np.array(results_df['likelihood_obs_std_at_dev'])
    rt = np.array(rt)
    mask = ~np.isnan(losad) & ~np.isnan(rt) & (rt>=0)
    # losad_clean = 1 - losad[mask] # NOTE: removed 1 - because of negative values...
    losad_clean = losad[mask]
    rt_clean = rt[mask]

    # plot association RT with likelihood_obs_std_at_dev
    plt.figure()
    plt.scatter(losad_clean, rt_clean)
    slope, intercept = np.polyfit(losad_clean, rt_clean, 1)
    plt.plot(losad_clean, slope * losad_clean + intercept, color='red', linewidth=2)  # regression line
    plt.xlabel("likelihood_obs_std_at_dev")
    plt.ylabel("RT")
    plt.title(f"Correlation coef: {ss.pearsonr(losad_clean, rt_clean)[0]:.2f}, p-value: {ss.pearsonr(losad_clean, rt_clean)[1]:.2e}")
    plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_std_sub-{sub}_ses-{sess}.png'))
    plt.close()

    # transform to array and mask out nans, plot association RT with likelihood_obs_dev_at_dev_over_rules
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


def aggregate_data(trials_path, logfiles_path, subject, session_num, likelihoods_path=None, pi_rule=None, obs_noise=None, file_prefix='kalman_predictions', sigma_r=None):
    """This function merges trials info (observations and parameters), model predictions and likelihoods,
    and behavioral data (logfiles) for a specific subject and session.
    NOTE: results df shows data for deviant positions only

    file_prefix selects the prediction files' filename stem. Defaults to 'kalman_predictions'
    (the Kalman pipeline); pass e.g. 'leaky_predictions' to read the leaky-integrator
    benchmarks written by compute_leaky_int_benchmarks.py (same schema, different stem).

    Two mutually exclusive filename suffixes, matching the two generations of KF outputs:
      sigma_r  -> `_sigma_r_{sigma_r}`, current files (sigma_r is a standard deviation).
      obs_noise -> `_obs_noise_{obs_noise}`, LEGACY files in kalman_predictions_noise_comparison*/
                   where the value recorded was a *variance*, and where the sweep passed stds into
                   a variance slot -- so those numbers are off by a square. Kept only so the old
                   folders still load; prefer sigma_r for anything regenerated.
    """
    if sigma_r is not None and obs_noise is not None:
        raise ValueError("Pass sigma_r (current files) or obs_noise (legacy files), not both.")

    suffix = _sigma_r_suffix(sigma_r) if obs_noise is None else f"_obs_noise_{obs_noise}"

    # `_ses-{n}*trials.csv` (no mandatory dash) matches both the 04/05/06 naming
    # (`sub-04_ses-1-d1-...trials.csv`) and the logfiles2clem cohort (`sub-11_ses-1_trials.csv`).
    # Accept either a per-subject subdir (`sub-XX/sub-XX_ses-...`) or a flat layout
    # (`sub-XX_ses-...`), so cohorts stored either way load through the same function.
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
        # The global prior must reserve mass for no-deviant trials whenever `pi_rule` carries a
        # null rule (04/05/06). Derived from the matrix, so a 2x2 no-null cohort (11/12/13) gets
        # exactly 0.0 and its numbers are unchanged.
        preds_liks_dev = compute_dev_likelihoods_over_dpos(
            trials_file, preds_liks_dev, subject, session_num,
            p_no_dev=null_rule_marginal(pi_rule))
        preds_liks_dev = compute_dev_likelihoods_over_rules(trials_file, preds_liks_dev, subject, session_num, pi_rule, results_save_path=None, inplace=False)
        # length: 220

        # When merging by index, keep a single copy of shared identifiers.
        # `trials` already has `trial_no`, and `preds`/`preds_liks_dev` also contain it.
        # If we keep multiple copies, later merges will attempt to suffix into names
        # like `trial_no_x` that may already exist, triggering a MergeError.
        if 'trial_no' in preds.columns:
            preds = preds.drop(columns=['trial_no'])
        if 'trial_no' in preds_liks_dev.columns:
            preds_liks_dev = preds_liks_dev.drop(columns=['trial_no'])

    logs = load_logs(logfiles_path, subject, session_num, filter=False, n_run=4)
    # mask = ~np.isnan(losad) & ~np.isnan(rt) & (rt>=0)

    # Now merge:
    df_trials_preds = pd.merge(trials, preds, left_index=True, right_index=True)
    # df_trials_preds[df_trials_preds['dpos']==df_trials_preds['relative_pos']] has the same length as logs (when filtered) --> 240
    # use df_trials_preds[df_trials_preds['contexts']==True] to get only deviant positions, should have same length as preds_liks_dev --> 220 (the difference are dpos==0)
    # Drop duplicate columns
    cols = logs.columns.difference(df_trials_preds.columns)
    df_all = pd.merge(df_trials_preds[df_trials_preds['contexts']==True].reset_index(drop=True), logs[cols], left_index=True, right_index=True)  
    df_all = pd.merge(df_all, preds_liks_dev.drop(columns=['dpos', 'rule']), left_index=True, right_index=True)
    # Futher filter: keep RTs >= 0 and not NaN, and corresponding likelihoods not NaN

    return df_all

def aggregate_data_all(trials_path, logfiles_path, with_likelihoods=False):
    # Create a dataframe merging trials info (observations and parameters) and behavioral data (logfiles) for ALL subjects and sessions
    pass

def get_session_types(trials_path):
    """
    Returns:
    session_types: 
        set listing all the unique (d, si_stat, si_r) configurations
    
    session_type_to_params answers:
        dictionary indexed by unique (d, si_stat, si_r) configurations,
        storing subjects IDs associated to them and at what session numbers
    """

    # Find unique session types (configurations of d, si_stat, si_r)
    all_trial_files = glob.glob(f"{trials_path}/sub-*/sub-*_ses-*-*trials.csv")
    session_types = set()
    session_type_to_params = {}  # Map from (d, si_stat, si_r) to dict of {subject: session_number}
    
    for trial_file in all_trial_files:
        filename = os.path.basename(trial_file)
        
        # Extract session type parameters
        param_match = re.search(r'-d(\d+)-si_stat([\d.]+)-si_r([\d.]+)', filename)
        # Extract subject and session number from filename
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
    """Significance stars for a p-value (GraphPad convention).

    ns: p > 0.05, *: p <= 0.05, **: p <= 0.01, ***: p <= 0.001, ****: p <= 0.0001.
    """
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def _fit_label(x, y):
    """Build a regplot legend label summarising the fit of likelihood x vs RT y.

    Reports Pearson rho, R^2 (= Pearson rho squared, the fraction of RT variance the
    linear fit explains), and the Pearson p-value as significance stars. All three depend
    on the transform actually applied to x and y (e.g. log RT).
    """
    r, p = ss.pearsonr(x, y)
    return rf"$\rho$: {r:.2f}, $R^2$: {r**2:.2f} {_p_stars(p)}"


def compare_likelihoods_with_RTs_global(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, take_dpos=True, take_rules=False, obs_noise=None, loglik=False, logrt=False, file_prefix='kalman_predictions', leaky_preds_path=None, leaky_file_prefix='leaky_predictions', correcttrials=False, indiv_kf_sir=False, sigma_r_files=False):
    """Compare Kalman likelihoods with reaction times from logfiles for multiple subjects and sessions per subject
    NOTE: so far visualization works best for 3 subjects and 4 sessions
    NOTE:
    - take_dpos and take_rules specify whether the likelihoods also include the deviant location prior (probability that there was a deviant at the deviant
    location selected, considering that previous observations were standards, see compute_dev_likelihoods_over_dpos and prior_dpos_given_prev_stds),
    without and with taking the rules into account, respectively.
    - take_dpos is True by default, meaning we're interested in taking prior about deviants by default, and take_rules shows likelihoods given prior location given prior rule as a comparison (optional)
    - leaky_preds_path (optional): if given, the leaky-integrator likelihood (read from that
      folder, filename stem leaky_file_prefix) is ALSO overlaid vs RT as the counterpart of
      the main likelihood line, using the same lik_col and the same loglik/logrt transforms.
      Setting it appends a '_leaky' suffix to the saved figure name.
    - correcttrials (optional): if True, narrow the trial selection from "valid RT" to "valid RT
      AND the deviant's position correctly identified" (see `_correct_dpos_mask`, which normalises
      the cohort-dependent coding of the logfile's `correct` column). Applied to both the main and
      leaky likelihoods, and appends a '_correcttrials' suffix to the saved figure name.
    - indiv_kf_sir (optional): if True, the KF likelihoods shown for each session (column)
      are those from the noise-comparison fit where all params were estimated but the
      observation noise was fixed to that session's TRUE observation noise (its si_r), instead
      of the single obs_noise passed for the whole figure. This requires preds_liks_path to
      hold per-obs-noise prediction files (keyed by si_r). Only the KF (main) likelihood is
      affected; the leaky overlay (if any) still uses obs_noise. Appends a '_indivkfsir' suffix.
    - sigma_r_files (optional): which filename generation preds_liks_path holds. True for files
      written by the current writer (`_sigma_r_{si_r}`, e.g. BehavModeling/kalman_pred); False
      for the legacy `_obs_noise_{si_r}` folders. Only matters when indiv_kf_sir is set --
      otherwise no suffix is used either way.
    """

    # Find unique session types (configurations of d, si_stat, si_r)
    session_types, session_type_to_params = get_session_types(trials_path)

    fig, axs = plt.subplots(len(subjects), len(session_types), figsize=(12, 10), sharex=True, sharey=True)
    plt.style.context('seaborn-poster')
    sessions = []  # List to store session type parameters for column headers

    for j, session_type in tqdm(enumerate(session_types), desc="Session Type"):
        d, si_stat, si_r = session_type
        sessions.append([d, si_stat, si_r])

        # If indiv_kf_sir, the KF uses this column's own true observation noise (si_r), i.e. the
        # fit where all params were estimated but obs noise was fixed to si_r; otherwise it uses
        # the single obs_noise passed for the whole figure.
        kf_obs_noise = si_r if indiv_kf_sir else obs_noise
        # Which filename generation to read: sigma_r_files -> `_sigma_r_{si_r}` (current, si_r is
        # a std); otherwise `_obs_noise_{si_r}` (legacy folders). See aggregate_data.
        kf_noise_kwargs = ({'sigma_r': kf_obs_noise} if sigma_r_files else {'obs_noise': kf_obs_noise})

        for i, sub in tqdm(enumerate(subjects), desc="Subject", leave=False):
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue

            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number

            # transform to array and mask out nans, also mask negative RTs
            df_all = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path, pi_rule, file_prefix=file_prefix, **kf_noise_kwargs) # dpos=0 are ALSO filtered out here
            lik_col = 'likelihood_obs_std_at_dev_over_dpos' if take_dpos else 'likelihood_obs_std_at_dev'
            mask = ~np.isnan(df_all[lik_col]) & ~np.isnan(df_all['rt']) & (df_all['rt'] >= 0)
            if correcttrials:
                mask &= _correct_dpos_mask(df_all)
            df_filtered = df_all[mask].reset_index(drop=True)

            # Log-transform variables before computing the correlation, so we study a linear
            # relationship against the log-transformed variable (the axis then needs no rescaling).
            # log(0) -> -inf, so drop rows that became non-finite after the transform.
            if loglik:
                df_filtered[lik_col] = np.log(df_filtered[lik_col])
                df_filtered['likelihood_obs_std_at_dev_over_rules'] = np.log(df_filtered['likelihood_obs_std_at_dev_over_rules'])
                df_filtered = df_filtered[np.isfinite(df_filtered[lik_col])].reset_index(drop=True)
            if logrt:
                df_filtered['rt'] = np.log(df_filtered['rt'])
                df_filtered = df_filtered[np.isfinite(df_filtered['rt'])].reset_index(drop=True)

            # Prepare the leaky-integrator dataframe once, so it can supply BOTH the
            # global-dpos-prior overlay (below) and, when take_rules, the rule-prior
            # overlay (further down). Same filtering and loglik/logrt transforms as the KF
            # df; RTs are identical (same logfiles), only the likelihood columns differ.
            df_leaky = None
            if leaky_preds_path is not None:
                # No noise suffix: compute_leaky_int_benchmarks writes ONE file per session, and
                # the LI takes its observation noise from the trials' sigma_r column (s_floor),
                # so its predictions are not keyed by the KF's assumed noise level.
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
                # plot association RT with likelihood_obs_std_at_dev (KF, global dpos prior)
                sns.regplot(df_filtered, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[0],
                            line_kws={'color': sns.color_palette("Paired")[1], 'linewidth': 2,
                                    'label': _fit_label(df_filtered[lik_col], df_filtered['rt'])})

                # Overlay the leaky-integrator counterpart (LI, global dpos prior): same
                # lik_col, same filtering/transforms as the KF line.
                if df_leaky is not None:
                    sns.regplot(df_leaky, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[4],
                                line_kws={'color': sns.color_palette("Paired")[5], 'linewidth': 2,
                                          'label': _fit_label(df_leaky[lik_col], df_leaky['rt'])})

            # Plot rules
            if take_rules:
                # KF, rule prior
                df_filtered_rules = df_filtered[np.isfinite(df_filtered['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                sns.regplot(df_filtered_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[2],
                            line_kws={'color': sns.color_palette("Paired")[3], 'linewidth': 2,
                                      'label': _fit_label(df_filtered_rules['likelihood_obs_std_at_dev_over_rules'], df_filtered_rules['rt'])})

                # Overlay the leaky-integrator counterpart (LI, rule prior)
                if df_leaky is not None:
                    df_leaky_rules = df_leaky[np.isfinite(df_leaky['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                    sns.regplot(df_leaky_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[6],
                                line_kws={'color': sns.color_palette("Paired")[7], 'linewidth': 2,
                                          'label': _fit_label(df_leaky_rules['likelihood_obs_std_at_dev_over_rules'], df_leaky_rules['rt'])})
            
            axs[i, j].set_xlabel("log likelihood" if loglik else "likelihood", fontsize=12)
            # Set ylabel as "RT" for all plots, but handle first column separately
            if j > 0:
                axs[i, j].set_ylabel("log RT" if logrt else "RT (s)", fontsize=12)
            leg = axs[i, j].legend(loc='upper right', fontsize=10)
            # Colour each fit label (incl. its significance stars) in its model's line
            # colour, so the stars can be traced to the right regression line.
            leg_handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
            for text, handle in zip(leg.get_texts(), leg_handles):
                text.set_color(handle.get_color())
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)

    # Label rows and columns, the label should appear only once per row/column
    for ax, session_params in zip(axs[0], sessions):
        # When indiv_kf_sir, the KF obs noise is this session's own si_r; note it in the header.
        obs_note = f"\n(KF obs noise: {session_params[2]})" if indiv_kf_sir else ""
        ax.set_title(f"d: {session_params[0]}, $\\sigma$_stat: {session_params[1]},\n$\\sigma$_r: {session_params[2]}{obs_note}", fontsize=16 if not indiv_kf_sir else 14, pad=20)

    # Add subject labels as text on the left side and RT labels for first column
    # subject_labels = [f"Subject {subj}" for subj in subjects]
    subject_labels = [f"Subject {subj}" for subj in ['1', '2', '3']] # NOTE: this is for publication

    for k, ax in enumerate(axs[:, 0]):
        ax.set_ylabel("log RT" if logrt else "RT (s)", fontsize=12, rotation=90, labelpad=6, va='center')
        # Add subject label to the left of the plot using figure coordinates
        fig.text(0.042, ax.get_position().y0 + ax.get_position().height/2, subject_labels[k], 
                fontsize=16, rotation=90, va='center', ha='right', weight='normal')

    # Build a single custom legend for line colors, crossing model (KF / leaky) with the
    # prior used (global dpos prior / rule prior). Only the lines actually plotted for the
    # current (take_dpos, take_rules, leaky) combination are listed. `plot_dpos` mirrors
    # the per-axes guard `not (take_rules and not take_dpos)` for the global-prior lines.
    plot_dpos = not (take_rules and not take_dpos)
    leaky = leaky_preds_path is not None
    custom_lines, custom_labels = [], []
    if leaky:
        # Up to four groups: KF/LI × global-prior/rule-prior.
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

    if obs_noise is not None:
        plt.suptitle(f"Obs noise: {obs_noise}", fontsize=18, y=1.02)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    save_name = f"{comparison_save_path}/RT_3sub_4sess{'_with_rules' if take_rules else ''}{'_wout_dpos' if not take_dpos else ''}{f'_obsnoise-{obs_noise}' if obs_noise is not None else ''}{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{'_leaky' if leaky_preds_path is not None else ''}{'_indivkfsir' if indiv_kf_sir else ''}{'_correcttrials' if correcttrials else ''}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")


def compare_likelihoods_with_RTs_global_indiv(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, take_dpos=True, take_rules=False, loglik=False, logrt=False, file_prefix='kalman_predictions', leaky_preds_path=None, leaky_file_prefix='leaky_predictions', sigma_r_files=False, correcttrials=False):
    """Same as compare_likelihoods_with_RTs_global, but instead of using a single observation
    noise level for the whole figure, each session (column) uses the likelihoods computed with
    the observation noise level that matches the session's TRUE observation noise (its si_r).

    This produces a single figure (rather than one per observation noise level), saved with an
    '_indivobsnoise' suffix.

    sigma_r_files: True when preds_liks_path holds current-generation `_sigma_r_{si_r}` files
    (e.g. BehavModeling/kalman_pred); False for the legacy `_obs_noise_{...}` folders.

    correcttrials: if True, narrow the trial selection from "valid RT" to "valid RT AND the
    deviant's position correctly identified" (see `_correct_dpos_mask`). Applied to the KF and
    leaky frames alike, and appends a '_correcttrials' suffix to the saved figure name.
    """

    # Find unique session types (configurations of d, si_stat, si_r)
    session_types, session_type_to_params = get_session_types(trials_path)

    fig, axs = plt.subplots(len(subjects), len(session_types), figsize=(12, 10), sharex=True, sharey=True)
    plt.style.context('seaborn-poster')
    sessions = []  # List to store session type parameters for column headers

    for j, session_type in tqdm(enumerate(session_types), desc="Session Type"):
        d, si_stat, si_r = session_type
        sessions.append([d, si_stat, si_r])

        # The session's true OBSERVATION noise is si_r (it equals the triallists' `sigma_r`
        # column), not si_stat -- si_stat is the state/process noise and is 10x larger here
        # (0.05/0.1 vs 0.005/0.01). This previously read si_stat, so each column was served the
        # predictions fixed at ~10x (in std terms) the session's real observation noise.
        obs_noise = si_r
        noise_kwargs = {'sigma_r': obs_noise} if sigma_r_files else {'obs_noise': obs_noise}

        for i, sub in tqdm(enumerate(subjects), desc="Subject", leave=False):
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue

            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number

            # transform to array and mask out nans, also mask negative RTs
            df_all = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path, pi_rule, file_prefix=file_prefix, **noise_kwargs) # dpos=0 are ALSO filtered out here
            lik_col = 'likelihood_obs_std_at_dev_over_dpos' if take_dpos else 'likelihood_obs_std_at_dev'
            mask = ~np.isnan(df_all[lik_col]) & ~np.isnan(df_all['rt']) & (df_all['rt'] >= 0)
            if correcttrials:
                mask &= _correct_dpos_mask(df_all)
            df_filtered = df_all[mask].reset_index(drop=True)

            # Log-transform variables before computing the correlation, so we study a linear
            # relationship against the log-transformed variable (the axis then needs no rescaling).
            # log(0) -> -inf, so drop rows that became non-finite after the transform.
            if loglik:
                df_filtered[lik_col] = np.log(df_filtered[lik_col])
                df_filtered['likelihood_obs_std_at_dev_over_rules'] = np.log(df_filtered['likelihood_obs_std_at_dev_over_rules'])
                df_filtered = df_filtered[np.isfinite(df_filtered[lik_col])].reset_index(drop=True)
            if logrt:
                df_filtered['rt'] = np.log(df_filtered['rt'])
                df_filtered = df_filtered[np.isfinite(df_filtered['rt'])].reset_index(drop=True)

            # Prepare the leaky-integrator dataframe once, so it can supply BOTH the
            # global-dpos-prior overlay (below) and, when take_rules, the rule-prior
            # overlay (further down). Same filtering and loglik/logrt transforms as the KF
            # df; RTs are identical (same logfiles), only the likelihood columns differ.
            df_leaky = None
            if leaky_preds_path is not None:
                # No noise suffix: compute_leaky_int_benchmarks writes ONE file per session, and
                # the LI takes its observation noise from the trials' sigma_r column (s_floor),
                # so its predictions are not keyed by the KF's assumed noise level.
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
                # plot association RT with likelihood_obs_std_at_dev (KF, global dpos prior)
                sns.regplot(df_filtered, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[0],
                            line_kws={'color': sns.color_palette("Paired")[1], 'linewidth': 2,
                                    'label': _fit_label(df_filtered[lik_col], df_filtered['rt'])})

                # Overlay the leaky-integrator counterpart (LI, global dpos prior): same
                # lik_col, same filtering/transforms as the KF line.
                if df_leaky is not None:
                    sns.regplot(df_leaky, x=lik_col, y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[4],
                                line_kws={'color': sns.color_palette("Paired")[5], 'linewidth': 2,
                                          'label': _fit_label(df_leaky[lik_col], df_leaky['rt'])})

            # Plot rules
            if take_rules:
                # KF, rule prior
                df_filtered_rules = df_filtered[np.isfinite(df_filtered['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                sns.regplot(df_filtered_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                            color=sns.color_palette("Paired")[2],
                            line_kws={'color': sns.color_palette("Paired")[3], 'linewidth': 2,
                                      'label': _fit_label(df_filtered_rules['likelihood_obs_std_at_dev_over_rules'], df_filtered_rules['rt'])})

                # Overlay the leaky-integrator counterpart (LI, rule prior)
                if df_leaky is not None:
                    df_leaky_rules = df_leaky[np.isfinite(df_leaky['likelihood_obs_std_at_dev_over_rules'])].reset_index(drop=True)
                    sns.regplot(df_leaky_rules, x='likelihood_obs_std_at_dev_over_rules', y='rt', scatter=True, ax=axs[i,j],
                                color=sns.color_palette("Paired")[6],
                                line_kws={'color': sns.color_palette("Paired")[7], 'linewidth': 2,
                                          'label': _fit_label(df_leaky_rules['likelihood_obs_std_at_dev_over_rules'], df_leaky_rules['rt'])})

            axs[i, j].set_xlabel("log likelihood" if loglik else "likelihood", fontsize=12)
            # Set ylabel as "RT" for all plots, but handle first column separately
            if j > 0:
                axs[i, j].set_ylabel("log RT" if logrt else "RT (s)", fontsize=12)
            leg = axs[i, j].legend(loc='upper right', fontsize=10)
            # Colour each fit label (incl. its significance stars) in its model's line
            # colour, so the stars can be traced to the right regression line.
            leg_handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
            for text, handle in zip(leg.get_texts(), leg_handles):
                text.set_color(handle.get_color())
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)

    # Label rows and columns, the label should appear only once per row/column.
    # Include the per-session observation noise used (si_r) in the column header.
    for ax, session_params in zip(axs[0], sessions):
        # session_params = [d, si_stat, si_r]; the KF's obs noise is si_r ([2]), not si_stat ([1]).
        ax.set_title(f"d: {session_params[0]}, $\\sigma$_stat: {session_params[1]},\n$\\sigma$_r: {session_params[2]}\n(KF obs noise: {session_params[2]})", fontsize=14, pad=20)

    # Add subject labels as text on the left side and RT labels for first column
    # subject_labels = [f"Subject {subj}" for subj in subjects]
    subject_labels = [f"Subject {subj}" for subj in ['1', '2', '3']] # NOTE: this is for publication

    for k, ax in enumerate(axs[:, 0]):
        ax.set_ylabel("log RT" if logrt else "RT (s)", fontsize=12, rotation=90, labelpad=6, va='center')
        # Add subject label to the left of the plot using figure coordinates
        fig.text(0.042, ax.get_position().y0 + ax.get_position().height/2, subject_labels[k],
                fontsize=16, rotation=90, va='center', ha='right', weight='normal')

    # Build a single custom legend for line colors, crossing model (KF / leaky) with the
    # prior used (global dpos prior / rule prior). Only the lines actually plotted for the
    # current (take_dpos, take_rules, leaky) combination are listed. `plot_dpos` mirrors
    # the per-axes guard `not (take_rules and not take_dpos)` for the global-prior lines.
    plot_dpos = not (take_rules and not take_dpos)
    leaky = leaky_preds_path is not None
    custom_lines, custom_labels = [], []
    if leaky:
        # Up to four groups: KF/LI × global-prior/rule-prior.
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
    save_name = f"{comparison_save_path}/RT_3sub_4sess{'_with_rules' if take_rules else ''}{'_wout_dpos' if not take_dpos else ''}{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{'_leaky' if leaky_preds_path is not None else ''}_indivobsnoise{'_correcttrials' if correcttrials else ''}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Saved fig as {save_name}")


# Model spec driving the per-run x per-model figure below. Each entry is one ROW of that
# figure. `src` picks which prediction dataframe the x-column is read from: 'kf' (Kalman) or
# 'li' (leaky integrator). The two prior-only rows are model-independent (the priors come from
# the trials, not from KF/LI), so they are read from the KF df by convention.
PER_RUN_MODELS = [
    {'name': 'KF alone',     'src': 'kf', 'col': 'likelihood_obs_std_at_dev'},
    {'name': 'LI alone',     'src': 'li', 'col': 'likelihood_obs_std_at_dev'},
    {'name': 'global prior', 'src': 'kf', 'col': 'prior_dpos_std'},
    {'name': 'rule prior',   'src': 'kf', 'col': 'prior_dpos_rules_std'},
    {'name': 'KF + global',  'src': 'kf', 'col': 'likelihood_obs_std_at_dev_over_dpos'},
    {'name': 'LI + global',  'src': 'li', 'col': 'likelihood_obs_std_at_dev_over_dpos'},
    {'name': 'KF + rule',    'src': 'kf', 'col': 'likelihood_obs_std_at_dev_over_rules'},
    {'name': 'LI + rule',    'src': 'li', 'col': 'likelihood_obs_std_at_dev_over_rules'},
]


def _rho_stars_annotation(ax, r, p, fontsize=9, color='black'):
    """Top-left 'rho{value}{stars}' readout with ONLY the significance stars in bold.

    A single Text can't mix font weights, so pack two TextAreas (normal value, bold stars)
    side by side with an HPacker anchored to the axes' upper-left corner.
    """
    val = TextArea(rf"$\rho${r:.2f}", textprops=dict(color=color, size=fontsize))
    stars = TextArea(_p_stars(p), textprops=dict(color=color, size=fontsize, weight='bold'))
    box = HPacker(children=[val, stars], align="baseline", pad=0, sep=1)
    ax.add_artist(AnchoredOffsetbox(loc='upper left', child=box, frameon=False,
                                    bbox_to_anchor=(0.02, 0.99), bbox_transform=ax.transAxes,
                                    borderpad=0.0, pad=0.0))


def _prep_rt_frame(df, logrt=True, correcttrials=False):
    """Trial selection shared by every per-run figure: valid RT (optionally also a correctly
    identified deviant position), followed by the optional log-RT transform.

    Factored out so the RT-vs-likelihood grids and the likelihood-DISTRIBUTION grids are
    guaranteed to describe the same trials -- a distribution figure that silently used a wider
    selection than the scatter it is meant to explain would be misleading.
    """
    keep = ~np.isnan(df['rt']) & (df['rt'] >= 0)
    if correcttrials:
        keep &= _correct_dpos_mask(df)
    df = df[keep].reset_index(drop=True)
    if logrt:
        df = df.copy()
        df['rt'] = np.log(df['rt'])
        df = df[np.isfinite(df['rt'])].reset_index(drop=True)
    return df


def _per_run_model_grid(groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                        pi_rule, comparison_save_path, save_stem, save_suffix='', n_run=4,
                        logrt=True, loglik=False, leaky_file_prefix='leaky_predictions',
                        figsize=(35, 12), correcttrials=False):
    """Shared engine for the per-run x per-model grids: PER_RUN_MODELS rows x one column per run.

    `groups` is the column layout, in order -- one entry per (subject, session), each contributing
    n_run columns. Each entry is a dict:
        sub           subject id, used for the tier-3 subject band
        sess_num      session number passed to aggregate_data
        noise_kwargs  kwargs selecting the KF prediction file (e.g. {'sigma_r': 0.02})
        band          tier-2 band text naming the session's fixed parameters

    Runs within a group are ordered by their GENERATIVE parameters (tau_std, then d), never by the
    order they were presented in, so the column order is reproducible across subjects/sessions.
    Runs are selected by `run_n` (the trials-side run id) rather than by tau_std value, because
    tau_std alone is not unique in every cohort (subject 12 has two tau=240 runs) and the logfile
    `runs` column is missing or constant for the fMRI logs.

    correcttrials: if True, narrow the trial selection from "valid RT" to "valid RT AND the
    deviant's position correctly identified" (see `_correct_dpos_mask`), and append a
    '_correcttrials' suffix to the saved figure name. Run ORDERING is unaffected -- it is read
    from the unfiltered frame -- so a run left with too few correct trials keeps its column and
    simply shows no regression line.
    """
    n_cols = len(groups) * n_run
    n_rows = len(PER_RUN_MODELS)
    # sharey=True: RT (y) is comparable everywhere, so one shared y-scale. x is NOT shared --
    # every subplot autoscales to its own likelihood range (each cell shows its own spread).
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False)
    colors = sns.color_palette('tab10', n_rows)
    # Left margin and row-label position are specified in INCHES and converted to figure
    # fractions, so a 12-column grid keeps the same physical left spacing as a 48-column one.
    left_frac = 1.12 / figsize[0]
    label_x = 0.49 / figsize[0]

    # Column -> metadata, filled as we advance the column counter. Used afterwards to place the
    # 3-tier column headers (subject band / session band / per-column run label).
    col_meta = []
    col = 0

    # Collect plotted y values to set one robust shared y-limit afterwards (sharey=True). x is
    # left to each subplot's own autoscale, so no per-row x collection is needed.
    all_y = []

    for g in tqdm(groups, desc="Subject/session"):
        sub, sess_num = g['sub'], g['sess_num']

        # Load KF and LI dataframes ONCE per (subject, session); reuse across the model rows and
        # the n_run runs.
        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, **g['noise_kwargs'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix)

        # Run ordering is taken from the UNFILTERED frame so a run whose RTs are all dropped still
        # gets its column and the grid stays aligned.
        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")
        d_varies = run_keys['d'].nunique() > 1  # label columns with d only when it varies

        # Base filter (RT only -- per-model x-column finiteness is handled per subplot) and
        # optional log-RT transform, matching the transforms used in `_global_indiv`.
        df_kf = _prep_rt_frame(df_kf, logrt=logrt, correcttrials=correcttrials)
        df_li = _prep_rt_frame(df_li, logrt=logrt, correcttrials=correcttrials)

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
                    # Default ci=95 draws the regression's bootstrap confidence band around the
                    # slope. It can flare for degenerate / high-leverage fits (the near-zero
                    # obs-only columns), but the robust shared y-limit set AFTER this loop clips
                    # it, so it no longer blows up the figure's y-scale. Dots are kept faint so
                    # the slope and its band read in front of them.
                    sns.regplot(x=x, y=y, ax=ax, color=colors[row],
                                scatter_kws={'s': 6, 'alpha': 0.25},
                                line_kws={'linewidth': 1.2})
                    # Compact readout instead of a per-axis legend (illegible at this many axes).
                    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                        r, p = ss.pearsonr(x, y)
                        _rho_stars_annotation(ax, r, p, fontsize=9)
                ax.tick_params(labelsize=7, length=2)
                # x is not shared, so every subplot draws its own ticks; cap them to a few and
                # prune the edges so neighbouring columns' labels don't collide.
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            col += 1

    assert col == n_cols, f"Filled {col} columns, expected {n_cols}"

    # Robust shared y-limit (RT) so a few outlier RTs don't compress every subplot; sharey=True
    # propagates it from axs[0, 0] to the whole grid. x is left to each subplot's own autoscale.
    if all_y:
        y_all = np.concatenate(all_y)
        ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
        if yhi > ylo:
            axs[0, 0].set_ylim(ylo, yhi)

    # Finalise geometry BEFORE reading axes positions for the band labels.
    # left : grid's left edge -- shrink to move the whole grid left (closer to the row labels).
    # top  : grid's top edge  -- lower it to make room for the 3 tiers of column headers. Cohorts
    #        that also label `d` get TWO-line column titles, which need more headroom or they
    #        collide with the tier-2 session band (fixed at 0.915).
    two_line_titles = any(m['show_d'] for m in col_meta)
    plt.subplots_adjust(left=left_frac, right=0.995, top=0.88 if two_line_titles else 0.9,
                        bottom=0.06, wspace=0.4, hspace=0.5)

    # --- Row labels (left): one model name per row (like the subject labels in `_global_indiv`).
    # `label_x` is the row-label x-position: RAISE it to sit closer to the grid (reduce the gap),
    # LOWER it to move labels further left. The gap ~= `left_frac` minus this value, minus room
    # for the y-tick labels between them.
    for row, m in enumerate(PER_RUN_MODELS):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, m['name'], fontsize=10, rotation=90,
                 va='center', ha='center', color=colors[row])
        # y-axis label on the FIRST column only (the rows already carry their model name further
        # left) -- names the RT transform once per row so the scatter's y is unambiguous.
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    def _band_centre(group):
        p0 = axs[0, group[0]['col']].get_position()
        p1 = axs[0, group[-1]['col']].get_position()
        return (p0.x0 + p1.x1) / 2

    # --- Column headers, 3 tiers.
    # Tier 1 (per column): tau_std -- the parameter distinguishing runs within a session. `d` is
    # added only for cohorts where it also varies run-to-run (it is fixed per session for 04/05/06,
    # but alternates +-1 run-to-run for the logfiles2clem cohort).
    for meta in col_meta:
        tau_txt = f"$\\tau$={meta['tau']}"
        if meta['show_d']:
            tau_txt += f"\nd={meta['d']}"
        axs[0, meta['col']].set_title(tau_txt, fontsize=9, pad=3)

    # Tier 2 (session band): centered over each group of n_run columns, naming the session's fixed
    # parameters (as in PreProParadigm/compare_RT_3subj_4sess.py -- not si_r alone).
    for start in range(0, n_cols, n_run):
        group = col_meta[start:start + n_run]
        fig.text(_band_centre(group), 0.915, group[0]['band'], fontsize=12,
                 va='bottom', ha='center')

    # Light dotted separators between consecutive session blocks, so each tier-2 band visually
    # owns its n_run columns. For a one-session-per-subject cohort these coincide with the
    # subject boundaries. Drawn in FIGURE coordinates (the gap between columns belongs to no
    # axes), spanning the grid upward. A separator that also divides two DIFFERENT subjects is
    # taken all the way up to the tier-3 "Subject ..." band (0.955); a within-subject session
    # separator stops just under the tier-2 session band (0.905).
    session_sep_top = 0.905
    subject_sep_top = 0.955
    sep_bottom = axs[-1, 0].get_position().y0
    for start in range(n_run, n_cols, n_run):
        x = (axs[0, start - 1].get_position().x1 + axs[0, start].get_position().x0) / 2
        crosses_subject = col_meta[start]['sub'] != col_meta[start - 1]['sub']
        sep_top = subject_sep_top if crosses_subject else session_sep_top
        fig.add_artist(Line2D([x, x], [sep_bottom, sep_top], transform=fig.transFigure,
                              color='gray', linestyle='--', linewidth=1, zorder=0))

    # Tier 3 (subject band): centered over each maximal run of consecutive same-subject columns,
    # so it adapts to however many sessions each subject contributes.
    start = 0
    while start < n_cols:
        end = start
        while end < n_cols and col_meta[end]['sub'] == col_meta[start]['sub']:
            end += 1
        group = col_meta[start:end]
        fig.text(_band_centre(group), 0.957, f"Subject {group[0]['sub']}", fontsize=13,
                 va='bottom', ha='center')
        start = end

    # Global x-axis label for the shared likelihood axis (suptitle-style, but at the bottom).
    fig.supxlabel("log likelihood" if loglik else "likelihood", fontsize=16, y=0.02)

    save_name = (f"{comparison_save_path}/{save_stem}_{n_cols}cols"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def _per_subject_model_grid(subject_specs, trials_path, logfiles_path, preds_liks_path,
                            leaky_preds_path, pi_rule, comparison_save_path, save_stem,
                            save_suffix='', logrt=True, loglik=False,
                            leaky_file_prefix='leaky_predictions', figsize=None,
                            correcttrials=False):
    """Per-model grid with ONE column per subject, POOLING all of that subject's runs/sessions.

    The subject-wise analogue of `_per_run_model_grid`: same PER_RUN_MODELS rows and the same
    RT-vs-likelihood scatter + regression + Pearson-rho readout, but where the per-run engine gives
    each (subject, session, run) its own column, this one collapses every trial a subject
    contributed -- across ALL its sessions and runs -- into a single column. With 3 subjects the
    grid is therefore 8 x 3, and each cell's scatter pools that subject's ~hundreds of deviant
    trials rather than the ~55 of one run.

    `subject_specs` is the column layout, in order -- one entry per subject/column, each a dict:
        sub       subject id, shown in the column band
        sessions  list of {'sess_num', 'noise_kwargs'} dicts, one per session to pool for this
                  subject. `noise_kwargs` selects that session's KF prediction file (e.g.
                  {'sigma_r': si_r}), so -- exactly as in the per-run figure -- each session's
                  likelihoods are the ones computed with its OWN true observation noise; pooling
                  them is the "individual observation noise" strategy carried up to the subject.

    correcttrials: if True, narrow the trial selection from "valid RT" to "valid RT AND the
    deviant's position correctly identified" (see `_correct_dpos_mask`), and append a
    '_correcttrials' suffix to the saved figure name.
    """
    n_cols = len(subject_specs)
    n_rows = len(PER_RUN_MODELS)
    if figsize is None:
        # Scale to the (usually few) subject columns rather than the 48-column per-run default.
        figsize = (2.6 * n_cols + 1.8, 1.55 * n_rows + 1.4)

    # sharey=True: RT (y) is comparable everywhere, so one shared y-scale. x is NOT shared -- each
    # subject's pooled likelihood range differs (different session si_r mixes), so every subplot
    # autoscales to its own spread, as in `_per_run_model_grid`.
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=False,
                            squeeze=False)
    colors = sns.color_palette('tab10', n_rows)
    # Left margin and row-label x, specified in INCHES then converted to figure fractions, so the
    # physical left spacing is stable whatever the (narrow) figure width works out to.
    left_frac = 1.12 / figsize[0]
    label_x = 0.49 / figsize[0]

    col_meta = []          # per column, for the subject band placed afterwards
    all_y = []             # every plotted y, for one robust shared y-limit (sharey=True)

    for col, spec in enumerate(tqdm(subject_specs, desc="Subject")):
        sub = spec['sub']

        # Pool every session/run of this subject. Each session is loaded with its OWN noise_kwargs
        # (its true si_r) and trial-filtered exactly as the per-run engine does, THEN concatenated
        # -- so the pooled column mixes sessions only after each has contributed the same trials it
        # would have shown as its own per-run columns.
        kf_parts, li_parts = [], []
        for s in spec['sessions']:
            df_kf = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], preds_liks_path,
                                   pi_rule, **s['noise_kwargs'])
            df_li = aggregate_data(trials_path, logfiles_path, sub, s['sess_num'], leaky_preds_path,
                                   pi_rule, file_prefix=leaky_file_prefix)
            kf_parts.append(_prep_rt_frame(df_kf, logrt=logrt, correcttrials=correcttrials))
            li_parts.append(_prep_rt_frame(df_li, logrt=logrt, correcttrials=correcttrials))
        df_kf = pd.concat(kf_parts, ignore_index=True)
        df_li = pd.concat(li_parts, ignore_index=True)
        col_meta.append({'sub': sub, 'n_sess': len(spec['sessions'])})

        for row, m in enumerate(PER_RUN_MODELS):
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
                sns.regplot(x=x, y=y, ax=ax, color=colors[row],
                            scatter_kws={'s': 5, 'alpha': 0.2},
                            line_kws={'linewidth': 1.4})
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r, p = ss.pearsonr(x, y)
                    _rho_stars_annotation(ax, r, p, fontsize=9)
                # Pooled trial count -- now a meaningful per-cell quantity (the per-run figure's
                # cells all held ~one run), so report it in the lower-right corner.
                ax.text(0.97, 0.03, f"n={len(x)}", transform=ax.transAxes, fontsize=6.5,
                        va='bottom', ha='right', color='dimgray')
            ax.tick_params(labelsize=7, length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Robust shared y-limit (RT), matching the per-run engine, so a few outlier RTs don't compress
    # every subplot; sharey=True propagates it from axs[0, 0].
    if all_y:
        y_all = np.concatenate(all_y)
        ylo, yhi = np.nanpercentile(y_all, [0.5, 99.5])
        if yhi > ylo:
            axs[0, 0].set_ylim(ylo, yhi)

    # Finalise geometry before reading axes positions for the subject band. Only ONE tier of column
    # header here (the subject), so less top headroom is needed than the per-run 3-tier grid.
    plt.subplots_adjust(left=left_frac, right=0.985, top=0.92, bottom=0.06, wspace=0.28, hspace=0.4)

    # --- Row labels (left): one model name per row, in its row colour, plus the RT-transform label
    # on the first column only -- identical convention to `_per_run_model_grid`.
    for row, m in enumerate(PER_RUN_MODELS):
        p = axs[row, 0].get_position()
        fig.text(label_x, p.y0 + p.height / 2, m['name'], fontsize=10, rotation=90,
                 va='center', ha='center', color=colors[row])
        axs[row, 0].set_ylabel("logRT" if logrt else "RT (s)", fontsize=8, labelpad=2)

    # --- Column header (single tier): the subject, centred over its column.
    for col, meta in enumerate(col_meta):
        p = axs[0, col].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.945, f"Subject {meta['sub']}", fontsize=13,
                 va='bottom', ha='center')

    fig.supxlabel("log likelihood" if loglik else "likelihood", fontsize=14, y=0.02)

    save_name = (f"{comparison_save_path}/{save_stem}_{n_cols}cols"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def compare_likelihoods_with_RTs_per_subject_models(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, logrt=True, loglik=False, sigma_r_files=True, figsize=None, correcttrials=False):
    """Per-model grid with one POOLED column per subject (subject-wise `compare_..._per_run_models`).

    Same 8 model rows as `compare_likelihoods_with_RTs_per_run_models`, but instead of one column
    per (subject, session, run) it draws one column per subject, pooling ALL of that subject's
    deviant trials across its sessions and runs. With 3 subjects that is an 8 x 3 grid.

    As in the per-run figure, each session's KF likelihoods are those computed with that session's
    TRUE observation noise (its si_r); `sigma_r_files` selects the current `_sigma_r_{si_r}`
    filename generation (True) vs the legacy `_obs_noise_{...}` folders. Pooling therefore combines,
    within a subject, trials whose KF likelihoods each used their own session's noise.

    correcttrials: if True, keep only trials with a valid RT AND the deviant's position correctly
    identified; the figure name then carries a '_correcttrials' suffix.
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
                'noise_kwargs': {'sigma_r': si_r} if sigma_r_files else {'obs_noise': si_r},
            })
        subject_specs.append({'sub': sub, 'sessions': sessions})

    return _per_subject_model_grid(
        subject_specs, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_persubject_8models', save_suffix='_indivobsnoise',
        logrt=logrt, loglik=loglik, figsize=figsize, correcttrials=correcttrials)


def compare_likelihoods_with_RTs_per_run_models(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, n_run=4, logrt=True, loglik=False, sigma_r_files=True, figsize=(35, 12), correcttrials=False):
    """Per-run x per-model RT-vs-likelihood grid (complements compare_likelihoods_with_RTs_global_indiv).

    Unlike the `_global_indiv` figure -- which is a subjects x session-types grid that POOLS all
    runs of a session and overlays models as colored lines -- this figure gives:
      - each MODEL its own row (PER_RUN_MODELS, 8 rows), and
      - each (subject, session, run) combination its own column (no pooling across runs).

    With 3 subjects x len(session_types) sessions x n_run runs columns, the default data give a
    single 8 x 48 grid. Each subplot is a trial-level RT-vs-likelihood scatter (~55 deviant
    trials for that run) with a regression line and a compact Pearson rho + significance-stars
    readout in the corner.

    Like `_global_indiv`, each session uses the KF likelihoods computed with that session's TRUE
    observation noise (its si_r); `sigma_r_files` selects the current `_sigma_r_{si_r}` filename
    generation (True, e.g. BehavModeling/kalman_pred) vs the legacy `_obs_noise_{...}` folders.

    correcttrials: if True, keep only trials with a valid RT AND the deviant's position correctly
    identified; the figure name then carries a '_correcttrials' suffix.
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
                'noise_kwargs': {'sigma_r': si_r} if sigma_r_files else {'obs_noise': si_r},
                'band': f"d={d_par}, $\\sigma_{{stat}}$={si_stat}\n$\\sigma_r$={si_r}",
            })

    return _per_run_model_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='RT_perrun_8models', save_suffix='_indivobsnoise',
        n_run=n_run, logrt=logrt, loglik=loglik, figsize=figsize, correcttrials=correcttrials)


def compare_likelihoods_with_RTs_per_run_models_single_session(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, leaky_preds_path, sess_num=1, sigma_r=0.02, n_run=4, logrt=True, loglik=False, leaky_file_prefix='leaky_predictions', save_stem='RT_perrun_8models_sub11-12-13', save_suffix='_cueprior', figsize=(13, 12), correcttrials=False):
    """Per-run x per-model grid for cohorts with ONE session per subject (logfiles2clem: 11/12/13).

    Same 8 model rows as `compare_likelihoods_with_RTs_per_run_models`, but the column layout is
    subjects x n_run runs (3 x 4 = 12 columns) since these subjects have a single session, all at
    a fixed observation noise sigma_r (0.02).

    Two cohort differences the shared engine handles:
      - `d` alternates +-1 run-to-run (it is fixed per session for 04/05/06), so it is shown in
        each column's header alongside tau_std.
      - tau_std is NOT unique within a subject (subject 12 has two tau=240 runs and no tau=160),
        so runs are selected by `run_n` and merely ORDERED by (tau_std, d). Column k therefore
        need not carry the same parameters across subjects -- read each column's own header.

    The rule-prior rows are CUE-AWARE here: `likelihood_obs_std_at_dev_over_rules` is built by
    `compute_dev_likelihoods_over_rules` from the cue-informed rule posterior for this cohort. The
    transition-only variant is available as `likelihood_obs_std_at_dev_over_rules_nocue`.

    correcttrials: if True, keep only trials with a valid RT AND the deviant's position correctly
    identified; the figure name then carries a '_correcttrials' suffix. This cohort's `correct`
    column is coded per subject (numeric for 11, words for 12/13) -- `_correct_dpos_mask` handles
    both.
    """
    groups = [{
        'sub': sub,
        'sess_num': sess_num,
        'noise_kwargs': {'sigma_r': sigma_r},
        # Only the generative parameters, as in the 04/05/06 band -- the session number is not a
        # model parameter, and these subjects have a single session anyway.
        'band': f"$\\sigma_r$={sigma_r}",
    } for sub in subjects]

    return _per_run_model_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem=save_stem, save_suffix=save_suffix, n_run=n_run,
        logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix, figsize=figsize,
        correcttrials=correcttrials)


# The two models the likelihood-distribution figure describes, in figure order: the first fills
# the TOP half of the grid, the second the BOTTOM half. Colours are taken from the same tab10
# palette as `_per_run_model_grid`'s rows, at the indices of its 'KF alone' / 'LI alone' rows, so
# a model reads the same colour in the distribution figure as in the RT scatter it explains.
LIK_DIST_MODELS = [
    {'name': 'KF alone', 'src': 'kf', 'row_color': 0},
    {'name': 'LI alone', 'src': 'li', 'row_color': 1},
]

# Below this, a likelihood is indistinguishable from 0 on the LINEAR x-axis the RT scatters use
# (their x-range runs up to ~20). The per-subplot readout reports what fraction of the run's
# trials sits under it -- i.e. how much of that scatter is piled on the left edge.
_LIK_ZERO_THRESHOLD = 1e-3


def _likelihood_dist_grid(groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                          pi_rule, comparison_save_path, save_stem, save_suffix='', n_run=4,
                          lik_col='likelihood_obs_std_at_dev', n_cols=12, n_bins=45, loglik=True,
                          leaky_file_prefix='leaky_predictions', figsize=None,
                          correcttrials=False):
    """Per-run distribution of the KF and LI likelihoods that feed the RT analyses.

    Companion to `_per_run_model_grid`: same `groups` layout, same data loading, same run ordering
    and (via `_prep_rt_frame`) the same trials -- but instead of regressing RT on the likelihood it
    shows what that likelihood's distribution actually looks like, one histogram per run.

    There is deliberately NO `logrt` parameter here: RT is not plotted. It only ever enters as
    trial SELECTION, and this function always applies the selection of the log-RT figures it
    explains (see the `_prep_rt_frame` call below).

    Layout differs from the RT grids because only TWO models are compared, so a model-per-row grid
    would be 2 rows x 48 columns. Runs are instead WRAPPED over `n_cols` columns and the models are
    stacked: the first `ceil(n_runs / n_cols)` rows hold every run's KF histogram, the same number
    of rows below hold the LI ones, in the same order. 48 runs at n_cols=12 give 4+4 rows; 12 runs
    at n_cols=6 give 2+2. Since a column no longer identifies a run, EVERY subplot carries its own
    subject / session / tau (/ d) title, and titles are coloured per subject so a subject's block
    of runs stays visible across a row break.

    Each subplot's readout gives the median likelihood, the median |z| = |y - mu| / sigma that
    produced it (only when `lik_col` is the bare likelihood -- the prior-weighted columns are that
    same density times a prior, so |z| would not describe them), and the percentage of trials
    below `_LIK_ZERO_THRESHOLD`.

    lik_col: which likelihood column to describe. Defaults to the bare observation likelihood,
        the 'KF alone' / 'LI alone' quantity. Pass 'likelihood_obs_std_at_dev_over_dpos' or
        'likelihood_obs_std_at_dev_over_rules' to describe the prior-weighted columns that
        `compare_likelihoods_with_RTs_global_indiv` plots instead.

    loglik: bin the likelihoods in log10 (default) or raw. Log is the informative default, and not
        merely cosmetic: `likelihood_obs_std_at_dev` is a Gaussian DENSITY,
        N(y_t; mu_std_t, sigma_std_t^2), evaluated at a deviant tone under the STANDARD context's
        prediction. Deviants sit many predicted SDs away, so the density falls off as exp(-z^2/2)
        and spans 100+ decades within a single run. loglik=False reproduces the RAW scale the RT
        scatters use -- almost every trial then stacks in the leftmost bin, which is the point of
        the comparison, but tells you nothing about the shape. Raw bins keep exact zeros (they are
        a legitimate value on that axis); log bins cannot, so they are counted in the readout
        instead.
    """
    bare_lik = (lik_col == 'likelihood_obs_std_at_dev')

    def _abs_z(df):
        """|z| of each deviant observation under the standard context's predictive Gaussian.

        The KF and LI prediction files both store the predictive mean/std at the deviant, so this
        is available for either model; NaN when a file predates those columns.
        """
        needed = {'observation', 'mu_pred_std_at_dev', 'sigma_pred_std_at_dev'}
        if not needed <= set(df.columns):
            return np.full(len(df), np.nan)
        return np.abs((df['observation'] - df['mu_pred_std_at_dev'])
                      / df['sigma_pred_std_at_dev']).to_numpy(dtype=float)

    # --- Data pass. Deliberately mirrors `_per_run_model_grid`'s loop: same aggregate_data calls,
    # same (tau_std, d) run ordering read from the UNFILTERED frame, same trial filter.
    records = []  # one entry per run, in figure order
    for g in tqdm(groups, desc="Subject/session"):
        sub, sess_num = g['sub'], g['sess_num']
        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, **g['noise_kwargs'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix)

        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")
        d_varies = run_keys['d'].nunique() > 1

        # logrt=True is passed for trial-SELECTION parity only: it lands the log transform on the
        # `rt` column, which this figure never reads, but it reproduces exactly the selection of
        # the log-RT figures being explained (log(0) -> -inf, so rt == 0 is dropped). No trial in
        # the current data has rt == 0, so this is parity insurance rather than a real filter.
        df_kf = _prep_rt_frame(df_kf, logrt=True, correcttrials=correcttrials)
        df_li = _prep_rt_frame(df_li, logrt=True, correcttrials=correcttrials)

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

    # --- Bins, shared by every subplot of the figure so histograms are directly comparable.
    # In log10, non-positive likelihoods (true underflow to 0) have no log: they are excluded from
    # the bins and counted in the readout instead, rather than silently dropped. On the raw scale
    # they are ordinary values and stay in.
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

    # --- Layout: runs wrapped over n_cols, models stacked (top half / bottom half).
    n_runs = len(records)
    half = int(np.ceil(n_runs / n_cols))
    n_rows = half * len(LIK_DIST_MODELS)
    if figsize is None:
        figsize = (1.55 * n_cols + 1.0, 1.45 * n_rows + 1.4)
    # sharex: one common log-likelihood axis (that IS the comparison). sharey: counts are ~60
    # trials per run everywhere, so a shared count axis keeps bar heights meaningful.
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True,
                            squeeze=False)
    colors = sns.color_palette('tab10', len(PER_RUN_MODELS))

    # Per-subject title colour: with runs wrapped over rows, a subject's block can break across a
    # row, and the colour is what keeps it readable as one block.
    subs_in_order = list(dict.fromkeys(r['sub'] for r in records))
    sub_colors = dict(zip(subs_in_order, sns.color_palette('Dark2', max(len(subs_in_order), 3))))

    for mi, m in enumerate(LIK_DIST_MODELS):
        for k, rec in enumerate(records):
            ax = axs[mi * half + k // n_cols, k % n_cols]
            x = rec[m['src']]
            n_zero = int(np.sum(x <= 0))  # underflowed to exactly 0: no log, binned nowhere
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
                # Median marker, so the bulk's position is readable without parsing the bars.
                if med > 0 or not loglik:
                    ax.axvline(_scaled(med), color=colors[m['row_color']], linestyle='-',
                               linewidth=1.1)
            # "Effectively zero on a linear axis" reference, drawn everywhere for comparability.
            # Only meaningful in log10 -- on the raw axis it sits on top of the spine.
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

        # Trailing axes of a half-filled last row carry no run: blank them out.
        for k in range(n_runs, half * n_cols):
            axs[mi * half + k // n_cols, k % n_cols].axis('off')

    # Headroom above the tallest bar so the top-left readout never sits on top of it. sharey=True
    # propagates this from axs[0, 0]; it matters most on the raw scale, where one bin holds
    # nearly every trial.
    axs[0, 0].set_ylim(top=axs[0, 0].get_ylim()[1] * 1.35)

    plt.subplots_adjust(left=0.045, right=0.995, top=0.885, bottom=0.075, wspace=0.25, hspace=0.62)

    # Axis labels only on the edges: the axes are shared, so labelling each one is noise. The
    # column being described is named once, in the suptitle -- spelling it out under every column
    # makes neighbouring labels overlap. Only the figure's LAST row is labelled: sharex already
    # hides every other row's tick labels, and leaving the top half's bottom row unlabelled is
    # what frees the gap for the second model's band.
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

    # Model band over each half, in the model's own colour, plus a separator between the halves.
    # Both are placed in the gap left by the unlabelled row above, in FIGURE coordinates -- the
    # band sits just under the separator so it clearly belongs to the half it introduces.
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

    # No '_logrt' tag: RT is not on either axis here (see the docstring).
    save_name = (f"{comparison_save_path}/{save_stem}_{n_runs}runs"
                 f"{'_loglik' if loglik else '_rawlik'}{save_suffix}"
                 f"{'_correcttrials' if correcttrials else ''}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def plot_likelihood_distributions_per_run(subjects, trials_path, preds_liks_path, logfiles_path,
                                          comparison_save_path, pi_rule, leaky_preds_path,
                                          lik_col='likelihood_obs_std_at_dev', n_run=4,
                                          n_cols=12, loglik=True, sigma_r_files=True,
                                          figsize=None, correcttrials=False):
    """Per-run KF/LI likelihood distributions for the 04/05/06 cohort (4 sessions x 4 runs each).

    Explains the likelihood axis of `compare_likelihoods_with_RTs_global_indiv` and of
    `compare_likelihoods_with_RTs_per_run_models`: same predictions, same per-session observation
    noise (each session's true si_r), same trials -- just the marginal distribution instead of the
    regression against RT.

    Groups are ordered SESSION-major (all subjects of one session type, then the next), unlike the
    subject-major RT grids. With 3 subjects x 4 runs = 12 runs per session type and n_cols=12, each
    row of the figure is then exactly one session type, read left to right as subject 04 / 05 / 06.

    The KF strategy is recorded in the filename as '_indivobsnoise': every session is served the
    predictions fixed at its OWN true si_r, so there is no single noise value to name.
    """
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
                'noise_kwargs': {'sigma_r': si_r} if sigma_r_files else {'obs_noise': si_r},
            })

    return _likelihood_dist_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem='likdist_perrun_KF_LI', save_suffix='_indivobsnoise',
        lik_col=lik_col, n_run=n_run, n_cols=n_cols, loglik=loglik, figsize=figsize,
        correcttrials=correcttrials)


def plot_likelihood_distributions_per_run_single_session(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path, sess_num=1, sigma_r=0.02, lik_col='likelihood_obs_std_at_dev',
        n_run=4, n_cols=6, loglik=True, leaky_file_prefix='leaky_predictions',
        save_stem='likdist_perrun_KF_LI_sub11-12-13', save_suffix=None, figsize=None,
        correcttrials=False):
    """Per-run KF/LI likelihood distributions for the logfiles2clem cohort (11/12/13).

    Explains the 'KF alone' / 'LI alone' rows of
    `compare_likelihoods_with_RTs_per_run_models_single_session`: one session per subject, KF fixed
    at this cohort's true si_r (0.02), 3 x 4 = 12 runs. At n_cols=6 that is 2 rows of KF histograms
    over 2 rows of LI ones. Each subject contributes 4 consecutive runs, so a subject's block
    breaks across the two rows -- read the (colour-coded) per-subplot titles, and note that column
    k does NOT carry the same parameters across subjects (tau_std is not unique in this cohort).

    save_suffix: defaults to the KF strategy actually used, `_sigma_r_{sigma_r}` -- the same tag the
    prediction files carry -- so a figure can never be mistaken for one built at a different fixed
    observation noise. Pass an explicit string to override.
    """
    if save_suffix is None:
        save_suffix = _sigma_r_suffix(sigma_r)
    groups = [{
        'sub': sub,
        'sess_num': sess_num,
        'noise_kwargs': {'sigma_r': sigma_r},
    } for sub in subjects]

    return _likelihood_dist_grid(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        comparison_save_path, save_stem=save_stem, save_suffix=save_suffix, lik_col=lik_col,
        n_run=n_run, n_cols=n_cols, loglik=loglik, leaky_file_prefix=leaky_file_prefix,
        figsize=figsize, correcttrials=correcttrials)


def _collect_perrun_models_R2(groups, trials_path, logfiles_path, preds_liks_path,
                              leaky_preds_path, pi_rule, n_run=4, logrt=True, loglik=False,
                              leaky_file_prefix='leaky_predictions', correcttrials=False):
    """Shared engine for the per-run x per-model R2 tables (numbers behind `_per_run_model_grid`).

    `groups` has the same shape as in `_per_run_model_grid` -- one entry per (subject, session)
    with `sub`, `sess_num`, `noise_kwargs` -- plus an optional `meta` dict whose keys are copied
    verbatim into every record of that group (e.g. the session-type parameters si_stat/si_r).
    `band` is ignored here (it is a figure-only field).

    The data loop is deliberately identical to the grid's: same run ordering by (tau_std, d),
    same run SELECTION by `run_n` (tau_std alone is not unique in every cohort -- subject 12 has
    two tau=240 runs), same RT filter / log transform, and the same >= 3-finite-points guard the
    grid uses before annotating rho. For a single-predictor OLS fit the coefficient of
    determination is just the squared Pearson correlation, so R2 = rho**2 -- the R2 of the very
    regression line drawn in that subplot.

    `correcttrials` mirrors the grid's flag: keep only trials with a valid RT AND the deviant's
    position correctly identified. Because it shrinks each run's trial count, more cells can fail
    the >= 3-points guard below, so the violins/heatmap built from this table may pool fewer than
    the nominal number of runs -- both figures already annotate the n they actually used.

    Returns a tidy DataFrame with one row per grid cell that had a well-defined fit:
        model, R2, rho, p, sub, sess_num, run_n, run_idx, tau, d, n_points (+ the group's meta)
    Cells failing the guard are skipped (they carry no rho/R2 in the grid either), so a model
    contributes at most one R2 per column and possibly fewer.
    """
    records = []
    for g in tqdm(groups, desc="Subject/session (R2)"):
        sub, sess_num = g['sub'], g['sess_num']

        df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path,
                               pi_rule, **g['noise_kwargs'])
        df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path,
                               pi_rule, file_prefix=leaky_file_prefix)

        # Run ordering taken from the UNFILTERED frame, exactly as the grid does.
        run_keys = df_kf.groupby('run_n')[['tau_std', 'd']].first().sort_values(['tau_std', 'd'])
        assert len(run_keys) == n_run, (
            f"sub {sub} ses {sess_num}: found {len(run_keys)} runs, expected {n_run}")

        def _prep(df):
            keep = ~np.isnan(df['rt']) & (df['rt'] >= 0)
            if correcttrials:
                keep &= _correct_dpos_mask(df)
            df = df[keep].reset_index(drop=True)
            if logrt:
                df = df.copy()
                df['rt'] = np.log(df['rt'])
                df = df[np.isfinite(df['rt'])].reset_index(drop=True)
            return df
        df_kf, df_li = _prep(df_kf), _prep(df_li)

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

                # Same guard as the grid's rho annotation.
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                    r, p = ss.pearsonr(x, y)
                    records.append({
                        'model': m['name'], 'R2': r ** 2, 'rho': r, 'p': p,
                        'sub': sub, 'sess_num': sess_num,
                        'run_n': int(run_n), 'run_idx': run_idx,
                        'tau': int(prow['tau_std']), 'd': int(prow['d']),
                        **g.get('meta', {}),
                        'n_points': int(len(x)),
                    })

    return pd.DataFrame.from_records(records)


def collect_perrun_models_R2(subjects, trials_path, preds_liks_path, logfiles_path, pi_rule,
                             leaky_preds_path, n_run=4, logrt=True, loglik=False,
                             sigma_r_files=True, correcttrials=False):
    """Per-(subject, session, run, model) R2 for the 04/05/06 cohort (4 session-types x 4 runs).

    Walks the EXACT same cells as `compare_likelihoods_with_RTs_per_run_models` -- same column
    layout, same per-session true observation noise (si_r) -- and returns the R2 = rho**2 each
    subplot annotates. See `_collect_perrun_models_R2` for the returned columns; `d` is the
    per-RUN value read from the trials frame, while the session-type parameters recovered from
    the filenames are carried as `d_par`, `si_stat` and `si_r`.
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
                'noise_kwargs': {'sigma_r': si_r} if sigma_r_files else {'obs_noise': si_r},
                'meta': {'d_par': d_par, 'si_stat': si_stat, 'si_r': si_r},
            })

    return _collect_perrun_models_R2(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        n_run=n_run, logrt=logrt, loglik=loglik, correcttrials=correcttrials)


def collect_perrun_models_R2_single_session(subjects, trials_path, preds_liks_path, logfiles_path,
                                            pi_rule, leaky_preds_path, sess_num=1, sigma_r=0.02,
                                            n_run=4, logrt=True, loglik=False,
                                            leaky_file_prefix='leaky_predictions',
                                            correcttrials=False):
    """Per-(subject, run, model) R2 for cohorts with ONE session per subject (logfiles2clem).

    The `compare_likelihoods_with_RTs_per_run_models_single_session` analogue of
    `collect_perrun_models_R2`: same 8 models, but the cells are subjects x n_run runs (3 x 4 = 12
    for subjects 11/12/13) at a single fixed observation noise sigma_r. As in that grid, runs are
    selected by `run_n` and merely ordered by (tau_std, d) -- tau_std is not unique within subject
    12 -- and the rule-prior models are the CUE-AWARE ones.
    """
    groups = [{
        'sub': sub,
        'sess_num': sess_num,
        'noise_kwargs': {'sigma_r': sigma_r},
        'meta': {'si_r': sigma_r},
    } for sub in subjects]

    return _collect_perrun_models_R2(
        groups, trials_path, logfiles_path, preds_liks_path, leaky_preds_path, pi_rule,
        n_run=n_run, logrt=logrt, loglik=loglik, leaky_file_prefix=leaky_file_prefix,
        correcttrials=correcttrials)


def _draw_perrun_models_R2_violins_on_ax(ax, df, logrt=True, loglik=False,
                                         order=None, palette=None, show_xticklabels=True):
    """Draw the one-violin-per-model R2 summary onto a single Axes.

    This is the per-ax body extracted from `_plot_perrun_models_R2_violins`: for the models in
    `order` (default PER_RUN_MODELS), a coloured violin of the `R2` column, the individual
    per-run R2 points as a strip, and a white-diamond mean marker per model, plus the axis
    labels / y-limit / spine styling. Only the Axes is touched -- no figure creation, no
    tight_layout, no savefig -- so the same routine styles the single all-subjects panel and
    each subject's panel in the stacked per-subject figure.

    `order` and `palette` are accepted so a multi-panel figure can share one model order and one
    colour mapping across panels. `show_xticklabels=False` hides (but keeps) the model tick
    labels, for every panel above the bottom one in a stacked figure.

    Returns the per-model run counts (a list aligned with `order`).
    """
    if order is None:
        order = [m['name'] for m in PER_RUN_MODELS]
    if palette is None:
        colors = sns.color_palette('tab10', len(order))
        palette = dict(zip(order, colors))

    # hue=x + legend=False is the seaborn>=0.12 way to colour one violin per category.
    # cut=0 keeps the KDE inside the observed range (R2 is bounded at 0, no negative tail).
    sns.violinplot(data=df, x='model', y='R2', order=order, hue='model', hue_order=order,
                   palette=palette, legend=False, cut=0, inner='quartile', ax=ax)
    # Overlay the individual per-run R2 points so the (few) values behind each violin are visible.
    sns.stripplot(data=df, x='model', y='R2', order=order, color='0.15', size=3.5,
                  alpha=0.5, jitter=0.15, ax=ax)

    # Mean marker per model (white diamond).
    counts = []
    for i, name in enumerate(order):
        vals = df.loc[df['model'] == name, 'R2']
        counts.append(len(vals))
        if len(vals):
            ax.scatter([i], [vals.mean()], marker='D', s=32, color='white',
                       edgecolor='black', zorder=5)

    ax.set_xlabel('')
    ylab = 'R$^2$ (per-run RT vs. ' + ('log-' if loglik else '') + 'likelihood)'
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_ylim(bottom=0)
    if show_xticklabels:
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    else:
        ax.tick_params(labelbottom=False)  # keep the ticks/locator, hide the model names
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return counts


def _plot_perrun_models_R2_violins(df, comparison_save_path, save_stem, save_suffix='',
                                   logrt=True, loglik=False):
    """Shared engine for the R2 violins: one violin per model of its per-run R2 distribution.

    `df` is a table from `_collect_perrun_models_R2` (either cohort); everything below depends
    only on its `model` and `R2` columns, so the figure is identical in construction whether the
    R2 values come from 48 runs (04/05/06) or 12 (11/12/13).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_perrun_models_R2_violins_on_ax(ax, df, logrt=logrt, loglik=loglik)
    plt.tight_layout()

    save_name = (f"{comparison_save_path}/{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def _plot_perrun_models_R2_violins_per_subject(df, comparison_save_path, save_stem, save_suffix='',
                                               logrt=True, loglik=False, subjects=None):
    """Per-subject version of `_plot_perrun_models_R2_violins`: one stacked panel per subject.

    Instead of pooling every (subject, session, run) R2 into one set of violins, this draws N
    panels stacked vertically (N = number of subjects), each showing that subject's own
    one-violin-per-model R2 distribution. Panels share the model x-axis and a common y-range,
    and the shared per-ax styling comes from `_draw_perrun_models_R2_violins_on_ax`, so every
    subject's panel is drawn exactly like the all-subjects figure.

    `subjects` fixes the panel order (top to bottom); default is the sorted `sub` values in `df`.
    """
    order = [m['name'] for m in PER_RUN_MODELS]
    colors = sns.color_palette('tab10', len(order))
    palette = dict(zip(order, colors))

    if subjects is None:
        subjects = sorted(df['sub'].unique())
    n = len(subjects)

    # sharey so the R2 scale is comparable across subjects; sharex so only the bottom panel
    # carries the (rotated) model names.
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[:, 0]

    for k, (sub, ax) in enumerate(zip(subjects, axes)):
        sub_df = df[df['sub'] == sub]
        _draw_perrun_models_R2_violins_on_ax(
            ax, sub_df, logrt=logrt, loglik=loglik, order=order, palette=palette,
            show_xticklabels=(k == n - 1))
        ax.set_title(f"subject {sub}", fontsize=11, loc='left')

    plt.tight_layout()

    save_name = (f"{comparison_save_path}/{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name


def plot_perrun_models_R2_violins(subjects, trials_path, preds_liks_path, logfiles_path,
                                  comparison_save_path, pi_rule, leaky_preds_path, n_run=4,
                                  logrt=True, loglik=False, sigma_r_files=True,
                                  correcttrials=False):
    """Summary of the per-run x per-model grid: one violin per model of its R2 distribution.

    Each violin pools the R2 values that model earns across every (subject, session, run)
    column of its row in `compare_likelihoods_with_RTs_per_run_models` -- i.e. the squared
    Pearson rho annotated in each subplot. Models sit side by side on the x-axis (in the same
    PER_RUN_MODELS order and tab10 colours as the grid's rows), so the figure reads as a
    single answer to "which model's RT-vs-likelihood coupling is strongest, and how much does
    it vary run to run".

    correcttrials: if True, the underlying per-run fits keep only trials with a valid RT AND the
    deviant's position correctly identified; the figure name then carries a '_correcttrials'
    suffix.
    """
    df = collect_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik, sigma_r_files=sigma_r_files,
        correcttrials=correcttrials,
    )
    save_name = _plot_perrun_models_R2_violins(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_violins',
        save_suffix='_indivobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)
    return save_name, df


def plot_perrun_models_R2_violins_per_subject(subjects, trials_path, preds_liks_path,
                                              logfiles_path, comparison_save_path, pi_rule,
                                              leaky_preds_path, n_run=4, logrt=True, loglik=False,
                                              sigma_r_files=True, correcttrials=False):
    """Per-subject stacked version of `plot_perrun_models_R2_violins`.

    Collects the exact same per-(subject, session, run, model) R2 table (04/05/06 cohort), then
    instead of pooling every subject into one set of violins, draws one stacked panel of violins
    per subject (each panel = that subject's own per-run R2 distribution across its sessions and
    runs). See `_plot_perrun_models_R2_violins_per_subject`.

    correcttrials: as in `plot_perrun_models_R2_violins`, restricts the underlying per-run fits to
    correctly localised deviants and appends a '_correcttrials' suffix to the figure name.
    """
    df = collect_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik, sigma_r_files=sigma_r_files,
        correcttrials=correcttrials,
    )
    save_name = _plot_perrun_models_R2_violins_per_subject(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_violins_persubject',
        save_suffix='_indivobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik, subjects=subjects)
    return save_name, df


def plot_perrun_models_R2_violins_single_session(subjects, trials_path, preds_liks_path,
                                                 logfiles_path, comparison_save_path, pi_rule,
                                                 leaky_preds_path, sess_num=1, sigma_r=0.02,
                                                 n_run=4, logrt=True, loglik=False,
                                                 leaky_file_prefix='leaky_predictions',
                                                 save_stem='RT_perrun_8models_R2_violins_sub11-12-13',
                                                 save_suffix='_cueprior', correcttrials=False):
    """R2 violins for the one-session-per-subject cohort (logfiles2clem: 11/12/13).

    Same figure as `plot_perrun_models_R2_violins`, summarising the 8 x 12 grid drawn by
    `compare_likelihoods_with_RTs_per_run_models_single_session` instead of the 8 x 48 one, so
    each violin pools 12 per-run R2 values (3 subjects x 4 runs) at the cohort's fixed sigma_r.

    correcttrials: if True, the underlying per-run fits keep only trials with a valid RT AND the
    deviant's position correctly identified; the figure name then carries a '_correcttrials'
    suffix. With only 12 runs here, watch the n annotated on the figure -- a run whose correct
    trials drop below 3 contributes no R2 at all.
    """
    df = collect_perrun_models_R2_single_session(
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
    """Signed Cohen's d (pooled-SD, 'row minus column') for two equal-length R2 samples.

    d = (mean(a) - mean(b)) / s_pooled, with s_pooled the pooled standard deviation. The two
    R2 vectors are paired run-by-run (same runs, same trials), but the pooled-SD d we report is
    the standard between-distributions effect size -- it measures how far apart the two R2
    *distributions* sit, which is what the heatmap is about. The pairing is instead carried by
    the Wilcoxon signed-rank p-value that labels each cell.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    s_pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if s_pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / s_pooled


def _model_R2_cohensd_matrices(df, order):
    """Pairwise Cohen's d / Wilcoxon matrices between models' per-run R2, paired by run identity.

    Extracted from `_plot_model_R2_cohensd_heatmap` so the single-panel and per-subject figures
    build their cells identically. Returns (D, P, N): the signed Cohen's d ("row minus column",
    antisymmetric off the diagonal), the two-sided Wilcoxon signed-rank p-value (symmetric), and
    the paired sample size actually used per cell. The diagonal and any pair with fewer than 2
    shared runs stay NaN in D and P.
    """
    # Align R2 by run identity (sub, session, run) so model columns are paired row-by-row. The key
    # is `run_n`, not tau_std: tau_std does not identify a run uniquely in every cohort (subject 12
    # has two tau=240 runs), and collapsing those two runs would silently average them away.
    mat = df.pivot_table(index=['sub', 'sess_num', 'run_n'], columns='model', values='R2')
    mat = mat.reindex(columns=order)

    n = len(order)
    D = np.full((n, n), np.nan)  # signed Cohen's d, full matrix off the diagonal
    P = np.full((n, n), np.nan)  # Wilcoxon signed-rank p-value
    N = np.zeros((n, n), dtype=int)  # paired sample size actually used per cell
    for i in range(n):
        for j in range(i + 1, n):
            pair = mat[[order[i], order[j]]].dropna()  # keep runs defined for BOTH models
            a = pair[order[i]].to_numpy()
            b = pair[order[j]].to_numpy()
            N[i, j] = N[j, i] = len(a)
            if len(a) >= 2:
                D[i, j] = _cohens_d_paired(a, b)
                try:
                    _, P[i, j] = ss.wilcoxon(a, b)  # paired: same runs/trials
                except ValueError:
                    P[i, j] = np.nan  # e.g. all paired differences are zero
                # Mirror into the lower triangle: d is antisymmetric under swapping the two
                # samples (it is "row minus column"), the two-sided p-value is symmetric. Both
                # halves are shown so a model's full row can be read left-to-right.
                D[j, i] = -D[i, j]
                P[j, i] = P[i, j]
    return D, P, N


def _draw_model_R2_cohensd_heatmap_on_ax(ax, D, P, order, vmax, show_yticklabels=True):
    """Draw one pairwise Cohen's d heatmap (cells + labels + ticks) onto a single Axes.

    The per-ax body extracted from `_plot_model_R2_cohensd_heatmap`: the diverging imshow of the
    signed d matrix `D` on a symmetric scale +/-`vmax` (red = the row model fits RT better, blue =
    the column), each finite cell overlaid with its d value and the Wilcoxon stars from `P`. No
    figure, colorbar or savefig -- the caller owns those and passes a shared `vmax` so several
    panels stay colour-comparable. `show_yticklabels=False` hides the model names on the y-axis,
    for every panel except the leftmost in a horizontal strip. Returns the AxesImage so the caller
    can hang a shared colorbar off it.
    """
    n = len(order)
    Dm = np.ma.masked_invalid(D)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white')  # blank diagonal (and any pair with too few shared runs)

    im = ax.imshow(Dm, cmap=cmap, vmin=-vmax, vmax=vmax)

    # Per-cell label: signed d value (normal weight) with the significance stars stacked below in
    # bold, matching the grid's rho+bold-stars convention.
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
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
    if show_yticklabels:
        ax.set_yticklabels(order, fontsize=9)
    else:
        ax.set_yticklabels([])
    # Thin white cell borders (imshow draws no grid of its own).
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _plot_model_R2_cohensd_heatmap(df, comparison_save_path, save_stem, save_suffix='',
                                   logrt=True, loglik=False):
    """Shared engine for the pairwise Cohen's d heatmap; `df` comes from `_collect_perrun_models_R2`.

    Depends only on the R2 values and the run identity used to pair them, so it serves either
    cohort unchanged.
    """
    order = [m['name'] for m in PER_RUN_MODELS]
    D, P, N = _model_R2_cohensd_matrices(df, order)

    # Symmetric diverging scale centred on 0 so colour direction reads as "who fits better".
    vmax = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1.0

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = _draw_model_R2_cohensd_heatmap_on_ax(ax, D, P, order, vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
                                               logrt=True, loglik=False, subjects=None):
    """Per-subject version of `_plot_model_R2_cohensd_heatmap`: one heatmap per subject, side by side.

    Instead of pooling every subject's runs into a single pairwise Cohen's d matrix, this computes
    a separate matrix per subject (its own runs, paired within subject) and lays the N heatmaps out
    HORIZONTALLY (N = number of subjects). All panels share one symmetric colour scale -- the
    global max |d| across subjects -- and a single colorbar, so a cell's colour means the same in
    every panel; only the leftmost panel carries the y-axis model names. Cells are drawn by
    `_draw_model_R2_cohensd_heatmap_on_ax`.

    `subjects` fixes the panel order (left to right); default is the sorted `sub` values in `df`.
    Returns (save_name, D_by_sub, P_by_sub) with the per-subject matrices keyed by subject.
    """
    order = [m['name'] for m in PER_RUN_MODELS]

    if subjects is None:
        subjects = sorted(df['sub'].unique())
    n_sub = len(subjects)

    # Compute every subject's matrices first so the colour scale can be shared across all panels.
    D_by_sub, P_by_sub = {}, {}
    for sub in subjects:
        D, P, _ = _model_R2_cohensd_matrices(df[df['sub'] == sub], order)
        D_by_sub[sub], P_by_sub[sub] = D, P
    finite_maxes = [np.nanmax(np.abs(D)) for D in D_by_sub.values() if np.isfinite(D).any()]
    vmax = max(finite_maxes) if finite_maxes else 1.0

    # constrained_layout handles a colorbar spanning several axes better than tight_layout.
    fig, axes = plt.subplots(1, n_sub, figsize=(8.5 * n_sub, 8), squeeze=False,
                             constrained_layout=True)
    axes = axes[0]

    im = None
    for k, (sub, ax) in enumerate(zip(subjects, axes)):
        im = _draw_model_R2_cohensd_heatmap_on_ax(
            ax, D_by_sub[sub], P_by_sub[sub], order, vmax, show_yticklabels=(k == 0))
        ax.set_title(f"subject {sub}", fontsize=12)

    # One shared colorbar spanning all panels (they all use the same +/- vmax scale).
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.046 / n_sub, pad=0.02)
    cbar.set_label(r"Cohen's $d$", fontsize=10) #  (row − column):  >0 row model fits RT better
    cbar.outline.set_visible(False)

    save_name = (f"{comparison_save_path}/{save_stem}"
                 f"{'_loglik' if loglik else ''}{'_logrt' if logrt else ''}{save_suffix}.png")
    plt.savefig(save_name, dpi=150)
    print(f"Saved fig as {save_name}")
    return save_name, D_by_sub, P_by_sub


def plot_model_R2_cohensd_heatmap(subjects, trials_path, preds_liks_path, logfiles_path,
                                  comparison_save_path, pi_rule, leaky_preds_path, n_run=4,
                                  logrt=True, loglik=False, sigma_r_files=True,
                                  correcttrials=False):
    """Heatmap of pairwise Cohen's d between models' per-run R2 distributions.

    Companion to the R2 violins. For every ordered pair of models (row i, column j, both
    triangles, diagonal blank), the box shows the signed Cohen's d = (mean R2_i - mean
    R2_j)/pooled_sd (diverging colormap,
    so red = the row model fits RT better, blue = the column model does), and overlaps it with a
    significance label from a Wilcoxon SIGNED-RANK test on the run-by-run paired R2 differences
    (`ss.wilcoxon`) -- paired because both models are fit on the SAME runs/trials. The stars
    follow the same GraphPad convention as the grid figure (`_p_stars`: ns / * / ** / *** / ****).

    The R2 samples are exactly those pooled into each violin (one per subject x session x run),
    aligned across models by run identity so the pairing is well defined.

    correcttrials: if True, the underlying per-run fits keep only trials with a valid RT AND the
    deviant's position correctly identified; the figure name then carries a '_correcttrials'
    suffix. The pairing stays valid -- both models of a pair are still fit on the same runs and
    the same (now narrower) trials.
    """
    df = collect_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik, sigma_r_files=sigma_r_files,
        correcttrials=correcttrials,
    )
    return _plot_model_R2_cohensd_heatmap(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_cohensd_heatmap',
        save_suffix='_indivobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik)


def plot_model_R2_cohensd_heatmap_per_subject(subjects, trials_path, preds_liks_path,
                                              logfiles_path, comparison_save_path, pi_rule,
                                              leaky_preds_path, n_run=4, logrt=True, loglik=False,
                                              sigma_r_files=True, correcttrials=False):
    """Per-subject side-by-side version of `plot_model_R2_cohensd_heatmap`.

    Collects the exact same per-(subject, session, run, model) R2 table (04/05/06 cohort), then
    instead of pooling every subject into one pairwise Cohen's d matrix, draws one heatmap per
    subject laid out horizontally, all on a shared colour scale with a single colorbar. See
    `_plot_model_R2_cohensd_heatmap_per_subject`.

    correcttrials: as in `plot_model_R2_cohensd_heatmap`, restricts the underlying per-run fits to
    correctly localised deviants and appends a '_correcttrials' suffix to the figure name.
    """
    df = collect_perrun_models_R2(
        subjects, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
        n_run=n_run, logrt=logrt, loglik=loglik, sigma_r_files=sigma_r_files,
        correcttrials=correcttrials,
    )
    return _plot_model_R2_cohensd_heatmap_per_subject(
        df, comparison_save_path, save_stem='RT_perrun_8models_R2_cohensd_heatmap_persubject',
        save_suffix='_indivobsnoise' + ('_correcttrials' if correcttrials else ''),
        logrt=logrt, loglik=loglik, subjects=subjects)


def plot_model_R2_cohensd_heatmap_single_session(subjects, trials_path, preds_liks_path,
                                                 logfiles_path, comparison_save_path, pi_rule,
                                                 leaky_preds_path, sess_num=1, sigma_r=0.02,
                                                 n_run=4, logrt=True, loglik=False,
                                                 leaky_file_prefix='leaky_predictions',
                                                 save_stem='RT_perrun_8models_R2_cohensd_heatmap_sub11-12-13',
                                                 save_suffix='_cueprior', correcttrials=False):
    """Pairwise Cohen's d heatmap for the one-session-per-subject cohort (logfiles2clem: 11/12/13).

    Same construction as `plot_model_R2_cohensd_heatmap`, on the 12 per-run R2 values (3 subjects
    x 4 runs) behind `plot_perrun_models_R2_violins_single_session`. With only 12 paired runs the
    Wilcoxon signed-rank test is much less powerful than in the 48-run cohort, so read the effect
    sizes (d) as the primary signal and the stars as secondary.

    correcttrials: if True, the underlying per-run fits keep only trials with a valid RT AND the
    deviant's position correctly identified; the figure name then carries a '_correcttrials'
    suffix. Already-thin power thins further -- the title's paired-run count is the n to read.
    """
    df = collect_perrun_models_R2_single_session(
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
    # Find unique session types (configurations of d, si_stat, si_r)
    session_types, session_type_to_params = get_session_types(trials_path)
    
    for sub in tqdm(subjects, desc="Subject"):
        for session_type in tqdm(session_types, desc="Session Type"):
            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number
            trials_file = glob.glob(f"{trials_path}/sub-{sub}/sub-{sub}_ses-{sess_num}-*trials.csv")[0]
            run_KF_sub_sess(sub, sess_num, trials_file, kf_preds_save_path)




if __name__ == "__main__":
    print("in model_RTs.py main")
    pass
    # BELOW: OBSOLETE
    # --> Rather run appropriate script like compute_predictions_and_likelihoods.py and compare_RT_3subj_4sess.py

    # # =====================================================================
    # # CONFIGURATION: Define paths for reading trials and logfiles, saving results
    # # =====================================================================
    
    # # # Subject and session identifiers
    # # sub = '04'
    # # sess = 0
    # for sub in tqdm(['04', '05', '06'], desc="Subject"):

    #     for sess in tqdm(range(0,4), desc="Session", leave=False):
    #         # Path to read trials from
    #         # trials_path = f'/home/clevyfidel/Documents/Workspace/Jasmin/experiment2alexclem/trial_lists/sub-{sub}/sub-{sub}_ses-{sess+1}_trials.csv'
    #         trials_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/sub-04/sub-04_ses-1-d1-si_stat0.1-si_r0.01_ses-01_trials.csv'
            
    #         # Path to save Kalman predictions and likelihoods
    #         # results_save_path = os.path.join(os.path.dirname(__file__), 'results')
    #         results_save_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/kalman_predictions_new' #os.path.join(os.path.dirname(__file__), 'results')
    #         viz_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/viz_likelihoods'
            
    #         # Path to read logfiles from (can use glob patterns)
    #         # Option 1: Use glob pattern with *, **, or {{}} notation
    #         # logfiles_path = os.path.join(os.path.dirname(__file__), '../path/to/logfiles')  # or use glob: "logfiles/sub-{sub}-ses-{sess+1}*run*[0-9].tsv"
            
    #         # Path to save RT vs likelihood comparison results
    #         RT_results_path =  f'/home/clevyfidel/Documents/Workspace/PreProParadigm/RT_comparisons'
            
    #         # Rule transition matrix
    #         pi_rule = np.array([ # actual values of pilots so far
    #             [0.8, 0.1, 0.1], 
    #             [0.1, 0.8, 0.1], 
    #             [0.5, 0.5, 0.0]  
    #         ])
            
    #         # =====================================================================
    #         # STEP 1: Prepare trial data
    #         # =====================================================================
    #         trials = prepare_data(trials_path)

    #         # =====================================================================
    #         # STEP 2: Compute Kalman likelihoods at deviant positions
    #         # =====================================================================
    #         # UNCOMMENT BELOW to compute likelihoods at deviant positions (or comment if already computed)
    #         # compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=results_save_path)

    #         # =====================================================================
    #         # STEP 3: Load and analyze likelihood results
    #         # =====================================================================
    #         results_df = pd.read_csv(
    #             os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'),
    #             index_col=False
    #         )
    #         results_df_long = pd.read_csv(
    #             os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'),
    #             index_col=False
    #         )

    #         # Study likelihood_obs_std_at_dev across deviant positions
    #         print("\n=== likelihood_obs_std_at_dev grouped by deviant position ===")
    #         print(results_df.groupby('dpos')['likelihood_obs_std_at_dev'].mean())

    #         print("\n=== 1 - likelihood_obs_std_at_dev grouped by deviant position ===")

    #         # Replace absolute 0 with a very small value to be able to plot on log scale
    #         results_df.loc[np.isclose(results_df['likelihood_obs_std_at_dev'], 0), 'likelihood_obs_std_at_dev'] = 1e-30
            
    #         # Get dev likelihood
    #         results_df['comp_likelihood_obs_std_at_dev'] = 1 - results_df['likelihood_obs_std_at_dev']

    #         # plt.figure(figsize=(10, 5))
    #         # sns.swarmplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    #         # plt.savefig(os.path.join(results_save_path, f'likelihood_std_swarmplot_sub-{sub}_ses-{sess+1}.png'))
    #         # plt.close()
            
    #         plt.figure(figsize=(10, 5))
    #         sns.violinplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    #         plt.savefig(os.path.join(viz_path, f'likelihood_std_violinplot_sub-{sub}_ses-{sess+1}.png'))
    #         plt.close()

    #         # plt.figure(figsize=(10, 5))
    #         # sns.swarmplot(results_df.dropna(subset='comp_likelihood_obs_std_at_dev', how='any'), x='comp_likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    #         # plt.savefig(os.path.join(results_save_path, f'likelihood_dev_swarmplot_sub-{sub}_ses-{sess+1}.png'))
    #         # plt.close()
            
    #         plt.figure(figsize=(10, 5))
    #         sns.violinplot(results_df.dropna(subset='comp_likelihood_obs_std_at_dev', how='any'), x='comp_likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    #         plt.xlabel("1 - likelihood_obs_std_at_dev (higher means more deviant)")
    #         plt.savefig(os.path.join(viz_path, f'likelihood_dev_violinplot_sub-{sub}_ses-{sess+1}.png'))
    #         plt.close()

    #         # =====================================================================
    #         # STEP 4: Compute likelihoods accounting for rule transitions
    #         # =====================================================================
    #         results_df = compute_dev_likelihoods_over_rules(trials_path, sub, sess, pi_rule, results_save_path=results_save_path, inplace=True)
    #         print("\n=== (1 - likelihood_obs_std_at_dev) * p(dev)(over rules) grouped by deviant position ===")
    #         print(results_df.groupby('dpos')['likelihood_obs_dev_at_dev_over_rules'].mean())

    #         # Plot likelihoods over rules
    #         # plt.figure(figsize=(10, 5))
    #         # sns.swarmplot(results_df.dropna(subset='likelihood_obs_dev_at_dev_over_rules', how='any'), x='likelihood_obs_dev_at_dev_over_rules', hue='dpos', log_scale=False)
    #         # plt.savefig(os.path.join(results_save_path, f'likelihood_obs_dev_at_dev_over_rules_swarmplot_sub-{sub}_ses-{sess+1}.png'))
    #         # plt.close()
            
    #         plt.figure(figsize=(10, 5))
    #         sns.violinplot(results_df.dropna(subset='likelihood_obs_dev_at_dev_over_rules', how='any'), x='likelihood_obs_dev_at_dev_over_rules', hue='dpos', log_scale=False)
    #         plt.savefig(os.path.join(viz_path, f'likelihood_obs_dev_at_dev_over_rules_violinplot_sub-{sub}_ses-{sess+1}.png'))
    #         plt.xlabel("1 - likelihood_obs_std_at_dev (higher means more deviant)*p(dev) (over rules)")
    #         plt.close()

    #         # =====================================================================
    #         # STEP 5: Compare likelihoods with reaction times
    #         # =====================================================================
    #         # Configure logfiles path before running
    #         # Example glob patterns:
    #         #   logfiles_path = 'logfiles/sub-{sub}-ses-{sess+1}*run{run+1}*.tsv'
    #         #   logfiles_path = os.path.join('/path/to/logs', f'sub-{sub}-ses-{sess+1}*run*.tsv')

    #         logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles' #sub-{sub}-ses-{sess+1}*run{run+1}*.tsv'

    #         compare_likelihoods_with_RTs(
    #             results_df,
    #             logfiles_path=logfiles_path,
    #             sub=sub,
    #             sess=sess,
    #             RT_results_path=RT_results_path,
    #             n_run=4
    #         )