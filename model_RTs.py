import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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



def compute_likelihood_obs_being_std_given_rule(pos, ctx, rule):
    """Compute the likelihood of the observation at position pos being generated 
    by the predicted distribution for the standard context, given the rule this position.
    
    If here we consider t spans over one trial only. hat{theta}^c_t refers to the estimated state parameters (mu and sigma) of the context c at time t.
    The probability can be obtained recusively like this:

    P(y_t | (hat{theta}^dev_{t-1}, y_{1:t-1}, rule) = P(y_t | hat{theta}^dev_{t-1}, y_{1:t-1}, rule, y_{t-1} being std) * PRODUCT over previous t(P(y_{t-1} is std | y_{1:t-2}, rule)) + P(y_t | (hat{theta}^dev_{t-1}, y_{1:t-1}, rule, y_{t-1} being dev) * PRODUCT over previous pos(P(y_{t==pos} | y_{1:t-1}, rule, y_{t-1} is dev) * P(y_{t-1} is dev | y_{1:t-2}, rule))
    The second term is always 0 because if the current position is deviant, the previous position cannot be deviant, so we only need to consider the first product, which is the product of the probabilities of all previous positions being standard, which is given by the rule and the position in the trial.
    
    Can be rewritten as:
    P(y_t | (hat{theta}^dev_{t-1}, y_{1:t-1}, rule) = P(y_t | hat{theta}^dev_{t-1}, y_{1:t-1}, rule, y_{t-1} being std) * PRODUCT over previous t( ( 1 - P(y_{t-1} |  hat{theta}^dev_{t-2}, y_{1:t-2}, rule))
    Note: the first possible position for a deviant is at position t=3 so the first "previous" is t-1=2,
    
    """    
    pass

def prepare_data(trials_path):
    # Load generated sequences
    trials = pd.read_csv(trials_path)
    trials['dpos'] = trials['dpos'].fillna(0).astype(int)
    n_trials = len(trials)/8
    trials['trial_n'] = np.repeat(np.arange(int(n_trials)), 8) # 0 to n_trials-1 for each run
    trials['relative_pos'] = np.tile(np.arange(8), int(n_trials)) # 1 to 8 for each run, to match dpos indexing
    trials['contexts'] = trials['relative_pos'] == trials['dpos'] # 0: standard, 1: deviant
    # dpos = 0 is actually not a deviant but an omission, so convert it to standard
    trials.loc[trials.dpos == 0, 'contexts'] = False
    return trials


def compute_likelihoods_at_deviants(trials_path, sub, sess):
    # Prepare data
    trials = prepare_data(trials_path)

    # Prepare storage for predictions across runs
    results = {
        'run_id': [],
        'mu_pred': [],
        'sigma_pred': [],
        'per_ctx_mu_pred_std': [],
        'per_ctx_sigma_pred_std': [],
        'per_ctx_mu_pred_dev': [],
        'per_ctx_sigma_pred_dev': []
    }

    # Prepare storage for predictions and likelihoods at deviant position across runs
    results_devpos = {
        'per_ctx_mu_pred_std_at_dev': [],
        'per_ctx_sigma_pred_std_at_dev': [],
        'per_ctx_mu_pred_dev_at_dev': [],
        'per_ctx_sigma_pred_dev_at_dev': [],
        'likelihood_obs_std_at_dev': [],
        'likelihood_obs_dev_at_dev': [],
        'dpos': []
    }

    # Separate into runs
    for run_id in trials['run_n'].unique():
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
        mu_pred, sigma_pred, kfs_fitted, per_ctx_mu_pred, per_ctx_sigma_pred = kalman_online_fit_predict_multicontext(
            observations, ctx_probabilities, n_iter=5, return_per_ctx=True
        )

        # Save predictions:
        results['run_id'].append(np.full(len(mu_pred), run_id))
        results['mu_pred'].append(mu_pred)
        results['sigma_pred'].append(sigma_pred)
        results['per_ctx_mu_pred_std'].append(per_ctx_mu_pred[:, 0])
        results['per_ctx_sigma_pred_std'].append(per_ctx_sigma_pred[:, 0])
        results['per_ctx_mu_pred_dev'].append(per_ctx_mu_pred[:, 1])
        results['per_ctx_sigma_pred_dev'].append(per_ctx_sigma_pred[:, 1])

        # Note: 
        # - First MIN_OBS_FOR_EM + 1 predictions are np.nan
        # - So per_ctx_mu_pred[:MIN_OBS_FOR_EM + 1, 0] and per_ctx_mu_pred[:8*(MIN_OBS_FOR_EM)+(dpos[0]+1), 1] are np.nan
        

        # Evaluate per_ctx_mu_pred[ctx=std] at positions of deviant
        # per_ctx_mu_pred has dimensions (n_timesteps, n_ctx), so need to slice for timestep = when contexts is 1
        per_ctx_mu_pred_std_at_dev = per_ctx_mu_pred[contexts, 1] # should have len=sum(contexts==True)= #deviants in run
        per_ctx_sigma_pred_std_at_dev = per_ctx_sigma_pred[contexts, 1]
        per_ctx_mu_pred_dev_at_dev = per_ctx_mu_pred[contexts, 0] # should have len=sum(contexts==True)= #deviants in run
        per_ctx_sigma_pred_dev_at_dev = per_ctx_sigma_pred[contexts, 0]


        # Evaluate likelihood for observations at positions of deviant if they were generated by the predicted distribution for the standard context:
        observations_at_dev = observations[contexts]
        likelihood_obs_std_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=per_ctx_mu_pred_std_at_dev,
            sigma=per_ctx_sigma_pred_std_at_dev
        )

        likelihood_obs_dev_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=per_ctx_mu_pred_dev_at_dev,
            sigma=per_ctx_sigma_pred_dev_at_dev
        )

        # Store likelihood results for this run
        results_devpos['per_ctx_mu_pred_std_at_dev'].append(per_ctx_mu_pred_std_at_dev)
        results_devpos['per_ctx_sigma_pred_std_at_dev'].append(per_ctx_sigma_pred_std_at_dev)
        results_devpos['per_ctx_mu_pred_dev_at_dev'].append(per_ctx_mu_pred_dev_at_dev)
        results_devpos['per_ctx_sigma_pred_dev_at_dev'].append(per_ctx_sigma_pred_dev_at_dev)
        results_devpos['likelihood_obs_std_at_dev'].append(likelihood_obs_std_at_dev)
        results_devpos['likelihood_obs_dev_at_dev'].append(likelihood_obs_dev_at_dev)
        results_devpos['dpos'].append(run['dpos'][contexts].values)

    # Concatenate results across runs
    results = pd.DataFrame({
        k: np.concatenate(v) for k, v in results.items()
    })
    results.to_csv(os.path.join(os.path.dirname(__file__), f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'), index=False)


    results_devpos = pd.DataFrame({
        k: np.concatenate(v) for k, v in results_devpos.items()
    })
    results_devpos.to_csv(os.path.join(os.path.dirname(__file__), f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index=False)


def get_valid_positions_per_rule():
    """
    Returns the set of valid deviant positions for each rule.
    
    Rule 1: Deviant can appear at positions {2, 3, 4}
    Rule 2: Deviant can appear at positions {4, 5, 6}
    Rule 0 (or unknown): No valid positions
    """
    return {
        1: [2, 3, 4],
        2: [4, 5, 6]
    }


def prior_dpos_given_rule(d, r):
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
    """
    valid_positions = get_valid_positions_per_rule()
    
    if r == 0 or r not in valid_positions:
        return 0
    
    if d in valid_positions[r]:
        return 1.0 / len(valid_positions[r])
    else:
        return 0


def prior_dev_given_rule_prev_stds(t, r):
    """
    Prior probability that position t is deviant, given rule r and all previous positions are standard.
    
    Mathematical formulation (derived from prior_dpos_given_rule via conditional probability):
    P(t = dev | r, S_{<t}) = P(dpos = t | r) / Σ_{s≥t} P(dpos = s | r)
    
    Equivalently:
    P(t = dev | r, S_{<t}) = { 1 / |D_r^{≥t}|  if t ∈ D_r
                             { 0               otherwise
    
    where:
      - t: current position
      - r: rule
      - S_{<t}: condition that all positions s < t are standard
      - D_r^{≥t} = {s ∈ D_r : s ≥ t}: valid positions from t onwards under rule r
      - |D_r^{≥t}|: number of remaining valid positions
    
    Interpretation: Posterior inference. Given that earlier positions were standard,
    the deviant must occur at t or later. Using the prior P(dpos | r), we normalize
    to only the remaining valid positions. This is "sampling without replacement" over time.
    """
    valid_positions = get_valid_positions_per_rule()
    
    if r == 0 or r not in valid_positions:
        return 0
    
    # Derive from prior_dpos_given_rule: normalize by positions ≥ t
    prior_dpos_given_r = prior_dpos_given_rule(t, r)
    if prior_dpos_given_r == 0:
        return 0
    
    # Sum prior for all positions s ≥ t (denominator normalizes over remaining positions)
    return prior_dpos_given_r / sum(prior_dpos_given_rule(s, r) for s in valid_positions[r] if s >= t)
    
    


def prior_dev_given_prev_stds(t):
    """
    Prior probability that position t is deviant, marginalizing over all rules.
    
    Mathematical formulation (law of total probability):
    P(t = dev | S_{<t}) = Σ_r P(t = dev | r, S_{<t}) · P(r | S_{<t})
    
    With uniform rule prior P(r) = 1/|R|:
    P(t = dev | S_{<t}) = (1/|R|) · Σ_r P(t = dev | r, S_{<t})
    """
    valid_positions = get_valid_positions_per_rule()
    rules = list(valid_positions.keys())
    n_rules = len(rules)
    
    # Marginalize: sum probabilities across rules with uniform rule prior
    prob_sum = sum(prior_dev_given_rule_prev_stds(t, r) for r in rules)
    
    return prob_sum / n_rules

def prior_dev_given_prev_rule_and_stds(t, pi_rule, r_prev):
    """
    Prior probability that position t is deviant, given previous rule, transition matrix,
    and all previous positions are standard.
    
    Mathematical formulation (sequential reasoning with rule uncertainty):
    P(t = dev | S_{<t}, r_prev, π) = Σ_r P(t = dev | r, S_{<t}) · P(r | r_prev)
    
    where:
      - t: current position
      - S_{<t}: condition that all positions s < t are standard
      - r_prev: previous rule
      - π: rule transition matrix
      - P(t = dev | r, S_{<t}): prior_dev_given_rule_prev_stds(t, r)
      - P(r | r_prev) = π[r, r_prev]: current rule probability given previous rule
    
    Interpretation: Sequential Bayesian reasoning as the trial unfolds. Marginalizes
    over uncertainty about the current rule (which depends on the transition from previous rule).
    """
    valid_positions = get_valid_positions_per_rule()
    rules = list(valid_positions.keys())
    
    # Sum over current rules, weighted by their transition probability from r_prev
    prob_sum = sum(
        prior_dev_given_rule_prev_stds(t, r) * pi_rule[r, r_prev]
        for r in rules
    )
    return prob_sum


def prior_dpos_given_prev_rule(d, pi_rule, r_prev):
    """
    Prior probability of deviant at position d, marginalizing over current rules.
    [PRIOR: Unconditional—no observation from current trial]
    
    Mathematical formulation:
    P(d | r_prev, π) = Σ_r P(d | r) · P(r | r_prev)
    
    where:
      - d: deviant position (we want to know how probable it is that d can be deviant)
      - r_prev: previous rule
      - π: rule transition matrix
      - P(d | r): prior_dpos_given_rule(d, r) — unconditional position prior
      - P(r | r_prev) = π[r, r_prev]: transition probability
    
    Interpretation: Based on the rule transition, where do we expect the deviant to be?
    This is the PRIOR—calculated before observing the current trial's positions.
    """
    valid_positions = get_valid_positions_per_rule()
    rules = list(valid_positions.keys())
    
    # Marginalize over current rules using transition matrix
    prob_sum = sum(
        prior_dpos_given_rule(d, r) * pi_rule[r, r_prev]
        for r in rules
    )
    return prob_sum


def posterior_dpos_given_prev_rule_and_stds(d, pi_rule, r_prev):
    """
    Posterior probability of deviant at position d, given previous rule, transitions, 
    AND that all positions before d are standard.
    [POSTERIOR: Conditional on observed data S_{<d}]
    
    Mathematical formulation:
    P(d | r_prev, π, S_{<d}) = Σ_r P(d | r, S_{<d}) · P(r | r_prev)
    
    where:
      - d: deviant position (what we want to know about)
      - r_prev: previous rule
      - π: rule transition matrix
      - S_{<d}: condition that all positions s < d are standard (OBSERVED DATA)
      - P(d | r, S_{<d}): prior_dev_given_rule_prev_stds(d, r) — posterior position probability
      - P(r | r_prev) = π[r, r_prev]: transition probability
    
    Interpretation: Given what we've observed (that earlier positions were standard),
    where do we think the deviant is? This is the POSTERIOR—updated belief after seeing data.
    
    """      
    valid_positions = get_valid_positions_per_rule()
    rules = list(valid_positions.keys())
    # Sum over current rules, weighted by transition probability    
    # Using posterior position priors (condition on S_{<d})
    prob_sum = sum(
        prior_dev_given_rule_prev_stds(d, r) * pi_rule[r, r_prev]
        for r in rules
    )
    return prob_sum




if __name__ == "__main__":

    sub = 'test'
    sess = 1
    trials_path = f'/home/clevyfidel/Documents/Workspace/Jasmin/experiment2alexclem/trial_lists/sub-test/sub-{sub}_ses-{sess}_trials.csv' # f'{trial_list_dir}sub-{participant}/sub-{participant}_ses-{session}_trials.csv'
    trials = prepare_data(trials_path)

    # # UNCOMMENT BELOW to compute likelihoods at deviant positions (or comment if already computed)
    # compute_likelihoods_at_deviants(trials_path, sub, sess)

    # Load results
    results_df = pd.read_csv(os.path.join(os.path.dirname(__file__), f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index_col=False)
    # results_df['dpos']=trials[trials['contexts']].dpos.reset_index(drop=True)

    # Study likelihood_obs_std_at_dev across deviant positions
    print(results_df.groupby('dpos')['likelihood_obs_std_at_dev'].mean())

    # Replace absolute 0 with a very small value to be able to plot on log scale
    results_df.loc[np.isclose(results_df['likelihood_obs_std_at_dev'], 0), 'likelihood_obs_std_at_dev'] = 1e-30

    sns.swarmplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    plt.show()
    plt.close()
    sns.violinplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    plt.show()
    plt.close()


    # Next steps
    # - Study (1-likelihoods(dpos))*SUM[over rules](P(dpos|actual rule)*P(rule|previous actual rule))) (we assume participants know the previous rule, the rule transition matrix and so the probability of the current rule given the previous rule, and the associations of position with the current rule)
    
    # Set rule transition matrix:
    pi_rule = np.array([ # Dummy example to replace
        [0.8, 0.1, 0.1], 
        [0.1, 0.8, 0.1], 
        [0.1, 0.1, 0.8]  
    ])

    # Identify previous rule for each trial (not just read rule from previous row but rule associated with previous n_trial row)
    trials['prev_rule'] = (trials['trial_n'] - 1).map(dict(zip(trials['trial_n'], trials['rule']))).astype('Int64')
    
    # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule)
    trials['prior_dpos'] = trials.apply(lambda row: posterior_dpos_given_prev_rule_and_stds(row['dpos'], pi_rule, row['prev_rule']) if pd.notna(row['prev_rule']) else np.nan, axis=1)

    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    results_df['likelihood_over_rules'] = (1 - results_df['likelihood_obs_std_at_dev']) * trials[trials['dpos'] != 0][['trial_n', 'dpos', 'prior_dpos']].drop_duplicates()['prior_dpos'].reset_index(drop=True)

    # Can produce the same violin/swarm plots as above..
    sns.swarmplot(results_df.dropna(subset='likelihood_over_rules', how='any'), x='likelihood_over_rules', hue='dpos', log_scale=False)
    plt.show()
    sns.violinplot(results_df.dropna(subset='likelihood_over_rules', how='any'), x='likelihood_over_rules', hue='dpos', log_scale=False)
    plt.show()


    # - Compare likelihood_obs_std_at_dev against reaction times
    # TODO: load RTs


    pass