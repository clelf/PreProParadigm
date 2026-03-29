import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from scipy import stats as ss # to get pearson: ss.pearsonr(x, y)

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


def compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=None):
    """
    Compute Kalman filter likelihoods at deviant positions.
    
    Args:
        trials_path: Path to load trials from (glob patterns supported)
        sub: Subject identifier
        sess: Session number (0-indexed)
        results_save_path: Path to save results. If None, saves in script directory.
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    else:
        os.makedirs(results_save_path, exist_ok=True)
    
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
        mu_pred_std_at_dev = per_ctx_mu_pred[contexts, 0] # should have len=sum(contexts==False)= #standards in run
        sigma_pred_std_at_dev = per_ctx_sigma_pred[contexts, 0]
        mu_pred_dev_at_dev = per_ctx_mu_pred[contexts, 1] # should have len=sum(contexts==True)= #deviants in run
        sigma_pred_dev_at_dev = per_ctx_sigma_pred[contexts, 1]


        # Evaluate likelihood for observations at positions of deviant if they were generated by the predicted distribution for the standard context:
        observations_at_dev = observations[contexts]
        likelihood_obs_std_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=mu_pred_std_at_dev,
            sigma=sigma_pred_std_at_dev
        )

        likelihood_obs_dev_at_dev = likelihood_observation(
            y=observations_at_dev,
            mu=mu_pred_dev_at_dev,
            sigma=sigma_pred_dev_at_dev
        )

        # Store likelihood results for this run
        results_devpos['mu_pred_std_at_dev'].append(mu_pred_std_at_dev)
        results_devpos['sigma_pred_std_at_dev'].append(sigma_pred_std_at_dev)
        results_devpos['mu_pred_dev_at_dev'].append(mu_pred_dev_at_dev)
        results_devpos['ctx_sigma_pred_dev_at_dev'].append(sigma_pred_dev_at_dev)
        results_devpos['likelihood_obs_std_at_dev'].append(likelihood_obs_std_at_dev)
        results_devpos['likelihood_obs_dev_at_dev'].append(likelihood_obs_dev_at_dev)
        results_devpos['dpos'].append(run['dpos'][contexts].values)

    # Concatenate results across runs
    results = pd.DataFrame({
        k: np.concatenate(v) for k, v in results.items()
    })
    results.to_csv(os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'), index=False)


    results_devpos = pd.DataFrame({
        k: np.concatenate(v) for k, v in results_devpos.items()
    })
    results_devpos.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index=False)


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
    
    if r == 2 or r not in valid_positions:
        return 0
    
    if d in valid_positions[r]:
        return 1.0 / (max(valid_positions[r])-d+1) # uniform over valid positions left
    else:
        return 0




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
        prior_dpos_given_rule(t, r) * pi_rule[r, r_prev]
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
        prior_dpos_given_rule(d, r) * pi_rule[r_prev, r]
        for r in rules
    )
    return prob_sum




def compute_dev_likelihoods_over_rules(trials_path, sub, sess, pi_rule, results_save_path=None):
    """
    Compute likelihoods accounting for rule transitions.
    
    Args:
        trials_path: Path to load trials from (glob patterns supported)
        sub: Subject identifier
        sess: Session number (0-indexed)
        pi_rule: Rule transition matrix
        results_save_path: Path to read/save likelihood results. If None, uses script directory.
    
    Returns:
        results_df: DataFrame with likelihood_obs_dev_at_dev_over_rules column
    """
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    
    # Load results
    results_df = pd.read_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index_col=False)
    # results_df['dpos']=trials[trials['contexts']].dpos.reset_index(drop=True)

    # Load exp data
    trials= prepare_data(trials_path)
    if "prev_rule" not in trials.columns or "proba_dpos" not in trials.columns:
        # Identify previous rule for each trial (not just read rule from previous row but rule associated with previous n_trial row)
        trials['prev_rule'] = (trials['trial_n'] - 1).map(dict(zip(trials['trial_n'], trials['rule']))).astype('Int64')
        
        # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule)
        trials['prior_dpos'] = trials.apply(lambda row: posterior_dpos_given_prev_rule_and_stds(row['dpos'], pi_rule, row['prev_rule']) if pd.notna(row['prev_rule']) else np.nan, axis=1)

    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    results_df['likelihood_obs_dev_at_dev_over_rules'] = (1 - results_df['likelihood_obs_std_at_dev']) * trials[trials['dpos'] != 0][['trial_n', 'dpos', 'prior_dpos']].drop_duplicates()['prior_dpos'].reset_index(drop=True)
    
    return results_df


def compare_likelihoods_with_RTs(results_df, logfiles_path, sub, sess, RT_results_path=None, n_run=4):
    """
    Compare Kalman likelihoods with reaction times from logfiles.
    
    Args:
        results_df: DataFrame with likelihood columns
        logfiles_path: Path to read logfiles from. Can use glob patterns with * or {}
        sub: Subject identifier
        sess: Session number (0-indexed)
        RT_results_path: Path to save RT comparison results. If None, uses script directory.
        n_run: Number of runs to load
    """
    if RT_results_path is None:
        RT_results_path = os.path.dirname(__file__)
    else:
        os.makedirs(RT_results_path, exist_ok=True)
    
    logy = []

    for r in range(0, n_run):
        # # Support both glob patterns and standard path format
        # if '*' in logfiles_path or '{' in logfiles_path:
        #     # Use glob pattern directly
        #     logfile_path_pattern = logfiles_path.format(sub=sub, sess=sess+1, run=r+1)
        #     logfile_paths = glob.glob(logfile_path_pattern)
        # else:
        #     # Construct path with glob pattern
        #     logfile_path_pattern = os.path.join(logfiles_path, f"sub-{sub}-ses-{sess+1}*run{r+1}*.tsv")
        #     logfile_paths = glob.glob(logfile_path_pattern)
        
        # if not logfile_paths:
        #     print(f"Warning: No logfile found matching pattern: {logfile_path_pattern}")
        #     continue
            
        # print(f"Loading: {logfile_paths[0]}")
        # log = pd.read_csv(logfile_paths[0], sep="\t")

        logfile_path = glob.glob(f"{logfiles_path}/sub-{sub}-ses-{sess+1}*run{r+1}*.tsv")
        print(logfile_path)
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
    losad_clean = 1 - losad[mask]
    rt_clean = rt[mask]

    # plot association RT with likelihood_obs_std_at_dev
    plt.figure()
    plt.scatter(losad_clean, rt_clean)
    slope, intercept = np.polyfit(losad_clean, rt_clean, 1)
    plt.plot(losad_clean, slope * losad_clean + intercept, color='red', linewidth=2)  # regression line
    plt.xlabel("likelihood_obs_dev_at_dev")
    plt.ylabel("RT")
    plt.title(f"Correlation coef: {ss.pearsonr(losad_clean, rt_clean)[0]:.2f}, p-value: {ss.pearsonr(losad_clean, rt_clean)[1]:.2e}")
    plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_dev_sub-{sub}_ses-{sess+1}.png'))
    plt.close()

    # transform to array and mask out nans, plot association RT with likelihood_obs_dev_at_dev_over_rules
    if 'likelihood_obs_dev_at_dev_over_rules' in results_df.columns:
        lor = np.array(results_df['likelihood_obs_dev_at_dev_over_rules'])
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
        plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_dev_rules_sub-{sub}_ses-{sess+1}.png'))
        plt.close()
        
    
    print(f"RT comparison plots saved to: {RT_results_path}")


if __name__ == "__main__":
    # =====================================================================
    # CONFIGURATION: Define paths for reading trials and logfiles, saving results
    # =====================================================================
    
    # Subject and session identifiers
    sub = '04'
    sess = 0

    # Path to read trials from
    # trials_path = f'/home/clevyfidel/Documents/Workspace/Jasmin/experiment2alexclem/trial_lists/sub-{sub}/sub-{sub}_ses-{sess+1}_trials.csv'
    trials_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/sub-04/sub-04_ses-1-d1-si_stat0.1-si_r0.01_ses-01_trials.csv'
    
    # Path to save Kalman predictions and likelihoods
    # results_save_path = os.path.join(os.path.dirname(__file__), 'results')
    results_save_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/kalman_predictions_new' #os.path.join(os.path.dirname(__file__), 'results')

    
    # Path to read logfiles from (can use glob patterns)
    # Option 1: Use glob pattern with *, **, or {{}} notation
    # logfiles_path = os.path.join(os.path.dirname(__file__), '../path/to/logfiles')  # or use glob: "logfiles/sub-{sub}-ses-{sess+1}*run*[0-9].tsv"
    
    # Path to save RT vs likelihood comparison results
    RT_results_path =  f'/home/clevyfidel/Documents/Workspace/PreProParadigm/RT_comparisons'
    
    # Rule transition matrix
    pi_rule = np.array([ # actual values of pilots so far
        [0.8, 0.1, 0.1], 
        [0.1, 0.8, 0.1], 
        [0.5, 0.5, 0.0]  
    ])
    
    # =====================================================================
    # STEP 1: Prepare trial data
    # =====================================================================
    trials = prepare_data(trials_path)

    # =====================================================================
    # STEP 2: Compute Kalman likelihoods at deviant positions
    # =====================================================================
    # UNCOMMENT BELOW to compute likelihoods at deviant positions (or comment if already computed)
    # compute_likelihoods_at_deviants(trials_path, sub, sess, results_save_path=results_save_path)

    # =====================================================================
    # STEP 3: Load and analyze likelihood results
    # =====================================================================
    results_df = pd.read_csv(
        os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'),
        index_col=False
    )
    results_df_long = pd.read_csv(
        os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'),
        index_col=False
    )

    # Study likelihood_obs_std_at_dev across deviant positions
    print("\n=== likelihood_obs_std_at_dev grouped by deviant position ===")
    print(results_df.groupby('dpos')['likelihood_obs_std_at_dev'].mean())

    print("\n=== 1 - likelihood_obs_std_at_dev grouped by deviant position ===")
    results_df['comp_likelihood_obs_std_at_dev'] = 1 - results_df['likelihood_obs_std_at_dev']
    print(results_df.groupby('dpos')['comp_likelihood_obs_std_at_dev'].mean())

    # Replace absolute 0 with a very small value to be able to plot on log scale
    results_df.loc[np.isclose(results_df['likelihood_obs_std_at_dev'], 0), 'likelihood_obs_std_at_dev'] = 1e-30

    plt.figure(figsize=(10, 5))
    sns.swarmplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    plt.savefig(os.path.join(results_save_path, f'likelihood_std_swarmplot_sub-{sub}_ses-{sess+1}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.violinplot(results_df.dropna(subset='likelihood_obs_std_at_dev', how='any'), x='likelihood_obs_std_at_dev', hue='dpos', log_scale=True)
    plt.savefig(os.path.join(results_save_path, f'likelihood_std_violinplot_sub-{sub}_ses-{sess+1}.png'))
    plt.close()

    # =====================================================================
    # STEP 4: Compute likelihoods accounting for rule transitions
    # =====================================================================
    results_df = compute_dev_likelihoods_over_rules(trials_path, sub, sess, pi_rule, results_save_path=results_save_path)
    print("\n=== (1 - likelihood_obs_std_at_dev) * p(dev)(over rules) grouped by deviant position ===")
    print(results_df.groupby('dpos')['likelihood_obs_dev_at_dev_over_rules'].mean())

    # Plot likelihoods over rules
    plt.figure(figsize=(10, 5))
    sns.swarmplot(results_df.dropna(subset='likelihood_obs_dev_at_dev_over_rules', how='any'), x='likelihood_obs_dev_at_dev_over_rules', hue='dpos', log_scale=False)
    plt.savefig(os.path.join(results_save_path, f'likelihood_obs_dev_at_dev_over_rules_swarmplot_sub-{sub}_ses-{sess+1}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.violinplot(results_df.dropna(subset='likelihood_obs_dev_at_dev_over_rules', how='any'), x='likelihood_obs_dev_at_dev_over_rules', hue='dpos', log_scale=False)
    plt.savefig(os.path.join(results_save_path, f'likelihood_obs_dev_at_dev_over_rules_violinplot_sub-{sub}_ses-{sess+1}.png'))
    plt.close()

    # =====================================================================
    # STEP 5: Compare likelihoods with reaction times
    # =====================================================================
    # Configure logfiles path before running
    # Example glob patterns:
    #   logfiles_path = 'logfiles/sub-{sub}-ses-{sess+1}*run{run+1}*.tsv'
    #   logfiles_path = os.path.join('/path/to/logs', f'sub-{sub}-ses-{sess+1}*run*.tsv')

    logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles' #sub-{sub}-ses-{sess+1}*run{run+1}*.tsv'

    compare_likelihoods_with_RTs(
        results_df,
        logfiles_path=logfiles_path,
        sub=sub,
        sess=sess,
        RT_results_path=RT_results_path,
        n_run=4
    )