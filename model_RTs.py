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
    Compute Kalman filter predictions, and predictions & likelihoods at deviant positions.
    
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

    ## Concatenate results across runs
    # Predictions over entire sequences
    results = pd.DataFrame({
        k: np.concatenate(v) for k, v in results.items()
    })
    results.to_csv(os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'), index=False)

    # Predictions and likelihoods at deviant locations
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


def prior_dpos_given_prev_rule(d, r):
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


def prior_dpos_given_prev_stds(d):
    probas = {
        0: 0,
        1: 0,
        2: 1/6,
        3: 1/5,
        4: 1/2,
        5: 1/2,
        6: 1,
        7: 0          
    }
    return probas[d]

def pior_dpos_given_prev_rule_and_stds(d, pi_rule, r_prev):
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
        prior_dpos_given_prev_rule(d, r) * pi_rule[r_prev, r]
        for r in rules
    )
    return prob_sum


def compute_dev_likelihoods_over_dpos(trials_path, results_df, sub, sess, results_save_path=None, inplace=False):
    """Get likelihood of observations at deviant positions, accounting for prior probability of deviant location given that previous observations must have been standards."""
    if results_save_path is None:
        results_save_path = os.path.dirname(__file__)
    
    # Load exp data
    trials = prepare_data(trials_path)

    # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule)
    trials['prior_dpos_dev'] = trials['relative_pos'].apply(lambda x: prior_dpos_given_prev_stds(x))
    # trials[['relative_dpos', 'dpos','prior_dpos_dev']]
    
    # For the same positions, compute prior of standard as 1 - prior of deviant
    trials['prior_dpos_std'] = 1 - trials['prior_dpos_dev']
    
    # Also save priors in results_df for later analysis
    results_df['prior_dpos_std'] = trials[(trials['dpos'] != 0) & (trials['contexts']==True)].drop_duplicates()['prior_dpos_std'].reset_index(drop=True)
    results_df['prior_dpos_dev'] = trials[(trials['dpos'] != 0) & (trials['contexts']==True)].drop_duplicates()['prior_dpos_dev'].reset_index(drop=True)

    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    results_df['likelihood_obs_std_at_dev_over_dpos'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_std']

    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index=False)
    return results_df



def compute_dev_likelihoods_over_rules(trials_path, results_df, sub, sess, pi_rule, results_save_path=None, inplace=False):
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
    
    # Load exp data
    trials= prepare_data(trials_path)
    if "prev_rule" not in trials.columns or "proba_dpos" not in trials.columns:
        # Identify previous rule for each trial (not just read rule from previous row but rule associated with previous n_trial row)
        trials['prev_rule'] = (trials['trial_n'] - 1).map(dict(zip(trials['trial_n'], trials['rule']))).astype('Int64')
        
        # For all deviant positions in trials, compute prior_dpos(dpos, pi_rule, previous_rule)
        trials['prior_dpos_rules_dev'] = trials.apply(lambda row: pior_dpos_given_prev_rule_and_stds(row['relative_pos'], pi_rule, row['prev_rule']) if pd.notna(row['prev_rule']) else np.nan, axis=1)
        
        # For the same positions, compute prior of standard as 1 - prior of deviant
        trials['prior_dpos_rules_std'] = 1 - trials['prior_dpos_rules_dev']
    
    # Also save priors in results_df for later analysis
    results_df['prior_dpos_rules_std'] = trials[(trials['dpos'] != 0) & (trials['contexts']==True)].drop_duplicates()['prior_dpos_rules_std'].reset_index(drop=True)
    results_df['prior_dpos_rules_dev'] = trials[(trials['dpos'] != 0) & (trials['contexts']==True)].drop_duplicates()['prior_dpos_rules_dev'].reset_index(drop=True)


    # Multiply prior_dpos with (1-results_df['likelihood_obs_std_at_dev'])
    # results_df['likelihood_obs_dev_at_dev_over_rules'] = (1 - results_df['likelihood_obs_std_at_dev']) * trials[trials['dpos'] != 0][['trial_n', 'dpos', 'prior_dpos_dev']].drop_duplicates()['prior_dpos_dev'].reset_index(drop=True) # This was probably inaccurate
    results_df['likelihood_obs_std_at_dev_over_rules'] = results_df['likelihood_obs_std_at_dev'] * results_df['prior_dpos_rules_std']
    
    if inplace:
        results_df.to_csv(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv'), index=False)
    return results_df


def load_log_RTs(logfiles_path, sub, sess, n_run=4):
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

    return rt


def compare_likelihoods_with_RTs(results_df, logfiles_path, sub, sess, RT_results_path=None, n_run=4):
    """
    Compare Kalman likelihoods with reaction times from logfiles for ONE subject, ONE session
    NOTE: not up-to-date with need for considering prior for deviant location, as in function below
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
    plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_std_sub-{sub}_ses-{sess+1}.png'))
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
        plt.savefig(os.path.join(RT_results_path, f'plot_rt_vs_likelihood_std_rules_sub-{sub}_ses-{sess+1}.png'))
        plt.close()
        
    
    print(f"RT comparison plots saved to: {RT_results_path}")



def compare_likelihoods_with_RTs_global(subjects, trials_path, results_save_path, logfiles_path, RT_results_path, pi_rule, take_dpos=True, take_rules=False):
    """Compare Kalman likelihoods with reaction times from logfiles for multiple subjects and sessions per subject
    NOTE: so far visualization works best for 3 subjects and 4 sessions
    NOTE:
    - take_dpos and take_rules specify whether the likelihoods also include the deviant location prior (probability that there was a deviant at the deviant 
    location selected, considering that previous observations were standards, see compute_dev_likelihoods_over_dpos and prior_dpos_given_prev_stds), 
    without and with taking the rules into account, respectively.
    - take_dpos is True by default, meaning we're interested in taking prior about deviants by default, and take_rules shows likelihoods given prior location given prior rule as a comparison (optional)
    """


    # Discover unique session types (configurations of d, si_stat, si_r)
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
    
    fig, axs = plt.subplots(len(subjects), len(session_types), figsize=(12, 10))
    plt.style.context('seaborn-poster')
    sessions = []  # List to store session type parameters for column headers

    for j, session_type in tqdm(enumerate(session_types), desc="Session Type"):
        d, si_stat, si_r = session_type
        sessions.append([d, si_stat, si_r])
        
        for i, sub in tqdm(enumerate(subjects), desc="Subject", leave=False):
            if sub not in session_type_to_params.get(session_type, {}):
                print(f"Warning: No data for subject {sub} in session type {session_type}")
                continue
            
            sess_num = session_type_to_params[session_type][sub]  # 1-indexed session number
            
            results_df = pd.read_csv(
                os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess_num}.csv'),
                index_col=False
            )
            # Compute likelihoods over rules if not already done and save in place
            trials_file = glob.glob(f"{trials_path}/sub-{sub}/sub-{sub}_ses-{sess_num}-*trials.csv")[0]
            
            # Compute likelihoods over rules
            # Note: compute_dev_likelihoods_over_rules expects 0-indexed session, so subtract 1
            if take_dpos:
                results_df = compute_dev_likelihoods_over_dpos(trials_file, results_df, sub, sess_num-1)
            if take_rules:
                results_df = compute_dev_likelihoods_over_rules(trials_file, results_df, sub, sess_num-1, pi_rule, results_save_path=results_save_path, inplace=False)

            rt = load_log_RTs(logfiles_path, sub, sess_num-1, n_run=4)

            # transform to array and mask out nans, also mask negative RTs
            if take_dpos:
                losad = np.array(results_df['likelihood_obs_std_at_dev_over_dpos'])
            else:
                losad = np.array(results_df['likelihood_obs_std_at_dev'])
            rt = np.array(rt)
            mask = ~np.isnan(losad) & ~np.isnan(rt) & (rt>=0)
            losad_clean = losad[mask]
            rt_clean = rt[mask]

            # plot association RT with likelihood_obs_std_at_dev
            axs[i,j].scatter(losad_clean, rt_clean, color=sns.color_palette("Paired")[0])
            slope, intercept = np.polyfit(losad_clean, rt_clean, 1)
            corr = ss.pearsonr(losad_clean, rt_clean)
            axs[i,j].plot(losad_clean, slope * losad_clean + intercept, 
                     color=sns.color_palette("Paired")[1], linewidth=2,
                     label=rf"$\rho$: {corr[0]:.2f}, p: {corr[1]:.1g}")  # regression line

            # Plot rules
            if take_rules:
                lor = np.array(results_df['likelihood_obs_std_at_dev_over_rules'])
                mask = ~np.isnan(lor) & ~np.isnan(rt) & (rt>=0)
                lor_clean = lor[mask]
                rt_clean = rt[mask]
                axs[i,j].scatter(lor_clean, rt_clean, color=sns.color_palette("Paired")[2])
                slope, intercept = np.polyfit(lor_clean, rt_clean, 1)
                corr_or = ss.pearsonr(lor_clean, rt_clean)
                axs[i,j].plot(lor_clean, slope * lor_clean + intercept, color=sns.color_palette("Paired")[3], linewidth=2,
                        label=rf"$\rho$: {corr_or[0]:.2f}, p: {corr_or[1]:.1g}")
            
            axs[i, j].set_xlabel("likelihood", fontsize=12)
            # Set ylabel as "RT" for all plots, but handle first column separately
            if j > 0:
                axs[i, j].set_ylabel("RT (s)", fontsize=12)
            axs[i, j].legend(loc='upper right', fontsize=10)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)

    # Label rows and columns, the label should appear only once per row/column
    for ax, session_params in zip(axs[0], sessions):
        ax.set_title(f"d: {session_params[0]}, $\\sigma$_stat: {session_params[1]},\n$\\sigma$_r: {session_params[2]}", fontsize=16, pad=20)
    
    # Add subject labels as text on the left side and RT labels for first column
    subject_labels = ['Subject 1', 'Subject 2', 'Subject 3']
    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel("RT (s)", fontsize=12, rotation=90, labelpad=6, va='center')
        # Add subject label to the left of the plot using figure coordinates
        fig.text(0.042, ax.get_position().y0 + ax.get_position().height/2, subject_labels[i], 
                fontsize=16, rotation=90, va='center', ha='right', weight='normal')

    if take_rules:
        # Create custom legend for line colors (without rules vs with rules)
        custom_lines = [Line2D([0], [0], color=sns.color_palette("Paired")[1], lw=2),
                        Line2D([0], [0], color=sns.color_palette("Paired")[3], lw=2)]
        fig.legend(custom_lines, ['without rules', 'with rules'], loc='lower center', ncol=2, 
                bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig(f"{RT_results_path}/RT_3sub_4sess{'_with_rules' if take_rules else ''}.png", dpi=600)


if __name__ == "__main__":
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