from model_RTs import prepare_trials_data, compute_likelihoods_at_deviants
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import ast
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add workspace directory to path for absolute imports
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace')

# plot_sample now lives in RNN_paradigm/RNN/train/pipeline_core_v2.py.
PIPELINE_CORE_V2 = '/home/clevyfidel/Documents/Workspace/RNN_paradigm/RNN/train/pipeline_core_v2.py'


def _load_plot_sample(path=PIPELINE_CORE_V2):
    """Load `plot_sample` from pipeline_core_v2.py without importing the package.

    The `RNN_paradigm.RNN` package currently fails to import (its __init__ and several
    module-level imports are stale), but `plot_sample` itself only depends on numpy and
    matplotlib. We extract just that function's source via AST and exec it in a clean
    namespace, so this visualization stays decoupled from the broken package. If the
    package import is fixed later, this can revert to a plain
    `from RNN_paradigm.RNN.train.pipeline_core_v2 import plot_sample`.
    """
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'plot_sample':
            func_src = ast.get_source_segment(src, node)
            ns = {'np': np, 'plt': plt, 'mpl': mpl, 'mpatches': mpatches}
            exec(func_src, ns)
            return ns['plot_sample']
    raise RuntimeError(f"plot_sample not found in {path}")


plot_sample = _load_plot_sample()

if __name__ == "__main__":

    subjects = ['04','05','06']
    
    # Optional: Specify observation noise levels to test.
    # If None, observation noise will be estimated by EM (default behavior).
    # If a list is provided, visualizations will be created for each noise level.
    observation_noise_levels = None
    observation_noise_levels = [0.005, 0.01, 0.05, 0.1]
    
    # If no noise levels specified, run with default (EM-estimated noise)
    if observation_noise_levels is None:
        observation_noise_levels = [None]  # None means EM estimates it
    
    # Load base folder paths
    results_load_folder = os.path.join(os.path.dirname(__file__), 'kalman_predictions_noise_comparison')
    results_save_folder = os.path.join(os.path.dirname(__file__), 'viz_KF_experimental_runs_noise_comparison')
    
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder, exist_ok=True)

    # Context colors, matching how pipeline_core_v2 builds them for plot_sample
    # (tab10 colormap indexed by context id). n_ctx=2: 0=standard, 1=deviant.
    n_ctx = 2
    ctx_cmap = plt.get_cmap('tab10')
    ctx_colors = {c: ctx_cmap(c) for c in range(n_ctx)}

    for sub in subjects:
        # Get number of sessions for this subject
        triallists_dir = os.path.join(os.path.dirname(__file__), f"triallists/sub-{sub}/")
        if not os.path.exists(triallists_dir):
            print(f"Triallists directory not found for sub-{sub}")
            continue
        
        n_sessions = len([f for f in os.listdir(triallists_dir) if f.startswith(f"sub-{sub}_ses-")])
        
        for sess in range(n_sessions):
            # Load trials data
            trials_path = glob.glob(os.path.join(os.path.dirname(__file__), f"triallists/sub-{sub}/sub-{sub}_ses-{sess+1}*.csv"))
            
            if not trials_path:
                print(f"No trials found for sub-{sub}, ses-{sess+1}")
                continue
            
            trials = prepare_trials_data(trials_path[0])
            
            # Process for each observation noise level
            for obs_noise in observation_noise_levels:
                # Generate filename suffix based on observation noise
                noise_suffix = "" if obs_noise is None else f"_obs_noise_{obs_noise}"
                
                # Check if results file exists
                results_file = os.path.join(results_load_folder, f'kalman_predictions_sub-{sub}_ses-{sess+1}{noise_suffix}.csv')
                
                if not os.path.exists(results_file):
                    print(f"Results file not found: {results_file}")
                    continue
                
                results_df = pd.read_csv(results_file, index_col=False)
                
                # Create visualizations for each run
                for run_id in trials['run_n'].unique():
                    
                    save_path = os.path.join(results_save_folder, f'sub-{sub}_ses-{sess+1}{noise_suffix}_run_{run_id}')
                    
                    obs = trials.loc[trials['run_n']==run_id,'observation'].reset_index(drop=True)
                    ctx = trials.loc[trials['run_n']==run_id,'contexts'].reset_index(drop=True)
                    mu_pred_std = results_df.loc[results_df['run_id']==run_id, 'mu_pred_std'].reset_index(drop=True)
                    sigma_pred_std = results_df.loc[results_df['run_id']==run_id, 'sigma_pred_std'].reset_index(drop=True)
                    mu_pred_dev = results_df.loc[results_df['run_id']==run_id, 'mu_pred_dev'].reset_index(drop=True)
                    sigma_pred_dev = results_df.loc[results_df['run_id']==run_id, 'sigma_pred_dev'].reset_index(drop=True)
                    
                    # Plot observations, mu_pred +/- sigma_pred and contexts using
                    # plot_sample from RNN_paradigm/RNN/train/pipeline_core_v2.py
                    plot_sample(id=run_id, i=0, obs=[obs], mu_estim=[mu_pred_std], sigma_estim=[sigma_pred_std], save_path=save_path, n_ctx=n_ctx, title="y_pred = std, y_kal = dev", params=None, kalman_mu=[mu_pred_dev], kalman_sigma=[sigma_pred_dev], hidden_states=None, contexts=[ctx], contexts_prob=None, contexts_pred=None, preds_aligned=True, shared_ylim=False, ylim=None, kal_x_start=0, ctx_colors=ctx_colors, dpos_colors={}, rule_colors={})



