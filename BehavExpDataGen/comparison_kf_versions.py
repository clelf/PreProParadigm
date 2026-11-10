from model_RTs import prepare_trials_data
from kalman_full import HAudIn, STD_CONTEXT
from BehavModeling.visualize_leaky_vs_kf_predictions import plot_obs_and_models

import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    home_dir = '/home/clevyfidel/Documents/Workspace/PreProParadigm'
    kf_pred_dir = os.path.join(home_dir, 'BehavModeling/kalman_pred')
    fullinf_pred_dir = os.path.join(os.path.dirname(__file__), 'fullinf_pred')
    figures_dir = os.path.join(os.path.dirname(__file__), 'fullinf_comp_viz')

    # One file per subject: the 4 runs of the session combined (run_n 0-3, matching run_id of the old KF predictions)
    exp_input = os.path.join(home_dir, 'BehavModeling/triallists_logfiles2clem')

    subjects = ['11', '12', '13']
    sigma_r = 0.02
    sigma_stat = ...

    # Column index of HAudIn's pi_cues for each cue of the trial files
    cue_to_index = {'cue_1': 1, 'cue_2': 0}  # TODO: fill here

    for sub in subjects:
        # Load experimental sequences (one session, 4 runs)
        trials = prepare_trials_data(os.path.join(exp_input, f'sub-{sub}_ses-1_trials.csv'))

        # Load "old" KF predictions (as a df)
        old_kf_df = pd.read_csv(os.path.join(kf_pred_dir, f'kalman_predictions_sub-{sub}_ses-1_sigma_r_{sigma_r}.csv'))

        for run_n in trials['run_n'].unique():
            run_trials = trials[trials['run_n'] == run_n].reset_index(drop=True)
            y = run_trials['observation'].to_numpy()
            q = run_trials['cue'].map(cue_to_index).to_numpy()

            # True generative parameters of this run (constant within a run), one value per process [std, dev]
            true_taus = run_trials.loc[0, ['tau_std', 'tau_dev']].to_numpy(dtype=float)
            true_process_noise_sds = run_trials.loc[0, ['sigma_q_std', 'sigma_q_dev']].to_numpy(dtype=float)
            # # Stationary sd of each process, from the stationarity condition sigma_q = sigma_stat * sqrt(2 tau - 1) / tau
            # true_stationary_sds = true_process_noise_sds * true_taus / np.sqrt(2 * true_taus - 1)
            # sigma_eff = sqrt(sigma_stat**2 + sigma_r**2), as in audit_gm.py
            true_effective_sd = np.sqrt(sigma_stat ** 2 + sigma_r ** 2)

            # Initialize new-KF (cheat for now: true tau, true process noise, and sigma_eff**2 as prior state variance)
            full_inf_config = {"y": y,
                               "q": q,
                               "estimate_process_noise": True,
                               "n_transitions_before_using_estimates": (24, 3),
                               "prior_state_means":  [y[0], y[0]],
                               "prior_state_vars": [true_effective_sd ** 2, true_effective_sd ** 2],
                               "tau_init": true_taus,
                               "b_init": run_trials.loc[0, ['lim_std', 'lim_dev']].to_numpy(dtype=float), # [y[0], y[0]],
                               "process_noise_init": true_process_noise_sds ** 2}
            
            full_inf = HAudIn(**full_inf_config)

            # Run new full inference
            inf_states = full_inf.inference()

            # Store predictions
            full_inf.save_inference(inf_states, os.path.join(fullinf_pred_dir, f'fullinf_predictions_sub-{sub}_ses-1_run_{run_n}.csv'))

            # Old KF predictions of this run
            old_kf_run_df = old_kf_df[old_kf_df['run_id'] == run_n].reset_index(drop=True)

            # Plot visually with plot_obs_and_models: (mean, standard deviation, colour, label) per model;
            # standard process only, for both old and new inference
            models = [
                (old_kf_run_df['mu_pred_std'], old_kf_run_df['sigma_pred_std'], 'purple', 'old KF'),
                (inf_states['hat_y_mean_ctx'][:, STD_CONTEXT], np.sqrt(inf_states['hat_y_var_ctx'][:, STD_CONTEXT]), 'green', 'new KF'),
            ]
            plot_obs_and_models(
                obs=run_trials['observation'],
                contexts=run_trials['contexts'],
                models=models,
                save_path=os.path.join(figures_dir, f'kf_versions_sub-{sub}_ses-1_run_{run_n}.png'),
                title=f'sub-{sub}, run {run_n}',
            )

    # Ignore for now: Compare variance of estimations and MSE to ground truth
