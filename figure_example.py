from audit_gm import NonHierarchicalAuditGM, HierarchicalAuditGM
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from model_RTs import prepare_trials_data


if __name__ == "__main__":

    # Example of the two consescutive trials
    trial_list_dir = os.path.join(os.path.dirname(__file__), 'triallists')
    sub = '04'
    
    for sess in range(4):
        
        trial_list_path = glob.glob(f"{trial_list_dir}/sub-{sub}/sub-{sub}_ses-{sess+1}*.csv")[0]
        # trials_df = pd.read_csv(trial_list_path, index_col=False)
        trials_df = prepare_trials_data(trial_list_path)
        
        # Index(['observation', 'frequency', 'state_std', 'state_dev', 'lim_std',
        #    'lim_dev', 'tau_std', 'tau_dev', 'd', 'rule', 'dpos', 'trial_type',
        #    'sigma_q_std', 'sigma_q_dev', 'sigma_r', 'ITI', 'duration_tones', 'ISI',
        #    'trial_no', 'run_n', 'session_n', 'relative_pos', 'contexts'],
        #   dtype='object')
        
        # Creat dummy GM
        config ={
                "participant_nr": '', # participant nr, as also read in by PsychPy exp.
                "N_samples": 1,
                "N_blocks": 60, # number of trials
                "N_tones": 8, # number of tones in each trial
                "rules_dpos_set": [[2, 3, 4], [4, 5, 6], None], # deviant positions per rule
                "mu_tau": 16, # unused for experiment
                "si_tau": 1, # unused for experiment
                "si_lim": 0.2, # unused for experiment
                "mu_rho_rules": 0.8, # unused for experiment
                "si_rho_rules": 0.05, # unused for experiment
                "mu_rho_timbres": 0.8, # unused for experiment
                "si_rho_timbres": 0.05, # unused for experiment
                # "si_q": 2,  # process noise variance
                "si_stat": 0.05,  # stationary process variance
                "si_r": 0.02,  # measurement noise variance
                "si_d_coef": 0.05, # unused for experiment
                "mu_d": 2, # unused for experiment
                "return_pi_rules": True,
                "fixed_rule_id": 2,
                "fixed_rule_p": 0.1,
                "rules_cmap": {0: sns.color_palette("Paired")[7], 1: sns.color_palette("Paired")[9], 2: "#ffffff"},
                "fix_process": True, # fix tau, lim, d to input values
                "tau_std_ind": None,
                "fix_tau_val": [16, 2], # tau std, tau dev
                "fix_lim_val": -0.6, # lim std
                "fix_d_val": 1, # effect size d
                "fix_pi_rules": True,
                "fix_pi_vals": [0.8, 0.1, 0], # fixed values to create transition matrix
                "n_sessions": 4, # number of sessions
                "n_runs": 4, # number of runs per session
                "isi": 0.65, # inter-stimulus interval
                "duration_tones": 0.1, # stimulus duration
        }
        gm = HierarchicalAuditGM(config)

        # Rule transition matrix
        pi_rule = np.array([ # actual values of pilots so far
            [0.8, 0.1, 0.1], 
            [0.1, 0.8, 0.1], 
            [0.5, 0.5, 0.0]  
        ])

        # Iterate over runs
        for run_id in range(4): # config['n_runs']
            pars = {
                "lim": [trials_df[trials_df['run_n']==run_id]['lim_std'].iloc[0], trials_df[trials_df['run_n']==run_id]['lim_dev'].iloc[0]],
                "tau": [trials_df[trials_df['run_n']==run_id]['tau_std'].iloc[0], trials_df[trials_df['run_n']==run_id]['tau_dev'].iloc[0]],
                "si_q": [trials_df[trials_df['run_n']==run_id]['sigma_q_std'].iloc[0], trials_df[trials_df['run_n']==run_id]['sigma_q_dev'].iloc[0]],
                "si_stat": trials_df[trials_df['run_n']==run_id]['sigma_r'].iloc[0]*10,
                "si_r": trials_df[trials_df['run_n']==run_id]['sigma_r'].iloc[0],
                "d": trials_df[trials_df['run_n']==run_id]['d'].iloc[0],
            }


            save_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/figures/example_sub{sub}_sess{sess}_run{run_id+1}.png'
            gm.plot_combined_with_matrix(trials_df[trials_df['run_n']==run_id]['state_std'].reset_index(drop=True),
                                        trials_df[trials_df['run_n']==run_id]['state_dev'].reset_index(drop=True),
                                        trials_df[trials_df['run_n']==run_id]['observation'].reset_index(drop=True),
                                        trials_df[trials_df['run_n']==run_id]['contexts'].reset_index(drop=True),
                                        trials_df[trials_df['run_n']==run_id].drop_duplicates('trial_no')['rule'].reset_index(drop=True),
                                        trials_df[trials_df['run_n']==run_id].drop_duplicates('trial_no')['dpos'].reset_index(drop=True),
                                        pars=pars, pi_rules=pi_rule, text=False, plot_obs=False, plot_dpos_dist=True, save_path=save_path,
                                        title=rf"d = {pars['d']}, $\sigma$_stat = {pars['si_stat']}, $\sigma_r$ = {pars['si_r']},  $\tau$_std = {pars['tau'][0]}, $\tau$_dev = {pars['tau'][1]}, $\sigma_q$_std = {pars['si_q'][0]}, $\sigma_q$_dev = {pars['si_q'][1]}")

    # An example of the states and observation sampling for one block
    # TODO:
    # - if more than one block, plot a vertical line to separate them
    # - make std line only made of dots at context==dvt
    # - locate black context dots not on horizontal lines but ON the corresponding state line
    # gm.plot_contexts_states_obs(trials_df['contexts'][0:2*gm.N_tones], trials_df['observation'][0:2*gm.N_tones], trials_df['state_std'][0:2*gm.N_tones], trials_df['state_dev'][0:2*gm.N_tones], 2*gm.N_tones, pars=pars, plot_obs=True)
