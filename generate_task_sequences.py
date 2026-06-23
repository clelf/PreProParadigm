# ---- imports ---- #
import numpy as np
import pandas as pd

import os
import time
import random
from itertools import permutations, product

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import audit_gm as gm
from model_RTs import compute_likelihoods_at_deviants, compute_dev_likelihoods_over_dpos, compute_dev_likelihoods_over_rules

# ---- define class ----#

class trials_master:

    def __init__(self):

        self.config_H = {

            "participant_nr": '', # participant nr, as also read in by PsychPy exp.
            "N_samples": 1,
            "N_blocks": 60, # number of trials
            "N_tones": 8, # number of tones in each trial
            "rules_dpos_set": [[2, 3, 4], [4, 5, 6]], # deviant positions per rule
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
            "fixed_rule_p": 0,
            "rules_cmap": {0: "tab:blue", 1: "tab:red", 2: "tab:gray"},
            "fix_process": True, # fix tau, lim, d to input values
            "fix_tau_val": [16, 2], # tau std, tau dev
            "fix_lim_val": -0.6, # lim std
            "fix_d_val": 1, # effect size d
            "fix_pi_rules": True,
            "fix_pi_vals": [0.85, 0.15], # fixed values to create transition matrix
            "n_sessions": 1, # number of sessions
            "n_runs": 4, # number of runs per session
            "isi": 0.65, # inter-stimulus interval
            "duration_tones": 0.1, # stimulus duration,
            "init": 'TN_3', # how to initialize std dev processes,
            "sub_dir": None,
            "plot_dir": None
        }

    def prep_dirs(self):
        '''create necessary output directories to save trial lists and plots'''

        os.makedirs('fMRI/sequences_per_condition/', exist_ok=True)

        sub_dir = f"fMRI/sequences_per_condition/{self.config_H["participant_nr"]}"
        os.makedirs(sub_dir, exist_ok=True)

        plot_dir = f"fMRI/sequences_per_condition/{self.config_H["participant_nr"]}/plots"
        os.makedirs(plot_dir, exist_ok=True)

        self.config_H['sub_dir'] = sub_dir
        self.config_H['plot_dir'] = plot_dir


    def compute_stationary(self, pi):
        '''compute the stationary of a given transition matix
        
        Args:
            pi: transition matrix

        Returns:
            stationary distrib: stationary distribution
        '''

        # from: https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b
        transition_matrix = pi
        transition_matrix_transp = transition_matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
        close_to_1_idx = np.isclose(eigenvals,1)
        target_eigenvect = eigenvects[:,close_to_1_idx]
        target_eigenvect = target_eigenvect[:,0]
        stationary_distrib = target_eigenvect / sum(target_eigenvect)

        return stationary_distrib
    
    def plot_probabilities(self, rules, dpos, num, type):
        '''plot the occurrences of rules and deviant positions per rule'''

        p_1 = sum(rules == 0)/len(rules)
        p_2 = sum(rules == 1)/len(rules)
        p_3 = sum(rules == 2)/len(rules)

        p_d3_r1 = np.where((dpos==2)&(rules==0))[0].size/np.sum(rules == 0)
        p_d4_r1 = np.where((dpos==3)&(rules==0))[0].size/np.sum(rules == 0)
        p_d5_r1 = np.where((dpos==4)&(rules==0))[0].size/np.sum(rules == 0)

        p_d5_r3 = np.where((dpos == 4)&(rules==1))[0].size/np.sum(rules == 1)
        p_d6_r3 = np.where((dpos == 5)&(rules==1))[0].size/np.sum(rules == 1)
        p_d7_r3 = np.where((dpos == 6)&(rules==1))[0].size/np.sum(rules == 1)

        x_label = ['P(rule 1)', 'P(rule 3)','P(no dev)', 'P(dev3|rule1)','P(dev4|rule1)','P(dev5|rule1)','P(dev5|rule3)','P(dev6|rule3)','P(dev7|rule3)']
        y_values = [p_1, p_2, p_3, p_d3_r1, p_d4_r1, p_d5_r1, p_d5_r3, p_d6_r3, p_d7_r3]

        colors = ['lightblue', 'lightcoral', 'grey','lightblue','lightblue','lightblue','lightcoral','lightcoral','lightcoral']

        plt.figure(figsize=(15, 7))
        plt.bar(x_label, y_values, color=colors)

        ymin, ymax = plt.ylim()

        n_trials_rule = [sum(rules == 0), sum(rules == 1), sum(rules == 2), sum(dpos[np.where(rules == 0)]==2), sum(dpos[np.where(rules == 0)]==3), sum(dpos[np.where(rules == 0)]==4),
                        sum(dpos[np.where(rules == 1)]==4), sum(dpos[np.where(rules == 1)]==5), sum(dpos[np.where(rules == 1)]==6)]

        for i in range(0, len(colors)):
            plt.text(i, ymax-0.01, f'N = {int(n_trials_rule[i])}', fontsize=6, ha='center', va='bottom')

        plt.xlabel('')
        plt.ylabel('P')
        plt.title(f"Probabilities ({type} {num})")

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.yticks(np.arange(0, 0.47, 0.01))
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/probs_{type}_{num}.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()

    def get_kalman_likelihoods_clem(self, trials_path, s, trials_final, tau_std, pi_rules):

        '''get Kalman-likelihoods using model_RTs
        
        Args:
            trials_path: path to trial lists
            s: session #
            trials_final: trials_final df
            tau_std: tau standard list
            pi_rules: 

        Returns: (plots likelihood, likelihoods over dpos, and likelihood over rules)
        '''

        results_df = compute_likelihoods_at_deviants(trials_path, self.config_H['participant_nr'], s, results_save_path=f'{self.config_H['sub_dir']}/')
        
        trials_final_no_null = trials_final[~trials_final["trial_n"].astype(str).str.contains("null", na=False)]
        results_df['rules'] = trials_final_no_null['rule'].iloc[::8].tolist()
        results_df['tau'] = trials_final_no_null['tau_std'].iloc[::8].tolist()
        results_df['d'] = trials_final_no_null['d'].iloc[::8].tolist()

        mask1 = (results_df['dpos'] == 4) & (results_df['rules'] == 0)
        mask2 = (results_df['dpos'] == 4) & (results_df['rules'] == 1)

        results_df.loc[mask1, 'dpos'] = 3.8
        results_df.loc[mask2, 'dpos'] = 4.2

        # scatter plot KF result per dpos
        chunks = np.array_split(results_df, self.config_H['n_runs'])
        fig, axes = plt.subplots(1, self.config_H['n_runs'] + 1, figsize=(20, 4))
        for i, chunk in enumerate(chunks):
            axes[i].scatter(chunk['dpos'], chunk['likelihood_obs_std_at_dev'])
            axes[i].set_title(f'run {i+1}, tau = {tau_std[s][i]}')

        # plot full dataframe
        axes[-1].scatter(results_df['dpos'], results_df['likelihood_obs_std_at_dev'])
        axes[-1].set_title('full session')

        plt.ylim()
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/KF_likelihoods_std_at_dev_session_{s+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

        chunks = np.array_split(results_df, self.config_H['n_runs'])
        fig, axes = plt.subplots(1, self.config_H['n_runs'] + 1, figsize=(20, 4))
        for i, chunk in enumerate(chunks):
            axes[i].scatter(chunk['dpos'], chunk['likelihood_obs_dev_at_dev'])
            axes[i].set_title(f'run {i+1}, tau = {tau_std[s][i]}')

        # plot full dataframe
        axes[-1].scatter(results_df['dpos'], results_df['likelihood_obs_dev_at_dev'])
        axes[-1].set_title('full session')

        plt.ylim()
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/KF_likelihoods_dev_at_dev_session_{s+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

        results_df_dpos = compute_dev_likelihoods_over_dpos(trials_path, results_df, self.config_H['participant_nr'], s, results_save_path=None, inplace=False)

        results_df_dpos['rules'] = trials_final_no_null['rule'].iloc[::8].tolist()
        results_df_dpos['tau'] = trials_final_no_null['tau_std'].iloc[::8].tolist()
        results_df_dpos['d'] = trials_final_no_null['d'].iloc[::8].tolist()

        mask1 = (results_df_dpos['dpos'] == 4) & (results_df_dpos['rules'] == 0)
        mask2 = (results_df_dpos['dpos'] == 4) & (results_df_dpos['rules'] == 1)

        results_df_dpos.loc[mask1, 'dpos'] = 3.8
        results_df_dpos.loc[mask2, 'dpos'] = 4.2

        chunks = np.array_split(results_df_dpos, self.config_H['n_runs'])
        fig, axes = plt.subplots(1, self.config_H['n_runs'] + 1, figsize=(20, 4))
        for i, chunk in enumerate(chunks):
            axes[i].scatter(chunk['dpos'], chunk['likelihood_obs_std_at_dev_over_dpos'])
            axes[i].set_title(f'run {i+1}, tau = {tau_std[s][i]}')

        # plot full dataframe
        axes[-1].scatter(results_df['dpos'], results_df['likelihood_obs_std_at_dev_over_dpos'])
        axes[-1].set_title('full session')

        plt.ylim()
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/KF_likelihoods_std_at_dev_over_dpos_session_{s+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

        results_df_rules = compute_dev_likelihoods_over_rules(trials_path, results_df, self.config_H['participant_nr'], s, pi_rules, results_save_path=None, inplace=False)

        results_df_rules['rules'] = trials_final_no_null['rule'].iloc[::8].tolist()
        results_df_rules['tau'] = trials_final_no_null['tau_std'].iloc[::8].tolist()
        results_df_rules['d'] = trials_final_no_null['d'].iloc[::8].tolist()

        mask1 = (results_df_rules['dpos'] == 4) & (results_df_rules['rules'] == 0)
        mask2 = (results_df_rules['dpos'] == 4) & (results_df_rules['rules'] == 1)

        results_df_rules.loc[mask1, 'dpos'] = 3.8
        results_df_rules.loc[mask2, 'dpos'] = 4.2

        results_df_rules.to_csv(f"{self.config_H['sub_dir']}/results_KF_session_{s+1}.csv", index=False)

        chunks = np.array_split(results_df_rules, self.config_H['n_runs'])
        fig, axes = plt.subplots(1, self.config_H['n_runs'] + 1, figsize=(20, 4))
        for i, chunk in enumerate(chunks):
            axes[i].scatter(chunk['dpos'], chunk['likelihood_obs_std_at_dev_over_rules'])
            axes[i].set_title(f'run {i+1}, tau = {tau_std[s][i]}')

        # plot full dataframe
        axes[-1].scatter(results_df['dpos'], results_df['likelihood_obs_std_at_dev_over_rules'])
        axes[-1].set_title('full session')

        plt.ylim()
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/KF_likelihoods_std_at_dev_over_rules_session_{s+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def frequency_transfer(self, x_input, x_min=-1, x_max=1, f_exp_min=500, f_exp_max=1300):
        
        '''compute frequency in Hz from input using the ERB transfer function
        as implemented according to Glasberg & Moore, 1990, Eq. 4
        
        Args:
            x_input: input value
            x_min: minimum of x scale
            x_max: maximum of x scale
            f_exp_min: min F0 of experiment in Hz
            f_exp_max: max F0 of experiment in Hz

        Returns:
            freq_out: F0 corresponding to x_input after applying ERB scale
        '''

        e_min = 21.4 * np.log10(4.37 * f_exp_min / 1000 + 1)
        e_max = 21.4 * np.log10(4.37 * f_exp_max / 1000 + 1)

        e = e_min + (e_max - e_min) * (x_input - (x_min)) / (x_max - (x_min))

        freq_out = (10 ** (e / 21.4) - 1) * (1000 / 4.37)       

        return freq_out 

    def plot_states(self, s, std, dev, tau_std, tau_dev, run_rules, si_q_arr, si_q_dev_arr, mu_tones, run_obs, run_contexts, name):
        '''plot the state values for one session (on input scale)
        
        Args:
            s: session #
            std:
            dev:
            tau_std:
            tau_dev:
            run_rules:
            si_q_arr:
            si_q_dev_arr:
            mu_tones:
            run_obs:
            run_contexts:
            name:

        Returns: (plots states of std and dev for one session)
        '''

        contexts = ['std', 'dev']

        tau_std_seq = [[tau_std[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_std[s]))]
        tau_std_seq = np.concatenate(tau_std_seq)

        tau_dev_seq = [[tau_dev[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_dev[s]))]
        tau_dev_seq = np.concatenate(tau_dev_seq)

        rules_long = np.repeat(run_rules, self.config_H["N_tones"])

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(std, label='standard', color='blue')
        ax.plot(dev, label='deviant', color='red')

        for j in range(len(rules_long)):

            if rules_long[j] == 0:
                ax.axvspan(j - 0.5, j + 0.5, color='lightblue', alpha=0.1)
            elif rules_long[j] == 1:
                ax.axvspan(j - 0.5, j + 0.5, color='lightcoral', alpha=0.1)
            elif rules_long[j] == 2:
                ax.axvspan(j - 0.5, j + 0.5, color='gray', alpha=0.1)

        colors = ['black', 'dimgrey', 'lightgrey', 'darkgrey']

        colors_tau = {
            tau: color
            for tau, color in zip(tau_std[s], colors[:len(tau_std[s])])
        }

        for k, r in enumerate(tau_std_seq):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)

        tau_legend_patches = [
        mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
        ]

        quarter_width = self.config_H["N_blocks"]*self.config_H["N_tones"]
        xmin = 0
        xmax = self.config_H["N_blocks"]*self.config_H["N_tones"]*self.config_H["n_runs"]

        if name == 'input':
            # Loop through each quarter
            for i in range(self.config_H["n_runs"]):

                line_values = mu_tones[s][i]
                colors = ['b','r']
                q_start = xmin + i * quarter_width
                q_end = q_start + quarter_width
                ind_col = -1

                for val in line_values:
                    ind_col += 1

                    if ind_col == 0:
                        stationary_std = si_q_arr[i] * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5)
                    else:
                        stationary_std = si_q_dev_arr[i] * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5)
                    
                    lbl = f"mu {contexts[ind_col]}" if i == 0 else None
                    lbl2 = f"stationary std {contexts[ind_col]}" if i == 0 else None
                    plt.hlines(y=val, xmin=q_start, xmax=q_end,
                            colors=colors[ind_col], linestyles='--', linewidth=0.5, label=lbl)
                    plt.fill_between(np.linspace(q_start, q_end, self.config_H["N_blocks"]),
                        val - stationary_std,
                        val + stationary_std,
                        color=colors[ind_col], alpha=0.2, label = lbl2)
                
        # add observations as dots
        indices_dev = np.where(run_contexts == 1)
        indices_dev = indices_dev[0]
        obs_hz_array = np.array(run_obs)
        std_data = obs_hz_array.copy().astype(float)
        std_data[indices_dev] = np.nan

        ax.plot(std_data,label='standard observation', color='darkblue', alpha=0.7,linestyle='None', marker='.', markersize=2)
        ax.plot(indices_dev, obs_hz_array[indices_dev],color='darkred', label='deviant observation', alpha=0.7,linestyle='None', marker='.', markersize=2)

        ax.set_xlabel('tone')
        ax.set_ylabel('state value')
        ax.set_title(f"linear gaussian dynamics for states of standard and deviant across tones (session {s+1})")
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=tau_legend_patches + handles, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/lgd_std_dev_session_{s+1}_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_observations(self, s, obs_hz, run_contexts, run_rules, tau_std, dpos):
        '''plot observations for one session (in Hz)
        
        Args:
            s:
            obs_hz:
            run_contexts:
            run_rules:
            tau_std:
            dpos:

        Returns: (plots observed F0s in Hz)
        '''

        tau_std_seq = [[tau_std[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_std[s]))]
        tau_std_seq = np.concatenate(tau_std_seq)
        
        indices_dev = np.where(run_contexts == 1)
        indices_dev = indices_dev[0]
        
        obs_hz_array = np.array(obs_hz)

        std_data = obs_hz_array.copy().astype(float)

        std_data[indices_dev] = np.nan

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(std_data, label='standard observation', color='blue')
        ax.plot(indices_dev, obs_hz_array[indices_dev], color='red', linestyle='--', marker='o', label='deviant observation')

        rules_long = np.repeat(run_rules, self.config_H["N_tones"])

        for l in range(len(rules_long)):

            if rules_long[l] == 0:
                ax.axvspan(l - 0.5, l + 0.5, color='lightblue', alpha=0.1)
            elif rules_long[l] == 1:
                ax.axvspan(l - 0.5, l + 0.5, color='lightcoral', alpha=0.1)
            elif rules_long[l] == 2:
                ax.axvspan(l - 0.5, l + 0.5, color='grey', alpha=0.1)

        colors = ['black', 'dimgrey', 'lightgrey', 'darkgrey']

        colors_tau = {
            tau: color
            for tau, color in zip(tau_std[s], colors[:len(tau_std[s])])
        }

        for k, r in enumerate(tau_std_seq):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)

        tau_legend_patches = [
        mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
        ]

        ax.set_xlabel('tone')
        ax.set_ylabel('observation in Hz (erb scale)')
        ax.set_title(f"sound observations (session {s+1})")
        ax.legend()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=tau_legend_patches + handles, loc='upper right')

        ymin, ymax = plt.ylim()

        dpos_seq_no_zero = dpos[dpos != None]

        for i in range(0, len(indices_dev)):
            plt.text(indices_dev[i], ymax-20, f'{int(dpos_seq_no_zero[i])}', fontsize=6, ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.config_H['plot_dir']}/observations_std_dev_session_{s+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_sessions(self, d, cue_prob, cues, single_run_config = None):

        ''' generates all experimental sessions

        Args:
            d: effect size (std-dev deparation)
            cue_prob: cue validity (0-1)

        Returns:
            writes csv files containing trial lists for each session/run
        
        '''

        self.prep_dirs()

        if self.config_H['n_runs'] > 1:
            df = self.randomize_design_abc(self.config_H["n_runs"], self.config_H["n_sessions"])
            mu_tones, taus_all, tau_std, tau_dev, sign_d = self.define_task_structure(df)
        
        elif self.config_H['n_runs'] == 1:
            tau_std = [[single_run_config[0]]]
            tau_dev = [[single_run_config[0]/8]]
            mu_tones = [[[single_run_config[1], None]]]
            sign_d = [[single_run_config[2]]]


        # for loop for sessions
        for s in range(0, self.config_H["n_sessions"]):
                
            session_nr = str(s + 1).zfill(2)
            print(f"Generating session: {session_nr}")

            run_rules = []
            run_cues = []
            run_dpos = []
            run_contexts = []
            run_states_std = []
            run_states_dev = []
            run_obs = []

            si_q_arr = []
            si_q_dev_arr = []

            for r in range(0, self.config_H["n_runs"]):

                print(f"Generating run: {r}")

                # change configuration dict for each run
                self.config_H["fix_tau_val"] = [tau_std[s][r], tau_std[s][r]/8]
                self.config_H["fix_lim_val"] = mu_tones[s][r][0]
                self.config_H["fix_d_val"] = d*sign_d[s][r]

                unbalanced = True

                while unbalanced:
                    
                    # use Cléms implementation to generate data per run
                    hgm = gm.HierarchicalAuditGM(self.config_H)
                    rules, _, dpos, _, _, contexts, states, obs, pars, pi_rules = hgm.generate_run(return_pars=True, return_pi_rules=self.config_H["return_pi_rules"])

                    # check if rules are balanced
                    p_r1 = sum(rules == 0)/len(rules)
                    p_r2 = sum(rules == 1)/len(rules)
                    p_rules = np.array([p_r1, p_r2])

                    stationary_distrib = self.compute_stationary(pi_rules)

                    if np.any((p_rules < (stationary_distrib[0]-0.02)) | (p_rules > (stationary_distrib[0]+0.02))): 
                        # allow for deviations of max. 2% in both directions from the stationary
                        continue
                    else:
                        # check if deviant positions are balanced within each rule
                        ind_r1 = np.where(rules == 0)
                        ind_r2 = np.where(rules == 1)

                        vals_r1 = [dpos[i] for i in ind_r1]
                        vals_r2 = [dpos[i] for i in ind_r2]

                        vals_r1 = np.array([dpos[i] for i in ind_r1])
                        dpos_set_r1 = np.array(self.config_H["rules_dpos_set"][0])
                        counts_r1 = [(vals_r1 == v).sum() for v in dpos_set_r1]

                        vals_r2 = np.array([dpos[i] for i in ind_r2])
                        dpos_set_r2 = np.array(self.config_H["rules_dpos_set"][1])
                        counts_r2 = [(vals_r2 == v).sum() for v in dpos_set_r2]

                        all_counts = counts_r1 + counts_r2 

                        max_count = max(all_counts)
                        min_count = min(all_counts)

                        if max_count - min_count > 1: # max. difference of 1 trial between occurrences of deviant positions
                            continue
                        else:
                            unbalanced = False

                            # plot run using Cléms plotting approach
                            hgm.plot_combined_with_matrix(states[0], states[1], obs, contexts, rules, dpos, pars, pi_rules=pi_rules, text=False, plot_dpos_dist=True, save_path = None)
                            fig = plt.gcf() 
                            fig.savefig(f"{self.config_H['plot_dir']}/lgd_std_dev_session_{s+1}_run{r+1}_plot_clem.png", dpi=300, bbox_inches='tight')
                            plt.close()                         
                            
                            si_q_arr.append(pars['si_q'][0])
                            si_q_dev_arr.append(pars['si_q'][1])

                            states_std = states[0]
                            states_dev = states[1]
                            mu_tones[s][r][1] = pars['lim'][1]

                # plot probabilities per run
                self.plot_probabilities(rules, dpos, r+1, f'session_{s+1}_run')

                t_1_1 = 0
                t_1_2 = 0
                t_2_1 = 0
                t_2_2 = 0

                for ru in range(1,len(rules)):
                    if rules[ru] == 0 and rules[ru-1] == 0:
                        t_1_1 += 1
                    elif rules[ru] == 1 and rules[ru-1] == 0:
                        t_1_2 += 1
                    elif rules[ru] == 0 and rules[ru-1] == 1:
                        t_2_1 += 1
                    elif rules[ru] == 1 and rules[ru-1] == 1:
                        t_2_2 += 1
                
                # add probabilistic cues
                r1_pos_3 = np.where((np.array(rules) == 0) & (np.array(dpos) == 2))[0]
                r1_pos_4 = np.where((np.array(rules) == 0) & (np.array(dpos) == 3))[0]
                r1_pos_5 = np.where((np.array(rules) == 0) & (np.array(dpos) == 4))[0]
                
                r2_pos_5 = np.where((np.array(rules) == 1) & (np.array(dpos) == 4))[0]
                r2_pos_6 = np.where((np.array(rules) == 1) & (np.array(dpos) == 5))[0]
                r2_pos_7 = np.where((np.array(rules) == 1) & (np.array(dpos) == 6))[0]

                cuesy = np.full(len(rules), cues[0], dtype=object)

                r1_cue_pos_3 = np.random.choice(r1_pos_3, size=int(cue_prob*len(r1_pos_3)), replace=False)
                r1_cue_pos_4 = np.random.choice(r1_pos_4, size=int(cue_prob*len(r1_pos_4)), replace=False)
                r1_cue_pos_5 = np.random.choice(r1_pos_5, size=int(cue_prob*len(r1_pos_5)), replace=False)
                
                r2_cue_pos_5 = np.random.choice(r2_pos_5, size=int(round((1-cue_prob),2)*len(r2_pos_5)), replace=False)
                r2_cue_pos_6 = np.random.choice(r2_pos_6, size=int(round((1-cue_prob),2)*len(r2_pos_6)), replace=False)
                r2_cue_pos_7 = np.random.choice(r2_pos_7, size=int(round((1-cue_prob),2)*len(r2_pos_7)), replace=False)

                r1_cue_1 = np.concatenate([r1_cue_pos_3, r1_cue_pos_4, r1_cue_pos_5])
                r2_cue_1 = np.concatenate([r2_cue_pos_5, r2_cue_pos_6, r2_cue_pos_7])

                cuesy[r1_cue_1] = cues[1]
                cuesy[r2_cue_1] = cues[1]  

                print(cuesy)

                # collect data across runs
                run_rules.append(rules)
                run_cues.append(cuesy)
                run_dpos.append(dpos)
                run_contexts.append(contexts)
                run_states_std.append(states_std)
                run_states_dev.append(states_dev)
                run_obs.append(obs) 
                
            # plot probabilities per session
            run_rules = np.concatenate(run_rules)
            rules_long = np.repeat(run_rules, self.config_H["N_tones"])

            run_cues = np.concatenate(run_cues)
            cues_long = np.repeat(run_cues, self.config_H["N_tones"])

            run_dpos = np.concatenate(run_dpos)

            run_contexts = np.concatenate(run_contexts)
            run_states_std = np.concatenate(run_states_std)
            run_states_dev = np.concatenate(run_states_dev)
            run_obs = np.concatenate(run_obs) 

            self.plot_probabilities(run_rules, run_dpos, s, 'overall_session')

            # generate some variables for output file
            tau_std_seq = [[tau_std[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_std[s]))]
            tau_std_seq = np.concatenate(tau_std_seq)

            tau_dev_seq = [[tau_dev[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_dev[s]))]
            tau_dev_seq = np.concatenate(tau_dev_seq)

            lim_std_seq = [[mu_tones[s][x][0]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(mu_tones[s]))]
            lim_std_seq = np.concatenate(lim_std_seq)

            lim_dev_seq = [[mu_tones[s][x][1]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(mu_tones[s]))]
            lim_dev_seq = np.concatenate(lim_dev_seq)

            # apply frequency transfer to observations and plot observations in Hz
            obs_hz = []
            state_s_hz = []
            state_d_hz = []

            for observation in run_obs:
                hz_val_erb = self.frequency_transfer(observation)
                obs_hz.append(hz_val_erb)

            for state_s in run_states_std:
                hz_val_erb = self.frequency_transfer(state_s)
                state_s_hz.append(hz_val_erb)

            for state_d in run_states_dev:
                hz_val_erb = self.frequency_transfer(state_d)
                state_d_hz.append(hz_val_erb)

            # plot states for each full session
            self.plot_states(s, run_states_std, run_states_dev, tau_std, tau_dev, run_rules, si_q_arr, si_q_dev_arr, mu_tones, run_obs, run_contexts, name = 'input')
            self.plot_states(s, state_s_hz, state_d_hz, tau_std, tau_dev, run_rules, si_q_arr, si_q_dev_arr, mu_tones, obs_hz, run_contexts, name = 'hz')
            self.plot_observations(s, obs_hz, run_contexts, run_rules, tau_std, run_dpos)   

            # save session output file to read in for experiment in psychopy
            trials_final = pd.DataFrame(columns=['observation', 'frequency', 'state_std', 'state_dev','lim_std','lim_dev','tau_std','tau_dev', 'd',
                                                 'rule','dpos','trial_type','sigma_q_std','sigma_q_dev','sigma_r','ITI','duration_tones','ISI','trial_n','run_n','session_n'])
            
            trials_final['observation'] = run_obs
            trials_final['frequency'] = obs_hz
            trials_final['state_std'] = run_states_std
            trials_final['state_dev'] = run_states_dev
            trials_final['state_std_hz'] = state_s_hz
            trials_final['state_dev_hz'] = state_d_hz
            trials_final['diff_std'] = trials_final['frequency'] - trials_final['state_std_hz']
            trials_final['diff_dev'] = trials_final['frequency'] - trials_final['state_dev_hz']
            
            trials_final['lim_std'] = lim_std_seq
            trials_final['lim_dev'] = lim_dev_seq

            trials_final['tau_std'] = tau_std_seq
            trials_final['tau_dev'] = tau_dev_seq

            trials_final['d'] = np.repeat(np.array(sign_d[s])*d, self.config_H["N_tones"]*self.config_H["N_blocks"])

            trials_final['rule'] = rules_long
            trials_final['cue'] = cues_long

            trials_final['dpos'] = np.repeat(run_dpos, self.config_H["N_tones"])
            trials_final['trial_type'] = run_contexts

            trials_final.loc[trials_final['trial_type'] == 1, 'diff_std'] = np.nan
            trials_final.loc[trials_final['trial_type'] == 0, 'diff_dev'] = np.nan

            trials_final['sigma_q_std'] = np.repeat([x for x in si_q_arr], self.config_H["N_tones"]*self.config_H["N_blocks"])
            trials_final['sigma_q_dev'] = np.repeat([x for x in si_q_dev_arr], self.config_H["N_tones"]*self.config_H["N_blocks"])
            trials_final['sigma_r'] = [self.config_H["si_r"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])

            trials_final['duration_tones'] = [self.config_H["duration_tones"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])
            trials_final['ISI'] = [self.config_H["isi"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])
            trials_final['trial_n'] = [i for i in range(self.config_H["N_blocks"]*len(tau_std[s])) for _ in range(self.config_H["N_tones"])]
            trials_final['run_n'] = np.repeat([range(0,len(tau_std[s]))], self.config_H["N_blocks"]*self.config_H["N_tones"])
            trials_final['session_n'] = [s]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])

            trials_path = f'{self.config_H['sub_dir']}/sub-{self.config_H['participant_nr']}_ses-{s+1}_trials.csv'
            trials_final.to_csv(trials_path, index=False, float_format="%.4f")
            
            # write out runs separately
            for r in range(0, self.config_H["n_runs"]):

                trials_run = trials_final.loc[trials_final['run_n']==r]
                trials_run.to_csv(f'{self.config_H['sub_dir']}/sub-{self.config_H['participant_nr']}_ses-{s+1}_run-{r+1}_trials.csv', index=False, float_format="%.4f")

                run_unique = trials_run.drop_duplicates(subset="trial_n")

            self.get_kalman_likelihoods_clem(trials_path, s, trials_final, tau_std, pi_rules)
            
if __name__ == "__main__":
    
    subs = [f"seq{i}" for i in range(1, 101)]

    for suby in subs:

        cue_prob = 0.8

        d = 1

        si_stat = 0.1
        si_r_rat = 5

        init = 'TN_3'

        cues = ['cue_3','cue_4']

        condy = 0

        # here: insert a loop over 16 conditional combinations for single runs --> test
        for tau_std in [16, 40, 160, 240]:
            for mu_cond in [-0.6, 0.3]:
                for dev_cond in [1, -1]:

                    condy += 1

                    print(f"=== Generating Trials with |d| = {d}, si_stat = {si_stat}, si_r = {si_stat/si_r_rat} ===")

                    task = trials_master()

                    task.config_H["participant_nr"] = f"{suby}_cond{condy}_tau{tau_std}_mu{mu_cond}_dev{dev_cond}"
                    task.config_H["si_stat"] = si_stat
                    task.config_H["si_r"] = si_stat/si_r_rat
                    task.config_H["n_sessions"] = 1
                    task.config_H['n_runs'] = 1
                    task.config_H["init"] = init

                    single_run_config = [tau_std, mu_cond, dev_cond]

                    start = time.time()

                    task.generate_sessions(d, cue_prob, cues, single_run_config) 

                    end = time.time()

                    print(f"=== Total Run Time Script: {(end-start)/60} ===")
           