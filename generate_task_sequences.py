# ---- imports ---- #
import audit_gm as gm

import numpy as np
from numpy import ma

import pandas as pd

from pykalman import KalmanFilter

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import os

import time

# ---- define class ----#

class trials_master:

    def __init__(self):

        self.config_H = {

            "participant_nr": 'test_participant', # participant nr, as also read in by PsychPy exp.
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
            "rules_cmap": {0: "tab:blue", 1: "tab:red", 2: "tab:gray"},
            "fix_process": True, # fix tau, lim, d to input values
            "fix_tau_val": [16, 2], # tau std, tau dev
            "fix_lim_val": -0.6, # lim std
            "fix_d_val": 2, # effect size d
            "fix_pi": True,
            "fix_pi_vals": [0.8, 0.1, 0], # fixed values to create transition matrix
            "n_sessions": 6, # number of sessions
            "n_runs": 4, # number of runs per session
            "isi": 0.65, # inter-stimulus interval
            "duration_tones": 0.1, # stimulus duration
        }

    def prep_dirs(self):
        '''create necessary output directories'''

        os.makedirs('trial_lists/', exist_ok=True)

        sub_dir = f"trial_lists/sub-{self.config_H["participant_nr"]}"
        os.makedirs(sub_dir, exist_ok=True)

        plot_dir = f"trial_lists/sub-{self.config_H["participant_nr"]}/plots"
        os.makedirs(plot_dir, exist_ok=True)

    def compute_stationary(self, pi):
        '''compute the stationary of a given transition matix'''

        # from: https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b
        transition_matrix = pi
        transition_matrix_transp = transition_matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
        close_to_1_idx = np.isclose(eigenvals,1)
        target_eigenvect = eigenvects[:,close_to_1_idx]
        target_eigenvect = target_eigenvect[:,0]
        stationary_distrib = target_eigenvect / sum(target_eigenvect)

        return stationary_distrib

    def define_task_structure(self):
        '''create a manually predefined order of mus and taus,
        NOTE: only roughly balanced for 6 sessions, might need change'''
        
        mu_tones = [[-0.6, None], [0.3, None]]
        taus = [16, 40, 160, 240]
        mu_tones_all = []

        for rep in range(self.config_H["n_sessions"]):
            row = []

            for i, _ in enumerate(taus):

                if (rep + i) % 2 == 0:
                    row.append(mu_tones[0])
                else:
                    row.append(mu_tones[1])

            mu_tones_all.append(row)
                
        mu_tones_all = np.array(mu_tones_all)
        tau_std_ind = np.array([[0, 1, 2, 3],
                                [3, 2, 1, 0],
                                [1, 3, 0, 2],
                                [3, 1, 2, 0],
                                [2, 3, 0, 1],
                                [1, 0, 3, 2]])
        
        tau_std = [[]]*self.config_H["n_sessions"]

        for rep in range(self.config_H["n_sessions"]):
                
                tau_row = [taus[i] for i in tau_std_ind[rep]]  # map indices to values
                tau_std[rep]=tau_row

        tau_std = np.array(tau_std)
        perm = np.random.permutation(len(mu_tones_all)) # shuffle sessions
        perm = np.array(range(self.config_H["n_sessions"])) # for test reasons keep order
        
        mu_tones_all = mu_tones_all[perm].tolist()
        
        tau_std = tau_std[perm]
        tau_dev = (tau_std // 8).astype(int)
        taus_all = [[x, y] for x, y in zip(tau_std, tau_dev)]

        return mu_tones_all, taus_all, tau_std, tau_dev
    
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
        plt.savefig(f"trial_lists/sub-{self.config_H['participant_nr']}/plots/probs_{type}_{num}.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()

    def apply_kalman(self, s, obs, contexts, si_q_arr, mu_tones, tau_std, figy, axy, n_iter = 10, tau_inits = 3.0, n_kalman = [8, 16, 24, 32, 40, 48, 56]):
        '''applies kalman filter separately for each run of each session'''

        run_length = self.config_H["N_blocks"] * 8
        n_runs = self.config_H["n_runs"]

        obs_use = obs.copy()
       
        obs_use[contexts == 1] = np.nan # we don't observe the standard in the position of the deviant

        iter = -1 

        for n in n_kalman:    

            iter+= 1    
            b_estimated_values = []

            for run in range(n_runs):

                tau_init = None
                b_init = None

                start_idx = run * run_length # get start of each run
                end_idx = start_idx + n # only take n onservations at the beginning of each run
                
                measurements = obs_use[start_idx:end_idx] # standard observations (incl. deviant position as nan)

                measurements = ma.asarray(measurements).reshape(-1, 1)
                measurements[np.isnan(measurements)] = ma.masked # mask deviant position for pykalman to handle as missing observation

                if tau_init is None:
                    tau_init = tau_inits  # Reasonable default initial guess for tau
                if b_init is None:
                    # Estimate b from final observations (rough steady state estimate)
                    if len(measurements) > 20:
                        b_init = np.mean(measurements[-20:])  # Use mean of final observations
                    else:
                        b_init = np.mean(measurements[-len(measurements)//2:])  # Use latter half

                # Convert to A matrix format: x_t = x_t-1 + (b - x_t-1)/tau
                A_init = np.array([[1.0 - 1.0/tau_init, b_init/tau_init], [0.0, 1.0]])

                # Ground truth noise covariances (FIXED, not estimated)
                # note: sigma q varies across runs according to the current implementation
                Q_true = np.array([[si_q_arr[run]**2, 0.0], [0.0, 0.0]])  # Only first state has noise
                R_true = np.array([[self.config_H["si_r"]**2]])  # 1D observation noise

                kf = KalmanFilter(
                    transition_matrices=A_init,  # Start with random initial guess
                    observation_matrices=np.array([[1.0, 0.0]]),  # Fixed - observe only x, not intercept
                    transition_covariance=Q_true,  # Ground truth (FIXED)
                    observation_covariance=R_true,  # Ground truth (FIXED)
                    #initial_state_mean=np.array([0.0, 1.0]),  # Second component should be 1
                    initial_state_mean=np.array([b_init, 1.0]),
                    initial_state_covariance=np.array([[1.0, 0.0], [0.0, 0.01]]),  # Small variance on intercept
                    n_dim_state=2,
                    n_dim_obs=1  # 1D observations
                )

                kf_fitted = kf.em(measurements, n_iter=n_iter,
                            em_vars=['transition_matrices', 'initial_state_mean', 'initial_state_covariance'])
                
                tau_est = 1.0 / (1.0 - kf_fitted.transition_matrices[0, 0])
                b_est = kf_fitted.transition_matrices[0, 1] * tau_est
                b_estimated_values.append(b_est)

                state_means_filt, state_covariances_filt = kf_fitted.filter(measurements)
                state_means_smooth, state_covariances_smooth = kf_fitted.smooth(measurements)


            # prep plot for each N and session all four runs (corresponding to different taus and mus)
            axy[iter, s].scatter(range(0,4), b_estimated_values, alpha=0.4, s=30, color='red', facecolors='red')
            axy[iter, s].scatter(range(0,4), [val[0] for val in mu_tones[s]], alpha=0.4, s=30, color='blue', facecolors='blue')
            axy[iter, s].set_xticks(range(4)) 
            axy[iter, s].set_xticklabels([f'{tau_std[s][0]}', f'{tau_std[s][1]}', f'{tau_std[s][2]}', f'{tau_std[s][3]}'])
            axy[iter, s].set_ylim(-2, 2)
            axy[iter, s].set_yticks(np.arange(-2, 1, 10)) 
            axy[iter, s].grid(True, alpha=0.3)
        

    def plot_kalman(self, figy, axy, n_kalman = [8, 16, 24, 32, 40, 48, 56]): 
        '''plots kalman results across sessions and n observations'''

        # just adding some labels and plot save the full plot across sessions
        for n in range(len(n_kalman)):
            axy[n, 0].set_ylabel(f'n = {n_kalman[n]}', fontsize=12, weight='bold')
            
        for s in range(0, self.config_H["n_sessions"]):
            axy[0, s].set_title(f'session {s+1}', fontsize=12, weight='bold')
            axy[-1, s].set_xlabel(f'tau', fontsize=12, weight='bold')


        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Estimated b',
                markerfacecolor='red', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='True mean',
                markerfacecolor='blue', markersize=6)
        ]

        figy.legend(handles=legend_elements, loc='upper right', fontsize=10) 
        figy.set_size_inches(25, 12) 
        
        plt.tight_layout()
        plt.savefig(f"trial_lists/sub-{self.config_H["participant_nr"]}/plots/kalman_results.png", dpi=300, bbox_inches='tight')
        #plt.show() 
        plt.close()  

    def frequency_transfer(self, x_input, x_min=-1, x_max=1, f_exp_min=500, f_exp_max=1300):
        '''compute frequency in Hz from input using the ERB transfer function
        as implemented according to Glasberg & Moore, 1990, Eq. 4
        '''

        e_min = 21.4 * np.log10(4.37 * f_exp_min / 1000 + 1)
        e_max = 21.4 * np.log10(4.37 * f_exp_max / 1000 + 1)

        e = e_min + (e_max - e_min) * (x_input - (x_min)) / (x_max - (x_min))

        freq_out = (10 ** (e / 21.4) - 1) * (1000 / 4.37)       

        return freq_out 

    def plot_states(self, s, std, dev, tau_std, tau_dev, run_rules, si_q_arr, si_q_dev_arr, mu_tones):
        '''plot states for one session'''

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

        colors_tau = {
            tau_std[s][0]: 'black',
            tau_std[s][1]: 'dimgrey',
            tau_std[s][2]: 'lightgrey',
            tau_std[s][3]: 'darkgrey'
        }

        for k, r in enumerate(tau_std_seq):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)

        tau_legend_patches = [
        mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
        ]

        quarter_width = self.config_H["N_blocks"]*self.config_H["N_tones"]
        xmin = 0
        xmax = self.config_H["N_blocks"]*self.config_H["N_tones"]*self.config_H["n_runs"]

        # Loop through each quarter
        for i in range(4):

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

        ax.set_xlabel('tone')
        ax.set_ylabel('state value')
        ax.set_title(f"linear gaussian dynamics for states of standard and deviant across tones (session {s+1})")
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=tau_legend_patches + handles, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"trial_lists/sub-{self.config_H['participant_nr']}/plots/lgd_std_dev_session_{s+1}.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()

    def plot_observations(self, s, obs_hz, run_contexts, run_rules, tau_std, dpos):
        '''plot observations in Hz'''

        tau_std_seq = [[tau_std[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_std[s]))]
        tau_std_seq = np.concatenate(tau_std_seq)
        
        indices_dev = np.where(run_contexts == 1)
        indices_dev = indices_dev[0]
        
        obs_hz_array = np.array(obs_hz)

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(obs_hz_array, label='sound observation', color='blue')
        plt.scatter(indices_dev, obs_hz_array[indices_dev], color='red', label='deviant')

        rules_long = np.repeat(run_rules, self.config_H["N_tones"])

        for l in range(len(rules_long)):

            if rules_long[l] == 0:
                ax.axvspan(l - 0.5, l + 0.5, color='lightblue', alpha=0.1)
            elif rules_long[l] == 1:
                ax.axvspan(l - 0.5, l + 0.5, color='lightcoral', alpha=0.1)
            elif rules_long[l] == 2:
                ax.axvspan(l - 0.5, l + 0.5, color='grey', alpha=0.1)

        colors_tau = {
            tau_std[s][0]: 'black',
            tau_std[s][1]: 'dimgrey',
            tau_std[s][2]: 'lightgrey',
            tau_std[s][3]: 'darkgrey'
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
        plt.savefig(f"trial_lists/sub-{self.config_H["participant_nr"]}/plots/observations_std_dev_session_{s+1}.png", dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()

    def generate_sessions(self):

        self.prep_dirs()
        mu_tones, taus_all, tau_std, tau_dev = self.define_task_structure()
        figy, axy = plt.subplots(7, self.config_H["n_sessions"], squeeze=False)

        # for loop for sessions
        for s in range(0, self.config_H["n_sessions"]):
                
            session_nr = str(s + 1).zfill(2)
            print(f"Generating session: {session_nr}")

            run_rules = []
            run_dpos = []
            run_contexts = []
            run_states_std = []
            run_states_dev = []
            run_obs = []

            si_q_arr = []
            si_q_dev_arr = []

            for r in range(0, self.config_H["n_runs"]):

                # change configuration dict for each run
                self.config_H["fix_tau_val"] = [tau_std[s][r], tau_std[s][r]/8]
                self.config_H["fix_lim_val"] = mu_tones[s][r][0]

                unbalanced = True

                while unbalanced:
                    
                    # use Cléms implementation to generate data per run
                    hgm = gm.HierarchicalAuditGM(self.config_H)
                    rules, _, dpos, _, _, contexts, states, obs, pars, pi_rules = hgm.generate_run(return_pars=True, return_pi_rules=self.config_H["return_pi_rules"])

                    # check if rules are balanced
                    p_r1 = sum(rules == 0)/len(rules)
                    p_r2 = sum(rules == 1)/len(rules)
                    p_r3 = sum(rules == 2)/len(rules)
                    p_rules = np.array([p_r1, p_r2])

                    stationary_distrib = self.compute_stationary(pi_rules)

                    if np.any((p_rules < (stationary_distrib[0]-0.02)) | (p_rules > (stationary_distrib[0]+0.02))) or p_r3 < (stationary_distrib[2]-0.02) or p_r3 > (stationary_distrib[2]+0.02): 
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
                            hgm.plot_combined_with_matrix(states[0], states[1], obs, contexts, rules, dpos, pars, pi_rules=pi_rules, text=False)
                            fig = plt.gcf() 
                            fig.savefig(f"trial_lists/sub-{self.config_H['participant_nr']}/plots/lgd_std_dev_session_{s+1}_run{r+1}_plot_clem.png", dpi=300, bbox_inches='tight')
                            plt.close()                             
                            
                            si_q_arr.append(pars[3][0])
                            si_q_dev_arr.append(pars[3][1])
                            states_std = states[0]
                            states_dev = states[1]
                            mu_tones[s][r][1] = pars[1][1]

                # plot probabilities per run
                self.plot_probabilities(rules, dpos, r+1, f'session_{s+1}_run')
                
                # collect data across runs
                run_rules.append(rules)
                run_dpos.append(dpos)
                run_contexts.append(contexts)
                run_states_std.append(states_std)
                run_states_dev.append(states_dev)
                run_obs.append(obs)   
                
            # plot probabilities per session
            run_rules = np.concatenate(run_rules)
            rules_long = np.repeat(run_rules, self.config_H["N_tones"])

            run_dpos = np.concatenate(run_dpos)
            run_contexts = np.concatenate(run_contexts)
            run_states_std = np.concatenate(run_states_std)
            run_states_dev = np.concatenate(run_states_dev)
            run_obs = np.concatenate(run_obs) 

            self.plot_probabilities(run_rules, run_dpos, s+1, 'overall_session')

            # generate some variables for output file
            tau_std_seq = [[tau_std[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_std[s]))]
            tau_std_seq = np.concatenate(tau_std_seq)

            tau_dev_seq = [[tau_dev[s][x]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(tau_dev[s]))]
            tau_dev_seq = np.concatenate(tau_dev_seq)

            lim_std_seq = [[mu_tones[s][x][0]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(mu_tones[s]))]
            lim_std_seq = np.concatenate(lim_std_seq)

            lim_dev_seq = [[mu_tones[s][x][1]]*self.config_H["N_blocks"]*self.config_H["N_tones"] for x in range(len(mu_tones[s]))]
            lim_dev_seq = np.concatenate(lim_dev_seq)
            
            # plot states for each full session
            self.plot_states(s, run_states_std, run_states_dev, tau_std, tau_dev, run_rules, si_q_arr, si_q_dev_arr, mu_tones)
                
            # apply Kalman filter to each session
            self.apply_kalman(s, run_obs, run_contexts, si_q_arr, mu_tones, tau_std, figy, axy)

            # apply frequency transfer to observations and plot observations in Hz
            obs_hz = []
            for observation in run_obs:
                hz_val_erb = self.frequency_transfer(observation)
                obs_hz.append(hz_val_erb)
            
            self.plot_observations(s, obs_hz, run_contexts, run_rules, tau_std, run_dpos)   

            # save session output file to read in for experiment in psychopy
            iti_range = np.arange(7, 12, 0.5) # ITI range (change for fMRI)
            ITI =[]

            for i in range(0,len(run_dpos)):
                ITI.append(np.random.choice(iti_range))

            trials_final = pd.DataFrame(columns=['observation', 'frequency', 'state_std', 'state_dev','lim_std','lim_dev','tau_std','tau_dev', 'd',
                                                 'rule','dpos','trial_type','sigma_q_std','sigma_q_dev','sigma_r','ITI','duration_tones','ISI','trial_n','run_n','session_n'])
            
            trials_final['observation'] = run_obs
            trials_final['frequency'] = obs_hz
            trials_final['state_std'] = run_states_std
            trials_final['state_dev'] = run_states_dev
            
            trials_final['lim_std'] = lim_std_seq
            trials_final['lim_dev'] = lim_dev_seq

            trials_final['tau_std'] = tau_std_seq
            trials_final['tau_dev'] = tau_dev_seq

            trials_final['d'] = np.repeat(self.config_H["fix_d_val"], self.config_H["N_tones"]*self.config_H["N_blocks"]*self.config_H["n_runs"])

            trials_final['rule'] = rules_long

            trials_final['dpos'] = np.repeat(run_dpos, self.config_H["N_tones"])
            trials_final['trial_type'] = run_contexts

            trials_final['sigma_q_std'] = np.repeat([x for x in si_q_arr], self.config_H["N_tones"]*self.config_H["N_blocks"])
            trials_final['sigma_q_dev'] = np.repeat([x for x in si_q_dev_arr], self.config_H["N_tones"]*self.config_H["N_blocks"])
            trials_final['sigma_r'] = [self.config_H["si_r"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])

            trials_final['ITI'] = [round(item,2) for item in ITI for _ in range(self.config_H["N_tones"])]

            trials_final['duration_tones'] = [self.config_H["duration_tones"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])
            trials_final['ISI'] = [self.config_H["isi"]]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])
            trials_final['trial_n'] = [i for i in range(self.config_H["N_blocks"]*len(tau_std[s])) for _ in range(self.config_H["N_tones"])]
            trials_final['run_n'] = np.repeat([range(0,len(tau_std[s]))], self.config_H["N_blocks"]*self.config_H["N_tones"])
            trials_final['session_n'] = [s]*self.config_H["N_blocks"]*self.config_H["N_tones"]*len(tau_std[s])

            trials_final.to_csv(f'trial_lists/sub-{self.config_H['participant_nr']}/sub-{self.config_H['participant_nr']}_ses-{session_nr}_trials.csv', index=False, float_format="%.4f")

            session_duration = ((((self.config_H["N_tones"]-1)*self.config_H["isi"])+(self.config_H["N_tones"]*self.config_H["duration_tones"]))*self.config_H["N_blocks"]*(len(tau_std[s]))) + sum(trials_final['ITI'][::8])
            print(f'Estimated session duration in minutes: {session_duration/60}')

        # plot Kalman results across all sessions
        self.plot_kalman(figy, axy, n_kalman = [8, 16, 24, 32, 40, 48, 56])        
            
if __name__ == "__main__":
    
    task = trials_master() 

    start = time.time()
    print("=== Generating Trials ===")
    task.generate_sessions() 
    end = time.time()
    print(f"=== Total Run Time Script: {(end-start)/60} ===")