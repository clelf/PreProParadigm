import numpy as np
from numpy import ma
import pandas as pd
import random
import math
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os
from pykalman import KalmanFilter

'''
GENERAL

- process: x_t = a*x_t-1 + d + w_t, x_t ~ N(0, sigma_q**2)
- a = 1-(1/tau), d = mu/tau
- stat_std = (sigma_q*tau)/sqrt(2*tau-1)
    --> sigma_q = (stat_std*sqrt(2*tau-1))/tau
- d_eff = (log(mu_std)-log(mu_dev))/stat_std
- TODO: use d_eff to generate mus?

'''

class trials_master:
    '''data generation process for auditPrePro task'''

    def __init__(self):
        self.params = {
                "participant_nr": '01',
                "n_sessions": 6,
                "n_trials": 60, # number of trials per run
                "rules": [0, 1, 2],
                "diag": 0.8, # self-trinsition of rules 1 and 2
                "tr_3": 0.1, # transition rules to trials w/o deviant
                "tr_3_3": 0, # transition pprobability trials w/o deviant
                "frequencies": [[1000, 1050], [700, 750]], # mu values for std and dev process
                "taus": [16, 40, 160, 240], # taus in each session
                "dpos": [[2,3,4],[4,5,6],[0]], # deviant positions per rule
                "contexts": ['std', 'dev'], 
                "stat_std": 25, # stationary standard deviation of the processes
                "si_r": 10, # observation noise
                "duration_tones": 0.1, # stimulus duration
                "isi": 0.65, # inter-stimulus interval
                "t_tones": True, # use tones as time steps 
                "dev_t_tones": False, # do not use tones as time steps for deviant
                "dev_process": True, # use a process for the deviants
                "n_kalman": [8, 16, 24, 32, 40, 48, 56] # time steps to use for Kalman filter
            }

    def sample_next_markov_state(self, current_state, states_values, states_trans_matrix):
            '''Clems function to sample the next Markov state from current state, state values and transition matrix'''

            return np.random.choice(states_values, p=states_trans_matrix[current_state])


    def get_markov_sequence(self, rule_init):
        '''Clems function to generate a Markov sequence based on the diagonal of the transition matrix,
        t_r3 = transition probability from rules one and two to rule w/o deviant, 
        t_r3_r3 is probability of trial w/0 deviant to repeat'''

        ii = np.array([
            [self.params["diag"], 1 - self.params["diag"] - self.params["tr_3"], self.params["tr_3"]],
            [1 - self.params["diag"] - self.params["tr_3"], self.params["diag"], self.params["tr_3"]],
            [(1 - self.params["tr_3_3"])/2, (1 - self.params["tr_3_3"])/2, self.params["tr_3_3"]]
        ])

        rules_seq = np.zeros(self.params["n_trials"], dtype = int)
        rules_seq[0] = int(rule_init)

        for r in range(1, self.params["n_trials"]):

            rules_seq[r] = self.sample_next_markov_state(
                    current_state=rules_seq[r - 1],
                    states_values=range(len(self.params["rules"])),
                    states_trans_matrix=ii,
                )

        return rules_seq, ii


    def compute_stationary(self, ii):
        '''compute the stationary of a given transition matix'''

        # from: https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b
        transition_matrix = ii
        transition_matrix_transp = transition_matrix.T
        eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
        close_to_1_idx = np.isclose(eigenvals,1)
        target_eigenvect = eigenvects[:,close_to_1_idx]
        target_eigenvect = target_eigenvect[:,0]
        stationary_distrib = target_eigenvect / sum(target_eigenvect)

        return stationary_distrib


    def prep_dirs(self):
        '''create output directories'''

        os.makedirs('trial_lists/', exist_ok=True)

        sub_dir = f"trial_lists/sub-{self.params['participant_nr']}"
        os.makedirs(sub_dir, exist_ok=True)

        plot_dir = f"trial_lists/sub-{self.params['participant_nr']}/plots"
        os.makedirs(plot_dir, exist_ok=True)

                                                                                                                                                                                                                                                                                                                            
    def get_task_structure(self):
        '''create a manually predefined order of mus and taus,
        NOTE: only roughly balanced for 6 sessions, might need change'''

        mu_tones = []

        for rep in range(self.params["n_sessions"]):
            row = []

            for i, _ in enumerate(self.params["taus"]):

                if (rep + i) % 2 == 0:
                    row.append(self.params["frequencies"][0])
                else:
                    row.append(self.params["frequencies"][1])

            mu_tones.append(row)

        mu_tones = np.array(mu_tones)

        tau_std_ind = np.array([[0, 1, 2, 3],
                                [3, 2, 1, 0],
                                [1, 3, 0, 2],
                                [3, 1, 2, 0],
                                [2, 3, 0, 1],
                                [1, 0, 3, 2]])
        
        tau_std = [[]]*self.params["n_sessions"]

        for rep in range(self.params["n_sessions"]):
                
                tau_row = [self.params["taus"][i] for i in tau_std_ind[rep]]  # map indices to values
                tau_std[rep]=tau_row

        tau_std = np.array(tau_std)
        perm = np.random.permutation(len(mu_tones)) # shuffle sessions
        perm = np.array(range(self.params["n_sessions"])) # for test reasons keep order
        mu_tones = mu_tones[perm].tolist()
        tau_std = tau_std[perm]

        if self.params["dev_t_tones"] == False:
            tau_dev = (tau_std // 8).astype(int)
        else:
            tau_dev = tau_std.copy()

        return mu_tones, tau_std, tau_dev
    

    def apply_kalman(self, s, obs, dpos_seq_long_flat_full, si_q_arr, mu_tones, tau_std, figy, axy, n_iter, tau_inits):
        '''applies kalman filter separately for each run of each session'''

        run_length = self.params["n_trials"] * 8
        n_runs = 4
        
        obs_std = obs.copy() # use the generated observations
        obs_std[dpos_seq_long_flat_full == 1] = np.nan # we don't observe the standard in the position of the deviant

        iter = -1 

        for n in self.params["n_kalman"]:    

            iter+= 1    
            b_estimated_values = []

            for run in range(n_runs):

                tau_init = None
                b_init = None

                start_idx = run * run_length # get start of each run
                end_idx = start_idx + n # only take n onservations at the beginning of each run
                
                measurements = obs_std[start_idx:end_idx] # standard observations (incl. deviant position as nan)

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
                R_true = np.array([[self.params["si_r"]**2]])  # 1D observation noise

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
            axy[iter, s].set_ylim(500, 1200)
            axy[iter, s].set_yticks(np.arange(500, 1201, 100)) 
            axy[iter, s].grid(True, alpha=0.3)
        

    def plot_kalman(self, figy, axy): 
        '''plots kalman results across sessions and n'''

        # just adding some labels and plot save the full plot across sessions
        for n in range(len(self.params["n_kalman"])):
            axy[n, 0].set_ylabel(f'n = {self.params["n_kalman"][n]}', fontsize=12, weight='bold')
            
        for ses in range(0, self.params["n_sessions"]):
            axy[0, ses].set_title(f'session {ses+1}', fontsize=12, weight='bold')
            axy[-1, ses].set_xlabel(f'tau', fontsize=12, weight='bold')


        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Estimated b',
                markerfacecolor='red', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='True mean',
                markerfacecolor='blue', markersize=6)
        ]

        figy.legend(handles=legend_elements, loc='upper right', fontsize=10) 
        figy.set_size_inches(25, 12) 
        
        plt.tight_layout()
        plt.savefig(f"trial_lists/sub-{self.params['participant_nr']}/plots/kalman_results.png", dpi=300, bbox_inches='tight')
        plt.show()      


    def generate_sessions(self):
        '''script for generating the trial sequence'''

        self.prep_dirs()

        mu_tones, tau_std, tau_dev = self.get_task_structure()  

        figy, axy = plt.subplots(7, self.params["n_sessions"], squeeze=False)

        ###------------ start generate the session

        for s in range(0, self.params["n_sessions"]):

            session_nr = str(s + 1).zfill(2)
            print(f"Generating session: {session_nr}")

            lim_std_log = np.repeat([mu_tones[s][i][0] for i in range(0,len(mu_tones[s]))],self.params["n_trials"]*8)
            lim_dev_log = np.repeat([mu_tones[s][i][1] for i in range(0,len(mu_tones[s]))],self.params["n_trials"]*8)   

            ###------------ find a rule sequence that is roughly balanced

            if self.params["t_tones"] == True:
                if self.params["dev_t_tones"] == True:
                    states = {
                        i: {
                            int(c): np.zeros(int(self.params["n_trials"] * 8))
                            for c in range(len(self.params["contexts"]))
                        }
                        for i in range(len(tau_std))
                    }
                elif self.params["dev_t_tones"] == False:
                    states = {
                        i: {
                            0: np.zeros(int(self.params["n_trials"] * 8)),
                            1: np.zeros(int(self.params["n_trials"]))
                        }
                        for i in range(len(tau_std))
                    }
            elif self.params["t_tones"] == False:
                states = {
                i: {
                    int(c): np.zeros(int(self.params["n_trials"]))
                    for c in range(len(self.params["contexts"]))
                }
                for i in range(len(tau_std))
                }

            dpos_seq_full = []
            dpos_seq_long_flat_full = []
            rules_seq_full = []
            rule_init = np.random.choice(self.params["rules"][0:-1]) # draw random rule to start

            dev_lim = [] # to collect dev states in case of dev_process = False
            proc_noise = [[[], []] for _ in range(len(tau_std))]

            si_q_arr = []
            si_q_dev_arr = []

            for i in range(0, len(tau_std[s])): # for each tau create a sequence of rules

                # compute sigma q for the standard based on current tau and stationary standard deviation
                si_q = (self.params["stat_std"]*((2 * tau_std[s][i] - 1) ** 0.5))/tau_std[s][i]
                si_q_arr.append(si_q)
                
                # create markov sequences until result is roughly balanced
                unbalanced = True
                while unbalanced == True:

                    rules_seq, ii = self.get_markov_sequence(rule_init)
                    p_r1 = sum(rules_seq == 0)/len(rules_seq)
                    p_r2 = sum(rules_seq == 1)/len(rules_seq)
                    p_r3 = sum(rules_seq == 2)/len(rules_seq)
                    p_rules = np.array([p_r1, p_r2])

                    stationary_distrib = self.compute_stationary(ii)

                    if np.any((p_rules < (stationary_distrib[0]-0.02)) | (p_rules > (stationary_distrib[0]+0.02))) or p_r3 < (stationary_distrib[2]-0.02) or p_r3 > (stationary_distrib[2]+0.02): 
                        # allow for deviations of max. 2% in both directions from the stationary
                        continue
                    else:
                        unbalanced = False
                        rules_seq_full.append(rules_seq)
                        rule_init = np.random.choice(self.params["rules"][0:-1])
                        # rule_init = rules_seq[-1] # save last rule to continue Markov provess across all trials --> comment out to initialize randomly in each block
                        
                        #if i != len(tau_std[s])-1:
                        #    print(rule_init)
                        

                ###------------ distribute deviant positions equally for each rule and overall

                dpos_task = [self.params["dpos"][0] * int(((sum(rules_seq == 0))-(sum(rules_seq == 0)%3))/3), self.params["dpos"][1] * int(((sum(rules_seq == 1))-(sum(rules_seq == 1)%3))/3), self.params["dpos"][2] * int(((sum(rules_seq == 2))-(sum(rules_seq == 2)%3))/3)]
                
                # spread deviant positions equally as far as possible, for the remainder draw randomly
                dpos_seq = np.zeros(self.params["n_trials"])

                for r in self.params["rules"]:
                    indices = np.where(rules_seq == r)[0]

                    for t in indices:

                        if len(dpos_task[r]) == 0:
                            dpos_seq[t] = np.random.choice(self.params["dpos"][r])
                            continue
                        choice = np.random.choice(dpos_task[r])
                        dpos_seq[t] = choice
                        dpos_task[r].remove(int(choice))

                dpos_seq_long = []

                for d in range(len(dpos_seq)):

                    if dpos_seq[d] != 0:
                        trial = [0]*8
                        trial[int(dpos_seq[d])] = 1
                        dpos_seq_long.append(trial)
                    else:
                        trial = [0]*8
                        dpos_seq_long.append(trial)

                dpos_seq_long_flat = [x for sublist in dpos_seq_long for x in sublist]

                dpos_seq_long_flat_full.append(dpos_seq_long_flat)
                dpos_seq_full.append(dpos_seq)

                ###------------ create dynamics for the standard and deviant process                    

                mu_std_dev = np.array(mu_tones[s][i])
                dev_lim.append(mu_std_dev[1])


                for c in range(len(self.params["contexts"])): 

                    # this would match the stationary standard deviation of the deviant to that of the standard in case of different taus
                    if self.params["t_tones"] == True and self.params["dev_t_tones"] == False:
                        # compute sigma q deviant based on tau and stationary
                        si_q_dev = ((si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5))*((2*tau_dev[s][i] - 1)**0.5))/tau_dev[s][i]
                        # print(f"sigma_q deviant: {si_q_dev}")
                    else:
                        si_q_dev = si_q

                    if c == 0:
                        states[i][c][0] = np.array(ss.norm.rvs(mu_std_dev[c], si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5))) # std
                        # print(f"stationary std standard: {si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5)}")
                    elif c == 1:
                        states[i][c][0] = np.array(ss.norm.rvs(mu_std_dev[c], si_q_dev * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5))) # dev
                        # print(f"stationary std deviant: {si_q_dev * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5)}")
                        si_q_dev_arr.append(si_q_dev)

                    if self.params["t_tones"] == True:

                        if self.params["dev_t_tones"] == True:
                            w = np.array(ss.norm.rvs(0, si_q_dev, int(self.params["n_trials"]*8)-1))
                            proc_noise[i][c] = w
                            proc_noise[i][c] = np.insert(proc_noise[i][c], 0, np.nan)
                            for t in range(1, int(self.params["n_trials"]*8)): 
                                states[i][c][t] = states[i][c][t - 1] + 1 / tau_std[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w[t - 1]
                        elif self.params["dev_t_tones"] == False: # this is what we want: std on every tone, dev once per trial (including omissions), tau_dev = tau_std/8
                            if c == 0:
                                w_std = np.array(ss.norm.rvs(0, si_q, int(self.params["n_trials"]*8)-1))
                                proc_noise[i][0] = w_std
                                proc_noise[i][0] = np.insert(proc_noise[i][0], 0, np.nan)
                                for t in range(1, int(self.params["n_trials"]*8)): 
                                    states[i][c][t] = states[i][c][t - 1] + 1 / tau_std[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w_std[t - 1]
                                    #print(states[i][c][t - 1])
                                    #print(tau_std[s][i])
                                    #print(w_std[t - 1])
                            elif c == 1:
                                w_dev = np.array(ss.norm.rvs(0, si_q_dev, int(self.params["n_trials"])-1))
                                proc_noise[i][1] = w_dev
                                proc_noise[i][1] = np.insert(proc_noise[i][1], 0, np.nan)
                                proc_noise[i][1] = np.repeat(proc_noise[i][1],8)
                                for t in range(1, int(self.params["n_trials"])): 
                                    states[i][c][t] = states[i][c][t - 1] + 1 / tau_dev[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w_dev[t - 1]        
                    elif self.params["t_tones"] == False: # here, we should use tau dev also for standards
                        w = np.array(ss.norm.rvs(0, si_q, int(self.params["n_trials"])-1))
                        proc_noise[i][c] = w
                        proc_noise[i][c] = np.insert(proc_noise[i][c], 0, np.nan)
                        proc_noise[i][c] = np.repeat(proc_noise[i][c],8)
                        for t in range(1, int(self.params["n_trials"])): 
                            states[i][c][t] = states[i][c][t - 1] + 1 / tau_dev[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w[t - 1]

                # compute effect size SNR
                # NOTE: this could be used as an input parameter to sample mus?
                d_eff = (math.log(mu_std_dev[1])-math.log(mu_std_dev[0]))/self.params["stat_std"]
            
            
            if self.params["t_tones"] == True:
                if self.params["dev_t_tones"] == True:
                    std = [states[0][0].tolist(), states[1][0].tolist(), states[2][0].tolist(), states[3][0].tolist()]
                    std = np.concatenate(std).tolist()
                    dev = [states[0][1].tolist(), states[1][1].tolist(), states[2][1].tolist(), states[3][1].tolist()]
                    dev= np.concatenate(dev).tolist()
                elif self.params["dev_t_tones"] == False:
                    std = [states[0][0].tolist(), states[1][0].tolist(), states[2][0].tolist(), states[3][0].tolist()]
                    std = np.concatenate(std).tolist()
                    dev = [np.repeat(states[0][1],8).tolist(), np.repeat(states[1][1],8).tolist(), np.repeat(states[2][1],8).tolist(), np.repeat(states[3][1],8).tolist()]
                    dev= np.concatenate(dev).tolist()
            elif self.params["t_tones"] == False:
                std = [np.repeat(states[0][0],8).tolist(), np.repeat(states[1][0],8).tolist(), np.repeat(states[2][0],8).tolist(), np.repeat(states[3][0],8).tolist()]
                std = np.concatenate(std).tolist()
                dev = [np.repeat(states[0][1],8).tolist(), np.repeat(states[1][1],8).tolist(), np.repeat(states[2][1],8).tolist(), np.repeat(states[3][1],8).tolist()]
                dev= np.concatenate(dev).tolist()


            if self.params["dev_process"] == False:
                dev = [item for item in dev_lim for _ in range(int(len(dev)/len(tau_dev[s])))]    


            dpos_seq_long_flat_full = np.concatenate(dpos_seq_long_flat_full)
            dpos_seq_full = np.concatenate(dpos_seq_full)
            rules_seq_full = np.concatenate(rules_seq_full)
            rules_long = np.repeat(rules_seq_full, 8)

            tau_std_seq = [[tau_std[s][x]]*len(states[x][0]) for x in range(len(tau_std[s]))]
            tau_std_seq = np.concatenate(tau_std_seq)

            tau_dev_seq = [[tau_dev[s][x]]*len(states[x][0]) for x in range(len(tau_dev[s]))]
            tau_dev_seq = np.concatenate(tau_dev_seq)


            ###------------ plot states across all tones

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

            if self.params["t_tones"] == True:
                for k, r in enumerate(tau_std_seq):
                    ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)
            elif self.params["t_tones"] == False:
                tau_seq_plot = np.repeat(tau_std_seq, 8)
                for k, r in enumerate(tau_seq_plot):
                    ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)      

            tau_legend_patches = [
            mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
            ]

            quarter_width = self.params["n_trials"]*8
            xmin = 0
            xmax = self.params["n_trials"]*8*4

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
                        # print(stationary_std)
                    else:
                        stationary_std = si_q_dev_arr[i] * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5)
                        # print(stationary_std)
                    lbl = f"mu {self.params["contexts"][ind_col]}" if i == 0 else None
                    lbl2 = f"stationary std {self.params["contexts"][ind_col]}" if i == 0 else None
                    plt.hlines(y=val, xmin=q_start, xmax=q_end,
                            colors=colors[ind_col], linestyles='--', linewidth=0.5, label=lbl)
                    plt.fill_between(np.linspace(q_start, q_end, self.params["n_trials"]),
                        val - stationary_std,
                        val + stationary_std,
                        color=colors[ind_col], alpha=0.2, label = lbl2)

            ax.set_xlabel('tone')
            ax.set_ylabel('state value')
            ax.set_title(f"Linear Gaussian Dynamics for States of Standard and Deviant across Tones (Session {session_nr})")
            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=tau_legend_patches + handles, loc='upper right')
            plt.tight_layout()
            plt.savefig(f"trial_lists/sub-{self.params['participant_nr']}/plots/lgd_std_dev_session_{session_nr}.png", dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close()

            ###------------  if t_tones == FALSE: plot states also across trials

            if self.params["t_tones"] == False:
                std_short = std[::8]
                dev_short = dev[::8]

                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(std_short, label='standard', color='blue')
                ax.plot(dev_short, label='deviant', color='red')

                for j in range(len(rules_seq_full)):

                    if rules_seq_full[j] == 0:
                        ax.axvspan(j - 0.5, j + 0.5, color='lightblue', alpha=0.1)
                    elif rules_seq_full[j] == 1:
                        ax.axvspan(j - 0.5, j + 0.5, color='lightcoral', alpha=0.1)
                    elif rules_seq_full[j] == 2:
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
                mpatches.Patch(color=clr, label=f'τ = {t}') for t, clr in colors_tau.items()
                ]    

                ax.set_xlabel('tone')
                ax.set_ylabel('state value')
                ax.set_title('Linear Gaussian Dynamics for States of Standard and Deviant across Trials')
                ax.legend()
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=tau_legend_patches + handles, loc='upper right')
                plt.tight_layout()
                #plt.show()

            ###------------ create observations by adding observation noise

            obs = dpos_seq_long_flat_full
            obs = np.array(obs)
            v = ss.norm.rvs(0, self.params["si_r"], self.params["n_trials"]*8*len(tau_std[s])) # create noise across all tones
            v = np.array(v)

            dev = np.array(dev)
            std = np.array(std)

            obs = obs.astype(float)

            indices_dev = np.where(obs == 1)[0]
            indices_std = np.where(obs == 0)[0]

            obs[indices_dev] = dev[indices_dev]
            obs[indices_std] = std[indices_std]

            obs = obs + v

            obs_noise = v

            ###------------ plot observations

            fig, ax = plt.subplots(figsize=(20, 5))
            ax.plot(obs, label='sound observation', color='blue')
            plt.scatter(indices_dev, obs[indices_dev], color='red', label='deviant')


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

            if self.params["t_tones"] == True:
                for k, r in enumerate(tau_std_seq):
                    ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)
            elif self.params["t_tones"] == False:
                tau_seq_plot = np.repeat(tau_std_seq, 8)
                for k, r in enumerate(tau_seq_plot):
                    ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)  

            tau_legend_patches = [
            mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
            ]

            ax.set_xlabel('tone')
            ax.set_ylabel('observation value')
            ax.set_title(f"Sound Observations (Session {session_nr})")
            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=tau_legend_patches + handles, loc='upper right')

            ymin, ymax = plt.ylim()

            dpos_seq_no_zero = dpos_seq_full[dpos_seq_full != 0]

            for i in range(0, len(indices_dev)):
                plt.text(indices_dev[i], ymax-20, f'{int(dpos_seq_no_zero[i])}', fontsize=6, ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f"trial_lists/sub-{self.params['participant_nr']}/plots/observations_std_dev_session_{session_nr}.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

            ###------------ create probabilities overall and per tau 

            p_1 = sum(rules_seq_full == 0)/len(rules_seq_full)
            p_2 = sum(rules_seq_full == 1)/len(rules_seq_full)
            p_3 = sum(rules_seq_full == 2)/len(rules_seq_full)

            p_d3_r1 = np.where((dpos_seq_full == 2)&(rules_seq_full==0))[0].size/np.sum(rules_seq_full == 0)
            p_d4_r1 = np.where((dpos_seq_full == 3)&(rules_seq_full==0))[0].size/np.sum(rules_seq_full == 0)
            p_d5_r1 = np.where((dpos_seq_full == 4)&(rules_seq_full==0))[0].size/np.sum(rules_seq_full == 0)

            p_d5_r3 = np.where((dpos_seq_full == 4)&(rules_seq_full==1))[0].size/np.sum(rules_seq_full == 1)
            p_d6_r3 = np.where((dpos_seq_full == 5)&(rules_seq_full==1))[0].size/np.sum(rules_seq_full == 1)
            p_d7_r3 = np.where((dpos_seq_full == 6)&(rules_seq_full==1))[0].size/np.sum(rules_seq_full == 1)

            ###------------ plot probabilities overall

            x_label = ['P(rule 1)', 'P(rule 3)','P(no dev)', 'P(dev3|rule1)','P(dev4|rule1)','P(dev5|rule1)','P(dev5|rule3)','P(dev6|rule3)','P(dev7|rule3)']
            y_values = [p_1, p_2, p_3, p_d3_r1, p_d4_r1, p_d5_r1, p_d5_r3, p_d6_r3, p_d7_r3]

            colors = ['lightblue', 'lightcoral', 'grey','lightblue','lightblue','lightblue','lightcoral','lightcoral','lightcoral']

            plt.figure(figsize=(15, 7))
            plt.bar(x_label, y_values, color=colors)

            ymin, ymax = plt.ylim()

            n_trials_rule = [sum(rules_seq_full == 0), sum(rules_seq_full == 1), sum(rules_seq_full == 2), sum(dpos_seq_full[np.where(rules_seq_full == 0)]==2), sum(dpos_seq_full[np.where(rules_seq_full == 0)]==3), sum(dpos_seq_full[np.where(rules_seq_full == 0)]==4),
                            sum(dpos_seq_full[np.where(rules_seq_full == 1)]==4), sum(dpos_seq_full[np.where(rules_seq_full == 1)]==5), sum(dpos_seq_full[np.where(rules_seq_full == 1)]==6)]

            for i in range(0, len(colors)):
                plt.text(i, ymax-0.01, f'N = {int(n_trials_rule[i])}', fontsize=6, ha='center', va='bottom')

            plt.xlabel('')
            plt.ylabel('P')
            plt.title(f"Probabilities (session {session_nr})")

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.yticks(np.arange(0, 0.47, 0.01))
            plt.tight_layout()
            plt.savefig(f"trial_lists/sub-{self.params['participant_nr']}/plots/probs_overall_session_{session_nr}.png", dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close()

            ###------------ plot probabilities per tau

            for tau_i in range(0,len(tau_std[s])):

                if self.params["t_tones"] == False:
                    ind_tau = np.where(tau_std_seq == tau_std[s][tau_i])
                elif self.params["t_tones"] == True:
                    tau_seq_trials = tau_std_seq[::8]
                    ind_tau = np.where(tau_seq_trials == tau_std[s][tau_i])
                
                p_1_tau = sum(rules_seq_full[ind_tau] == 0)/len(rules_seq_full[ind_tau])
                p_2_tau = sum(rules_seq_full[ind_tau] == 1)/len(rules_seq_full[ind_tau])
                p_3_tau = sum(rules_seq_full[ind_tau] == 2)/len(rules_seq_full[ind_tau])

                p_d3_r1_tau = np.where((dpos_seq_full[ind_tau] == 2)&(rules_seq_full[ind_tau]==0))[0].size/np.sum(rules_seq_full[ind_tau] == 0)
                p_d4_r1_tau = np.where((dpos_seq_full[ind_tau] == 3)&(rules_seq_full[ind_tau]==0))[0].size/np.sum(rules_seq_full[ind_tau] == 0)
                p_d5_r1_tau = np.where((dpos_seq_full[ind_tau] == 4)&(rules_seq_full[ind_tau]==0))[0].size/np.sum(rules_seq_full[ind_tau] == 0)

                p_d5_r3_tau = np.where((dpos_seq_full[ind_tau] == 4)&(rules_seq_full[ind_tau]==1))[0].size/np.sum(rules_seq_full[ind_tau] == 1)
                p_d6_r3_tau = np.where((dpos_seq_full[ind_tau] == 5)&(rules_seq_full[ind_tau]==1))[0].size/np.sum(rules_seq_full[ind_tau] == 1)
                p_d7_r3_tau = np.where((dpos_seq_full[ind_tau] == 6)&(rules_seq_full[ind_tau]==1))[0].size/np.sum(rules_seq_full[ind_tau] == 1)

                x_label = ['P(rule 1)', 'P(rule 3)','P(no dev)', 'P(dev3|rule1)','P(dev4|rule1)','P(dev5|rule1)','P(dev5|rule3)','P(dev6|rule3)','P(dev7|rule3)']
                y_values = [p_1, p_2, p_3, p_d3_r1, p_d4_r1, p_d5_r1, p_d5_r3, p_d6_r3, p_d7_r3]

                colors = ['lightblue', 'lightcoral', 'grey','lightblue','lightblue','lightblue','lightcoral','lightcoral','lightcoral']

                plt.figure(figsize=(15, 7))
                plt.bar(x_label, y_values, color=colors)

                ymin, ymax = plt.ylim()

                n_trials_rule = [sum(rules_seq_full[ind_tau] == 0), sum(rules_seq_full[ind_tau] == 1), sum(rules_seq_full[ind_tau] == 2), sum(dpos_seq_full[ind_tau][np.where(rules_seq_full[ind_tau] == 0)]==2), sum(dpos_seq_full[ind_tau][np.where(rules_seq_full[ind_tau] == 0)]==3), sum(dpos_seq_full[ind_tau][np.where(rules_seq_full [ind_tau]== 0)]==4),
                            sum(dpos_seq_full[ind_tau][np.where(rules_seq_full[ind_tau] == 1)]==4), sum(dpos_seq_full[ind_tau][np.where(rules_seq_full[ind_tau] == 1)]==5), sum(dpos_seq_full[ind_tau][np.where(rules_seq_full[ind_tau] == 1)]==6)]

                for i in range(0, len(colors)):
                    plt.text(i, ymax-0.01, f'N = {int(n_trials_rule[i])}', fontsize=6, ha='center', va='bottom')

                plt.xlabel('')
                plt.ylabel('P')
                plt.title(f'Probabilities tau std = {tau_std[s][tau_i]} (Session {session_nr}, Run {tau_i + 1})')

                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.yticks(np.arange(0, 0.47, 0.01))
                plt.tight_layout()
                plt.savefig(f"trial_lists/sub-{self.params['participant_nr']}/plots/probs_session_{session_nr}_run_{tau_i + 1}.png", dpi=300, bbox_inches='tight')
                #plt.show()
                plt.close()

            ###------------ compute transision probabilities in final sequence

            t_1_1 = 0
            t_1_2 = 0
            t_1_3 = 0
            t_2_3 = 0
            t_3_1 = 0
            t_3_2 = 0
            t_2_1 = 0
            t_2_2 = 0
            t_3_3 = 0

            for r in range(1,len(rules_seq_full)):
                if rules_seq_full[r] == 0 and rules_seq_full[r-1] == 0:
                    t_1_1 += 1
                elif rules_seq_full[r] == 1 and rules_seq_full[r-1] == 0:
                    t_1_2 += 1
                elif rules_seq_full[r] == 2 and rules_seq_full[r-1] == 0:
                    t_1_3 += 1
                elif rules_seq_full[r] == 2 and rules_seq_full[r-1] == 1:    
                    t_2_3 += 1
                elif rules_seq_full[r] == 0 and rules_seq_full[r-1] == 2:  
                    t_3_1 += 1
                elif rules_seq_full[r] == 1 and rules_seq_full[r-1] == 2:
                    t_3_2 += 1 
                elif rules_seq_full[r] == 0 and rules_seq_full[r-1] == 1:
                    t_2_1 += 1
                elif rules_seq_full[r] == 1 and rules_seq_full[r-1] == 1:
                    t_2_2 += 1
                elif rules_seq_full[r] == 2 and rules_seq_full[r-1] == 2:
                    t_3_3 += 1           

            ###------------ add ITI
            # increased because of the slider ratings in behavioral experiment --> decrease for fmri

            iti_range = np.arange(7, 12, 0.5)
            ITI =[]

            for i in range(0,len(dpos_seq_full)):
                ITI.append(np.random.choice(iti_range))
                            

            ###------------ add all variables relevant to the logfiles and save conditions file

            trials_final = pd.DataFrame(columns=['observation', 'state_std', 'state_dev','lim_std','lim_dev','tau_std','tau_dev','rule','dpos','trial_type','sigma_q_std','sigma_q_dev','sigma_r','noise_v','noise_w_std_t_minus_one','noise_w_dev_t_minus_one','ITI','duration_tones','ISI','trial_n','run_n','session_n'])
            trials_final['observation'] = obs
            trials_final['state_std'] = std
            trials_final['state_dev'] = dev
            trials_final['lim_std'] = lim_std_log
            trials_final['lim_dev'] = lim_dev_log

            if self.params["t_tones"] == True:
                trials_final['tau_std'] = tau_std_seq
                trials_final['tau_dev'] = tau_dev_seq
            elif self.params["t_tones"] == False:
                trials_final['tau_std'] = tau_seq_plot
                trials_final['tau_dev'] = tau_seq_plot

            trials_final['rule'] = rules_long
            trials_final['dpos'] = np.repeat([int(x) for x in dpos_seq_full], 8)
            trials_final['trial_type'] = dpos_seq_long_flat_full
            trials_final['sigma_q_std'] = np.repeat([x for x in si_q_arr], 8*self.params["n_trials"])
            trials_final['sigma_q_dev'] = np.repeat([x for x in si_q_dev_arr], 8*self.params["n_trials"])
            trials_final['sigma_r'] = [self.params["si_r"]]*self.params["n_trials"]*8*len(tau_std[s])
            trials_final['noise_v'] = obs_noise
            trials_final['noise_w_std_t_minus_one'] = np.concatenate([proc_noise[x][0] for x in range(len(tau_std[s]))])
            trials_final['noise_w_dev_t_minus_one'] = np.concatenate([proc_noise[x][1] for x in range(len(tau_std[s]))])
            trials_final['ITI'] = [round(item,2) for item in ITI for _ in range(8)]
            trials_final['duration_tones'] = [self.params["duration_tones"]]*self.params["n_trials"]*8*len(tau_std[s])
            trials_final['ISI'] = [self.params["isi"]]*self.params["n_trials"]*8*len(tau_std[s])
            trials_final['trial_n'] = [i for i in range(self.params["n_trials"]*len(tau_std[s])) for _ in range(8)]
            trials_final['run_n'] = np.repeat([range(0,len(tau_std[s]))], self.params["n_trials"]*8)
            trials_final['session_n'] = [s]*self.params["n_trials"]*8*len(tau_std[s])

            trials_final.to_csv(f'trial_lists/sub-{self.params['participant_nr']}/sub-{self.params['participant_nr']}_ses-{session_nr}_trials.csv', index=False, float_format="%.4f")

            session_duration = (((7*self.params["isi"])+(8*self.params["duration_tones"]))*self.params["n_trials"]*(len(tau_std[s]))) + sum(trials_final['ITI'][::8])
            print(f'Estimated session duration in minutes: {session_duration/60}')

            ###------------ apply adaptation of Alex' script for Kalman filter

            # apply for each session
            self.apply_kalman(s, obs, dpos_seq_long_flat_full, si_q_arr, mu_tones, tau_std, figy, axy, n_iter = 10, tau_inits = 3.0)
            
        # plot across session
        self.plot_kalman(figy, axy)


if __name__ == "__main__":
    
    task = trials_master()

    print("=== Generating Trials ===")
    task.generate_sessions()




                        








