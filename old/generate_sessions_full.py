import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import matplotlib.patches as mpatches
import os

###------------ functions from Cléms script to sample the next Markov state (for the rules)

# compute next Markov state based on transition matrix
def sample_next_markov_state(
        current_state, states_values, states_trans_matrix
    ):
        return np.random.choice(states_values, p=states_trans_matrix[current_state])

# get the full Markov sequence
def get_markov_sequence(diag, t_r3, n_trials, rule_init):

    # as discussed with Alex
    ii = np.array([
        [diag, 1 - diag - t_r3, t_r3],
        [1 - diag - t_r3, diag, t_r3],
        [(1 - t_r3_r3)/2, (1 - t_r3_r3)/2, t_r3_r3]
    ])

    rules_seq = np.zeros(n_trials, dtype = int)
    rules_seq[0] = int(rule_init)

    for r in range(1, n_trials):
        rules_seq[r] = sample_next_markov_state(
                current_state=rules_seq[r - 1],
                states_values=range(len(rules)),
                states_trans_matrix=ii,
            )

    return rules_seq, ii

# compute the stationary of the transition matrix
def compute_stationary(ii):
    # from: https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b
    transition_matrix = ii
    transition_matrix_transp = transition_matrix.T
    eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
    close_to_1_idx = np.isclose(eigenvals,1)
    target_eigenvect = eigenvects[:,close_to_1_idx]
    target_eigenvect = target_eigenvect[:,0]
    stationary_distrib = target_eigenvect / sum(target_eigenvect)

    return stationary_distrib

###------------ setup

t_tones = True # False: time steps for LGD = trials, True: time steps for LGD = tones
dev_t_tones = False # True: deviant process time steps = tones, False: deviant process time steps = trials
dev_process = True # True: LGD for deviant, False: no dynamic for deviant (just observation noise)

participant_nr = 'test' # participant number

sub_dir = f"trial_lists_full/sub-{participant_nr}"
os.makedirs(sub_dir, exist_ok=True)

plot_dir = f"trial_lists_full/sub-{participant_nr}/plots"
os.makedirs(plot_dir, exist_ok=True)

rules = [0, 1, 2] 
contexts = ['std', 'dev']

# possible implementation for roughly balanced design
mu_tones = np.array([[[1000, 1050],[700, 750],[1000, 1050],[700, 750]],
                    [[1000, 1050],[700, 750],[1000, 1050],[700, 750]],
                    [[700, 750],[1000, 1050],[700, 750],[1000, 1050]],
                    [[700, 750],[1000, 1050],[700, 750],[1000, 1050]],
                    [[1000, 1050],[700, 750],[1000, 1050],[700, 750]],
                    [[1000, 1050],[700, 750],[1000, 1050],[700, 750]]])

tau_std = np.array([[16, 40, 160, 240],
                    [240, 160, 40, 16],
                    [40, 240, 16, 160],
                    [240, 40, 160, 16],
                    [160, 240, 16, 40],
                    [40, 16, 240, 160]])

#perm = np.random.permutation(len(mu_tones)) # shuffle sessions
perm = np.array(range(6)) # for test reasons keep order

mu_tones = mu_tones[perm].tolist()
tau_std = tau_std[perm]

if dev_t_tones == False:
    tau_dev = (tau_std // 8).astype(int)
else:
    tau_dev = tau_std.copy()

n_sessions = len(mu_tones)

for s in range(0, n_sessions):

    session_nr = str(s + 1).zfill(2)
    print(f"Generating session: {session_nr}")

    si_r = 10 # set observation noise to 10 Hz 
    si_q = 10 # TBD, only for test --> best to pilot (10 seems quite high but still okay? but depending on mu for std and dev) 

    n_trials = 60 # 60 trials per tau  -> 240 trials in total which should be reasonable in like 45 mins
    n_rules = len(rules)
    n_contexts = len(contexts)

    lim_std_log = np.repeat([mu_tones[s][i][0] for i in range(0,len(mu_tones[s]))],n_trials*8)
    lim_dev_log = np.repeat([mu_tones[s][i][1] for i in range(0,len(mu_tones[s]))],n_trials*8)   

    diag = 0.80 # relatively sticky --> could be piloted and changed
    t_r3 = 0.1 # relatively rare --> could be piloted and changed
    t_r3_r3 = 0 # not ever happening

    duration_tones = 0.1 # tones for 100 ms
    isi = 0.65 # 650 ms ISI

    ###------------ find a rule sequence that is roughly balanced

    if t_tones == True:

        if dev_t_tones == True:
            states = {
                i: {
                    int(c): np.zeros(int(n_trials * 8))
                    for c in range(len(contexts))
                }
                for i in range(len(tau_std))
            }
        elif dev_t_tones == False:
            states = {
                i: {
                    0: np.zeros(int(n_trials * 8)),
                    1: np.zeros(int(n_trials))
                }
                for i in range(len(tau_std))
            }


    elif t_tones == False:
        states = {
        i: {
            int(c): np.zeros(int(n_trials))
            for c in range(len(contexts))
        }
        for i in range(len(tau_std))
        }

    dpos_seq_full = []
    dpos_seq_long_flat_full = []
    rules_seq_full = []
    rule_init = np.random.choice(rules[0:-1]) # draw random rule to start
    # print(rule_init)

    dev_lim = [] # to collect dev states in case of dev_process = False
    proc_noise = [[[], []] for _ in range(len(tau_std))]

    for i in range(0, len(tau_std[s])): # for each tau create a sequence of rules
        
        unbalanced = True
        while unbalanced == True:

            rules_seq, ii = get_markov_sequence(diag, t_r3, n_trials, rule_init)
            p_r1 = sum(rules_seq == 0)/len(rules_seq)
            p_r2 = sum(rules_seq == 1)/len(rules_seq)
            p_r3 = sum(rules_seq == 2)/len(rules_seq)
            p_rules = np.array([p_r1, p_r2])

            stationary_distrib = compute_stationary(ii)

            if np.any((p_rules < (stationary_distrib[0]-0.02)) | (p_rules > (stationary_distrib[0]+0.02))) or p_r3 < (stationary_distrib[2]-0.02) or p_r3 > (stationary_distrib[2]+0.02): 
                # allow for deviations of max. 2% in both directions from the stationary
                continue
            
            else:
                unbalanced = False
                rules_seq_full.append(rules_seq)
                rule_init = np.random.choice(rules[0:-1])
                # rule_init = rules_seq[-1] # save last rule to continue Markov provess across all trials --> comment out to initialize randomly in each block
                
                #if i != len(tau_std[s])-1:
                #    print(rule_init)
                

        ###------------ distribute deviant positions equally for each rule and overall

        dpos = [[2,3,4],[4,5,6],[0]]
        dpos_task = [dpos[0] * int(((sum(rules_seq == 0))-(sum(rules_seq == 0)%3))/3), dpos[1] * int(((sum(rules_seq == 1))-(sum(rules_seq == 1)%3))/3), dpos[2] * int(((sum(rules_seq == 2))-(sum(rules_seq == 2)%3))/3)]
        
        # spread deviant positions equally as far as possible, for the remainder draw randomly
        dpos_seq = np.zeros(n_trials)

        for r in rules:
            indices = np.where(rules_seq == r)[0] 
            for t in indices:
                if len(dpos_task[r]) == 0:
                    dpos_seq[t] = np.random.choice(dpos[r])
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


        for c in range(len(contexts)): 

            # this would match the stationary standard deviation of the deviant to that of the standard in case of different taus
            if t_tones == True and dev_t_tones == False:
                si_q_dev = ((si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5))*((2*tau_dev[s][i] - 1)**0.5))/tau_dev[s][i]
                print(f"sigma_q deviant: {si_q_dev}")
            else:
                si_q_dev = si_q

            if c == 0:
                states[i][c][0] = np.array(ss.norm.rvs(mu_std_dev[c], si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5))) # std
                print(f"stationary std standard: {si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5)}")
            elif c == 1:
                states[i][c][0] = np.array(ss.norm.rvs(mu_std_dev[c], si_q_dev * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5))) # dev
                print(f"stationary std deviant: {si_q_dev * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5)}")

            if t_tones == True:

                if dev_t_tones == True:
                    w = np.array(ss.norm.rvs(0, si_q_dev, int(n_trials*8)-1))
                    proc_noise[i][c] = w
                    proc_noise[i][c] = np.insert(proc_noise[i][c], 0, np.nan)
                    for t in range(1, int(n_trials*8)): 
                        states[i][c][t] = states[i][c][t - 1] + 1 / tau_std[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w[t - 1]

                elif dev_t_tones == False: # this is what we want: std on every tone, dev once per trial (including omissions), tau_dev = tau_std/8
                    if c == 0:
                        w_std = np.array(ss.norm.rvs(0, si_q, int(n_trials*8)-1))
                        proc_noise[i][0] = w_std
                        proc_noise[i][0] = np.insert(proc_noise[i][0], 0, np.nan)
                        for t in range(1, int(n_trials*8)): 
                            states[i][c][t] = states[i][c][t - 1] + 1 / tau_std[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w_std[t - 1]
                            #print(states[i][c][t - 1])
                            #print(tau_std[s][i])
                            #print(w_std[t - 1])
                    elif c == 1:
                        w_dev = np.array(ss.norm.rvs(0, si_q_dev, int(n_trials)-1))
                        proc_noise[i][1] = w_dev
                        proc_noise[i][1] = np.insert(proc_noise[i][1], 0, np.nan)
                        proc_noise[i][1] = np.repeat(proc_noise[i][1],8)
                        for t in range(1, int(n_trials)): 
                            states[i][c][t] = states[i][c][t - 1] + 1 / tau_dev[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w_dev[t - 1]        

            elif t_tones == False: # here, we should use tau dev also for standards
                w = np.array(ss.norm.rvs(0, si_q, int(n_trials)-1))
                proc_noise[i][c] = w
                proc_noise[i][c] = np.insert(proc_noise[i][c], 0, np.nan)
                proc_noise[i][c] = np.repeat(proc_noise[i][c],8)
                for t in range(1, int(n_trials)): 
                    states[i][c][t] = states[i][c][t - 1] + 1 / tau_dev[s][i] * (mu_std_dev[c] - states[i][c][t - 1]) + w[t - 1]


    if t_tones == True:
        if dev_t_tones == True:
            std = [states[0][0].tolist(), states[1][0].tolist(), states[2][0].tolist(), states[3][0].tolist()]
            std = np.concatenate(std).tolist()
            dev = [states[0][1].tolist(), states[1][1].tolist(), states[2][1].tolist(), states[3][1].tolist()]
            dev= np.concatenate(dev).tolist()
        elif dev_t_tones == False:
            std = [states[0][0].tolist(), states[1][0].tolist(), states[2][0].tolist(), states[3][0].tolist()]
            std = np.concatenate(std).tolist()
            dev = [np.repeat(states[0][1],8).tolist(), np.repeat(states[1][1],8).tolist(), np.repeat(states[2][1],8).tolist(), np.repeat(states[3][1],8).tolist()]
            dev= np.concatenate(dev).tolist()


    elif t_tones == False:
        std = [np.repeat(states[0][0],8).tolist(), np.repeat(states[1][0],8).tolist(), np.repeat(states[2][0],8).tolist(), np.repeat(states[3][0],8).tolist()]
        std = np.concatenate(std).tolist()
        dev = [np.repeat(states[0][1],8).tolist(), np.repeat(states[1][1],8).tolist(), np.repeat(states[2][1],8).tolist(), np.repeat(states[3][1],8).tolist()]
        dev= np.concatenate(dev).tolist()


    if dev_process == False:
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

    if t_tones == True:
        for k, r in enumerate(tau_std_seq):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)
    elif t_tones == False:
        tau_seq_plot = np.repeat(tau_std_seq, 8)
        for k, r in enumerate(tau_seq_plot):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)      

    tau_legend_patches = [
    mpatches.Patch(color=clr, label=f'τ std = {t}') for t, clr in colors_tau.items()
    ]

    quarter_width = n_trials*8
    xmin = 0
    xmax = n_trials*8*4

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
                stationary_std = si_q * tau_std[s][i] / ((2 * tau_std[s][i] - 1) ** 0.5)
            else:
                stationary_std = si_q_dev * tau_dev[s][i] / ((2 * tau_dev[s][i] - 1) ** 0.5)
            lbl = f"mu {contexts[ind_col]}" if i == 0 else None
            lbl2 = f"stationary std {contexts[ind_col]}" if i == 0 else None
            plt.hlines(y=val, xmin=q_start, xmax=q_end,
                    colors=colors[ind_col], linestyles='--', linewidth=0.5, label=lbl)
            plt.fill_between(np.linspace(q_start, q_end, n_trials),
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
    plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/lgd_std_dev_session_{session_nr}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    ###------------  if t_tones == FALSE: plot states also across trials

    if t_tones == False:

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
    v = ss.norm.rvs(0, si_r, n_trials*8*len(tau_std[s])) # create noise across all tones
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

    if t_tones == True:
        for k, r in enumerate(tau_std_seq):
            ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.05, color=colors_tau[r], alpha=0.9)
    elif t_tones == False:
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
    plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/observations_std_dev_session_{session_nr}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    ###------------ plot rule development across trials

    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax.plot(rules_seq_full, label='rule', color='blue')

    #ax.set_xlabel('trial')
    #ax.set_ylabel('rule')
    #ax.set_title('development rules across trials')
    #plt.tight_layout()
    #plt.show()

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
    plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/probs_overall_session_{session_nr}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    ###------------ plot probabilities per tau

    for tau_i in range(0,len(tau_std[s])):

        if t_tones == False:
            ind_tau = np.where(tau_std_seq == tau_std[s][tau_i])
        elif t_tones == True:
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
        plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/probs_session_{session_nr}_run_{tau_i + 1}.png", dpi=300, bbox_inches='tight')
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

    iti_range = np.arange(7, 12, 0.5) # increased because of the slider
    dev_dist_dev =[]
    dev_dist_s = []
    ITI =[]

    for i in range(0,len(dpos_seq_full)):
        ITI.append(np.random.choice(iti_range))
                    

    ###------------ add all variables relevant to the logfiles and save conditions file

    trials_final = pd.DataFrame(columns=['observation', 'state_std', 'state_dev','lim_std','lim_dev','tau_std','tau_dev','rule','dpos','trial_type','sigma_q','sigma_r','noise_v','noise_w_std_t_minus_one','noise_w_dev_t_minus_one','ITI','duration_tones','ISI','trial_n','run_n','session_n'])
    trials_final['observation'] = obs
    trials_final['state_std'] = std
    trials_final['state_dev'] = dev
    trials_final['lim_std'] = lim_std_log
    trials_final['lim_dev'] = lim_dev_log

    if t_tones == True:
        trials_final['tau_std'] = tau_std_seq
        trials_final['tau_dev'] = tau_dev_seq
    elif t_tones == False:
        trials_final['tau_std'] = tau_seq_plot
        trials_final['tau_dev'] = tau_seq_plot

    trials_final['rule'] = rules_long
    trials_final['dpos'] = np.repeat([int(x) for x in dpos_seq_full], 8)
    trials_final['trial_type'] = dpos_seq_long_flat_full
    trials_final['sigma_q'] = [si_q]*n_trials*8*len(tau_std[s])
    trials_final['sigma_r'] = [si_r]*n_trials*8*len(tau_std[s])
    trials_final['noise_v'] = obs_noise
    trials_final['noise_w_std_t_minus_one'] = np.concatenate([proc_noise[x][0] for x in range(len(tau_std[s]))])
    trials_final['noise_w_dev_t_minus_one'] = np.concatenate([proc_noise[x][1] for x in range(len(tau_std[s]))])
    trials_final['ITI'] = [round(item,2) for item in ITI for _ in range(8)]
    trials_final['duration_tones'] = [duration_tones]*n_trials*8*len(tau_std[s])
    trials_final['ISI'] = [isi]*n_trials*8*len(tau_std[s])
    trials_final['trial_n'] = [i for i in range(n_trials*len(tau_std[s])) for _ in range(8)]
    trials_final['run_n'] = np.repeat([range(0,len(tau_std[s]))], n_trials*8)
    trials_final['session_n'] = [s]*n_trials*8*len(tau_std[s])

    # maybe also record process and observation noise on each tone for sanity check?
    trials_final.to_csv(f'trial_lists_full/sub-{participant_nr}/sub-{participant_nr}_ses-{session_nr}_trials.csv', index=False, float_format="%.4f")

    session_duration = (((7*isi)+(8*duration_tones))*n_trials*(len(tau_std[s]))) + sum(trials_final['ITI'][::8])
    print(f'estimated session duration in minutes: {session_duration/60}')


    ###------------ try Kalman filter
    from pykalman import KalmanFilter
    from numpy import ma

    run_length = n_trials * 8
    n_runs = 4
    obs_std = obs.copy()
    obs_std[dpos_seq_long_flat_full == 1] = np.nan

    obs_dev = obs.copy()
    obs_dev[dpos_seq_long_flat_full == 0] = np.nan

    #std[dpos_seq_long_flat_full == 1] = np.nan
    dev[dpos_seq_long_flat_full == 0] = np.nan
        

    for run in range(n_runs):

        start_idx = run * run_length
        end_idx = (run + 1) * run_length
        
        measurements = obs_std[start_idx:end_idx]
        measurements_dev = obs_dev[start_idx:end_idx]
        std_state = std[start_idx:end_idx]
        dev_state = dev[start_idx:end_idx]
        dev_state = dev_state[~np.isnan(dev_state)]

        measurements = ma.asarray(measurements)
        measurements[np.isnan(measurements)] = ma.masked

        measurements_dev = measurements_dev[~np.isnan(measurements_dev)] # -> here only values treated as one process
        #measurements_dev = ma.asarray(measurements_dev)
        #measurements_dev[np.isnan(measurements_dev)] = ma.masked # --> mask would treat it as missing time steps so probably not good?

        kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])
        kf = kf.em(measurements, n_iter=10)
        print("Initial state mean:", kf.initial_state_mean)
        print("Initial state covariance:", kf.initial_state_covariance)
        print("Transition covariance Q:", kf.transition_covariance)
        print("Observation covariance R:", kf.observation_covariance)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

        kf_dev = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]])
        kf_dev = kf_dev.em(measurements_dev, n_iter=10)
        (filtered_state_means_dev, filtered_state_covariances_dev) = kf_dev.filter(measurements_dev)
        (smoothed_state_means_dev, smoothed_state_covariances_dev) = kf_dev.smooth(measurements_dev)

        plt.figure(figsize=(20, 6))
        plt.plot(std_state, label='state standard',  marker='o', linestyle='', alpha=0.5, color='black')
        plt.plot(measurements, label='observation standard', marker='o', linestyle='', alpha=0.5)
        plt.plot(filtered_state_means, label='filtered estimate', color='blue')
        plt.plot(smoothed_state_means, label='smoothed estimate', color='red')
        plt.legend()
        plt.xlabel('tone')
        plt.ylabel('state (Hz)')
        plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/kalman_std_session_{session_nr}.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 6))
        plt.plot(dev_state, label='state deviant',  marker='o', linestyle='', alpha=0.5, color='black')
        plt.plot(measurements_dev, label='observation deviant', marker='o', linestyle='', alpha=0.5)
        plt.plot(filtered_state_means_dev, label='filtered estimate', color='blue')
        plt.plot(smoothed_state_means_dev, label='smoothed estimate', color='red')
        plt.legend()
        plt.xlabel('tone')
        plt.ylabel('state (Hz)')
        plt.savefig(f"trial_lists_full/sub-{participant_nr}/plots/kalman_dev_session_{session_nr}.png", dpi=300, bbox_inches='tight')
        plt.close()

