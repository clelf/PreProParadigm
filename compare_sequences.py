import numpy as np
import pandas as pd
import random
import json
import glob
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import shutil
from pathlib import Path

#---------------SUBJECT INFORMATION & CONFIG----------------#
subject = '01' # participant number
session = 4 # session number
method = 'kalman'#'kalman' # 'activations_probs'
network = 'population_network_all_bn8_trainh0_lr0.002'#'population_network_all_bn8_trainh0_fixedsir_lr0.001'#'population_network_all_bn8_lr0.001_dposweight'
expo = 2 # exponent for sum measure
take_dpos = False # multiply by dpos prior?
#-----------------------------------------------------------#


#---------------LOAD RUN PARAMETERS--------------------------#

# get assigned experimental design and cue pairing from files
design = f"fMRI/randomizations/task_structure_sub-{subject}.json"
cue_pairings = f"fMRI/randomizations/cue_pairings_sub-{subject}.csv"

with open(design, "r") as f:
    data_design = json.load(f)

tau = data_design['tau_std'][session-1]
mu = data_design['mu_tones_all'][session-1]
d = data_design['sign_d'][session-1]


#---------------SELECT BASED ON MODEL ACTIVATION AND PROBABILITIES--------------------------#

if method == 'activations_probs':

    win = []

    mean_discard = []

    # loop over runs
    for run in range(0,len(tau)):

        sum_all = []
        activations_all = []

        # find 100 instances created for each condition assigned to the current subject and session
        sequences = glob.glob(f"fMRI/sequences_per_condition_sub-{subject}_ses-{session}/*/sub-seq*_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials.csv", recursive=True)

        print(len(sequences))

        for n, s in enumerate(sequences):

            discard = []

            seq_n = re.findall(r"seq\d+", s)
        
            # for each sequence get corresponding model activations
            activations = glob.glob(f"/Users/steinj/Documents/RNN_paradigm/RNN/{network}/activations_sub-{subject}_ses-{session}/sub-{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials_activations.csv")
            activations_all.append(activations)
            activations = pd.read_csv(activations[0])

            # compute pairwise correlations between modules
            cols = ["obs_norm","ctx_norm","dpos_norm","rule_norm"]
            rhos = activations[cols].corr(method="spearman") 
            rhos_all = rhos.where(np.triu(np.ones(rhos.shape), k=1).astype(bool)).stack()
            rhos_arr = rhos_all.to_numpy()

            sum_all.append(np.sum(rhos_all**expo))

            # load model probabilities
            probabilities = glob.glob(f"/Users/steinj/Documents/RNN_paradigm/RNN/{network}/probabilities_deviant_sub-{subject}_ses-{session}/sub-{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials_probabilities_deviant.csv")
            probabilities = pd.read_csv(probabilities[0])

            dpos_cols = probabilities[['dpos_p0','dpos_p1','dpos_p2','dpos_p3','dpos_p4']]

            dpos_cols.plot(kind="box")
            plt.show()

            probs = probabilities['lik_dpos']
            probs = [float(x) for x in probs]

            for i, p in enumerate(probs):
                if p < 0.2 or p > 0.9:
                    discard.append(1)
                else:
                    discard.append(0)

            mean_discard.append(np.mean(discard))        

            if np.mean(discard) > 0.4:
                print(f"discarded sequence: {s}")
                sequences[n] = 'discarded'
                sum_all[n] = -10
            else:
                print(f"kept sequence: {s}")   

        sequences_sorted = [s for _, s in sorted(zip(sum_all, sequences))]
        sum_sorted = sorted(sum_all)

        win.append(next(x for x in sequences_sorted if x != 'discarded'))


#---------------SELECT BASED ON KALMAN LIKELIHOODS--------------------------#

elif method == 'kalman':

    selected_run = [[] for _ in range(4)]

    win = []
    
    for run in range(0,len(tau)):

        # all sequences matching this run
        sequences = glob.glob(f"fMRI/sequences_per_condition_sub-{subject}_ses-{session}/*/sub-seq*_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials.csv", recursive=True)

        # loop across possible sequences
        for n, s in enumerate(sequences):
            
            seq_n = re.findall(r"seq\d+", s)
            
            # get kalman likelihoods
            kalman_likelihoods = glob.glob(f"fMRI/sequences_per_condition_sub-{subject}_ses-{session}/{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}/results_KF*.csv") if take_dpos else glob.glob(f"fMRI/sequences_per_condition_sub-{subject}_ses-{session}/{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}/kalman_predictions_and_likelihoods_at_deviants_sub-{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}*.csv")
            kalman_lik_data = pd.read_csv(kalman_likelihoods[0])

            likelihood_std_at_dpos = kalman_lik_data['likelihood_obs_std_at_dev_over_dpos'] if take_dpos else kalman_lik_data['likelihood_obs_std_at_dev']

            # define exclusion criterion
            fraction = np.mean((likelihood_std_at_dpos > 0) & (likelihood_std_at_dpos <= 1) & (np.mean(likelihood_std_at_dpos)<1.05) & (np.mean(likelihood_std_at_dpos)>0.7)) # manually determined based on data inspection for now --> exclude on the more difficult tail (>= 0.75 to <= 1.05 for the old version of the KF scripts)
            # for the new version of the KF script add mean criterion to make mean likelihood more similar (ensure a few difficult trials in any case)
            
            if fraction >= 0.78: #(0.8 for sub-01 ses-1 and 0.76 for sub-01 ses-2 and 0.75 for sub-01 ses-3)

                selected_run[run].append(s)
                print(np.mean(likelihood_std_at_dpos))


                '''
                plt.figure()
                likelihood_std_at_dpos.plot(kind="hist", bins=30, edgecolor="black")
                plt.title(f"likelihood std at dev {os.path.basename(s)[:-11]}")
                plt.xlim(0, 1.5) if take_dpos else plt.xlim(0, 2)
                plt.show()                

                plt.figure()
                plt.scatter(kalman_lik_data['dpos'],likelihood_std_at_dpos)
                plt.ylim(-0.1, 1.5) if take_dpos else plt.ylim(0, 2)
                plt.show()
                '''


        # randomly sample from run list and append to win list
        print(len(selected_run[run]))
        win.append(random.choice(selected_run[run]))


#---------------ADD ITIS + CUES TO SELECTED SEQUENCES--------------#

# load ITIs and nulls
iti = np.loadtxt("itis.txt")
nulls = np.loadtxt("null_events.txt")

# loop over runs
for run, file in enumerate(win):
    
    trial_data = pd.read_csv(file)

    iti = np.loadtxt("itis.txt")
    nulls = np.loadtxt("null_events.txt")

    # shuffle ITIs randomly
    random.shuffle(iti)
    iti = iti.tolist()

    positions = []
    last = -3 # trials distance between nulls

    n = len(iti)

    min_pos = 1 # min position for null
    max_pos = n - 2 # max position for null

    while len(positions) < len(nulls):
        max_start = max_pos - (len(nulls) - len(positions) - 1) * 3

        pos = random.randint(max(last + 3, min_pos), max_start)

        positions.append(pos)
        last = pos

    positions.sort()

    for pos, value in zip(reversed(positions), nulls):
        iti.insert(pos, value)

    # find positions for merging nulls w/ surrounding ITIs
    merge_positions = []
    nonmerge = np.zeros(len(iti))

    for i, v in enumerate(iti):
        if v == 6.1:
            start = i - 1
            end = i + 2
            nonmerge[i] = 1
            nonmerge[i-1] = 1
            nonmerge[i+1] = 1
            merge_positions.append((start, end))

    # get merged nulls
    final_nulls = [sum(iti[start:end]) for start, end in merge_positions] 

    # remove itis that are now merged with nulls
    n = len(nonmerge)
    remove_idx = []

    i = 0
    while i < n:
        if nonmerge[i] == 1:
            start = i
            while i < n and nonmerge[i] == 1:
                i += 1
            end = i - 1
            remove_idx.append(start)
            remove_idx.append(end)
        else:
            i += 1

    remove_set = set(remove_idx)
    final_itis = [val for i, val in enumerate(iti) if i not in remove_set]

    j = 0
    for i, x in enumerate(final_itis):
        if x == 6.1:
            final_itis[i] = final_nulls[j]
            j += 1     

    # write final ITIs to tone-wise trial list
    trial_data['ITI'] = np.repeat(final_itis, 8)
    trial_data['ITI'] = trial_data['ITI'].round(3)

    # get cues from pre-determined cue-mappings
    cue_data = pd.read_csv(cue_pairings)

    cue1, cue2 = (cue_data.loc[cue_data['session'] == session, ['cue1', 'cue2']].T.values)

    if subject == '01' and session == 1: # attention: for subject one session 1, cues are reversed in generated trial lists!
        trial_data['cue'] = trial_data['cue'].replace({
            'cue_1': cue2[0],
            'cue_2': cue1[0]
        })
    else:
        trial_data['cue'] = trial_data['cue'].replace({
        'cue_1': cue1[0],
        'cue_2': cue2[0]
        })

    # save final logfiles
    diry = f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/"
    os.makedirs(diry, exist_ok=True)

    trial_data.to_csv(f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/sub-{subject}_ses-{session}_run-{run+1}_trials.csv", index=False)

    path = Path(file)
    parent = path.parent

    shutil.copytree(parent, f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/selected_ses-{session}_run-{run+1}_tau_{tau[run]}")
