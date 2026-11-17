'''
NOTE: code used for selecting trial sequences consistently starting from participant 02 ses-3

- participant 01: sequence selection based on Kalman filter (models were not fully ready yet)
- participant 02: sequence selection model-based but unlike here, based on maximum correlation including priors and posteriors
'''

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


# loop over subjects and sessions
for subjects in ['04']:
    for sessions in [1,2,3,4,5,6]:

        #---------------SUBJECT INFORMATION & CONFIG----------------#
        subject = subjects # participant number
        session = sessions # session number
        network = 'population_network_all_bn8_trainh0_epochs300_lr0.002' # use the model trained on varying noise levels
        expo = 2 # exponent for sum measure; should be even to account for sign mismatches!
        exclude_rule = True # exclude rule module from sequence selection process
        use_derivs = False # use derivatives of the activation values
        use_dev_plus_one = True # use probability of deviant and tone after
        #-----------------------------------------------------------#


        #---------------LOAD RUN PARAMETERS--------------------------#

        # get assigned experimental design and cue pairing from files (stored separately for each participant, see generate_designs.py)
        design = f"fMRI/randomizations/task_structure_sub-{subject}.json"
        cue_pairings = f"fMRI/randomizations/cue_pairings_sub-{subject}.csv"

        with open(design, "r") as f:
            data_design = json.load(f)

        tau = data_design['tau_std'][session-1]
        mu = data_design['mu_tones_all'][session-1]
        d = data_design['sign_d'][session-1]


        #---------------SELECT BASED ON MODEL ACTIVATION AND PROBABILITIES--------------------------#

        win = []
        mean_discard = []

        # loop over runs
        for run in range(0,len(tau)):

            max_all = []
            activations_all = []

            # find 100 instances created for each condition assigned to the current subject and session
            sequences = glob.glob(f"fMRI/sequences_per_condition_sub-{subject}_ses-{session}/*/sub-seq*_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials.csv", recursive=True)

            for n, s in enumerate(sequences):

                discard = []

                seq_n = re.findall(r"seq\d+", s)
            
                # for each sequence get corresponding model activations
                activations_file = glob.glob(f"/data/p_03132/AuditPrePro_MRT/code/RNN_HierarchicalAuditory/RNN/exp_seq_act_output_sub-{subject}_ses-{session}/{network}/activations/sub-{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials_activations.csv")
                activations_all.append(activations_file)
                activations = pd.read_csv(activations_file[0])

                # compute pairwise correlations between modules (w/ or w/o rule module)
                if not use_derivs:
                    if not exclude_rule:
                        cols = ["obs_norm","ctx_norm","dpos_norm","rule_norm"]
                    else:
                        cols = ["obs_norm","ctx_norm","dpos_norm"]
                elif use_derivs:  
                    if not exclude_rule:
                        cols = ["obs_deriv","ctx_deriv","dpos_deriv","rule_deriv"] 
                    else:
                        cols = ["obs_deriv","ctx_deriv","dpos_deriv"]      

                rhos = activations[cols].corr(method="spearman")
                
                rhos_all = rhos.where(np.triu(np.ones(rhos.shape), k=1).astype(bool)).stack()

                print(rhos_all)

                max_all.append(np.max(rhos_all**expo)) # selection criterion: maximum of squared rhos

                # load model probabilities
                probabilities_file = glob.glob(f"/data/p_03132/AuditPrePro_MRT/code/RNN_HierarchicalAuditory/RNN/exp_seq_act_output_sub-{subject}_ses-{session}/{network}/probabilities_deviant/sub-{seq_n[0]}_cond*_tau{tau[run]}_mu{mu[run][0]}_dev{d[run]}_ses-1_trials_probabilities_deviant.csv")
                probabilities = pd.read_csv(probabilities_file[0])

                if not use_dev_plus_one:
                    probabilities = probabilities[probabilities["ctx"] == 1]

                probs = probabilities['lik_dpos']
                probs = [float(x) for x in probs]

                # identify probabilities that are too high or too low (preregistered cutoff)
                for i, p in enumerate(probs):
                    if p < 0.3 or p > 0.9:
                        discard.append(1)
                    else:
                        discard.append(0)

                mean_discard.append(np.mean(discard))        

                # discard if > 30% of probabilities are too high or too low (preregistered cutoff)
                if np.mean(discard) > 0.3:
                    sequences[n] = 'discarded'
                    max_all[n] = -10
                else:
                    pass  

            # sort the remaining sequences accoring to max_all and chose lowest of non-discarded sequences
            sequences_sorted = [s for _, s in sorted(zip(max_all, sequences))]
            max_sorted = sorted(max_all)

            win.append(next(x for x in sequences_sorted if x != 'discarded'))
            print(f"selected sequence: {win[-1]}")

        print(win)    

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
            np.random.shuffle(iti)
            iti = iti.tolist()

            n = len(iti)
            num_nulls = len(nulls)
            min_gap = 3

            available = n - 2

            free = available - (num_nulls - 1) * (min_gap - 1)

            base = sorted(random.sample(range(free), num_nulls))

            positions = [
                b + i * (min_gap - 1) + 1
                for i, b in enumerate(base)
            ] 

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

            # save final trial lists
            diry = f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/"
            os.makedirs(diry, exist_ok=True)

            trial_data.to_csv(f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/sub-{subject}_ses-{session}_run-{run+1}_trials.csv", index=False)

            path = Path(file)
            parent = path.parent

            shutil.copytree(parent, f"fMRI/selected_trial_lists_scanning_sub-{subject}_ses-{session}/sub-{subject}/selected_ses-{session}_run-{run+1}_tau_{tau[run]}")
