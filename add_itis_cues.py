import numpy as np
import random
import pandas as pd

participant = "sub-01"

n_sess = 6
n_run = 4

for sess in range(0, n_sess):
    for run in range(0, n_run):


        triallist = f"trial_lists/{participant}/{participant}_ses-{sess+1}_run-{run+1}_trials.csv"
        trial_data = pd.read_csv(triallist)

        iti = np.loadtxt("itis.txt")
        nulls = np.loadtxt("null_events.txt")

        random.shuffle(iti)
        iti = iti.tolist()

        positions = []
        last = -3

        n = len(iti)

        min_pos = 1
        max_pos = n - 2

        while len(positions) < len(nulls):
            max_start = max_pos - (len(nulls) - len(positions) - 1) * 3

            pos = random.randint(max(last + 3, min_pos), max_start)

            positions.append(pos)
            last = pos

        positions.sort()

        for pos, value in zip(reversed(positions), nulls):
            iti.insert(pos, value)

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

        final_nulls = [sum(iti[start:end]) for start, end in merge_positions] 

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

        print(final_itis)        

        trial_data['ITI'] = np.repeat(final_itis, 8)
        trial_data['ITI'] = trial_data['ITI'].round(3)

        # get cues
        cue_file = f"cue_pairings_{participant}.csv"
        cue_data = pd.read_csv(cue_file)

        cue1, cue2 = (cue_data.loc[cue_data['session'] == 1, ['cue1', 'cue2']].T.values)

        trial_data['cue'] = trial_data['cue'].replace({
            'cue_1': cue1[0],
            'cue_2': cue2[0]
        })

        print(trial_data)

        trial_data.to_csv(f"trial_lists/{participant}/{participant}_ses-{sess+1}_run-{run+1}_trials.csv", index=False)