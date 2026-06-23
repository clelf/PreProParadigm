# ----------------------------------------------------------#
# IMPORTS
# ----------------------------------------------------------#
import numpy as np
import pandas as pd
import random
from itertools import permutations, product
from pathlib import Path
import json

# ----------------------------------------------------------#
# DEFINE CLASS
# ----------------------------------------------------------#
class rando_master:

    def __init__(self):

        # ----------------------------------------------------------#
        # CONFIG
        # ----------------------------------------------------------#

        self.config = {
            'participant': '01',
            'n_runs': 4,
            'n_sessions': 6,
            'out_dir': 'fMRI/randomizations'
        }

    # ----------------------------------------------------------#
    # RANDOMIZATION FUNCTION
    # ----------------------------------------------------------#

    def randomize_design_abc(self):

        '''
        generates experimental design with n_runs and n_sessions of conditions A, B, C in the specific case of:
            - n_runs = N conditions A (each run belongs to a condition A)
            - there are two levels of condition B and C
            - this script makes sure that each condition A occurs roughly equally across different run positions across sessions
            - makes sure that AxB and AxC are balanced
            - even number of runs and sessions required

        what this does not DO:
            - account for transition probabilities between conditions and higher-order order effects!!    

        Args:
            - self (containing config specifying n_runs and n_sessions)

        Returns:
            df: data frame containing experimental design info     
        '''

        # only works with even numbers
        if self.config['n_runs'] % 2 != 0:
                raise ValueError(
                    f"self.config['n_runs'] must be even. Got n_runs={self.config['n_runs']}"
                )

        if self.config['n_sessions'] % 2 != 0:
            raise ValueError(
                f"n_sessions must be even. Got n_sessions={self.config['n_sessions']}"
            )
        
        #-------------------------------------------------------------------------------------------------#
        # draw conditions such that each condition occurs roughly balanced across positions across sessions
        #-------------------------------------------------------------------------------------------------#
        
        def draw_positions(self):

            positions_unbalanced = True

            max_attempts = 10000
            attempt = 0

            while positions_unbalanced:

                attempt += 1

                if attempt > max_attempts:
                    raise RuntimeError(
                        "Could not generate balanced design within max_attempts"
                    )

                # min and max N of occurence in each position
                minv = self.config['n_sessions']//self.config['n_runs']
                maxv = minv + (1 if self.config['n_sessions'] % self.config['n_runs'] else 0)
                
                chars = ['A' + chr(ord('1') + i) for i in range(self.config['n_runs'])]
                
                # generate all permutations, shuffle and use first N_session ones
                full = [list(p) for p in permutations(chars)]
                np.random.shuffle(full)
                a_all = full[0:self.config['n_sessions']]
                
                out = []

                # count occurence across positions
                for c in chars:

                    cnt = [sum(row[i] == c for row in a_all) for i in range(len(a_all[0]))]
                    out.append(cnt)
                    
                # check for roughly equal occurence
                check = all([minv <= x <= maxv for row in out for x in row])

                if check:
                    positions_unbalanced = False

            return a_all
                        
        #-------------------------------------------------------------------------------------------------#
        # make sure that AxB and AxC are balanced
        #-------------------------------------------------------------------------------------------------#
        
        conditions_unbalanced = True

        while conditions_unbalanced:

            # get combinations of B and C
            b_group = ['B' + chr(ord('1') + i) for i in range(self.config['n_runs']//2)]
            c_group = ['C' + chr(ord('1') + i) for i in range(self.config['n_runs']//2)]

            groups = [b_group, c_group]

            bc_combos = list(product(*groups))

            # redraw a_all every time to find a_all for which a counterbalanced solution is possible
            a_all = draw_positions(self)

            # create random run order
            run_full = [list(p) for p in permutations(np.arange(0,self.config['n_runs']))] 
            np.random.shuffle(run_full) 
            run_orders = run_full[0:self.config['n_sessions']]

            rows = []

            for s in range(self.config['n_sessions']):

                # assign A levels to BC combos
                current_a = a_all[s]

                session_trials = []

                for i in range(self.config['n_runs']):

                    B, C = bc_combos[i]

                    session_trials.append({
                        "session": s + 1,
                        "run": i + 1,
                        "a_level": current_a[i],
                        "b_level": B,
                        "c_level": C,
                    })

                # apply session-specific run order 
                reordered = [session_trials[idx] for idx in run_orders[s]] 

                for r, cond in enumerate(reordered): 
                    cond["run"] = r + 1 
                    rows.append(cond)

            # create df with session infos
            df = pd.DataFrame(rows)
            df = df[["session", "run", "a_level", "b_level", "c_level"]]

            # check balance

            print('----- experimental design -----')

            print("\nA counts:")
            print(df["a_level"].value_counts())

            print("\nB counts:")
            print(df["b_level"].value_counts())

            print("\nC counts:")
            print(df["c_level"].value_counts())

            print("\nA x B:")
            print(pd.crosstab(df["a_level"], df["b_level"]))
            ct1 = np.array(pd.crosstab(df["a_level"], df["b_level"])).flatten()

            print("\nA x C:")
            print(pd.crosstab(df["a_level"], df["c_level"]))
            ct2 = np.array(pd.crosstab(df["a_level"], df["c_level"])).flatten()
            
            if all(x == ct1[0] for x in ct1) and all(y == ct2[0] for y in ct2):
                conditions_unbalanced = False
                print('A x B and A x C balanced')

            print("\nB x C:")
            print(pd.crosstab(df["b_level"], df["c_level"]))

            print("\nFull cells:")
            print(pd.crosstab(
                [df["a_level"], df["b_level"]],
                df["c_level"]
            ))

            print('---------------------------------')

        
        df.to_csv(f'{self.config['out_dir']}/{self.config['participant']}_design.csv', index=False)
        
        return df
    
    def define_task_structure(self, df, mu_tones = [[-0.6, None], [0.3, None]], taus = [16, 40, 160, 240]):

        '''assigns correct values to generic randomization scheme created by randomize_design_abc

        Args:
            df: df returned by randomize_design_abc
            mu_tones: mu for standards in high and low condition (list of lists)
            taus: list of 4 taus to be used

        Returns:
            mu_tones_all: list of mus for all runs
            taus_all: list of taus for all runs (list of list, includes tau for std and dev)
            tau_std: np array of taus for std tone
            tau_dev: np array of taus for dev tone
            sign_d: list for signed effect size to constitute deviant up/down conditions
        
        '''

        mu_tones_all = [[[0, 0] for _ in range(self.config["n_runs"])] for _ in range(self.config["n_sessions"])]

        tau_std = [[] for _ in range(self.config["n_sessions"])]
        tau_std_ind = [df["a_level"].tolist()[i:i+self.config["n_sessions"]] for i in range(0, len(df), self.config["n_sessions"])]

        tau_std_ind_col = df["a_level"].str.extract(r"(\d+)").astype(int)[0] - 1
        tau_std_ind = [tau_std_ind_col.tolist()[i:i+self.config["n_runs"]] for i in range(0, len(tau_std_ind_col), self.config["n_runs"])]

        mu_tones_ind_col = df["b_level"].str.extract(r"(\d+)").astype(int)[0] - 1
        mu_tones_ind = [mu_tones_ind_col.tolist()[i:i+self.config["n_runs"]] for i in range(0, len(mu_tones_ind_col), self.config["n_runs"])] 

        sign_d_col = (
            df["c_level"]
            .str.extract(r"(\d+)").astype(int)[0]
            .map(lambda x: -1 if x == 1 else 1)
        )

        sign_d = [sign_d_col.tolist()[i:i+self.config["n_runs"]] for i in range(0, len(sign_d_col), self.config["n_runs"])] 

        for s in range(self.config["n_sessions"]):
            
            tau_row = [taus[i] for i in tau_std_ind[s]]  # map indices to values
            tau_std[s]=tau_row

            for r in range(self.config["n_runs"]):
                mu_tones_all[s][r] = mu_tones[mu_tones_ind[s][r]].copy()
        
        tau_std = np.array(tau_std)
        tau_dev = (tau_std // 8).astype(int)

        taus_all = [[x, y] for x, y in zip(tau_std, tau_dev)]

        data = {
            "mu_tones_all": np.asarray(mu_tones_all).tolist(),
            "taus_all": np.asarray(taus_all).tolist(),
            "tau_std": np.asarray(tau_std).tolist(),
            "tau_dev": np.asarray(tau_dev).tolist(),
            "sign_d": np.asarray(sign_d).tolist(),
        }

        with open(Path(self.config['out_dir']) / f"task_structure_{self.config['participant']}.json", "w") as f:
            json.dump(data, f)

    def get_cue_combs(self, cue_names):
        
        random.shuffle(cue_names)

        df = pd.DataFrame({
            "session": range(1, 7),
            "cue1": cue_names[::2],
            "cue2": cue_names[1::2]
        })

        csv_path = Path(self.config['out_dir']) / f"cue_pairings_{self.config['participant']}.csv"
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":

    subs = [f"sub-{i:02d}" for i in range(1, 11)]
    cue_names = ['triangle', 'cicle', 'square', 'diamond','hourglass','dome','x','star','crown','heart','hexagon','flower']

    for sub in subs:

        rando = rando_master()
        rando.config['participant'] = sub
        rand_df = rando.randomize_design_abc()
        rando.define_task_structure(rand_df)
        rando.get_cue_combs(cue_names)