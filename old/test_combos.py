import generate_task_sequences as gen
import numpy as np

d = [0.5, 1, 2, 3]
si_stati = np.linspace(0.01,0.5, num = 5)
si_stati = si_stati.tolist()

ratio = [5,2,1]



for eff in d:

    for si_stat in si_stati:

        for si_rat in ratio:

            si_r = si_stat/si_rat

            participant_id = f'd_{eff:.3f}_sistat_{si_stat:.3f}_si_r{si_r:.3f}'

            master = gen.trials_master()

            master.config_H = {
                "participant_nr": participant_id, # participant nr, as also read in by PsychPy exp.
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
                "si_stat": si_stat,  # stationary process variance
                "si_r": si_r,  # measurement noise variance
                "si_d_coef": 0.05, # unused for experiment
                "mu_d": 2, # unused for experiment
                "return_pi_rules": True,
                "fixed_rule_id": 2,
                "fixed_rule_p": 0.1,
                "rules_cmap": {0: "tab:blue", 1: "tab:red", 2: "tab:gray"},
                "fix_process": True, # fix tau, lim, d to input values
                "fix_tau_val": [16, 2], # tau std, tau dev
                "fix_lim_val": -0.6, # lim std
                "fix_d_val": eff, # effect size d
                "fix_pi": True,
                "fix_pi_vals": [0.8, 0.1, 0], # fixed values to create transition matrix
                "n_sessions": 1, # number of sessions
                "n_runs": 4, # number of runs per session
                "isi": 0.65, # inter-stimulus interval
                "duration_tones": 0.1, # stimulus duration
                }
            
            master.generate_sessions()
