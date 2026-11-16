


def get_condition_gm_config(subject_id, tau_std, mu_cond, dev_cond, d, si_stat, si_r, init):
    '''Configuration dict of the hierarchical generative model for one single-run condition.

    Fixed values are those of trials_master.config_H in generate_task_sequences.py; the
    process parameters are set as generate_sessions does for one run.

    Args:
        subject_id: name of the sequence (not used by the generative model)
        tau_std: time constant of the standard process (the deviant process gets tau_std / 8)
        mu_cond: stationary value of the standard process (lim_std)
        dev_cond: sign of the effect size d
        d: absolute effect size
        si_stat: stationary standard deviation of the processes
        si_r: observation noise standard deviation
        init: initialization of the standard and deviant processes

    Returns:
        gm_config: dict passed to the hierarchical generative model
    '''

    gm_config = {

        "subject_id": subject_id,
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
        "si_stat": si_stat,  # stationary process variance
        "si_r": si_r,  # measurement noise variance
        "si_d_coef": 0.02, # unused for experiment
        "mu_d": 2, # unused for experiment
        "return_pi_rules": True,
        "fixed_rule_id": 2,
        "fixed_rule_p": 0,
        "rules_cmap": {0: "tab:blue", 1: "tab:red", 2: "tab:gray"},
        "fix_process": True, # fix tau, lim, d to input values
        "fix_tau_val": [tau_std, tau_std / 8], # tau std, tau dev
        "fix_lim_val": mu_cond, # lim std
        "fix_d_val": d * dev_cond, # effect size d
        "fix_pi_rules": True,
        "fix_pi_vals": [0.85, 0.15], # fixed values to create transition matrix
        "n_sessions": 1, # number of sessions
        "n_runs": 1, # number of runs per session
        "isi": 0.65, # inter-stimulus interval
        "duration_tones": 0.1, # stimulus duration,
        "init": init, # how to initialize std dev processes,
    }

    return gm_config

if __name__=='__main__':
    pass