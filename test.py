import audit_gm as gm
import numpy as np

config_H = {
    "N_samples": 1,
    "N_blocks": 160,
    "N_tones": 8,
    "rules_dpos_set": np.array([[3, 4, 5], [5, 6, 7]]),
    "mu_tau": 16,
    "si_tau": 1,
    "si_lim": 0.2,
    "mu_rho_rules": 0.8,
    "si_rho_rules": 0.05,
    "mu_rho_timbres": 0.8,
    "si_rho_timbres": 0.05,
    # "si_q": 2,  # process noise variance
    "si_stat": 0.05,  # stationary process variance
    "si_r": 0.02,  # measurement noise variance
    "si_d_coef": 0.05,
    "mu_d": 2,
    "return_pi_rules": True,
    "fix_process": True, # fix tau, lim, d
    "fix_tau_val": [16,2],
    "fix_lim_val": -0.6,
    "fix_d_val": 2,
    "fix_pi": True,
    "fix_pi_vals": [0.8, 0.1, 0]
}


config_H_nullrule = config_H.copy()
config_H_nullrule["mu_tau"] = 16
config_H_nullrule["rules_dpos_set"] = [[3, 4, 5], [5, 6, 7], None]
config_H_nullrule["fixed_rule_id"] = 2
config_H_nullrule["fixed_rule_p"] = 0.1
config_H_nullrule["rules_cmap"] = {0: "tab:blue", 1: "tab:red", 2: "tab:gray"}

gm.example_HGM(config_H_nullrule)