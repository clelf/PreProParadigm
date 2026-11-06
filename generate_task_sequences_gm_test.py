# ---- imports ---- #
import os

import numpy as np
import pandas as pd

import audit_gm as gm


# ---- classes and functions ---- #

class HierarchicalAuditGMCorrectedStates(gm.HierarchicalAuditGM):
    '''Hierarchical audit generative model whose hidden states are sampled with
    sample_states_corr: generate_run calls it in place of sample_states.'''

    sample_states = gm.AuditGenerativeModel.sample_states_corr


def get_condition_gm_config(participant_nr, tau_std, mu_cond, dev_cond, d, si_stat, si_r, init):
    '''Configuration dict of the hierarchical generative model for one single-run condition.

    Fixed values are those of trials_master.config_H in generate_task_sequences.py; the
    process parameters are set as generate_sessions does for one run.

    Args:
        participant_nr: name of the sequence (not used by the generative model)
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

        "participant_nr": participant_nr,
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


def get_stationary_rule_probabilities(pi_rules):
    '''Stationary distribution over the rules of a transition matrix between rules.

    Args:
        pi_rules: transition matrix between rules

    Returns:
        stationary_distrib: stationary probability of each rule
    '''

    # from: https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b
    eigenvals, eigenvects = np.linalg.eig(pi_rules.T)
    close_to_1_idx = np.isclose(eigenvals, 1)
    target_eigenvect = eigenvects[:, close_to_1_idx]
    target_eigenvect = target_eigenvect[:, 0]
    stationary_distrib = target_eigenvect / sum(target_eigenvect)

    return stationary_distrib


def rules_are_balanced(rules, pi_rules):
    '''True if the proportion of trials of each rule is within 2% of the stationary
    probability of the first rule, as required by generate_sessions.'''

    p_rule_1 = sum(rules == 0) / len(rules)
    p_rule_2 = sum(rules == 1) / len(rules)
    p_rules = np.array([p_rule_1, p_rule_2])

    stationary_distrib = get_stationary_rule_probabilities(pi_rules)

    return not np.any((p_rules < (stationary_distrib[0] - 0.02)) | (p_rules > (stationary_distrib[0] + 0.02)))


def deviant_positions_are_balanced(rules, dpos, rules_dpos_set):
    '''True if the number of trials of the deviant positions of both rules differ by at most 1.'''

    all_counts = []

    for rule in [0, 1]:
        dpos_of_rule = dpos[rules == rule]
        for dpos_value in rules_dpos_set[rule]:
            all_counts.append((dpos_of_rule == dpos_value).sum())

    return max(all_counts) - min(all_counts) <= 1


def get_rules_and_dpos(gm_config, balanced):
    '''Rules and deviant positions of the trials of a run. If balanced, they are sampled again
    until both are balanced (rules_are_balanced and deviant_positions_are_balanced), as
    required by generate_sessions; otherwise they are sampled once.

    Only about 1 in 30000 samples is balanced. The states and observations depend on the
    rules and deviant positions but not the other way round, so they are sampled once
    afterwards (get_run_with_sampled_states) rather than for each rejected sample.

    Args:
        gm_config: configuration dict of the generative model
        balanced: if True, sample again until the rules and deviant positions are balanced

    Returns:
        rules: rule of each trial
        pi_rules: transition matrix between rules
        dpos: deviant position of each trial
    '''

    hgm = gm.HierarchicalAuditGM(gm_config)

    while True:
        rules, pi_rules = hgm.sample_rules(hgm.N_blocks, hgm.N_rules, hgm.mu_rho_rules, hgm.si_rho_rules, hgm.fixed_rule_id, hgm.fixed_rule_p, True)

        if balanced and not rules_are_balanced(rules, pi_rules):
            continue

        dpos = hgm.sample_dpos(rules, hgm.rules_dpos_set)

        if not balanced or deviant_positions_are_balanced(rules, dpos, hgm.rules_dpos_set):
            return rules, pi_rules, dpos


def get_run_with_sampled_states(gm_class, gm_config, rules, pi_rules, dpos):
    '''Run with the given rules and deviant positions, whose states and observations are
    sampled by gm_class (as in the end of generate_run).

    Args:
        gm_class: hierarchical generative model class
        gm_config: configuration dict of the generative model
        rules: rule of each trial
        pi_rules: transition matrix between rules
        dpos: deviant position of each trial

    Returns:
        run_obj: dict with the rules, deviant positions, contexts, states, observations,
            parameters and transition matrix between rules of the run
    '''

    hgm = gm_class(gm_config)

    contexts = hgm.get_contexts(dpos, hgm.N_blocks, hgm.N_tones)
    states, pars = hgm.sample_states(contexts, return_pars=True)
    obs = hgm.sample_observations(contexts, states)

    pars['si_r'] = hgm.si_r

    run_obj = {
        'rules': rules,
        'dpos': dpos,
        'contexts': contexts.flatten(),
        'states': [states[0].flatten(), states[1].flatten()],
        'obs': obs.flatten(),
        'pars': pars,
        'pi_rules': pi_rules
    }

    return run_obj


def get_cue_of_each_trial(run_obj, rules_dpos_set, cue_prob, cues):
    '''Probabilistic cue of each trial of a run: cues[0] is given to a proportion cue_prob of the
    trials of each deviant position of the first rule and to the complementary proportion of
    the trials of each deviant position of the second rule, cues[1] to all other trials.

    Args:
        run_obj: run with rules and deviant positions
        rules_dpos_set: deviant positions of each rule
        cue_prob: cue validity (0-1)
        cues: names of the two cues

    Returns:
        cue_of_each_trial: cue name of each trial
    '''

    cue_of_each_trial = np.full(len(run_obj['rules']), cues[1], dtype=object)

    proportion_cued_per_rule = [cue_prob, round(1 - cue_prob, 2)]

    for rule in [0, 1]:
        for dpos in rules_dpos_set[rule]:
            trials_of_rule_and_dpos = np.where((run_obj['rules'] == rule) & (run_obj['dpos'] == dpos))[0]
            n_cued = int(proportion_cued_per_rule[rule] * len(trials_of_rule_and_dpos))
            cued_trials = np.random.choice(trials_of_rule_and_dpos, size=n_cued, replace=False)
            cue_of_each_trial[cued_trials] = cues[0]

    return cue_of_each_trial


def convert_states_to_frequencies_hz(x_input, x_min=-1, x_max=1, f_exp_min=500, f_exp_max=1300):
    '''Frequencies in Hz of states or observations, from the ERB transfer function
    (Glasberg & Moore, 1990, Eq. 4).

    Args:
        x_input: array of states or observations
        x_min: minimum of x scale
        x_max: maximum of x scale
        f_exp_min: min F0 of experiment in Hz
        f_exp_max: max F0 of experiment in Hz

    Returns:
        freq_out: array of the F0 in Hz corresponding to x_input
    '''

    e_min = 21.4 * np.log10(4.37 * f_exp_min / 1000 + 1)
    e_max = 21.4 * np.log10(4.37 * f_exp_max / 1000 + 1)

    e = e_min + (e_max - e_min) * (x_input - x_min) / (x_max - x_min)

    freq_out = (10 ** (e / 21.4) - 1) * (1000 / 4.37)

    return freq_out


def get_sequence_trials_df(run_obj, cue_of_each_trial, gm_config, tau_std, dev_cond, d):
    '''Trial list (one row per tone) of a single-run sequence, with the columns of the
    trials CSV written by generate_sessions in generate_task_sequences.py.

    Args:
        run_obj: balanced run
        cue_of_each_trial: cue name of each trial
        gm_config: configuration dict used to generate the run
        tau_std: time constant of the standard process
        dev_cond: sign of the effect size d
        d: absolute effect size

    Returns:
        trials_df: trial list of the sequence
    '''

    obs_hz = convert_states_to_frequencies_hz(run_obj['obs'])
    state_std_hz = convert_states_to_frequencies_hz(run_obj['states'][0])
    state_dev_hz = convert_states_to_frequencies_hz(run_obj['states'][1])

    trials_df = pd.DataFrame({
        'observation': run_obj['obs'],
        'frequency': obs_hz,
        'state_std': run_obj['states'][0],
        'state_dev': run_obj['states'][1],
        'lim_std': gm_config['fix_lim_val'],
        'lim_dev': run_obj['pars']['lim'][1],
        'tau_std': tau_std,
        'tau_dev': tau_std / 8,
        'd': d * dev_cond,
        'rule': np.repeat(run_obj['rules'], gm_config['N_tones']),
        'dpos': np.repeat(run_obj['dpos'], gm_config['N_tones']),
        'trial_type': run_obj['contexts'],
        'sigma_q_std': run_obj['pars']['si_q'][0],
        'sigma_q_dev': run_obj['pars']['si_q'][1],
        'sigma_r': gm_config['si_r'],
        'ITI': np.nan, # kept empty, as in generate_sessions
        'duration_tones': gm_config['duration_tones'],
        'ISI': gm_config['isi'],
        'trial_n': np.repeat(np.arange(gm_config['N_blocks']), gm_config['N_tones']),
        'run_n': 0,
        'session_n': 0,
    })

    trials_df['state_std_hz'] = state_std_hz
    trials_df['state_dev_hz'] = state_dev_hz
    trials_df['diff_std'] = trials_df['frequency'] - trials_df['state_std_hz']
    trials_df['diff_dev'] = trials_df['frequency'] - trials_df['state_dev_hz']
    trials_df['cue'] = np.repeat(cue_of_each_trial, gm_config['N_tones'])

    trials_df.loc[trials_df['trial_type'] == 1, 'diff_std'] = np.nan
    trials_df.loc[trials_df['trial_type'] == 0, 'diff_dev'] = np.nan

    return trials_df


if __name__ == "__main__":

    # folder where the two batches of sequences are stored, each in its own sub-folder
    output_dir = "/home/clevyfidel/Documents/Workspace/PreProParadigm/sequences_gm_test"
    batch_folder_names = ['gm_original', 'gm_corrected']
    batch_gm_classes = [gm.HierarchicalAuditGM, HierarchicalAuditGMCorrectedStates]

    n_sequences_per_condition = 100

    # True: rules and deviant positions are sampled again until balanced (about 16 s per sequence)
    # False: they are sampled once, without any check (instant)
    BALANCED = False
    if BALANCED:
        output_dir = output_dir + "_balanced"

    # conditions: all combinations of these values, numbered as in generate_task_sequences.py
    tau_std_values = [16, 40, 160, 240]
    mu_cond_values = [-0.6, 0.3]
    dev_cond_values = [1, -1]

    d = 1

    si_stat = 0.1
    si_r_rat = 5

    cue_prob = 0.8
    cues = ['cue_1', 'cue_2']

    init = 'TN_3'

    batch_dirs = []

    for batch_folder_name in batch_folder_names:
        batch_dir = f"{output_dir}/{batch_folder_name}"
        os.makedirs(batch_dir, exist_ok=True)
        batch_dirs.append(batch_dir)

    condition_nr = 0

    for tau_std in tau_std_values:
        for mu_cond in mu_cond_values:
            for dev_cond in dev_cond_values:

                condition_nr += 1

                for sequence_nr in range(1, n_sequences_per_condition + 1):

                    np.random.seed(100 * condition_nr + sequence_nr)

                    participant_nr = f"seq{sequence_nr}_cond{condition_nr}_tau{tau_std}_mu{mu_cond}_dev{dev_cond}"

                    print(f"=== {participant_nr}, |d| = {d}, si_stat = {si_stat}, si_r = {si_stat / si_r_rat} ===")

                    gm_config = get_condition_gm_config(participant_nr, tau_std, mu_cond, dev_cond, d, si_stat, si_stat / si_r_rat, init)

                    # rules and deviant positions are shared by the sequences of both batches
                    rules, pi_rules, dpos = get_rules_and_dpos(gm_config, BALANCED)

                    random_state_after_rules_and_dpos = np.random.get_state()

                    for batch_dir, gm_class in zip(batch_dirs, batch_gm_classes):

                        # both batches continue from the same random state: their sequences share all
                        # random draws (observation noise, cues) and differ by the state sampling only
                        np.random.set_state(random_state_after_rules_and_dpos)

                        run_obj = get_run_with_sampled_states(gm_class, gm_config, rules, pi_rules, dpos)

                        cue_of_each_trial = get_cue_of_each_trial(run_obj, gm_config['rules_dpos_set'], cue_prob, cues)

                        trials_df = get_sequence_trials_df(run_obj, cue_of_each_trial, gm_config, tau_std, dev_cond, d)

                        trials_df.to_csv(f"{batch_dir}/sub-{participant_nr}_ses-1_trials.csv", index=False, float_format="%.4f")
