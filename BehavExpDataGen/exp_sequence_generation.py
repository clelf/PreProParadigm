import itertools

import numpy as np

from audit_gm import HierarchicalAuditGMCorrectedStates
from BehavExpDataGen.kalman_full import HAudIn
from BehavExpDataGen.data_generation_core import (
    get_condition_gm_config,
)
from generate_task_sequences_gm_test import (
    get_rules_and_dpos,
    get_run_with_sampled_states,
    get_cue_of_each_trial,
)
# from ..model_RTs import ...


if __name__ == '__main__':

    # Define base GM config: the fixed values of the experiment (N_blocks, N_tones, rules_dpos_set, pi_rules, ...)
    # are those of get_condition_gm_config (generate_task_sequences_gm_test.py); only the parameters below vary.
    # Parameters ranges: the sequences of every combination of these values are generated
    tau_std_values = ...   # e.g. [16, 40, 160, 240]
    mu_cond_values = ...   # stationary value of the standard process (lim_std)
    dev_cond_values = ...  # sign of the effect size d
    d_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1]         # absolute effect size
    si_stat_values = [0.0]   # stationary standard deviation of the processes
    si_r_values = ...      # observation noise standard deviation
    init = ...             # initialization of the standard and deviant processes (see audit_gm.py)

    n_sequences_per_condition = 100
    BALANCED = True         # True: rules and deviant positions are sampled again until balanced (slow)

    cue_prob = ...         # cue validity (0-1) # TODO: what is this?
    cues = ['cue_1', 'cue_2']             # names of the two cues, e.g. ['cue_1', 'cue_2']
    cue_to_index = {'cue_1': 0, 'cue_2': 1}     # column index of HAudIn's pi_cues for each cue name, e.g. {'cue_1': 1, 'cue_2': 0}

    # Start values and settings of HAudIn, one value per process [std, dev] where relevant
    estimate_process_noise = True
    min_transitions_before_PE = [24, 3]
    prior_state_means = ...
    prior_state_vars = ...
    tau_init = ...
    b_init = ...
    process_noise_init = ...
    observation_noise_sir = 0.02

    random_seed = ...

    # Define GM: the generative model class with the CORRECTED sample_states
    gm_class = HierarchicalAuditGMCorrectedStates

    # Generate batch of sequences (generate_run steps, with CORRECTED sample_states) and apply candidate models
    # Candidate models: so far, only HAudIn
    sequences = []
    haudin_inferences = []

    condition_values = itertools.product(tau_std_values, mu_cond_values, dev_cond_values, d_values, si_stat_values, si_r_values)

    for condition_nr, (tau_std, mu_cond, dev_cond, d, si_stat, si_r) in enumerate(condition_values, start=1):

        for sequence_nr in range(1, n_sequences_per_condition + 1):

            np.random.seed(random_seed + 100 * condition_nr + sequence_nr)

            sequence_name = f"seq{sequence_nr}_cond{condition_nr}_tau{tau_std}_mu{mu_cond}_dev{dev_cond}_d{d}"

            gm_config = get_condition_gm_config(sequence_name, tau_std, mu_cond, dev_cond, d, si_stat, si_r, init)

            rules, pi_rules, dpos = get_rules_and_dpos(gm_config, BALANCED)
            run_obj = get_run_with_sampled_states(gm_class, gm_config, rules, pi_rules, dpos)

            cue_of_each_trial = get_cue_of_each_trial(run_obj, gm_config['rules_dpos_set'], cue_prob, cues)

            # HAudIn inputs: observations and cue index of each tone
            y = run_obj['obs']
            q = np.array([cue_to_index[cue] for cue in np.repeat(cue_of_each_trial, gm_config['N_tones'])])

            haudin = HAudIn(y=y,
                            q=q,
                            estimate_process_noise=estimate_process_noise,
                            min_transitions_before_PE=min_transitions_before_PE,
                            prior_state_means=prior_state_means,
                            prior_state_vars=prior_state_vars,
                            tau_init=tau_init,
                            b_init=b_init,
                            process_noise_init=process_noise_init,
                            observation_noise_sir=observation_noise_sir)
            inf_states = haudin.inference()

            sequences.append({'sequence_name': sequence_name, 'run_obj': run_obj, 'cue_of_each_trial': cue_of_each_trial})
            haudin_inferences.append(inf_states)

    # For later: Compute correlation between models

    pass
