"""
Heuristics:
- distance of deviant $F0$ to $F0$ of preceding standard tone
- or to $F0$ of the first tone in the current trial
- or to trial average $F0$
- or to all previous history

Priors:
Global prior: prior_dpos_given_prev_stds. It marginalizes over rules, using prior_dpos_given_prev_rule for each rule and the no-deviant share from null_rule_marginal.
Rule prior, transition-only (04/05/06, no cue): pior_dpos_given_prev_rule_and_stds. The function name is misspelled "pior" in the code, so a grep for prior_dpos_given_prev_rule_and_stds will miss it.
Rule prior, cue-aware (11/12/13): pior_dpos_given_prev_rule_cue_and_stds, with ththe cue-rule association in DEFAULT_CUE_RULE_LIK.
--> example of computation:  scripts_data_modeling/compute_rule_priors_sub11-12-13.py
"""

import numpy as np


# Heuristics
# All functions take y_trials of shape (N_tones, N_trials) for one run and return an array of the same shape.

def distance_to_immediate_last(y_trials):
    """Distance of each tone's F0 to the F0 of the tone heard just before it (for position 0,
    the last tone of the previous trial).
    NaN for the first tone of the run."""
    y_run = y_trials.flatten(order='F')  # tones in the order they were heard
    distances_run = np.full(y_run.shape, np.nan)
    distances_run[1:] = np.diff(y_run)
    return distances_run.reshape(y_trials.shape, order='F')


def distance_to_first(y_trials):
    """Distance of each tone's F0 to the F0 of the first tone of the same trial.
    NaN at position 0."""
    distances = y_trials - y_trials[0, :]
    distances[0, :] = np.nan
    return distances


def distance_to_average_trial(y_trials):
    """Distance of each tone's F0 to the average F0 of the previous tones of the same trial.
    NaN at position 0.
    (Note: this is a vectorized implementation)
    """
    n_tones = y_trials.shape[0]
    previous_sums = np.cumsum(y_trials, axis=0)[:-1, :]
    previous_counts = np.arange(1, n_tones)[:, np.newaxis]
    distances = np.full(y_trials.shape, np.nan)
    distances[1:, :] = y_trials[1:, :] - previous_sums / previous_counts
    return distances


def distance_to_average_all(y_trials):
    """Distance of each tone's F0 to the average F0 of all the previous tones of the run.
    NaN for the first tone of the run."""
    y_run = y_trials.flatten(order='F')
    previous_means = np.cumsum(y_run)[:-1] / np.arange(1, y_run.size)
    distances_run = np.full(y_run.shape, np.nan)
    distances_run[1:] = y_run[1:] - previous_means
    return distances_run.reshape(y_trials.shape, order='F')


def distance_to_average_all_stds(y_trials, context_probabilities, standard_context_index):
    """Distance of each tone's F0 to the average F0 of the previous tones of the run that were
    classified as standards. As in the leaky integrator (leaky_average_masked), the tones
    classified as deviants are masked out of the average.
    NaN while no previous tone of the run was classified as a standard.

    context_probabilities : array (N_tones, N_trials, N_contexts), probability of each context at
        each tone, whatever their origin. Each tone is classified into its most probable context
        (argmax over the last axis).
    standard_context_index : index of the standard context along the last axis of
        context_probabilities.
    """
    y_run = y_trials.flatten(order='F')  # tones in the order they were heard
    is_std = np.argmax(context_probabilities, axis=-1) == standard_context_index
    is_std_run = is_std.flatten(order='F')

    # Sum and number of standards among the tones heard before each tone (the tone itself excluded)
    previous_std_sums = np.concatenate([[0.0], np.cumsum(y_run * is_std_run)[:-1]])
    previous_std_counts = np.concatenate([[0], np.cumsum(is_std_run)[:-1]])

    distances_run = np.full(y_run.shape, np.nan)
    has_previous_std = previous_std_counts > 0
    distances_run[has_previous_std] = (y_run[has_previous_std]
                                       - previous_std_sums[has_previous_std] / previous_std_counts[has_previous_std])
    return distances_run.reshape(y_trials.shape, order='F')
