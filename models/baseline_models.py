"""
Heuristics:
- distance of deviant $F0$ to $F0$ of preceding standard tone
- or to $F0$ of the first tone in the current trial
- or to trial average $F0$
- or to all previous history

Priors: probability that the tone at position j is the deviant, given that the earlier tones of the
trial were standards ("prev_stds").
- Given the rule of the trial: prior_tone_is_dev_given_rule_and_prev_stds
- Global (ignoring rules/cues): prior_tone_is_dev_given_prev_stds
- Given the previous rule (04/05/06): prior_tone_is_dev_given_prev_rule_and_prev_stds
- Given the previous rule and the cue (11/12/13): prior_tone_is_dev_given_prev_rule_cue_and_prev_stds
"""

import numpy as np


### Heuristics
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

    --> Needs decision criteria about contexts (here: context probabilities
    converted into most probable contexts by argmax)

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




### Priors
# Notation: trial k, tone position j, rule r_k, deviant position d_k, cue q_k,
# and S_j: the tones at positions 0,...,j-1 of trial k are all standards.
# Equation numbers refering to BehavModeling/deviant_prior_maths.md.

def prior_tone_is_dev_given_prev_stds(j, p_no_dev=0.0, valid_positions=None):
    """Probability that the tone at position j is the deviant, given S_j only: p(d_k = j | S_j), Eq. (5).

    Unaware of rules dynamics (so ignoring previous rule and possible cue).
    The rule belief before any tone is equal to p_no_dev for the null rule,
    and to (1 - p_no_dev) split uniformly over the remaining rules.
    When p_no_dev = 0, the values at j = 2,...,6 are 1/6, 1/5, 2/4, 1/2, 1.

    p_no_dev : probability that a trial has no deviant, e.g. prob_null_trial(pi_rule).
    valid_positions : {rule: deviant positions}; defaults to get_valid_positions_per_rule().
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    rule_belief = {r: (1.0 - p_no_dev) / len(valid_positions) for r in valid_positions}
    rule_belief['null'] = p_no_dev
    return prior_tone_is_dev_given_rule_belief_and_prev_stds(j, rule_belief, valid_positions)


def prior_tone_is_dev_given_prev_rule_and_prev_stds(j, pi_rule, r_prev, valid_positions=None):
    """Probability that the tone at position j is the deviant, given S_j and the previous rule:
    p(d_k = j | S_j, r_{k-1} = r_prev), Eqs. (3)-(4).

    The rule belief before any tone is the row of the transition matrix: rho_k(r) = pi_rule[r_prev, r].

    pi_rule : transition matrix, pi_rule[r_prev, r] = p(r_k = r | r_{k-1} = r_prev)
        (including the null rule if there is one.)
    r_prev : rule of the previous trial, r_{k-1}.
    valid_positions : {rule: deviant positions}; defaults to get_valid_positions_per_rule().
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    r_prev = int(r_prev)
    rule_belief = {r: pi_rule[r_prev, r] for r in range(np.shape(pi_rule)[0])}
    return prior_tone_is_dev_given_rule_belief_and_prev_stds(j, rule_belief, valid_positions)


def prior_tone_is_dev_given_prev_rule_cue_and_prev_stds(j, pi_rule, r_prev, cue, cue_rule_lik=None,
                                                        valid_positions=None):
    """Probability that the tone at position j is the deviant, given S_j, the previous rule and the cue:
    p(d_k = j | S_j, r_{k-1} = r_prev, q_k = cue), Eqs. (3)-(4).

    Rule belief before any tone of the trial is heard = the rule belief after the cue (Bayes' rule):

        rho_k(r) = p(r_k = r | r_{k-1} = r_prev, q_k) = p(r_k = r, q_k | r_{k-1} = r_prev) / p(q_k | r_{k-1} = r_prev)
        p(r_k = r, q_k | r_{k-1} = r_prev) = p(q_k | r_k = r) * pi_rule[r_prev, r]

    pi_rule : transition matrix, pi_rule[r_prev, r] = p(r_k = r | r_{k-1} = r_prev).
    r_prev : rule of the previous trial, r_{k-1}.
    cue : key of cue_rule_lik, e.g. 'cue_1'.
    cue_rule_lik : {cue: {rule: p(q_k = cue | r_k = rule)}}; defaults to DEFAULT_CUE_RULE_LIK.
    valid_positions : {rule: deviant positions}; defaults to get_valid_positions_per_rule().
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    if cue_rule_lik is None:
        cue_rule_lik = DEFAULT_CUE_RULE_LIK
    r_prev = int(r_prev)
    prob_rule_and_cue = {r: cue_rule_lik[cue][r] * pi_rule[r_prev, r] for r in valid_positions}
    prob_cue = sum(prob_rule_and_cue.values())
    rule_belief = {r: prob_rule_and_cue[r] / prob_cue for r in valid_positions}
    return prior_tone_is_dev_given_rule_belief_and_prev_stds(j, rule_belief, valid_positions)


def prior_tone_is_dev_given_rule_belief_and_prev_stds(j, rule_belief, valid_positions):
    """Probability that the tone at position j is the deviant, given S_j and the rule belief before
    any tone rho_k: p(d_k = j | S_j), Eqs. (3)-(4).

        p(d_k = j | S_j) = sum_r p(d_k = j | S_j, r_k = r) * p(r_k = r | S_j)      (3)
        p(r_k = r | S_j) = p(r_k = r, S_j) / p(S_j)                                (4), Bayes' rule
        p(r_k = r, S_j)  = p(S_j | r_k = r) * rho_k(r),   p(S_j) = sum_r p(r_k = r, S_j)

    The standards heard so far thus update the rule belief: e.g. standards at positions 0,...,4
    exclude rule 0.

    rule_belief : {rule: rho_k(rule)}. A null rule is any key absent from valid_positions.
    valid_positions : {rule: deviant positions}.

    p(S_j) = 0 at a deviant position only when rule_belief has zeros (e.g. all its weight on rule 0,
    and j = 5): the result is then NaN.
    """
    # For positions without possible deviants (0, 1, 7).
    if not any(j in D for D in valid_positions.values()):
        return 0.0
    # For other positions:
    prob_rule_and_no_dev_so_far = {r: prob_no_dev_so_far(j, r, valid_positions) * rule_belief[r] for r in rule_belief}
    prob_no_dev_so_far_any_rule = sum(prob_rule_and_no_dev_so_far.values())
    rule_belief_given_no_dev_so_far = {r: prob_rule_and_no_dev_so_far[r] / prob_no_dev_so_far_any_rule
                                       for r in rule_belief}
    return sum(prior_tone_is_dev_given_rule_and_prev_stds(j, r, valid_positions) * rule_belief_given_no_dev_so_far[r]
               for r in rule_belief)



### Helper for priors

# p(q_k = cue | r_k = rule) for subjects 11/12/13, keyed by the value of the `cue` column, then by rule.
# cue_1 is associated with rule 1, cue_2 with rule 0, with probability 0.8. Rules and cues are equally
# frequent in this design, so these values are also p(r_k = rule | q_k = cue).
DEFAULT_CUE_RULE_LIK = {
    'cue_1': {0: 0.2, 1: 0.8},
    'cue_2': {0: 0.8, 1: 0.2},
}


def get_valid_positions_per_rule():
    """Deviant positions allowed by each rule, D_r: {rule: positions}.

    Rule 0: {2, 3, 4}. Rule 1: {4, 5, 6}. The null rule (rule 2 of subjects 04/05/06) is absent.
    """

    return {
        0: [2, 3, 4],
        1: [4, 5, 6]
    }


def prior_tone_is_dev_given_rule_and_prev_stds(j, r, valid_positions=None):
    """Probability that the tone at position j is the deviant, given S_j and the rule:
    p(d_k = j | S_j, r_k = r), Eq. (1).

    The deviant position is uniform over D_r, so:

        p(d_k = j | S_j, r_k = r) = p(d_k = j | r_k = r) / p(S_j | r_k = r) = 1 / #{m in D_r : m >= j}

    for j in D_r, and 0 otherwise (and for the null rule).

    valid_positions : {rule: deviant positions}; defaults to get_valid_positions_per_rule().
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()

    if r not in valid_positions:
        return 0

    if j in valid_positions[r]:
        return 1.0 / (max(valid_positions[r]) - j + 1)  # number of positions of D_r from j on
    else:
        return 0




def prob_no_dev_so_far(j, r, valid_positions=None):
    """p(S_j | r_k = r): probability that the tones at positions 0,...,j-1 are all standards under rule r, Eq. (2).

        p(S_j | r_k = r) = #{m in D_r : m >= j} / |D_r|

    Specific cases:
    - Takes 1 for the null rule
    - Takes 0 when every deviant position of rule r is before j: S_j then excludes rule r.
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    D = valid_positions.get(r, [])
    if len(D) == 0:
        return 1.0
    return sum(1 for m in D if m >= j) / len(D)


def prob_null_trial(pi_rule, valid_positions=None):
    """Stationary probability that a trial has no deviant (null rule).

    --> 1/11 for the 3x3 pi_rule of subjects 04/05/06
    --> 0 when every rule has deviant positions (2x2 pi_rule of subjects 11/12/13).
    """
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    pi_rule = np.asarray(pi_rule, dtype=float)
    null_rules = [r for r in range(pi_rule.shape[0]) if len(valid_positions.get(r, [])) == 0]
    if not null_rules:
        return 0.0

    # Stationary distribution: left eigenvector of pi_rule for eigenvalue 1, normalized to sum to 1.
    eigenvalues, eigenvectors = np.linalg.eig(pi_rule.T)
    stationary_dist = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    stationary_dist = stationary_dist / stationary_dist.sum()
    return float(sum(stationary_dist[r] for r in null_rules))
