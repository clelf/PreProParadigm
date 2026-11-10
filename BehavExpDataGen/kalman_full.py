import audit_gm
import numpy as np
import pandas as pd
import sys
import os

# Check which folder exists and append the correct path
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, WORKSPACE)
if os.path.exists(os.path.join(WORKSPACE, 'Kalman')):
    import Kalman.kalman as kf
elif os.path.exists(os.path.join(WORKSPACE, 'KalmanFilterViz1D')):
    import KalmanFilterViz1D.kalman as kf
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")


# Index of each process (context) along the N_ctx axes
STD_CONTEXT = 0
DEV_CONTEXT = 1


def combine_kf_predictions_until_j(inf_states, trial_k, j):
    """Combine fixed Kalman filter predictions of previous trials's timesteps
    with m-hypothesis predictons of the current trial until timestep j.

    So that for each context, x^c_{:j}(m) = x_{t=1:end of trial k-1} + x_{t=t_k:j}(m)

    """

    kf_k_m_j = np.stack([inf_states['hat_x_trials'][:, :trial_k, :, np.newaxis], # First duplicate predictions of previous trials for each m hypothesis
        inf_states['hat_x_trials_m'][:j, trial_k, :, :]], axis=0) # shape: (N_tones, N_trials, N_ctx, N_dpos)

    return kf_k_m_j


def combine_pdk_until_j(inf_states, trial_k, j):
    """Combine fixed p(d_k = m | H_t) of previous trials's timesteps
    with m-hypothesis p(d_k = m | H_t) of the current trial until timestep j.

    So that for each context, p(d_k = m | H_t)_{:j} = p(d_k = m | H_t)_{t=1:end of trial k-1} + p(d_k = m | H_t)_{t=t_k:j}

    """
    # N_tones, N_trials, N_dpos
    p_dk_m_j = np.stack([inf_states['hat_dk_trials'][:, :trial_k, :, :], # First duplicate predictions of previous trials for each m hypothesis
        inf_states['hat_dk_trials'][:j, trial_k, :, :]], axis=0) # shape: (N_tones, N_trials, N_ctx, N_dpos)

    return p_dk_m_j

def combine_pctx_until_j(inf_states, trial_k, j):
    """Combine fixed p(c_t = c | H_t) of previous trials's timesteps
    with m-hypothesis p(c_t = c | H_t) of the current trial until timestep j.

    So that for each context, p(c_t = c | H_t)_{:j} = p(c_t = c | H_t)_{t=1:end of trial k-1} + p(c_t = c | H_t)_{t=t_k:j}

    """
    pass
    # return p_c_m_j




class HAudIn():
    """Inference over the hierarchical auditory generative model.

    TODO:
    - Check deviant prior from previous work
    - Import and use KF
    - Deal with the fact that we use KF for predictions of y_{t+1}|y_{1:t}, not just x_t|y_{1:t}, so we should study p(y_{t+1}|y_{1:t})

    """

    def __init__(self,
                 y,
                 q,
                 estimate_process_noise,
                 n_transitions_before_using_estimates,
                 prior_state_means,
                 prior_state_vars,
                 tau_init,
                 b_init,
                 process_noise_init,
                 N_ctx=2,
                 N_tones=8,
                 N_rules=2,
                 rules_dpos_set = np.array([[2, 3, 4], [4, 5, 6]]),
                 pi_rules = np.array([[0.85, 0.15], [0.15, 0.85]]),
                 pi_cues = np.array([[0.8, 0.2], [0.2, 0.8]]),
                 observation_noise_sir = 0.02
    ):

        # Observed inputs
        self.y = y
        self.q = q
        self.N_ctx = N_ctx
        self.N_tones = N_tones
        self.N_rules = N_rules
        self.pi_rules = pi_rules
        self.pi_cues = pi_cues
        self.rules_dpos_set = rules_dpos_set
        self.dpos_set = np.unique(self.rules_dpos_set) # possible deviant positions m, in the order of the N_dpos axes below
        self.N_dpos = self.dpos_set.size
        self.dpos_min = np.min(self.dpos_set)
        self.N_cues = self.pi_cues.shape[1]
        self.N_trials = self.y.shape[0] // self.N_tones

        # Parameters
        self.observation_noise_sir = observation_noise_sir
        self.observation_noise_var = self.observation_noise_sir ** 2

        # Online Kalman filter (kf.kalman_online_step_multicontext):
        # - process noise estimated or given (if given, its value is the one of the start state of trial 0)
        # - and start-up rule of the estimates (one int, or one per process, shape (N_ctx,))
        self.estimate_process_noise = estimate_process_noise
        self.n_transitions_before_using_estimates = n_transitions_before_using_estimates

        # Start of the online Kalman filter (6a), one value per process, shape (N_ctx,), index STD_CONTEXT / DEV_CONTEXT:
        # - prior_state_means, prior_state_vars: mean and variance of the belief about each state at its first tone
        #   (x^std at the first tone of trial 0; x^dev during trial 0)
        # - tau_init, b_init: start values of tau and b
        # - process_noise_init: process noise variance (sigma_q**2); start value if estimate_process_noise, fixed value otherwise
        self.prior_state_means = np.asarray(prior_state_means, dtype=float)
        self.prior_state_vars = np.asarray(prior_state_vars, dtype=float)
        self.tau_init = np.asarray(tau_init, dtype=float)
        self.b_init = np.asarray(b_init, dtype=float)
        self.process_noise_init = np.asarray(process_noise_init, dtype=float)

        # Prior p(d_k = m | r_k = r): uniform over the deviant positions of rule r; shape (N_rules, N_dpos)
        self.P_D = np.zeros((self.N_rules, self.N_dpos))
        for r in range(self.N_rules):
            is_dpos_of_rule = np.isin(self.dpos_set, self.rules_dpos_set[r])
            self.P_D[r, is_dpos_of_rule] = 1 / self.rules_dpos_set[r].size

        # Minimum observations / trials required for EM (only used by kalman_filter_trial_draft)
        self.min_obs_for_em     = kf.MIN_OBS_FOR_EM
        self.min_trials_for_em  = self.min_obs_for_em

        # Re-organize y and q data in trials long of N_tones timesteps (flat order: t = N_tones * k + j):
        self.y_trials = self.y.reshape(self.N_trials, self.N_tones).T # shape: (N_tones, N_trials)
        self.q_trials = self.q.reshape(self.N_trials, self.N_tones).T # shape: (N_tones, N_trials); one cue per tone, the same within a trial

        # # Inferred outputs
        # self.inf_states = self.initialize_inf()

    def initialize_inf(self):
        # Inferred outputs, one row per tone t = N_tones * k + j: p(hat{d_k_t}|H_t), p(hat{r_k_t}|H_t), p(hat{c_t}|H_t), mean and variance of p(hat{y_t}|H_t)
        # hat_x = np.zeros(self.y.shape[0], self.N_ctx) # one for std, one for dev; inferred by Kalman
        hat_dk = np.zeros((self.y.shape[0], self.N_dpos)) # probabilities: N_tones x N_trials, N_dpos
        hat_rk = np.zeros((self.y.shape[0], self.N_rules)) # probabilities: N_tones x N_trials, N_rules
        hat_ck = np.zeros((self.y.shape[0], self.N_ctx)) # probabilities: N_tones x N_trials, N_ctx
        hat_y_mean = np.zeros(self.y.shape) # (15)
        hat_y_var = np.zeros(self.y.shape) # (16)
        # Mean and VARIANCE of the prediction of y_t by each process, given that the tone comes from it, p(y_t | c_t = c, H_t),
        # marginalized over m; shape (N_tones x N_trials, N_ctx); NaN where the process cannot be heard at t
        hat_y_mean_ctx = np.zeros((self.y.shape[0], self.N_ctx))
        hat_y_var_ctx = np.zeros((self.y.shape[0], self.N_ctx))

        # Inferred outputs, organized by trials: views of the arrays above (filling one fills the other)
        hat_dk_trials = hat_dk.reshape(self.N_trials, self.N_tones, -1).transpose(1, 0, 2) # shape: N_tones, N_trials, N_dpos
        hat_rk_trials = hat_rk.reshape(self.N_trials, self.N_tones, -1).transpose(1, 0, 2) # shape: N_tones, N_trials, N_rules
        hat_ck_trials = hat_ck.reshape(self.N_trials, self.N_tones, -1).transpose(1, 0, 2) # shape: N_tones, N_trials, N_ctx
        hat_y_mean_ctx_trials = hat_y_mean_ctx.reshape(self.N_trials, self.N_tones, -1).transpose(1, 0, 2) # shape: N_tones, N_trials, N_ctx
        hat_y_var_ctx_trials = hat_y_var_ctx.reshape(self.N_trials, self.N_tones, -1).transpose(1, 0, 2) # shape: N_tones, N_trials, N_ctx
        hat_y_mean_ctx_trials_m = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos)) # shape: N_tones, N_trials,  self.N_ctx, N_dpos; mean of the prediction of y_t by each process (= mean of the predicted state, (6b)), under each hypothesis m (self.dpos_set) // Probability distribution mean
        hat_y_var_ctx_trials_m = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos))  # shape: N_tones, N_trials, N_ctx, N_dpos; VARIANCE of the same prediction of y_t (= variance of the predicted state (6b) + observation noise) // Probability distribution variance
        lik_y_trials_m = np.zeros((self.N_tones, self.N_trials, self.N_dpos)) # shape: N_tones, N_trials, N_dpos; l_t(m) (7), one per hypothesis m (self.dpos_set)
        # self.hat_y_trials = np.zeros((self.N_tones, self.N_trials)) # shape: N_tones, N_trials; prediction marginalized over hypotheses m

        # # Theoretical
        # # 1: p(c_t = dev | d_k = m) for the tone at position j: 1 if j == m, else 0; 0: p(c_t = std | d_k = m) shape (N_tones, N_dpos)
        # self.P_C = (np.arange(self.N_tones)[:, np.newaxis] == self.dpos_set[np.newaxis, :]).astype(float)

        # p(c_t = c | H_t, d_k = m); shape (N_tones, N_trials, N_ctx, N_dpos) # NOTE: only used by kalman_filter_trial_draft
        p_c_m = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos))

        # p(c_t = c | H_t); shape (N_tones, N_trials, N_ctx) # NOTE: inferred at end of trials only. Only used by kalman_filter_trial_draft
        p_c = np.zeros((self.N_tones, self.N_trials, self.N_ctx))

        # hat_dk at end of trials # NOTE: only used by kalman_filter_trial_draft
        hat_dk_end = np.zeros(self.N_trials)

        # Beliefs after the last tone of each trial: p(r_k = r | H_{<k+1}) (21) and w_k(m) = p(d_k = m | H_{<k+1}) (22)
        rk_end_trials = np.zeros((self.N_trials, self.N_rules))
        dk_end_trials = np.zeros((self.N_trials, self.N_dpos))

        # Storing variable
        inf_states = {
            'hat_dk': hat_dk_end,
            'hat_rk': hat_rk,
            'hat_ck': hat_ck,
            'hat_y_mean': hat_y_mean,
            'hat_y_var': hat_y_var,
            'hat_dk_trials': hat_dk_trials,
            'hat_rk_trials': hat_rk_trials,
            'hat_ck_trials': hat_ck_trials,
            'hat_y_mean_ctx': hat_y_mean_ctx,
            'hat_y_var_ctx': hat_y_var_ctx,
            'hat_y_mean_ctx_trials': hat_y_mean_ctx_trials,
            'hat_y_var_ctx_trials': hat_y_var_ctx_trials,
            'hat_y_mean_ctx_trials_m': hat_y_mean_ctx_trials_m,
            'hat_y_var_ctx_trials_m': hat_y_var_ctx_trials_m,
            'lik_y_trials_m': lik_y_trials_m,
            # 'P_C': P_C,
            'p_c_m': p_c_m,
            'p_c': p_c,
            'kf_params_m': [],
            'rk_end_trials': rk_end_trials,
            'dk_end_trials': dk_end_trials,
            'kf_state': None, # merged predictor after the last trial processed, from which every hypothesis m of the next trial starts (6a), (23)-(24)
            'kf_states_m': [], # predictor after the last tone of the current trial, one per hypothesis m (self.dpos_set order)
            'kf_states_end_trials': [], # merged predictor after each trial
            # 'hat_y_trials': self.hat_y_trials
        }

        return inf_states


    def p_ctx_i(self, p_dk_trials, j, k, m):
            """Compute the probability of each context given previous history, at timestep j of trial k.
            At any timestep j, p(c_t = dev | H_t) = p(d_k = j | H_t)
            p_ctx[0]: std context probability, p_ctx[1]: dev context probability
            p_dk_trials is (N_tones, N_trials, N_dpos), so we access p_dk_trials[j, k, j - self.dpos_min]
            """
            return np.array([1 - p_dk_trials[j, k, m], p_dk_trials[j, k, m]])


    def p_ctx(self, p_dk_trials):
        """Same as p_ctx_i but for all timesteps and trials available in p_dk_trials.
        Return p_ctx of shape (N_tones, N_trials, N_ctx), where p_ctx[:, :, 0] is the std context probability and p_ctx[:, :, 1] is the dev context probability.
        """
        p_ctx = np.zeros((p_dk_trials.shape[0], p_dk_trials.shape[1], self.N_ctx))
        for k in range(p_dk_trials.shape[1]): # p_dk_trials.shape[1] ~ N_trials
            for j in range(self.dpos_min, self.dpos_min + p_dk_trials.shape[2]): # p_dk_trials.shape[0] ~ N_tones
                p_ctx[j, k, :] = self.p_ctx_i(p_dk_trials, j, k, j - self.dpos_min) # p(d_k = j | H_t) for every j that can be considered a d_k (i.e. the m dimension)
        return p_ctx

    def kalman_filter_trial(self, inf_states, k, observation_noise=None):
        """Step 4 of deviant_position_inference_maths.md (preparatory loop): for each hypothesis m about the deviant
        position of trial k, runs the online Kalman filter of both processes (kf.kalman_online_step_multicontext,
        tau and b estimated online) over the N_tones tones of trial k, starting from the merged predictor of
        trial k-1, inf_states['kf_state'] (6a).

        Under m, the tone at position j comes from dev if j == m and from std otherwise (context probabilities 0 or
        1); std moves at every tone, dev moves once per trial, at j == m, from trial 1 (table of Step 4).

        Fills, for trial k and each m: inf_states['hat_y_mean_ctx_trials_m'], inf_states['hat_y_var_ctx_trials_m'] (prediction of
        y_t by each process, (6): mean of the predicted state, and its VARIANCE plus the observation noise), and
        inf_states['kf_states_m'] (filtered state and running sums after the last tone, one dict per m, in the order
        of self.dpos_set).
        """
        if observation_noise is None:
            observation_noise = self.observation_noise_var

        if k == 0:
            # Beginning of trial 0 (6a): belief about each state at its first tone, start values of the parameters,
            # empty running sums (keys of kf._init_online_kf_state)
            inf_states['kf_state'] = {
                'mu': self.prior_state_means,
                'var': self.prior_state_vars,
                'tau': self.tau_init,
                'b': self.b_init,
                'process_noise': self.process_noise_init,
                'tau_numerator_sum': np.zeros(self.N_ctx),
                'tau_denominator_sum': np.zeros(self.N_ctx),
                'b_sum': np.zeros(self.N_ctx),
                'process_noise_sum': np.zeros(self.N_ctx),
                'n_transitions': np.zeros(self.N_ctx, dtype=int),
            }

        kf_states_m = []
        for i_m, m in enumerate(self.dpos_set):
            # Every hypothesis starts from the same state and running sums (6a)
            kf_state = inf_states['kf_state']

            for j in range(self.N_tones):
                # Under m, the tone at position j comes from one process only (context probabilities 0 or 1)
                context_probabilities = np.zeros(self.N_ctx)
                if j == m:
                    context_probabilities[DEV_CONTEXT] = 1
                else:
                    context_probabilities[STD_CONTEXT] = 1

                # std moves at every tone except the first tone of trial 0, whose state is the prior itself; dev moves once
                # per trial, at the deviant's position, from trial 1 (during trial 0 its state is the prior itself)
                contexts_moving = np.zeros(self.N_ctx, dtype=bool)
                contexts_moving[STD_CONTEXT] = not (k == 0 and j == 0)
                contexts_moving[DEV_CONTEXT] = (j == m) and (k >= 1)

                pred_means, pred_vars, kf_state = kf.kalman_online_step_multicontext(
                    self.y_trials[j, k], context_probabilities, kf_state, observation_noise,
                    self.estimate_process_noise, self.n_transitions_before_using_estimates, contexts_moving
                )
                inf_states['hat_y_mean_ctx_trials_m'][j, k, :, i_m] = pred_means
                inf_states['hat_y_var_ctx_trials_m'][j, k, :, i_m] = pred_vars

            kf_states_m.append(kf_state)

        inf_states['kf_states_m'] = kf_states_m
        return inf_states

    def kalman_filter_trial_draft(self, inf_states, k, observation_noise=None, n_iter=5):
        """Applies the online EM-fitting Kalman filter to previous observations
        (of the current trial, under the hypothesis that m is the dpos of the current trial;
        the previous trials are already fixed)
        """
        if observation_noise is None:
            observation_noise = self.observation_noise_var

        # Get most recent estimation of d_{k-1}
        hat_dk_prev = inf_states['hat_dk'][-1]

        # Update p(ctx_t = c)_{:end of trial k-1} by adding last trial's p(ctx_t = c)_{t=j for every j in k-1} according to k-1 end's hat_dk
        inf_states['p_c'][:, k-1, :] = inf_states['p_c_m'][:, k-1, :, hat_dk_prev - self.dpos_min]

        # Duplicate p(ctx_t = c)_{:end of trial k-1} m times
        # and add p(ctx_t = c | d_k = m)_{t in trial k} to it for each m
        p_c_k_m = np.repeat(inf_states['p_c'][:, :k, :, np.newaxis], self.N_dpos, axis=-1) # shape: N_tones, N_trials, N_ctx, N_dpos
        p_dk_k_m = []
        for m in range(self.N_dpos):
            p_dk_m = np.array([1 if m==j else 0 for j in self.N_tones])
            p_dk_k_m.append(p_dk_m)
        p_dk_k_m = np.array([p_dk_k_m]).transpose() # shape: (N_dpos, N_tones)
        p_c_k_m = np.stack([p_c_k_m, p_dk_k_m], axis=0) # shape: (N_tones, N_trials, N_ctx, N_dpos)

        # Flatten p_c_k_m (N_tones, N_trials) axes into a continuous one:
        p_c_k_m_flat = p_c_k_m.transpose(1, 0, 2, 3).reshape(self.N_trials * self.N_tones, self.N_ctx, self.N_dpos)

        # Infer for each m
        for m in range(self.N_dpos):
            # Run online-fit KF
            _, _, _, per_ctx_mu_pred, per_ctx_var, kf_params = kf.kalman_online_fit_predict_multicontext(
                self.y[:(k+1)*self.N_tones], p_c_k_m_flat[:,:,m],
                return_per_ctx=True, return_params=True,
                observation_noise=observation_noise, n_iter=5,
            )

            # Only store the predictions for the current trial k
            inf_states['hat_y_mean_trials_m'][:, k, :, m - self.dpos_min] = per_ctx_mu_pred[-self.N_tones:,:]
            inf_states['hat_y_var_trials_m'][:, k, :, m - self.dpos_min] = per_ctx_var[-self.N_tones:,:]
            inf_states['kf_params_m'].append(kf_params)

        return inf_states


    def merge_kf_states_over_dpos(self, kf_states_m, dpos_weights, kf_state_trial_start):
        """Step 7 of deviant_position_inference_maths.md: predictor from which every hypothesis m of the next trial
        starts, merged per process over the hypotheses m of the current trial.

        State: mean and variance of the mixture of the branches' filtered states, weighted by w_k(m) (23).
        Running sums: averaged with the same weights (24). Estimates: recomputed from the merged sums as in (6e)
        (kf.online_parameter_estimates); a process whose start-up is not over keeps the values of
        kf_state_trial_start, which are the same in every branch.

        Parameters
        ----------
        kf_states_m : list of dict
            Predictor after the last tone of the trial, one per hypothesis m (self.dpos_set order).
        dpos_weights : np.array, shape (N_dpos,)
            w_k(m) = p(d_k = m | H_{<k+1}) (22).
        kf_state_trial_start : dict
            Predictor from which every hypothesis of the trial started (6a).

        Returns
        -------
        merged_kf_state : dict
            Same keys as each dict of kf_states_m.
        """
        weights = dpos_weights[:, np.newaxis] # (N_dpos, 1), against arrays of shape (N_dpos, N_ctx)

        mu_m = np.array([kf_state['mu'] for kf_state in kf_states_m]) # (N_dpos, N_ctx)
        var_m = np.array([kf_state['var'] for kf_state in kf_states_m]) # (N_dpos, N_ctx)
        merged_mu = np.sum(weights * mu_m, axis=0)
        merged_var = np.sum(weights * var_m, axis=0) + np.sum(weights * (mu_m - merged_mu) ** 2, axis=0)

        merged_kf_state = {'mu': merged_mu, 'var': merged_var}
        for sum_name in ['tau_numerator_sum', 'tau_denominator_sum', 'b_sum', 'process_noise_sum']:
            sums_m = np.array([kf_state[sum_name] for kf_state in kf_states_m]) # (N_dpos, N_ctx)
            merged_kf_state[sum_name] = np.sum(weights * sums_m, axis=0)

        # Number of moves of each process: the same under every m
        merged_kf_state['n_transitions'] = kf_states_m[0]['n_transitions']

        merged_kf_state['tau'], merged_kf_state['b'], merged_kf_state['process_noise'] = kf.online_parameter_estimates(
            merged_kf_state['tau_numerator_sum'], merged_kf_state['tau_denominator_sum'], merged_kf_state['b_sum'],
            merged_kf_state['process_noise_sum'], merged_kf_state['n_transitions'],
            kf_state_trial_start['tau'], kf_state_trial_start['b'], kf_state_trial_start['process_noise'],
            np.ones(self.N_ctx, dtype=bool), self.estimate_process_noise, self.n_transitions_before_using_estimates
        )
        return merged_kf_state


    def trial_inference(self, inf_states, trial_k):

        # Trial start:
        # - predict prior about current trial's rule r_k given rule transition
        #   and last believes about possible previous trial's rule r_{k-1}
        #   --> for a given r: rho_k_00(r) = p(r_k = r | H_k_0) = sum_r'[ p(r_{k-1} = r' | H_k_0) * p(r_k = r | r_{k-1} = r') ]
        if trial_k == 0:
            Rk_t_0_prior = np.ones(self.N_rules) / self.N_rules # uniform prior for first trial, e.g. if N_rules = 2, then [0.5, 0.5]; shape: (N_rules,)
        else:
            Rk_t_0_prior = inf_states['rk_end_trials'][trial_k-1, :] @ self.pi_rules # belief after the last tone of trial k-1 (21); shape: (N_rules,)

        # - observe cue q_k
        # - update prior about r_k
        #   --> for a given r:  rho_k_t0(r) = p(r_k = r | H_k_0, q_k) = p(q_k | r_k = r) * rho_k_00(r) / sum_r'[ p(q_k | r_k = r') * rho_k_00(r') ]
        q_k = self.q_trials[0, trial_k] # cue of trial k: column index of pi_cues
        Rk_t_0_posterior = Rk_t_0_prior * self.pi_cues[:, q_k] / np.sum(Rk_t_0_prior * self.pi_cues[:, q_k]) # operation: (N_rules,) * (N_rules,) [element-wise] / (1) = (N_rules,)

        # - predict prior believes about possible deviant locations: P_k(m)  = p(d_k = m | H_{t_k}) = Σ_r p(d_k = m | r_k = r) · ρ_k(r)
        Dk_m = Rk_t_0_posterior @ self.P_D # shape: (N_dpos,)

        # Prepare KF predictions for each m hypothesis (Step 4)
        kf_state_trial_start = inf_states['kf_state'] # merged predictor of trial k-1, also needed by the merge (Step 7)
        inf_states = self.kalman_filter_trial(inf_states, trial_k)
        if trial_k == 0:
            kf_state_trial_start = inf_states['kf_state'] # set at the beginning of trial 0 by kalman_filter_trial

        # Under m, the tone at position j is predicted by the process that m places there (6): dev if j == m, std otherwise
        i_m_all = np.arange(self.N_dpos)

        # Trial loop
        for j in range(0, self.N_tones):
            t = trial_k * self.N_tones + j

            # (L_t(m)) Compute joint likelihood of all previously heard y (y_{t_k:t-1}) according to dpos_k = m hypothesis (8); empty product = 1 at j = 0
            joint_hat_y_m = np.prod(inf_states['lik_y_trials_m'][:j, trial_k, :], axis=0) # shape: (N_dpos,)

            # (Z_t) Likelihood of all previously heard y, marginalizing over m (9)
            joint_hat_y = np.sum(Dk_m * joint_hat_y_m) # sum((N_dpos,) * (N_dpos,)) = scalar

            # Probability of r_k = r and d_k = m given history until t/j (excluded): B_rm(r, m) = p(r_k = r, d_k = m | H_t) (10)
            B_rm = Rk_t_0_posterior[:, np.newaxis] * self.P_D * joint_hat_y_m[np.newaxis, :] / joint_hat_y # shape(N_rules, N_dpos) because: (N_rules, 1) * (N_rules, N_dpos) * (1, N_dpos) / (1,) = (N_rules, N_dpos) (adding to replace (...,) to (..., 1) for matrix multiplication)

            # Updated beliefs about r_k and d_k (11), (12)
            p_dk_j = np.sum(B_rm, axis=0) # sum over r; verify that it yields the same as Dk_m * joint_hat_y_m / joint_hat_y // shape: (N_dpos,)
            p_rk_j = np.sum(B_rm, axis=1) # sum over m; verify that it yields the same as rho_k_t0 * (np.sum(P_D * joint_hat_y_m) / joint_hat_y // shape: (N_rules,)

            # Store p_dk_j and p_rk_j in relevant associated variables:
            inf_states['hat_dk_trials'][j, trial_k, :] = p_dk_j
            inf_states['hat_rk_trials'][j, trial_k, :] = p_rk_j

            # p(c_t = c | H_t) (13): p(c_t = dev | H_t) = p(d_k = j | H_t), 0 if j is not a possible deviant position
            is_dpos_j = self.dpos_set == j
            if np.any(is_dpos_j):
                inf_states['hat_ck_trials'][j, trial_k, :] = self.p_ctx_i(inf_states['hat_dk_trials'], j, trial_k, np.flatnonzero(is_dpos_j)[0])
            else:
                inf_states['hat_ck_trials'][j, trial_k, :] = np.array([1., 0.])

            # Prediction of y_t under each m (6): mu_t(m), S_t(m), from the process that m places at position j
            context_of_tone_m = np.where(is_dpos_j, DEV_CONTEXT, STD_CONTEXT) # shape: (N_dpos,)
            hat_y_mean_m = inf_states['hat_y_mean_ctx_trials_m'][j, trial_k, context_of_tone_m, i_m_all] # shape: (N_dpos,)
            hat_y_var_m = inf_states['hat_y_var_ctx_trials_m'][j, trial_k, context_of_tone_m, i_m_all] # shape: (N_dpos,)

            # Mean and variance of the predictive density p(y_t | H_t), mixture over m (15), (16)
            inf_states['hat_y_mean'][t] = np.sum(p_dk_j * hat_y_mean_m)
            inf_states['hat_y_var'][t] = np.sum(p_dk_j * hat_y_var_m) + np.sum(p_dk_j * (hat_y_mean_m - inf_states['hat_y_mean'][t]) ** 2)

            # Prediction of y_t by each process, given that the tone comes from it: mean and variance of
            # p(y_t | c_t = c, H_t) = sum_m p(y_t | c_t = c, d_k = m, H_t) p(d_k = m | c_t = c, H_t), where
            # p(d_k = m | c_t = c, H_t) = p(c_t = c | d_k = m) p(d_k = m | H_t) / p(c_t = c | H_t) (Bayes' rule; (11), (13)),
            # and p(c_t = c | d_k = m) is 1 if m places process c at position j, 0 otherwise
            # - std: hypotheses m != j, whose predictions (6) are std's; NaN if p(c_t = std | H_t) = 0
            p_std_j = inf_states['hat_ck_trials'][j, trial_k, STD_CONTEXT]
            if p_std_j > 0:
                std_weights_m = np.where(is_dpos_j, 0., p_dk_j) / p_std_j # shape: (N_dpos,)
                hat_y_mean_std = np.sum(std_weights_m * hat_y_mean_m)
                hat_y_var_std = np.sum(std_weights_m * hat_y_var_m) + np.sum(std_weights_m * (hat_y_mean_m - hat_y_mean_std) ** 2)
            else:
                hat_y_mean_std = np.nan
                hat_y_var_std = np.nan
            inf_states['hat_y_mean_ctx_trials'][j, trial_k, STD_CONTEXT] = hat_y_mean_std
            inf_states['hat_y_var_ctx_trials'][j, trial_k, STD_CONTEXT] = hat_y_var_std

            # - dev: hypothesis m = j only, prediction (6); NaN if j is not a possible deviant position
            if np.any(is_dpos_j):
                inf_states['hat_y_mean_ctx_trials'][j, trial_k, DEV_CONTEXT] = hat_y_mean_m[is_dpos_j][0]
                inf_states['hat_y_var_ctx_trials'][j, trial_k, DEV_CONTEXT] = hat_y_var_m[is_dpos_j][0]
            else:
                inf_states['hat_y_mean_ctx_trials'][j, trial_k, DEV_CONTEXT] = np.nan
                inf_states['hat_y_var_ctx_trials'][j, trial_k, DEV_CONTEXT] = np.nan

            # # Evaluate likelihood of y_t, marginalized over the different m hypotheses -- the value stored at j is thus the prediction from j-1
            # inf_states['lik_y_trials'][j, trial_k, :] = np.sum(p_dk_j * inf_states['lik_y_trials_m'][j, trial_k, :]) # sum over m; verify that it yields the same as: p(d_k = j | H_t) · N(y_t; μ^dev_k, S^dev_k)  +  Σ_{m≠j} p(d_k = m | H_t) · N(y_t; μ^std_t(m), S^std_t(m))

            # Evaluate likelihood of y_t under the KF predictions under different m hypotheses (7)
            inf_states['lik_y_trials_m'][j, trial_k, :] = kf.likelihood_observation(self.y_trials[j, trial_k], hat_y_mean_m, hat_y_var_m) # Evaluate y_t under the Kalman-estimated predictions for y_t, for different m hypotheses

            # End of trial:
            if j == self.N_tones - 1:
                # Keep one belief about r_k and d_k (Step 6), and one KF track for this trial (Step 7)

                # - L_end(m) = p(y_{8k}, ..., y_{8k+7} | d_k = m, H_{t_k}): product over ALL N_tones tones (18)
                L_end_m = np.prod(inf_states['lik_y_trials_m'][:, trial_k, :], axis=0) # shape: (N_dpos,)

                # - Z_end = p(y_{8k}, ..., y_{8k+7} | H_{t_k}) = sum_m P_k(m) * L_end(m) (19)
                Z_end = np.sum(Dk_m * L_end_m)

                # - b_end(r, m) = p(r_k = r, d_k = m | H_{<k+1}) = rho_k(r) * p(d_k = m | r_k = r) * L_end(m) / Z_end (20)
                B_end_rm = Rk_t_0_posterior[:, np.newaxis] * self.P_D * L_end_m[np.newaxis, :] / Z_end # shape: (N_rules, N_dpos); verify: same as B_rm (j = 7) * lik_y_trials_m[7, trial_k, :] / p(y_{8k+7} | H_t)
                inf_states['rk_end_trials'][trial_k] = np.sum(B_end_rm, axis=1) # p(r_k = r | H_{<k+1}), sum over m (21); used by the rule prior of trial k+1
                inf_states['dk_end_trials'][trial_k] = np.sum(B_end_rm, axis=0) # w_k(m) = p(d_k = m | H_{<k+1}), sum over r (22)

                # - Merge the predictors of the hypotheses m: start of every hypothesis of trial k+1 (23), (24)
                inf_states['kf_state'] = self.merge_kf_states_over_dpos(
                    inf_states['kf_states_m'], inf_states['dk_end_trials'][trial_k], kf_state_trial_start
                )
                inf_states['kf_states_end_trials'].append(inf_states['kf_state'])

        return inf_states


    def inference(self, init_states=True):
        if init_states:
            inf_states = self.initialize_inf()

        # No EM warm-up with the online Kalman filter: every trial is inferred, from trial 0
        for trial_k in range(self.N_trials):
            inf_states = self.trial_inference(inf_states, trial_k)

        return inf_states

    
    def save_inference(self, inf_states, saving_path):
        """Save the per-tone inference outputs of one sequence as a CSV at saving_path (full file path), one row per
        tone t = N_tones * k + j, with the HAudIn start values and settings as extra (constant) columns."""
        inf_df = pd.DataFrame({
            'trial_no': np.repeat(np.arange(self.N_trials), self.N_tones),
            'relative_pos': np.tile(np.arange(self.N_tones), self.N_trials),
            'observation': self.y,
            'cue_index': self.q,
            'hat_y_mean': inf_states['hat_y_mean'],
            'hat_y_var': inf_states['hat_y_var'],
            'hat_y_mean_std': inf_states['hat_y_mean_ctx'][:, STD_CONTEXT],
            'hat_y_var_std': inf_states['hat_y_var_ctx'][:, STD_CONTEXT],
            'hat_y_mean_dev': inf_states['hat_y_mean_ctx'][:, DEV_CONTEXT],
            'hat_y_var_dev': inf_states['hat_y_var_ctx'][:, DEV_CONTEXT],
            'p_dev': inf_states['hat_ck'][:, DEV_CONTEXT],
        })

        # p(d_k = m | H_t), one column per deviant position m; p(r_k = r | H_t), one column per rule r
        p_dk = inf_states['hat_dk_trials'].transpose(1, 0, 2).reshape(-1, self.N_dpos)
        for i_m, m in enumerate(self.dpos_set):
            inf_df[f'p_dpos_{m}'] = p_dk[:, i_m]
        for r in range(self.N_rules):
            inf_df[f'p_rule_{r}'] = inf_states['hat_rk'][:, r]

        # Start values and settings (one value per process -> one column per process)
        inf_df['estimate_process_noise'] = self.estimate_process_noise
        inf_df['observation_noise_sir'] = self.observation_noise_sir
        start_values = {
            'n_transitions_before_using_estimates': np.broadcast_to(self.n_transitions_before_using_estimates, (self.N_ctx,)),
            'prior_state_mean': self.prior_state_means,
            'prior_state_var': self.prior_state_vars,
            'tau_init': self.tau_init,
            'b_init': self.b_init,
            'process_noise_init': self.process_noise_init,
        }
        for name, values in start_values.items():
            inf_df[f'{name}_std'] = values[STD_CONTEXT]
            inf_df[f'{name}_dev'] = values[DEV_CONTEXT]

        inf_df.to_csv(saving_path, index=False)


if __name__ == '__main__':
    pass
