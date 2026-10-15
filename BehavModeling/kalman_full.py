import audit_gm
import numpy as np
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




class HAudIN():
    """Inference over the hierarchical auditory generative model.
    
    TODO:
    - Check deviant prior from previous work
    - Import and use KF
    - Deal with the fact that we use KF for predictions of y_{t+1}|y_{1:t}, not just x_t|y_{1:t}, so we should study p(y_{t+1}|y_{1:t})
    
    """

    def __init__(self,
                 y,
                 q, 
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

        # Prior p(d_k = m | r_k = r): uniform over the deviant positions of rule r; shape (N_rules, N_dpos)
        self.P_D = np.zeros((self.N_rules, self.N_dpos))
        for r in range(self.N_rules):
            is_dpos_of_rule = np.isin(self.dpos_set, self.rules_dpos_set[r])
            self.P_D[r, is_dpos_of_rule] = 1 / self.rules_dpos_set[r].size

        # Re-organize y and q data in trials long of N_tones timesteps:
        self.y_trials = self.y.reshape(self.N_tones, self.N_trials) # shape: (N_tones, N_trials)
        self.q_trials = self.q.reshape(self.N_tones, self.N_trials) # shape: (N_tones, N_trials)

        # # Inferred outputs
        # self.inf_states = self.initialize_inf()

    def initialize_inf(self):
        # Inferred outputs: # p(hat{x_t}|H_t), p(hat{d_k_t}|H_t), p(hat{r_k_t}|H_t), p(hat{y_t}|H_t)
        hat_x = np.zeros(self.y.shape[0], self.N_ctx) # one for std, one for dev; inferred by Kalman
        hat_dk = np.zeros(self.y.shape[0], self.N_dpos) # probabilities
        hat_rk = np.zeros(self.y.shape[0], self.N_rules) # probabilities
        hat_ck = np.zeros(self.y.shape[0], self.N_ctx) # probabilities
        hat_y_mean = np.zeros(self.y.shape)
        hat_y_var = np.zeros(self.y.shape)

        # Inferred outputs, organized by trials
        hat_dk_trials = self.hat_dk.reshape(self.N_tones, self.N_trials, -1) # shape: N_tones, N_trials, N_dpos
        hat_rk_trials = self.hat_rk.reshape(self.N_tones, self.N_trials, -1) # shape: N_tones, N_trials, N_rules
        hat_ck_trials = self.hat_ck.reshape(self.N_tones, self.N_trials, -1) # shape: N_tones, N_trials, N_ctx
        hat_y_mean_trials = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos)) # shape: N_tones, N_trials,  self.N_ctx, N_dpos; one prediction per context and hypothesis m (self.dpos_set) // Probability distribution mean
        hat_y_var_trials = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos))  # shape: N_tones, N_trials, N_dpos; one prediction per hypothesis m (self.dpos_set) // Probability distribution variance
        lik_y_trials_m = np.zeros((self.N_tones, self.N_trials, self.N_dpos)) # shape: N_tones, N_trials, N_dpos; one prediction per hypothesis m (self.dpos_set)
        # self.hat_y_trials = np.zeros((self.N_tones, self.N_trials)) # shape: N_tones, N_trials; prediction marginalized over hypotheses m
        
        # # Theoretical
        # # 1: p(c_t = dev | d_k = m) for the tone at position j: 1 if j == m, else 0; 0: p(c_t = std | d_k = m) shape (N_tones, N_dpos)
        # self.P_C = (np.arange(self.N_tones)[:, np.newaxis] == self.dpos_set[np.newaxis, :]).astype(float)    

        # p(c_t = c | H_t, d_k = m); shape (N_tones, N_trials, N_ctx, N_dpos)
        p_c_m = np.zeros((self.N_tones, self.N_trials, self.N_ctx, self.N_dpos))

        # p(c_t = c | H_t); shape (N_tones, N_trials, N_ctx) # NOTE: inferred at end of trials only.
        p_c = np.zeros((self.N_tones, self.N_trials, self.N_ctx))

        # hat_dk at end of trials
        hat_dk = np.zeros(self.N_trials)
                
        # Storing variable
        inf_states = {
            'hat_dk': hat_dk,
            'hat_rk': hat_rk,
            'hat_ck': hat_ck,
            'hat_y_mean': hat_y_mean,
            'hat_y_var': hat_y_var,
            'hat_dk_trials': hat_dk_trials,
            'hat_rk_trials': hat_rk_trials,
            'hat_ck_trials': hat_ck_trials,
            'hat_y_mean_trials': hat_y_mean_trials,
            'hat_y_var_trials': hat_y_var_trials,
            'lik_y_trials_m': lik_y_trials_m,
            # 'P_C': P_C,
            'p_c_m': p_c_m,
            'p_c': p_c,
            'hat_dk': hat_dk,
            'kf_params_m': []
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
    

    def kalman_filter_m(self, inf_states, k, j, m, observation_noise=None, n_iter=5):
        """
        Obsolete: draft
        
        Applies the online EM-fitting Kalman filter to previous observations
        (of the current trial, under the hypothesis that m is the dpos of the current trial; 
        the previous trials are already fixed)
        """
        if observation_noise is None:
            observation_noise = self.observation_noise

        # Combine Kalman filter predictions until timestep j included 
        y_m_j = self.y[: k*self.N_tones + j + 1] # TODO: check if correct
        p_dk_j = combine_pdk_until_j(inf_states['hat_dk_trials'], k, j)
        p_dk_m_j = p_dk_j[:, :, m - self.dpos_min]
        p_c_m_j = self.p_ctx(p_dk_m_j)
        _, _, _, per_ctx_mu_pred, per_ctx_var, kf_params = kf.kalman_online_fit_predict_multicontext(
            y_m_j, p_c_m_j, n_iter=5, return_per_ctx=True, return_params=True, observation_noise=observation_noise
        )
        # TODO: handle return dimensions (if input flattened --> restructure in _trials shape)
        return per_ctx_mu_pred, per_ctx_var, kf_params

    def kalman_filter_trial(self, inf_states, k, observation_noise=None, n_iter=5):
        """Applies the online EM-fitting Kalman filter to previous observations
        (of the current trial, under the hypothesis that m is the dpos of the current trial; 
        the previous trials are already fixed)
        """
        if observation_noise is None:
            observation_noise = self.observation_noise

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
            # Run online fit KF
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
    
    
    def inference(self, init_states=True):
        if init_states:
            inf_states = self.initialize_inf()

        # To-do: deal with first trial
        
        for trial_k in range(1, self.N_trials):
            self.trial_inference(inf_states, trial_k)


    def trial_inference(self, inf_states, trial_k):

        # Trial start:
        # - predict prior about current trial's rule r_k given rule transition
        #   and last believes about possible previous trial's rule r_{k-1}
        #   --> for a given r: rho_k_00(r) = p(r_k = r | H_k_0) = sum_r'[ p(r_{k-1} = r' | H_k_0) * p(r_k = r | r_{k-1} = r') ]
        if trial_k == 0:
            Rk_t_0_prior = np.ones((1, self.N_rules)) / self.N_rules # uniform prior for first trial, e.g. if N_rules = 2, then [0.5, 0.5]
        else:
            Rk_t_0_prior = inf_states['hat_rk_trials'][-1, trial_k-1, :] @ self.pi_rules # shape: (1, N_rules)
        
        # - observe cue q_k
        # - update prior about r_k
        #   --> for a given r:  rho_k_t0(r) = p(r_k = r | H_k_0, q_k) = p(q_k | r_k = r) * rho_k_00(r) / sum_r'[ p(q_k | r_k = r') * rho_k_00(r') ]
        Rk_t_0_posterior = Rk_t_0_prior * self.pi_cues[:, self.q_trials[0, trial_k]] / np.sum(Rk_t_0_prior * self.pi_cues[:, self.q_k[trial_k]]) # operation: (1, N_rules) * (N_rules, N_rules) [diag: numpy element-wise product] / (1) = (1, N_rules) # TODO: decide shape of q and q_trials

        # - predict prior believes about possible deviant locations: P_k(m)  = p(d_k = m | H_{t_k}) = Σ_r p(d_k = m | r_k = r) · ρ_k(r)
        Dk_m = Rk_t_0_posterior @ self.P_D # shape: (1, N_dpos)

        # Prepare KF predictions for each m hypothesis
        inf_states = self.kalman_filter_trial(inf_states, trial_k)

        # Trial loop
        for j in range(0, self.N_tones):
            if j == 0:
                p_rk_j = Rk_t_0_posterior  # (1, N_rules)
                p_dk_j = Dk_m  # (1, N_dpos)
                            
            
            # (L_t(m)) Compute joint likelihood of all previously inferred y (hat{y_{1:j-1}}) according to dpos_k = m hypothesis
            joint_hat_y_m = np.prod(inf_states['lik_y_trials_m'][:j, trial_k, m - self.dpos_min]) # TODO: what happens when j=0?

            # (Z_t) Likelihood of all previously inferred y, marginalizing over m
            joint_hat_y = np.sum(Dk_m * np.array(joint_hat_y_m)) # shape? --> sum((1, N_dpos) x (N_dpos,) = (1,)) = (1,)
            
            # Probability of r_k = r and d_k = m given history until t/j (excluded): B_rm(r, m) = p(r_k = r, d_k = m | H_t)
            B_rm = Rk_t_0_posterior[:, np.newaxis] * self.P_D * joint_hat_y_m[np.newaxis, :] / joint_hat_y # shape(N_rules, N_dpos) because: (N_rules, 1) * (N_rules, N_dpos) * (1, N_dpos) / (1,) = (N_rules, N_dpos) (adding to replace (...,) to (..., 1) for matrix multiplication)

            # Updated beliefs about r_k and d_k
            p_dk_j = np.sum(B_rm, axis=0) # sum over r; verify that it yields the same as Dk_m * joint_hat_y_m / joint_hat_y // shape: (N_dpos,)
            p_rk_j = np.sum(B_rm, axis=1) # sum over m; verify that it yields the same as rho_k_t0 * (np.sum(P_D * joint_hat_y_m) / joint_hat_y // shape: (N_rules,)

            # Store p_dk_j and p_rk_j in relevant associated variables:
            inf_states['hat_dk_trials'][j, trial_k, :] = p_dk_j
            inf_states['hat_rk_trials'][j, trial_k, :] = p_rk_j

            # # Evaluate likelihood of y_t, marginalized over the different m hypotheses -- the value stored at j is thus the prediction from j-1
            # inf_states['lik_y_trials'][j, trial_k, :] = np.sum(p_dk_j * inf_states['lik_y_trials_m'][j, trial_k, :]) # sum over m; verify that it yields the same as: p(d_k = j | H_t) · N(y_t; μ^dev_k, S^dev_k)  +  Σ_{m≠j} p(d_k = m | H_t) · N(y_t; μ^std_t(m), S^std_t(m))
            
            # Evaluate likelihood of y_t under the KF predictions under different m hypotheses
            for m in self.N_dpos:
                inf_states['lik_y_trials_m'][j, trial_k, m - self.dpos_min] = kf.likelihood_observation(self.y_trials[j, trial_k], inf_states['hat_y_mean_trials'][j, trial_k, m - self.dpos_min], inf_states['hat_y_var_trials'][j, trial_k, m - self.dpos_min]) # Evaluate y_t under the Kalman-estimated predictions for y_t, for different m hypotheses

            # End of trial:
            if j == self.N_tones - 1:
                # Keep one belief about r_k and d_k, and one KF track for this trial (inf)

                # - L_end(m) = p(y_{8k}, ..., y_{8k+7} | d_k = m, H_{t_k}): product over ALL N_tones tones
                L_end_m = np.prod(inf_states['lik_y_trials_m'][:, trial_k, :], axis=0) # shape: (N_dpos,)

                # - Z_end = p(y_{8k}, ..., y_{8k+7} | H_{t_k}) = sum_m P_k(m) * L_end(m)
                Z_end = np.sum(Dk_m * L_end_m)

                # - b_end(r, m) = p(r_k = r, d_k = m | H_{<k+1}) = rho_k(r) * p(d_k = m | r_k = r) * L_end(m) / Z_end
                B_end_rm = Rk_t_0_posterior[:, np.newaxis] * self.P_D * L_end_m[np.newaxis, :] / Z_end # shape: (N_rules, N_dpos); verify: same as B_rm (j = 7) * lik_y_trials_m[7, trial_k, :] / p(y_{8k+7} | H_t)
                inf_states['rk_end_trials'][trial_k] = np.sum(B_end_rm, axis=1) # p(r_k = r | H_{<k+1}), sum over m; l.212 becomes inf_states['rk_end_trials'][trial_k - 1] @ self.pi_rules
                inf_states['dk_end_trials'][trial_k] = np.sum(B_end_rm, axis=0) # p(d_k = m | H_{<k+1}), sum over r
