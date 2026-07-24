import os
import numpy as np
import scipy.stats as ss
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count

# Try to import pathos for better multiprocessing (handles lambdas/closures)
# Falls back to standard multiprocessing if not available
try:
    from pathos.multiprocessing import ProcessingPool
    HAS_PATHOS = True
except ImportError:
    HAS_PATHOS = False



##### UTILS

def reshape_batch_variable(var):
    if isinstance(var[0], dict):
        # Recursively process each dict key, stacking arrays along a new batch axis
        result = {}
        for key in var[0]:
            elements = [batch[key] for batch in var]
            # Check if elements are scalars or arrays
            if np.isscalar(elements[0]):
                result[key] = np.array(elements)
            else:
                result[key] = np.stack(elements, axis=0)
        return result
    elif isinstance(var[0], tuple):
        # Handle tuples (e.g., pars) by stacking each element separately
        # This handles mixed types (arrays and scalars) within the tuple
        n_elements = len(var[0])
        result = []
        for i in range(n_elements):
            elements = [batch[i] for batch in var]
            # Check if elements are scalars or arrays
            if np.isscalar(elements[0]):
                result.append(np.array(elements))
            else:
                result.append(np.stack(elements, axis=0))
        return tuple(result)
    else:
        # Direct array
        return np.stack([x for x in var], axis=0)



##### FMRI SUITED FEATURES

def draw_iti_exponential(N, rate=0.9, low=2.25, high=4):
    samples = []
    while len(samples) < N:
        s = np.random.exponential(scale=1/rate, size=N) + low
        s = s[s <= high]
        samples.extend(s.tolist())
    intervals = np.array(samples[:N])
    return intervals


##### GENERATIVE MODEL CLASSES

class AuditGenerativeModel:
    """Generic class for building a generative model.
    Not meant to be used on its own but provide reusable methods for contexts, process and observation states generation.
    NonHierachicalGenerativeGM and HierachicalGenerativeGM build on top of it.

    Attributes
    ----------
        N_samples: int; number of data batches to generate
        N_blocks: int; number of blocks a batch contains
        N_tones: int; number of tones per block (usually 8)
        tones_values: list-like; a set of tone frequencies to sample a pair from, to assign to the pair of standard and deviant tones
        mu_tau, si_tau, si_lim: float; define the linear Gaussian dynamics of the standard and deviant processes
        si_q: float; variance of the noise in the process LGD
        si_r: float; variance of the noise in the observation LGD

    """

    def __init__(self, params):
        
        # States initialization strategy
        if "init" in params.keys():
            self.init = params["init"]
        else:
            self.init = 'MU' # options: 'N' (sample from normal distribution), 'MU' (set to the mean), 'TN' (sample from truncated normal distribution), 'TN_3' (sample from truncated normal distribution with bounds at 3*si_stat from the mean)

        # Samples / sessions parameters
        self.N_samples = params["N_samples"]
        self.N_blocks = params["N_blocks"]
        self.N_tones = params["N_tones"]

        # Parallel processing: max_cores controls the number of workers
        # None or 1 = sequential (safe default), >1 = parallel with that many workers
        self.max_cores = params.get("max_cores", None)
        
        # Dynamics parameters
        if "tones_values" in params.keys():
            self.tones_values = params["tones_values"]
        if "mu_tau" in params.keys():
            self.mu_tau = params["mu_tau"]  # Std and dvt process
        self.si_tau = params["si_tau"]  # Std and dvt process
        self.si_lim = params["si_lim"]  # Std and dvt process
        # self.si_q = params["si_q"] # Obsolete
        if "si_stat" in params.keys():
            self.si_stat = params["si_stat"]
        if "si_r" in params.keys():    
            self.si_r = params["si_r"]

        # Context parameters
        if "N_ctx" not in params.keys(): self.N_ctx = 2 # context refers to being a std / dvt
        else: self.N_ctx = params["N_ctx"]

        # Couple the two process timescales (HierarchicalGM std/dev, i.e. N_ctx == 2):
        # when True, tau is sampled once as the (faster) deviant timescale and the
        # standard timescale is set to N_tones * tau_dev, i.e. tau_dev = tau_std / N_tones.
        # This matches how the experimental sequences are generated. When False (or when
        # N_ctx != 2) the previous behaviour is kept (both timescales equal in
        # params_testing, or sampled independently otherwise). Default False so that
        # callers that don't opt in are unaffected; the HierarchicalGM config passes True.
        self.tau_std_dev_dependency = params.get("tau_std_dev_dependency", False)

        # In case of 2 contexts (std and dvt), define stationary values for both processes
        if "si_d_coef" in params.keys() and "mu_d" in params.keys() and "d_bounds" in params.keys():
            self.mu_d = params["mu_d"]
            si_d_ub = (params["d_bounds"]["high"] - self.mu_d)/3     # we want most of d (i.e. 3*si_d) to be <= 4
            si_d_lb = (self.mu_d - params["d_bounds"]["low"])/3   # we want most of d (i.e. 3*si_d) to be >= 0.1
            # NOTE si_d_coef controls how close to its bounds the value of si_d is (bounds defined by us above)
            self.si_d = si_d_lb + params["si_d_coef"] * (si_d_ub - si_d_lb)

        # NOTE: this only happens in contexts where N_ctx is also == 1
        if "params_testing" in params.keys():
            self.params_testing = True
            self.mu_tau_bounds, self.si_stat_bounds, self.si_r_bounds = None, None, None
            if "mu_tau_bounds" in params.keys():
                self.mu_tau_bounds = params["mu_tau_bounds"]
            if "si_stat_bounds" in params.keys():
                self.si_stat_bounds = params["si_stat_bounds"]
            if "si_r_bounds" in params.keys():
                self.si_r_bounds = params["si_r_bounds"]
                
            if "si_d" in params.keys() and params["si_d"] and "mu_d" in params.keys():
                self.mu_d = params["mu_d"]
                si_d_ub = (4 - self.mu_d)/3 # from mu_d + 3 * si_d <= 4 --> so that 95% of d <= 4
                si_d_lb = (self.mu_d - 0.1)/3 # from mu_d - 3 * si_d >= 0.1 --> so that 95% of d >= 0.1
                self.si_d_set = np.random.uniform(si_d_lb, si_d_ub, self.N_samples)
                
        else:
            self.params_testing = False
        
        # fix taus std/dev, lim, and d
        if "fix_process" in params.keys():
            self.fix_process = params["fix_process"]
            if "fix_lim_val" in params.keys() and "fix_tau_val" in params.keys():
                self.fix_lim_val = params["fix_lim_val"]
                self.fix_tau_val = params["fix_tau_val"]
        else:
            self.fix_process = False


    # Auxiliary samplers from goin.coin.GenerativeModel
    def _sample_N_(self, mu, si, size=1):
        """Samples from a normal distribution

        Parameters
        ----------
        mu : float
            Mean of the normal distribution
        si : float
            Standard deviation of the normal distribution
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.norm.rvs(mu, si, size))

    def _sample_TN_(self, a, b, mu, si, size=1):
        """Samples from a truncated normal distribution

        Parameters
        ----------
        a  : float
            low truncation point
        b  : float
            high truncation point
        mu : float
            Mean of the normal distribution before truncation (i.e, location)
        si : float
            Standard deviation of the normal distribution before truncation (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.truncnorm.rvs((a - mu) / si, (b - mu) / si, mu, si, size))
    
    def _sample_logN_(self, min, mu, si, size=1):
        """Samples from a log-normal distribution
        Parameters
        ----------
        min : float
            Minimum value (location parameter of the log-normal distribution)
        mu : float
            Mean of the normal distribution before exponentiation
        si : float
            Standard deviation of the normal distribution before exponentiation (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        return np.array(ss.lognorm.rvs(s=si, loc=min, scale=mu, size=size))
    
    def _sample_halfN(self, mu, si, size=1):
        """Samples from a half-normal distribution
        Parameters
        ----------
        mu : float
            Mean of the normal distribution before absolute value (i.e, location)
        si : float
            Standard deviation of the normal distribution before absolute value (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        return np.array(ss.halfnorm.rvs(loc=mu, scale=si, size=size))

    def _sample_biN_(self, mu, si, size=1):
        """Samples from a bimodal normal distribution
        Parameters
        ----------
        mu : float
            Mean of one of the two normal distributions (i.e, location)
        si : float
            Standard deviation of each of the normal distributions (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        # return np.array(np.random.choice([-1, 1], size=size) * ss.norm.rvs(mu, si, size))
        return np.random.choice([-1, 1], size=size) * self._sample_N_(mu, si, size)

    def sample_uniform_choice(self, set_values):
        """ Sample a choice of one value from a set of possible values
        """
        return np.random.choice(set_values)

    def sample_uniform_set(self, set_values, N=2):
        """ Sample a set of N value choices without replacement from a set of possible values
        Note: len(values) should be > N
        """
        set = np.random.choice(set_values, size=(N,), replace=False)
        return set

    def sample_next_markov_state(
        self, current_state, states_values, states_trans_matrix
    ):
        return np.random.choice(states_values, p=states_trans_matrix[current_state])

    def compute_fixed_pi(self, fixed_pi_vals):
        """Compute a fixed transition matrix with given diagonal values, for 2 states.
        """
        pi = np.array([
            [fixed_pi_vals[0], 1 - fixed_pi_vals[0]],
            [1 - fixed_pi_vals[0], fixed_pi_vals[0]]
            ])  
        
        return pi


    def sample_pi(self, N, mu_rho, si_rho, fixed_id=None, fixed_p=None):
        """A transition matrix with a sticky diagonal, controlled by the concentration parameter rho.
        rho is comprised between 0 and 1 and is is sampled from a truncated normal distribution of 
        mean mu_rho and standard deviation si_rho

        Parameters
        ----------
        N : int
            Number of states (transition matrix thus has shape NxN).
        mu_rho : float
            Mean parameter of the normal distribution to sample from
        si_rho : float
            Std of the truncated-normal used to sample rho.
        fixed_id : int, optional
            If provided together with ``fixed_p``, set the diagonal element for this
            state to ``fixed_p`` and renormalize the remaining off-diagonal entries.
        fixed_p : float, optional
            Fixed probability to assign to the diagonal element ``fixed_id``.

        Returns
        -------
        np.ndarray
            (N, N) Markov transition matrix with sticky diagonal.
        """

        if N>1:
            # Sample parameters
            rho = self._sample_TN_(0, 1, mu_rho, si_rho).item()

            # eps = [np.random.uniform() for n in range(N)]
            # # Delta has a zero diagonal and the rest of the elements of a row (for a rule) are partitions from 1 using the corresponding eps[row] (parameter for that rule), controlling for the sum to be 1
            # delta = np.array([[(0 if i == j else (eps[i] * (1 - eps[i]) ** j if (j < i and j < N - 2) else (eps[i] * (1 - eps[i]) ** (j - 1) if i < j < N - 1 else 1 - sum([eps[i] * (1 - eps[i]) ** k for k in range(N - 2)])))) for j in range(N)] for i in range(N)])
            
            # Delta has a zero diagonal and the rest of the elements of a row (for a rule) are partitions from 1 using the corresponding eps[row] (parameter for that rule), controlling for the sum to be 1
            # Create delta: zero diagonal, off-diagonal values partitioned by eps
            delta = np.zeros((N, N))
            for i in range(N):
                off_diag = np.random.dirichlet(np.ones(N - 1))
                idx = 0
                for j in range(N):
                    if i != j:
                        delta[i, j] = off_diag[idx]
                        idx += 1
            # delta rows sum to 1, diagonal is zero
            
            # Transition matrix
            pi = rho * np.eye(N) + (1 - rho) * delta

            if fixed_id is not None and fixed_p is not None:
                # Fixed diagonal value for the specified context
                pi[fixed_id, fixed_id] = fixed_p
                for j in range(N):
                    if j != fixed_id:
                        # Recompute other values in the row to ensure sum to 1
                        pi[fixed_id, j] = (1 - fixed_p) * delta[fixed_id, j] / sum([delta[fixed_id, k] for k in range(N) if k != fixed_id])

        else:
            # if N==1:
            pi = np.eye(N)
        
        return pi

    def sample_events(self, N_evt, N_val, pi_evt):
        """Sample a sequence of discrete events evolving under a Markov chain.

        Parameters
        ----------
        N_evt : int
            Number of events to generate.
        N_val : int
            Number of distinct discrete values (states) each event can take.
        pi_evt : array-like, shape (N_val, N_val)
            Row-stochastic transition matrix (rows sum to 1).

        Returns
        -------
        np.ndarray
            Integer array of length ``N_evt`` with values in ``range(N_val)``.
        """


        # Sequence of N contexts
        evts = np.zeros(N_evt, dtype=np.int64)

        # Initilize context (assign to 0, randomly, or from the distribution from which the transition probas also come from)
        evts[0] = np.random.choice(N_val)

        for s in range(1, N_evt):
            # Markov chain
            evts[s] = self.sample_next_markov_state(
                current_state=evts[s - 1],
                states_values=range(N_val),
                states_trans_matrix=pi_evt,
            )

        return evts

    def sample_states(self, contexts, return_pars=False):
        """Generates a dictionary of data sequence for each context (std or dvt) dynamics given a sequence of contexts

        Here contexts is the sequence of tone-by-tone boolean value standing for {std: 0, dvt: 1} that has been
        hierarchically defined prior to the call of sample_states

        Parameter sampling strategy:
        - tau and si_stat are sampled once per run.
        - lim is sampled per block.

        Processes dynamics:
        - states[1] (dvt) is updated at every block, and follows the same across the entire block
        - states[0] (std) is updated at every timestep.


        Parameters
        ----------
        contexts : ndarray
            2-D integer array of shape (N_blocks, N_tones) with values {0, 1} indicating
            standard (0) or deviant (1) at each tone.
        return_pars : bool, optional
            If True, also return a tuple with sampled parameters ``(tau, lim, si_stat, si_q)``.

        Returns
        -------
        states : dict
            Mapping from context id to a 2-D array of shape (N_blocks, N_tones) with the
            hidden-state trajectory for that context.
        pars : tuple, optional
            When ``return_pars`` is True, returns ``(tau, lim, si_stat, si_q)`` where
            ``si_q`` is the process noise computed from ``tau`` and ``si_stat``.
        """

        # Sample tau and si_stat for all of the run's blocks

        if not self.fix_process:
            if self.params_testing:
                if self.tau_std_dev_dependency and self.N_ctx == 2:
                    # Sampled base = deviant timescale; standard timescale is N_tones
                    # slower (tau_dev = tau_std / N_tones). tau[0]=std, tau[1]=dev.
                    tau_dev = self.mu_tau
                    tau = np.array([tau_dev * self.N_tones, tau_dev])
                else:
                    tau = self.mu_tau * np.ones(self.N_ctx) # for compatibility with later, since when self.params_testing, N_ctx can only be 1
                si_stat = self.si_stat
            else:
                # NOTE: for mu_tau=64, si_tau=0.5 the distributions covers well the range of values from 1 to 256
                if self.tau_std_dev_dependency and self.N_ctx == 2:
                    # Sample the deviant timescale once (base) and derive the standard
                    # timescale as N_tones * tau_dev (tau_dev = tau_std / N_tones, >= 1).
                    tau_dev = self._sample_logN_(min=1, mu=self.mu_tau, si=self.si_tau, size=1).item()
                    tau = np.array([tau_dev * self.N_tones, tau_dev]) # [tau_std, tau_dev]
                else:
                    tau = self._sample_logN_(min=1, mu=self.mu_tau, si=self.si_tau, size=self.N_ctx) # size = (N_ctx,)
                si_stat = self._sample_logN_(min=0, mu=self.si_stat, si=0.2).item() # std and dvt share the same stationary variance

        elif self.fix_process:
            tau = np.array(self.fix_tau_val)
            si_stat = self.si_stat

        # Compute si_q (for both processes)
        si_q = si_stat * ((2 * tau - 1) ** 0.5) / tau
        
        # lim_Cs = self.sample_uniform_set(self.tones_values, N=self.N_ctx)
        # for c in range(self.N_ctx):
        #     lim[c] = self._sample_N_(lim_Cs[c], self.si_lim)
        
        # Sample lim once for the entire run
        lim = np.zeros((self.N_ctx,))        
        
        if self.N_ctx == 1:
            lim[0] = self._sample_N_(0, 1).item()

        # Sample d
        if self.N_ctx == 2:
            # effective standard deviation considering both process and observation noise
            si_eff = np.sqrt(si_stat**2 + self.si_r**2)
            if not self.fix_process:
                d = self._sample_biN_(self.mu_d, self.si_d).item()
                lim[0] = self._sample_N_(0.5, 0.5).item()
                # lim[0] = -0.6
                # d = 2
                if np.sign(lim[0]) == np.sign(d): lim[0] *= -1 # this ensures the sampled lim[0] and d have opposite signs so lim[1] is on the other side of lim[0]
                lim[1] = lim[0] + d * si_eff 

            else:
                d = self.fix_d_val
                lim[0] = self.fix_lim_val
                #if np.sign(lim[0]) == np.sign(d): lim[0] *= -1
                lim[1] = lim[0] + d * si_eff  
        
        # Initialize states
        states = dict([(int(c), np.zeros(contexts.shape)) for c in range(self.N_ctx)])

        # --- STD process: update at every timestep ---
        # Randomly initialize state for std process at first tone of first block only
        if self.init == 'N':
            states[0][0, 0] = self._sample_N_(lim[0], si_stat).item()
        elif self.init == 'MU':    
            states[0][0,0] = lim[0]
        elif self.init == 'TN':    
            states[0][0, 0] = self._sample_TN_(lim[0]-si_stat, lim[0]+si_stat, lim[0], si_stat).item()
        elif self.init == 'TN_3':
            states[0][0, 0] = self._sample_TN_(lim[0]-(si_stat/3), lim[0]+(si_stat/3), lim[0], si_stat).item()  

        for b in range(self.N_blocks):
            # Initial state for std process as the last value of the previous block
            states[0][b, 0] = states[0][b - 1, -1] if b > 0 else states[0][0, 0]

            # Sample noise for std process
            w_std = self._sample_N_(0, si_q[0], contexts.shape[1])
            for t in range(1, contexts.shape[1]):
                # LGD update at individual tone level: x[t] = x[t-1] + 1/tau * (lim - x[t-1]) + noise
                states[0][b, t] = states[0][b, t - 1] + 1 / tau[0] * (lim[0] - states[0][b, t - 1]) + w_std[t - 1]

        # --- DVT process: update only at deviant position in each block ---
        if self.N_ctx > 1:
            # Sample the first value around the process' stationary value
            if self.init == 'N':
                states[1][0, :] = self._sample_N_(lim[1], si_stat, size=1)
            elif self.init == 'MU':    
                states[1][0,:] = lim[1]
            elif self.init == 'TN':    
                states[1][0,:] = self._sample_TN_(lim[1]-si_stat, lim[1]+si_stat, lim[1], si_stat).item()
            elif self.init == 'TN_3':
                states[1][0,:] = self._sample_TN_(lim[1]-(si_stat/3), lim[1]+(si_stat/3), lim[1], si_stat).item()

            for b in range(1,self.N_blocks):
                # LGD update at block level: x[b] = x[b-1] + 1/tau * (lim - x[b-1]) + noise
                w_dvt = self._sample_N_(0, si_q[1], size=1)
                states[1][b, :] = states[1][b - 1, :] + 1 / tau[1] * (lim[1] - states[1][b - 1, :]) + w_dvt
                

        if return_pars:
            # Return parameters as a labeled dictionary for clarity and compatibility
            pars = {
                'tau': tau.squeeze(),       # time constant (scalar if N_ctx=1, array if N_ctx>1)
                'lim': lim.squeeze(),       # stationary value (scalar if N_ctx=1, array if N_ctx>1)
                'si_stat': si_stat,         # stationary std (scalar)
                'si_q': si_q.squeeze()      # process noise std (scalar if N_ctx=1, array if N_ctx>1)
            }
            return states, pars
        else:
            return states

    def sample_observations(self, contexts, states):
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).

        Returns
        -------
        y  : np.array
            2-dimensional sequence of observations of size (N_blocks, N_tones)
        """

        # Sample observation noise all at once
        v = self._sample_N_(0, self.si_r, contexts.shape)

        # Vectorized observation sampling using context-based indexing
        if self.N_ctx == 1:
            # Single context case - direct indexing
            obs = states[0] + v
        else:
            # Multi-context case - use np.where or advanced indexing
            # Stack states into a 3D array: (N_ctx, N_blocks, N_tones)
            states_stacked = np.stack([states[c] for c in range(self.N_ctx)], axis=0)
            # Use contexts as index into first dimension
            obs = np.take_along_axis(states_stacked, contexts[np.newaxis, :, :], axis=0)[0] + v
        
        return obs

    def plot_contexts_states_obs(self, Cs, ys, x_stds, x_dvts, T, pars, figsize=(10, 6), plot_obs=False):
        """For a non-hierarchical situation (only contexts std/dvt, no rules)

        Parameters
        ----------
        Cs : _type_
            sequence of contexts
        ys : _type_
            observations
        x_stds : _type_
            states of std
        x_dvts : _type_
            states of dvt
        """

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(x_stds, label="x_std", color="blue", linestyle="-", linewidth=2)
        ax1.plot(x_dvts, label="x_dvt", color="red", linestyle="-", linewidth=2)
        if plot_obs:
            ax1.plot(ys, label="y", color="green", linestyle="-", linewidth=2)
        ax1.set_ylabel("y")

        ax2 = ax1.twinx()
        ax2.plot(range(T), Cs, "o", color="black", label="context")
        ax2.set_ylabel("context")
        ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        # Plot horizontal lines for lim_std and lim_dvt
        ax1.hlines(pars['lim'][0], xmin=0, xmax=len(x_stds)-1, color="blue", linestyle="--", alpha=0.5, label="lim_std")
        ax1.hlines(pars['lim'][1], xmin=0, xmax=len(x_dvts)-1, color="red", linestyle="--", alpha=0.5, label="lim_dvt")

        # Fill margin between lim ± si_stat for both processes
        ax1.fill_between(
            range(len(x_stds)),
            pars['lim'][0] - pars['si_stat'],
            pars['lim'][0] + pars['si_stat'],
            color="blue",
            alpha=0.2,
            label="lim_std ± si_stat"
        )
        ax1.fill_between(
            range(len(x_dvts)),
            pars['lim'][1] - pars['si_stat'],
            pars['lim'][1] + pars['si_stat'],
            color="red",
            alpha=0.2,
            label="lim_dvt ± si_stat"
        )

        ax1.legend(bbox_to_anchor=(1.1, 1))
        plt.tight_layout()
        #plt.show()

    def _generate_single_sample(self, samp_idx):
        """Generate one sample, optionally using pre-sampled parameters.
        
        This helper is called by generate_batch and can be parallelized.
        
        Parameters
        ----------
        samp_idx : int
            Index of the sample to generate (used to fetch pre-sampled parameters).
        
        Returns
        -------
        list
            Result from generate_run as a list.
        """
        # Re-seed RNG per worker so that parallel (forked) processes don't
        # share the same numpy random state and produce identical draws.
        # Combining os.getpid() with samp_idx ensures uniqueness across
        # workers and across samples within the same worker.
        np.random.seed((os.getpid() + samp_idx) % (2**32))

        # Override parameters for this sample if testing
        if self.params_testing:
            if self.mu_tau_set is not None:
                self.mu_tau = self.mu_tau_set[samp_idx]
            if self.si_stat_set is not None:
                self.si_stat = self.si_stat_set[samp_idx]
            if self.si_r_set is not None:
                self.si_r = self.si_r_set[samp_idx]
        
        res = self.generate_run(return_pars=self._return_pars)
        return res

    def generate_batch(self, N_samples=None, return_pars=False):
        """Generate a batch of runs by repeatedly calling ``generate_run``.

        Parameters
        ----------
        N_samples : int, optional
            Number of runs to generate. If None, uses ``self.N_samples``.
        return_pars : bool, optional
            If True, each generated sample will include the associated dynamics parameters of runs.

        Returns
        -------
        dict
            A dictionary with keys for each data type produced by ``generate_run``, with values
            stacked along a new leading axis (so arrays become shape (N_samples, ...)).
            Keys include: 'contexts', 'states', 'obs', and optionally 'pars', 'rules', 
            'rules_long', 'dpos', 'timbres', 'timbres_long', 'cues', 'cues_long', 'pi_rules'.
            Dictionaries (e.g. ``states``) are preserved as mappings to stacked arrays.
        """

        # Store latent rules and timbres, states and observations from N_samples batches
        # TODO: find a better way to store batches

        if N_samples is None:
            N_samples = self.N_samples
        
        # Store return_pars for use by _generate_single_sample
        self._return_pars = return_pars
        
        # Create sets of parameters to sample from if testing
        self.mu_tau_set,self.si_stat_set,self.si_r_set = None, None, None
        if self.params_testing:
            if self.mu_tau_bounds is not None:
                self.mu_tau_set = 10 ** np.random.uniform(
                    low=np.log10(self.mu_tau_bounds["low"]),
                    high=np.log10(self.mu_tau_bounds["high"]),
                    size=N_samples
                )
            if self.si_stat_bounds is not None:
                self.si_stat_set = np.random.uniform(
                    low=self.si_stat_bounds["low"],
                    high=self.si_stat_bounds["high"],
                    size=N_samples
                )
            if self.si_r_bounds is not None:
                self.si_r_set = np.random.uniform(
                    low=self.si_r_bounds["low"],
                    high=self.si_r_bounds["high"],
                    size=N_samples
                )

        batch = []

        # Determine number of workers
        # max_cores=None (default / unset) → sequential
        # max_cores=1 → sequential (explicitly disabled)
        # max_cores>1 → parallel with that many workers (requires pathos)
        if self.max_cores is not None and self.max_cores > 1 and HAS_PATHOS and N_samples > 1:
            use_parallel = True
            n_workers = min(self.max_cores, N_samples)
        else:
            use_parallel = False

        # Generate samples
        if use_parallel:
            pool = ProcessingPool(nodes=n_workers)
            try:
                batch = pool.map(self._generate_single_sample, range(N_samples))
            finally:
                pool.close()
                pool.join()
                pool.clear()  # pathos-specific: clears the pool cache
        else:
            # Sequential processing
            for samp in range(N_samples):
                batch.append(self._generate_single_sample(samp))

        # Reorganize data: batch is now a list of dictionaries
        # Convert to a dictionary of lists (one list per key), then reshape each variable
        batch_dict = {}
        if batch:  # Check batch is not empty
            # Get all keys from the first dictionary
            for key in batch[0].keys():
                batch_dict[key] = [sample[key] for sample in batch]
        
        # Now reshape each variable in the batch dictionary
        res_reshaped = {key: reshape_batch_variable(var_list) for key, var_list in batch_dict.items()}

        return res_reshaped # TODO: should return pars here if params_testing as they're not sampled further down the pipeline



class NonHierarchicalAuditGM(AuditGenerativeModel):
    """A generative model that only presents one level of context for a tone: to be a standard or a deviant tone
    Since data is not clustered in blocks defined by rules, there is only one "block" (N_block = 1)
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        mu_rho_ctx :
            Mean of the truncated normal distribution to sample rho, the concentration (sticky) parameter of the transition matrix of contexts
        si_rho_ctx :
            Std of rho
        """

        super().__init__(params)
        self.N_blocks = 1
        self.mu_rho_ctx = params["mu_rho_ctx"]
        self.si_rho_ctx = params["si_rho_ctx"]

    def generate_run(
        self, return_pars=False
    ):
        """Generate data for one run of experiment: contexts, hidden states dynamics, observation

        Returns
        -------
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List ynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and sationary value parameters for each state at each block
        """
        # Sample transition matrix between contexts from a parametric distribution
        pi_ctx = self.sample_pi(self.N_ctx, mu_rho=self.mu_rho_ctx, si_rho=self.si_rho_ctx)

        # Get std/dvt contexts
        contexts = self.sample_events(
            N_evt=self.N_tones, N_val=self.N_ctx, pi_evt=pi_ctx
        )
        contexts = contexts.reshape((self.N_blocks, self.N_tones))

        # Sample states
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)
        
        # Sample observations
        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, (states, ) timbres and obs
        contexts = contexts.flatten()
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        # Build run_obj dictionary
        run_obj = {
            'contexts': contexts,
            'states': states,
            'obs': obs
        }
        
        if return_pars:
            # Add si_r to the parameters dictionary
            pars['si_r'] = self.si_r
            run_obj['pars'] = pars
            
        return run_obj


class HierarchicalAuditGM(AuditGenerativeModel):

    def __init__(self, params):

        super().__init__(params)

        self.N_blocks = params["N_blocks"]

        # Rules
        self.rules_dpos_set = params["rules_dpos_set"]
        self.N_rules = len(self.rules_dpos_set)
        if "return_pi_rules" in params.keys():
            self.return_pi_rules = params["return_pi_rules"]
        else:
            self.return_pi_rules = False

        # Transition matrix between rules
        # Set transition matrix manually
        if "fix_pi_rules" in params.keys():
            self.fix_pi_rules = params["fix_pi_rules"]
        else:
            self.fix_pi_rules = False
        if "fix_pi_vals" in params.keys():
            self.fix_pi_vals = params["fix_pi_vals"]

        # fix taus std/dev, lim, and d
        if self.fix_process:
            self.fix_d_val   = params["fix_d_val"]
        
        # Or define transition matrix stochastically
        if "mu_rho_rules" in params.keys():
            self.mu_rho_rules = params["mu_rho_rules"]
        if "si_rho_rules" in params.keys():
            self.si_rho_rules = params["si_rho_rules"]


        # Optionally, set one null rule
        if "fixed_rule_id" in params.keys() and "fixed_rule_p" in params.keys():
            self.fixed_rule_id = params["fixed_rule_id"]
            self.fixed_rule_p = params["fixed_rule_p"]
        else:
            self.fixed_rule_id = None
            self.fixed_rule_p = None

        # Rules color map
        if "rules_cmap" in params.keys():
            self.rules_cmap = params["rules_cmap"]
        else:
            colors = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}
            self.rules_cmap = {i: colors[i] for i in range(self.N_rules)}
            pass


        # Timbres
        if "mu_rho_timbres" in params.keys():
            self.mu_rho_timbres = params["mu_rho_timbres"]
        # else:
        #     self.mu_rho_timbres = 0.8 # Default value for compatibility
        if "si_rho_timbres" in params.keys():
            self.si_rho_timbres = params["si_rho_timbres"]
        # else:
        #     self.si_rho_timbres = 0.05 # Default value for compatibility
        
        # Cues
        # `cues_set` is the full pool of possible cue values (e.g. 12 values). Each
        # generated sequence only uses a subset of `N_cues_per_seq` cues sampled from
        # this pool. `N_cues` (used for the one-hot encoding dimension) reflects the
        # full pool size so that the encoding dimension stays fixed regardless of which
        # subset a given sequence draws.
        if "p_cues" in params.keys():
            self.p_cues = params["p_cues"]
        if "cues_set" in params.keys():
            self.cues_set = params["cues_set"]
            self.N_cues = len(self.cues_set)
            # Number of distinct cues used within a single sequence (subset size).
            # Defaults to 2; must not exceed the size of the full pool.
            self.N_cues_per_seq = params.get("N_cues_per_seq", 2)
            if self.N_cues_per_seq > self.N_cues:
                raise ValueError(
                    f"N_cues_per_seq ({self.N_cues_per_seq}) cannot exceed the size of "
                    f"the cue pool cues_set ({self.N_cues})."
                )

        # -----------------------------------------------------------------
        # Inter-trial silences (OPTIONAL, disabled by default)
        # -----------------------------------------------------------------
        # When `iti` is True, each trial (block of N_tones tones) is followed by a
        # variable number of "silence" timesteps, emulating the inter-trial interval
        # (ITI) shown to human participants. Silence carries information (it is a
        # labelled state), so it is encoded explicitly at every level:
        #   - obs (y): sampled observation noise ~ N(0, si_r) (never a constant 0)
        #   - discrete levels (contexts / dpos / rules / cues / timbres): a dedicated
        #     "silence" token appended as an extra category at that level.
        # Sequences stay rectangular: every run is padded with trailing silence to a
        # fixed length so that batches can still be stacked (see _apply_iti).
        self.iti = params.get("iti", False)
        if self.iti:
            # Physical timing used to convert an ITI in seconds into a number of
            # timesteps. One timestep == one tone slot == tone_dur + tone_isi seconds.
            self.tone_dur = params.get("tone_dur", 0.1)   # tone duration (s)
            self.tone_isi = params.get("tone_isi", 0.65)  # inter-tone interval (s)
            self.dt_slot = self.tone_dur + self.tone_isi  # seconds per timestep
            # ITI sampling bounds (seconds), consumed by draw_iti_exponential.
            self.iti_bounds = params.get("iti_bounds", {"rate": 0.9, "low": 2.25, "high": 4.0})
            # Maximum number of silence timesteps a single ITI can span. Used to size
            # the fixed padded sequence length so that every run fits.
            self.max_iti_steps = int(np.ceil(self.iti_bounds["high"] / self.dt_slot))

    @property
    def N_dpos(self):
        """Number of unique deviant positions (only for HierarchicalGM)."""
        if self.rules_dpos_set is None:
            return None
        # Flatten all position sets and count unique values
        all_positions = set()
        for positions in self.rules_dpos_set:
            if positions is not None:  # Handle None entries in rules_dpos_set
                all_positions.update(positions)
        return len(all_positions)

    @property
    def dpos_silence_token(self):
        """Raw dpos value used to mark silence timesteps.

        dpos is stored as raw positions (e.g. 3..7), so the silence token must be a
        value distinct from every valid position. Using max(position) + 1 guarantees
        this and, once the pipeline converts dpos to 0-based indices (value - min),
        maps to N_dpos, i.e. an extra "silence" class appended after the real ones.
        """
        all_positions = set()
        for positions in self.rules_dpos_set:
            if positions is not None:
                all_positions.update(positions)
        return int(max(all_positions)) + 1

    def _apply_iti(self, run_obj, n_tones=None):
        """Expand a single run with inter-trial silences (in place on a fresh dict).

        Each trial (block of ``n_tones`` tones) is followed by a variable number of
        silence timesteps drawn from ``draw_iti_exponential`` (converted from seconds
        to timesteps via ``dt_slot``). Every run is padded with trailing silence to a
        fixed length ``T' = N_blocks * (n_tones + max_iti_steps)`` so that runs of a
        batch remain stackable.

        The per-timestep streams are rewritten to length T':
          - obs: real tones kept, silence filled with observation noise ~ N(0, si_r)
          - contexts / rules_long / dpos_long / cues_long / timbres_long: real values
            kept, silence filled with that level's dedicated silence token
          - states: real values kept, silence filled with NaN (undefined during silence)

        Two extra per-timestep arrays are added:
          - 'is_silence' (T',) bool: True on silence/padding timesteps
          - 'within_trial_pos' (T',) int: within-trial tone index on real timesteps,
            -1 on silence timesteps (used downstream, e.g. for the dpos response window)

        Per-block streams (rules, dpos, cues, timbres) are left unchanged.
        """
        if n_tones is None:
            n_tones = self.N_tones

        L = self.N_blocks * n_tones
        T_prime = self.N_blocks * (n_tones + self.max_iti_steps)

        # Number of silence timesteps after each trial
        iti_sec = draw_iti_exponential(self.N_blocks, **self.iti_bounds)
        n_sil = np.rint(iti_sec / self.dt_slot).astype(int)
        n_sil = np.clip(n_sil, 0, self.max_iti_steps)

        # Destination index of each real tone in the padded sequence + bookkeeping
        dest = np.empty(L, dtype=int)
        within_trial_pos = np.full(T_prime, -1, dtype=int)
        is_silence = np.ones(T_prime, dtype=bool)
        pos, k = 0, 0
        for b in range(self.N_blocks):
            for t in range(n_tones):
                dest[k] = pos
                within_trial_pos[pos] = t
                is_silence[pos] = False
                pos += 1
                k += 1
            pos += int(n_sil[b])

        def _scatter(values, fill):
            full = np.full(T_prime, fill, dtype=values.dtype)
            full[dest] = values
            return full

        # Silence tokens per discrete level (one extra category appended after reals)
        silence_tokens = {
            'contexts': self.N_ctx,
            'rules_long': self.N_rules,
            'dpos_long': self.dpos_silence_token,
            'cues_long': self.N_cues,
            'timbres_long': self.N_rules,
        }

        expanded = dict(run_obj)

        # Observations: silence carries sampled observation noise (not a constant 0)
        obs_full = self._sample_N_(0, self.si_r, T_prime)
        obs_full[dest] = run_obj['obs']
        expanded['obs'] = obs_full

        # Discrete per-timestep streams
        for key, fill in silence_tokens.items():
            if key in run_obj and run_obj[key] is not None:
                expanded[key] = _scatter(np.asarray(run_obj[key]), fill)

        # Hidden states: undefined during silence -> NaN
        expanded['states'] = {
            c: _scatter(np.asarray(x, dtype=float), np.nan) for c, x in run_obj['states'].items()
        }

        expanded['is_silence'] = is_silence
        expanded['within_trial_pos'] = within_trial_pos

        return expanded

    def sample_rules(
        self, N_blocks, N_rules, mu_rho_rules, si_rho_rules, fixed_rule_id=None, fixed_rule_p=None, return_pi=False
    ):
        """Sample rules for a run consisting in a sequence of blocks of tones (each sequence being associated with one rule).
        Rules evolve in a Markov chain manner.

        Parameters
        ----------
        N_blocks : int
            Number of blocks of tones
        N_rules : int
            Number of rules
        mu_rho_rules : float
            _description_
        si_rho_rules : float
            _description_

        Returns
        -------
        np.array (N_blocks,)
            Sequence of rules associated with blocks for each block of N_blocks blocks
        """
        if self.fix_pi_rules and self.fix_pi_vals is not None:
            # If a pi_rules has been manually specified, use it
            pi_rules = self.compute_fixed_pi(self.fix_pi_vals)
        else:
            # Sample pi stochastically 
            pi_rules = self.sample_pi(N_rules, mu_rho_rules, si_rho_rules, fixed_id=fixed_rule_id, fixed_p=fixed_rule_p)


        if return_pi:
            return self.sample_events(N_evt=N_blocks, N_val=N_rules, pi_evt=pi_rules), pi_rules
        else:
            return self.sample_events(N_evt=N_blocks, N_val=N_rules, pi_evt=pi_rules)

    def sample_timbres(self, rules_seq, N_timbres, mu_rho_timbres, si_rho_timbres):
        """Sample timbres, mediated by a Markov chain transition process too

        TODO: to check if correct

        Parameters
        ----------
        rules_seq : np.array
            _description_
        N_timbres : int
            _description_
        mu_rho_timbres : float
            _description_
        si_rho_timbres : float
            _description_

        Returns
        -------
        list
            _description_
        """

        # Sample timbres transition (emission from rule) matrix
        pi_timbre = self.sample_pi(N_timbres, mu_rho_timbres, si_rho_timbres)

        # timbres = np.array([np.random.choice(range(N_timbres), p=pi_timbre[seq]) for seq in rules_seq])
        timbres = np.array(
            [
                self.sample_next_markov_state(
                    current_state=seq,
                    states_values=range(N_timbres),
                    states_trans_matrix=pi_timbre,
                )
                for seq in rules_seq
            ]
        )

        return timbres

    def sample_cues(self, rules, p_cues=None, cues_set=None):
        """Sample cues for a sequence.

        The set of cues is sampled probabilistically: `cues_set` is the full pool of
        possible cue values, and each sequence first draws a subset of `N_cues_per_seq`
        (default 2) distinct cues from this pool. Each block is then associated with one
        of the subset's cues with probability p_cues[current rule] (the remaining
        probability mass going to the other subset cue).

        NOTE: the rules sequence needs to be the trial-by-trial sequence of rules, not
        the tone-by-tone storing of rules (rules_long).
        """

        if p_cues is None:
            p_cues = self.p_cues
        if cues_set is None:
            cues_set = self.cues_set

        # Sample, once per sequence, the subset of cues actually used in this sequence
        # from the full pool (without replacement).
        seq_cues = self.sample_uniform_set(cues_set, N=self.N_cues_per_seq)

        cues = []
        for rule in rules:
            cue = np.random.choice(seq_cues, p=[p_cues[rule], 1 - p_cues[rule]])
            cues.append(cue)
        return np.array(cues)

    def sample_dpos(self, rules, rules_dpos_set):
        """Sample positions of the deviant tones for each block of tones

        Parameters
        ----------
        rules : np.array
            Sequence of blocks rules_
        rules_dpos_set : np,array (N_rules, 3)
            Mapping of the 3 indexes of possible deviant positions for each of the rules

        Returns
        -------
        list
            List of deviant position indexes for each block of tones
        """

        # Handle None case: if rules_dpos_set[rule] is None, return None for that block
        dpos = []
        for rule in rules:
            rule_set = rules_dpos_set[rule]
            if rule_set is None:
                dpos.append(None)
            else:
                dpos.append(self.sample_uniform_choice(rule_set))
        return np.array(dpos, dtype=int)
    


    def get_contexts(self, dpos, N_blocks, N_tones):
        contexts = np.zeros((N_blocks, N_tones), dtype=np.int64)
        for i, pos in enumerate(dpos):
            # i is block index, pos is deviant position index within the block
            if dpos[i] is not None:
                contexts[i, pos] = 1
        return contexts

    def generate_run(
        self,
        return_pars = False,
        return_pi_rules = False
    ):
        """Generate data for one run of experiment: rules, dvt positions, timbres, contexts (std or dvt),
        states (hidden states dynamics), observations

        Returns
        -------
        rules:
            List of the rules that apply to each block of N_tones tones --> rules[b] = rule for current block (length = N_blocks)
        rules_long:
            List of rules that apply to each tone in the whole list of tones --> rules_long[t] is the same for every 8 consecutive t (length = N_tones*N_blocks)
        dpos:
            Positions of the dvt within each block (length = N_blocks)
        timbres:
            List of timbres associated with each block (NOTE: dynamics correct but physical values TBD, not implemented ATM) (length = N_blocks)
        timbres_long:
            Same as in rules_long, per tone list of timbres (length = N_tones*N_blocks)
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List of the dynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and stationary value parameters for each state at each block
        """
        if return_pi_rules is None:
            return_pi_rules = self.return_pi_rules

        # Sample sequence of rules ids
        res = self.sample_rules(self.N_blocks, self.N_rules, self.mu_rho_rules, self.si_rho_rules, self.fixed_rule_id, self.fixed_rule_p, self.return_pi_rules)
        if return_pi_rules:
            rules, pi_rules = res
        else:
            rules = res
        rules_long = np.tile(rules[:, np.newaxis], (1, self.N_tones)).flatten()

        # OPTIONAL: sample timbres
        if hasattr(self, "mu_rho_timbres") and hasattr(self, "si_rho_timbres"):
            # Sample timbres (here we consider that there are as many different timbres as there are different rules -- self.N_rules)
            timbres = self.sample_timbres(rules, self.N_rules, self.mu_rho_timbres, self.si_rho_timbres)
            # Store timbres in a per-tone array # This is equivalent to matlab's repmat
            timbres_long = np.tile(timbres[:, np.newaxis], (1, self.N_tones))
            timbres_long = timbres_long.flatten()
        else:
            timbres = None
            timbres_long = None

        # OPTIONAL: sample cues
        if hasattr(self, "p_cues") and hasattr(self, "cues_set"):
            cues = self.sample_cues(rules, p_cues=self.p_cues, cues_set=self.cues_set)
            cues_long = np.tile(cues[:, np.newaxis], (1, self.N_tones)).flatten()
        else:
            cues = None
            cues_long = None


        # Sample deviant position
        dpos = self.sample_dpos(rules, self.rules_dpos_set)
        dpos_long = np.tile(dpos[:, np.newaxis], (1, self.N_tones)).flatten()

        # Get contexts
        contexts = self.get_contexts(dpos, self.N_blocks, self.N_tones)

        # Sample states
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)
        
        # Sample observations
        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, states, and obs
        contexts = contexts.flatten()
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        # Build run_obj dictionary
        run_obj = {
            'rules': rules,
            'rules_long': rules_long,
            'dpos': dpos,
            'dpos_long': dpos_long,
            'contexts': contexts,
            'states': states,
            'obs': obs
        }
        if timbres is not None and timbres_long is not None:
            run_obj['timbres'] = timbres
            run_obj['timbres_long'] = timbres_long
        if cues is not None and cues_long is not None:
            run_obj['cues'] = cues
            run_obj['cues_long'] = cues_long
        
        # Return parameters
        if return_pars:
            # Add si_r to the parameters dictionary
            pars['si_r'] = self.si_r            
            run_obj['pars'] = pars
        
        # Return transition rules
        if return_pi_rules:
            run_obj['pi_rules'] = pi_rules

        # OPTIONAL: insert inter-trial silences (expands the per-timestep streams to a
        # fixed padded length and adds 'is_silence' / 'within_trial_pos'). Disabled by
        # default so all existing use cases are unaffected.
        if getattr(self, "iti", False):
            run_obj = self._apply_iti(run_obj)

        return run_obj

    def plot_contexts_rules_states_obs(self, x_stds, x_dvts, ys, Cs, rules, dpos, pars, text=True):
        """For the hierachical evolution of rules and contexts (NOTE: timbres not included in this viz atm)

        Parameters
        ----------
        x_stds : _type_
            _description_
        x_dvts : _type_
            _description_
        ys : _type_
            _description_
        Cs : _type_
            _description_
        rules : _type_
            _description_
        """

        # TODO: sort legend positioning, and blank space around plot within rectangle

        # Visualize tone frequencies
        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax1.set_xlim(0, len(x_stds)-1)
        ax1.set_ylim(min(np.min(x_stds), np.min(x_dvts), np.min(ys)) - 0.5, max(np.max(x_stds), np.max(x_dvts), np.max(ys)) + 0.5)
        ax1.plot(
            x_stds,
            label="x_std",
            color="blue",
            marker="o" if text else None,
            markersize=4,
            linestyle="-",
            linewidth=2 if text else 1,
            alpha=0.9,
        )
        ax1.plot(
            x_dvts,
            label="x_dvt",
            color="red",
            marker="o" if text else None,
            markersize=4,
            linestyle="-",
            linewidth=2 if text else 1,
            alpha=0.9,
        )
        ax1.plot(
            ys,
            label="y",
            color="green",
            marker="o" if text else None,
            markersize=4,
            linewidth=2 if text else 1,
            alpha=0.9
        )
        ax1.set_ylabel("processes and observations")

        if text:
            ax2 = ax1.twinx()
            ax2.plot(Cs, "o", color="black", label="context", markersize=2)
            ax2.set_ylabel("context")
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        for i, rule in enumerate(rules):
            ax1.axvspan(
                i * self.N_tones,
                i * self.N_tones + self.N_tones,
                facecolor=self.rules_cmap[rule],
                alpha=0.25,
            )

        for i in range(self.N_blocks):
            ax1.axvline(i * self.N_tones, color="tab:gray", linewidth=0.9)

        if text:
            text_y_position = 1.15  # Position above the plot
            for i in range(self.N_blocks):
                ax2.text(
                    x=i * self.N_tones + 0.35 * self.N_tones,
                    y=text_y_position,
                    s=f"rule {rules[i]}",
                    color=self.rules_cmap[rules[i]],
                    transform=ax2.transData,
                    ha="center",
                )
                ax2.text(
                    x=i * self.N_tones + 0.35 * self.N_tones,
                    y=text_y_position - 0.075,
                    s=f"dvt {dpos[i]}",
                    color=self.rules_cmap[rules[i]],
                    transform=ax2.transData,
                    ha="center",
                )
        else:
            ax1.set_xticks(np.arange(0, self.N_blocks * self.N_tones + 1, 50))

        ax1.hlines(pars['lim'][0], xmin=0, xmax=len(x_stds)-1, color="blue", linestyle="--", linewidth=2, alpha=0.5, label="lim_std")
        ax1.hlines(pars['lim'][1], xmin=0, xmax=len(x_dvts)-1, color="red", linestyle="--", linewidth=2,alpha=0.5, label="lim_dvt")

        ax1.fill_between(
            range(len(x_stds)),
            pars['lim'][0] - pars['si_stat'],
            pars['lim'][0] + pars['si_stat'],
            color="blue",
            alpha=0.2,
            label="lim_std ± si_stat"
        )
        ax1.fill_between(
            range(len(x_dvts)),
            pars['lim'][1] - pars['si_stat'],
            pars['lim'][1] + pars['si_stat'],
            color="red",
            alpha=0.2,
            label="lim_dvt ± si_stat"
        )

        tau_str = f"std: {pars['tau'][0]:.2f}, dvt: {pars['tau'][1]:.2f}" if self.N_ctx == 2 else f"{pars['tau']:.2f}"
        si_q_str = f"std: {pars['si_q'][0]:.2f}, dvt: {pars['si_q'][1]:.2f}" if self.N_ctx == 2 else f"{pars['si_q']:.2f}"

        title_line1 = f"tau: {tau_str}  |  si_stat: {pars['si_stat']:.2f}  |  si_q: {si_q_str}"
        title_line2 = f"(mu_tau: {self.mu_tau:.2f}, mu_si_stat: {self.si_stat:.2f}, mu_si_q: {self.si_stat * ((2 * self.mu_tau - 1) ** 0.5) / self.mu_tau:.2f}, si_r: {self.si_r:.2f})"
        ax1.set_title(f"{title_line1}\n{title_line2}", y=-0.2)
        ax1.legend(bbox_to_anchor=(1.1, 1))
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
        #plt.show()


    def plot_combined_with_matrix(self, x_stds, x_dvts, ys, Cs, rules, dpos, pars, pi_rules=None, text=True, plot_obs=False, plot_dpos_dist=False, save_path=None, title=None):
        """
        Plots plot_contexts_rules_states_obs and plot_rules_dpos as subplots, and pi_rules as a matrix on the side.
        Includes histograms with KDE of dpos distribution to the right of the dpos plot.
        """

        fig = plt.figure(figsize=(13, 6))
        # Main gridspec: 2 rows x 2 columns for proper alignment
        # Left column: ax1 and ax3 (height_ratios [1,1] for equal heights)
        # Right column: ax4 (top) and ax3_hist (bottom, aligned with ax3)
        gs_main = gridspec.GridSpec(2, 2, width_ratios=[5, 0.6], height_ratios=[1, 1], figure=fig)
        
        # Top left: ax1
        ax1 = fig.add_subplot(gs_main[0, 0])
        
        # Bottom left: ax3 (shares x-axis with ax1 so trial positions line up exactly)
        ax3 = fig.add_subplot(gs_main[1, 0], sharex=ax1)
        
        # Top right: ax4 (will appear smaller naturally)
        ax4 = fig.add_subplot(gs_main[0, 1])
        
        # Bottom right: ax3_hist (aligned with ax3)
        ax3_hist = fig.add_subplot(gs_main[1, 1], sharey=ax3)
        
        # Note: Need to move the ax1 plotting code after ax1 is created

        # Top subplot: contexts, rules, states, obs
        ax1.set_xlim(0, len(x_stds)-1)
        # ax1.set_ylim(min(np.min(x_stds), np.min(x_dvts), np.min(ys)) - 0.5, max(np.max(x_stds), np.max(x_dvts), np.max(ys)) + 0.5)
        ax1.plot(x_stds, label="standard", color="tab:blue", linestyle="-", linewidth=2, alpha=0.8) # label="x_std"
        ax1.plot(x_dvts, label="deviant", color="tab:red", linestyle="-", linewidth=2, alpha=0.8) # label="x_dvt"
        if plot_obs:
            ax1.plot(ys, linestyle='--', label="obs", color="tab:green", linewidth=2, alpha=0.9) # label="y"
        ax1.set_ylabel(f"states {'and observations' if plot_obs else ''}")
        if text:
            ax2 = ax1.twinx()
            ax2.plot(Cs, "o", color="black", label="context", markersize=2)
            ax2.set_ylabel("context")
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])
        for i, rule in enumerate(rules):
            ax1.axvspan(i * self.N_tones, i * self.N_tones + self.N_tones, facecolor=self.rules_cmap[rule], alpha=0.25)
        for i in range(self.N_blocks):
            ax1.axvline(i * self.N_tones, color="tab:gray", linewidth=0.9)
        if text:
            text_y_position = 1.15
            for i in range(self.N_blocks):
                ax2.text(x=i * self.N_tones + 0.35 * self.N_tones, y=text_y_position, s=f"rule {rules.reset_index(drop=True)[i]}", color=self.rules_cmap[rules.reset_index(drop=True)[i]], transform=ax2.transData, ha="center")
                ax2.text(x=i * self.N_tones + 0.35 * self.N_tones, y=text_y_position - 0.075, s=f"dvt {dpos.reset_index(drop=True)[i]}", color=self.rules_cmap[rules.reset_index(drop=True)[i]], transform=ax2.transData, ha="center")
        else:
            ax1.set_xticks(np.arange(0, self.N_blocks * self.N_tones + 1, 50))
        ax1.hlines(pars['lim'][0], xmin=0, xmax=len(x_stds)-1, color="tab:blue", linestyle="--", linewidth=2, alpha=0.5) #  label="lim_std"
        ax1.hlines(pars['lim'][1], xmin=0, xmax=len(x_dvts)-1, color="tab:red", linestyle="--", linewidth=2,alpha=0.5) #  label="lim_dvt"
        ax1.fill_between(range(len(x_stds)), pars['lim'][0] - pars['si_stat'], pars['lim'][0] + pars['si_stat'], color="tab:blue", alpha=0.2) # label="lim_std ± si_stat"
        ax1.fill_between(range(len(x_dvts)), pars['lim'][1] - pars['si_stat'], pars['lim'][1] + pars['si_stat'], color="tab:red", alpha=0.2) # label="lim_dvt ± si_stat"
        # ax1.legend(bbox_to_anchor=(1.0, 1))
        ax1.legend(loc='lower left')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # If title is provided:
        if title is not None:
            ax1.set_title(title) #, y=-0.25)

        # Bottom subplot: rules/dpos
        # Trial centers, in the same tone-index units as ax1, so each dot/vline
        # falls midway between the two trial-boundary vlines drawn on ax1.
        trial_centers = np.arange(len(dpos)) * self.N_tones + self.N_tones / 2
        for i, y in enumerate(dpos):
            ax3.vlines(x=trial_centers[i], ymin=0, ymax=y, color="tab:gray", linewidth=0.9, zorder=1, alpha=0.5)
        ax3.scatter(trial_centers, dpos, c=[self.rules_cmap[rule] for rule in rules], zorder=2)
        ax3.set_ylabel("deviant location")
        ax3.set_xlabel("trials")
        ax3.set_ylim(0, 7) # but exclude tick for 8:
        ax3.set_yticks(range(0, 8))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10) for color in self.rules_cmap.values()]
        labels = self.rules_cmap.keys()
        ax3.legend(handles, labels, title="rule", loc='lower left')
        ax3.set_xticks(np.arange(0, self.N_blocks * self.N_tones + 1, 50))
        # Remove borders while keeping axes:
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        if plot_dpos_dist:
            # Histogram subplot: dpos distributions
            ax3_hist.tick_params(axis='y', labelleft=False)
            
            # Prepare data for histograms - filter out None values
            dpos_array = np.array(dpos)
            rules_array = np.array(rules)
            
            # Filter out None values for KDE calculations (convert to float)
            # dpos_all_clean = np.array([x for x in dpos_array if x is not None], dtype=float)
            dpos_rule0_clean = np.array([x for i, x in enumerate(dpos_array) if x is not None and rules_array[i] == 0], dtype=float)
            dpos_rule1_clean = np.array([x for i, x in enumerate(dpos_array) if x is not None and rules_array[i] == 1], dtype=float)

            # Create histograms with KDE for each group
            y_range = np.linspace(0, 7, 100)

            # All dpos histogram
            # if len(dpos_all_clean) > 0:
            #     ax3_hist.hist(dpos_all_clean, bins=7, range=(0, 7), orientation='horizontal', alpha=0.3,
            #                 color='gray', label='All', density=True, edgecolor='black', linewidth=0.5, align='mid')
            #     if len(dpos_all_clean) > 1:
            #         kde_all = ss.gaussian_kde(dpos_all_clean)
            #         ax3_hist.plot(kde_all(y_range), y_range, color='gray', linewidth=2, linestyle='--', alpha=0.7)

            # Rule 0 histogram
            if len(dpos_rule0_clean) > 0:
                ax3_hist.hist(dpos_rule0_clean, bins=8, range=(-0.5, 7.5), orientation='horizontal', alpha=0.3,
                            color=self.rules_cmap[0], label='Rule 0', density=True, edgecolor='black', linewidth=0.5, align='mid')
                if len(dpos_rule0_clean) > 1:
                    kde_rule0 = ss.gaussian_kde(dpos_rule0_clean)
                    ax3_hist.plot(kde_rule0(y_range), y_range, color=self.rules_cmap[0], linewidth=2)
            
            # Rule 1 histogram
            if len(dpos_rule1_clean) > 0:
                ax3_hist.hist(dpos_rule1_clean, bins=8, range=(-0.5, 7.5), orientation='horizontal', alpha=0.3,
                            color=self.rules_cmap[1], label='Rule 1', density=True, edgecolor='black', linewidth=0.5, align='mid')
                if len(dpos_rule1_clean) > 1:
                    kde_rule1 = ss.gaussian_kde(dpos_rule1_clean)
                    ax3_hist.plot(kde_rule1(y_range), y_range, color=self.rules_cmap[1], linewidth=2)
            
            ax3_hist.set_xlabel("Empirical density")
            ax3_hist.legend(loc='upper left', fontsize=8)
            ax3_hist.set_xlim(0, ax3_hist.get_xlim()[1])
            # Remove borders while keeping axes:
            ax3_hist.spines['top'].set_visible(False)
            ax3_hist.spines['right'].set_visible(False)

        # Right subplot: transition matrix pi_rules
        if pi_rules is not None:
            im = ax4.imshow(pi_rules, cmap="Blues", vmin=0, vmax=1)
            ax4.set_title("Rules transition \nprobabilities")
            ax4.set_xlabel("To rule")
            ax4.set_ylabel("From rule")
            ax4.set_xticks(np.arange(pi_rules.shape[1]))
            ax4.set_yticks(np.arange(pi_rules.shape[0]))
            # Annotate matrix values
            #print(pi_rules)
            for i in range(pi_rules.shape[0]):
                for j in range(pi_rules.shape[1]):
                    ax4.text(j, i, f"{pi_rules[i, j]:.2f}", ha="center", va="center", color="black")

        # Set figure title
        tau_str = f"std: {pars['tau'][0]:.2f}, dvt: {pars['tau'][1]:.2f}" if self.N_ctx == 2 else f"{pars['tau']:.2f}"
        si_q_str = f"std: {pars['si_q'][0]:.2f}, dvt: {pars['si_q'][1]:.2f}" if self.N_ctx == 2 else f"{pars['si_q']:.2f}"
        title_line1 = f"tau: {tau_str}  |  si_stat: {pars['si_stat']:.2f}  |  si_q: {si_q_str}"
        title_line2 = f"(mu_tau: {self.mu_tau:.2f}, mu_si_stat: {self.si_stat:.2f}, mu_si_q: {self.si_stat * ((2 * self.mu_tau - 1) ** 0.5) / self.mu_tau:.2f}, si_r: {self.si_r:.2f})"
        # fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=12)


        # Alt title: $x_{\text{c}} = $
        equations =  r"$x_{t+1}^{c} = x_{t}^{c}+\frac{1}{\tau^{c}}(\mu^{c}-x_{t}^{c})+\sigma_{q}\epsilon_{t}^{c}$ (c = std, dvt)" \
            + '\n' + r'$y_{t+1} = x_{t+1}^{c_{t}}+\sigma_{r}\nu_{t}$'

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.08, wspace=0.08, hspace=0.3)
        #plt.show()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)

    def plot_rules_dpos(self, rules, dpos, pars, text=True):

        # TODO: sort blank space around plot within rectangle


        # Visualize hierarchical information: dvt pos and rule
        fig, ax = plt.subplots(figsize=(20, 6))
        for i, y in enumerate(dpos):
            ax.vlines(
                x=i,
                ymin=0,
                ymax=y,
                color="tab:gray",
                linewidth=0.9,
                zorder=1,
                alpha=0.5,
            )
        ax.scatter(
            range(len(dpos)), dpos, c=[self.rules_cmap[rule] for rule in rules], zorder=2
        )
        ax.set_ylabel("dvt pos")
        ax.set_xlabel("trial")
        ax.set_ylim(1, 8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(len(dpos)))

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
            )
            for color in self.rules_cmap.values()
        ]
        labels = self.rules_cmap.keys()
        ax.legend(handles, labels, title="rule")

        if not text:
            # Set x ticks every 50 tones if too many tones
            ax.set_xticks(np.arange(0, self.N_blocks + 1, 50))

        tau_str = f"std: {pars['tau'][0]:.2f}, dvt: {pars['tau'][1]:.2f}" if self.N_ctx == 2 else f"{pars['tau']:.2f}"
        si_q_str = f"std: {pars['si_q'][0]:.2f}, dvt: {pars['si_q'][1]:.2f}" if self.N_ctx == 2 else f"{pars['si_q']:.2f}"
        title_line1 = f"tau: {tau_str}; si_stat: {pars['si_stat']:.2f}; si_q: {si_q_str}"
        title_line2 = f"(mu_tau: {self.mu_tau:.2f}, mu_si_stat: {self.si_stat:.2f}, mu_si_q: {self.si_stat * ((2 * self.mu_tau - 1) ** 0.5) / self.mu_tau:.2f}, si_r: {self.si_r:.2f})"
        plt.title(f"{title_line1}\n{title_line2}", y=-0.2)
        fig.tight_layout()
        #plt.show()



def example_HGM(config_H, plot_obs=False, plot_dpos_dist=False):
    gm = HierarchicalAuditGM(config_H)

    run_obj = gm.generate_run(return_pars=True, return_pi_rules=config_H.get("return_pi_rules", False))

    # States, current blocks' rules, contexts based on rules and sampled deviant positions, and observations sampled from states based on context
    if config_H["N_blocks"] <= 20:
        text=True
    else:
        text=False
    
    # gm.plot_contexts_rules_states_obs(states[0], states[1], obs, contexts, rules, dpos, pars, text=text)

    # Deviant position for each rule
    # gm.plot_rules_dpos(rules, dpos, pars, text=text)

    gm.plot_combined_with_matrix(run_obj['states'][0], run_obj['states'][1], run_obj['obs'], run_obj['contexts'], run_obj['rules'], run_obj['dpos'], run_obj['pars'], pi_rules=run_obj.get('pi_rules'), text=text, plot_obs=plot_obs, plot_dpos_dist=plot_dpos_dist)

    # An example of the states and observation sampling for one block
    gm.plot_contexts_states_obs(run_obj['contexts'][0:gm.N_tones], run_obj['obs'][0:gm.N_tones], run_obj['states'][0][0:gm.N_tones], run_obj['states'][1][0:gm.N_tones], gm.N_tones, pars=run_obj['pars'])



def example_NHGM(config_NH):
    gm_NH = NonHierarchicalAuditGM(config_NH)

    run_obj = gm_NH.generate_run(return_pars=True)

    # States and observation sampled based on contexts
    gm_NH.plot_contexts_states_obs(run_obj['contexts'], run_obj['obs'], run_obj['states'][0], run_obj['states'][1], gm_NH.N_tones, pars=run_obj['pars'], figsize=(20, 6))


def example_single(config_single):

    tau_values = [1, 1.5, 2, 4, 8, 16, 32, 50]

    fig, axs = plt.subplots(len(tau_values), 1)

    for i, tau in enumerate(tau_values):
        config_single["mu_tau"]=tau
        gm = NonHierarchicalAuditGM(config_single)
        run_obj = gm.generate_run(return_pars=True)

        # Plot process states
        axs[i].plot(range(len(run_obj['states'][0])), run_obj['states'][0], label='x_hid', color='orange', linewidth=2)
            
        # Plot observation
        axs[i].plot(range(len(run_obj['obs'])), run_obj['obs'], color='tab:blue', label='y_obs')

        axs[i].set_title(f"mu_tau = {tau}, tau = {run_obj['pars']['tau']:.2f}")

    plt.tight_layout()
    #plt.show()

    


if __name__ == "__main__":

    # Example hierachical GM (rules, timbres [not implemented], std/dvt)
    config_H = {
        "N_samples": 1,
        "N_blocks": 120, # TODO: adapt here!
        "N_tones": 8,
        # "rules_dpos_set": np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]]),
        "rules_dpos_set": np.array([[3, 4, 5], [5, 6, 7]]),
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 0.2,
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
        "mu_rho_timbres": 0.8,
        "si_rho_timbres": 0.05,
        # "si_q": 2,  # process noise variance
        "si_stat": 0.5,  # stationary process variance
        "si_r": 0.2,  # measurement noise variance
        "si_d_coef": 0.05,
        "d_bounds": {"high": 4, "low": 0.1},
        "mu_d": 2,
        "return_pi_rules": True,
        "fix_process": False, # fix tau, lim, d
        "fix_tau_val": [16,2],
        "fix_lim_val": -0.6,
        "fix_d_val": 2,
        "fix_pi_rules": True,
        "fix_pi_vals": [0.85, 0.15],
        "init": 'N' # TN = initialize from truncated normal truncated at si_stat, TN_3 = truncated at si_stat/3, N = from full normal, MU = initialize as mean
    }
    example_HGM(config_H, plot_obs=True, plot_dpos_dist=True)


    config_H_nullrule = config_H.copy()
    config_H_nullrule["mu_tau"] = 160
    config_H_nullrule["rules_dpos_set"] = [[3, 4, 5], [5, 6, 7], None]
    config_H_nullrule["fixed_rule_id"] = 2
    config_H_nullrule["fixed_rule_p"] = 0.1
    config_H_nullrule["rules_cmap"] = {0: "tab:blue", 1: "tab:red", 2: "tab:gray"}
    example_HGM(config_H_nullrule, plot_obs=True, plot_dpos_dist=True)
    
    # Example non-hierachical GM (no rules, std/dvt)
    config_NH = {
        "N_samples": 1,
        "N_blocks": 1,
        "N_tones": 160,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "mu_tau": 4, # tau = 1 / (1 - a) = x_lim / b
        "si_tau": 1,
        "si_lim": 5,
        # "si_q": 2,  # process noise variance
        "si_stat": 0.2,  # stationary process variance
        "si_r": 0.2,  # measurement noise
        "si_d_coef": 0.05,
        "d_bounds": {"high": 4, "low": 0.1},
        "mu_d": 2
    }
    example_NHGM(config_NH)


    # Example 1 single process (1 context)
    config_single = {
        "N_samples": 1,
        "N_blocks": 1,
        "N_ctx": 1,
        "N_tones": 160,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "mu_tau": 4, # tau = 1 / (1 - a) = x_lim / b
        "si_tau": 1,
        "si_lim": 5,
        "si_stat": 0.2,  # stationary process variance
        "si_r": 0.2,  # measurement noise
    }
    # example_single(config_single)



