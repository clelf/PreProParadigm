import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import scipy.stats as ss

class KalmanEM:
    """2D Linear Gaussian State Space Model with built-in EM parameter estimation"""

    def __init__(self):
        self.true_params = {}
        self.true_tau = None
        self.true_b = None

    def generate_observations(self, T=100, tau=None, b=None, sigma_q=None, sigma_r=None, C=None, x0=None):
        """Generate synthetic observations using x_t = (b - x_t-1)/tau + noise
        Uses 2D state: [x_t, 1] where second dimension is constant for intercept handling
        In steady state: x --> b"""

        stat_std = 25 # fix the stationary

        # Default parameters for tau and b (scalars)
        if tau is None:
            tau = 1.2
        if b is None:
            b = 0.5
        if sigma_q is None:
            sigma_q = (stat_std*np.sqrt(2*tau-1))/tau  # compute sigma_q from the stationary
        if sigma_r is None:
            sigma_r = 10  # Observation noise standard deviation
        if C is None:
            C = np.array([[1.0, 0.0]])  # Observe only first state (1D observation)
        if x0 is None:
            x0 = np.array([ss.norm.rvs(b,stat_std), 1.0]) # draw x0 from the stationary

        # Construct noise covariance matrices from sigma parameters
        Q = np.array([[sigma_q**2, 0.0], [0.0, 0.0]])  # Only first state has noise
        R = np.array([[sigma_r**2]])  # 1D observation noise

        # Store true tau and b
        self.true_tau = tau
        self.true_b = b

        # Convert x_{t+1} = x_t + (b - x_t)/tau + noise to state space form
        # x_{t+1} = x_t(1 - 1/tau) + b/tau + noise
        # Using augmented state [x_t, 1]:
        # [x_t  ]   [1-1/tau  b/tau] [x_t-1]
        # [1    ] = [0        1    ] [1    ] + noise
        A = np.array([[1.0 - 1.0/tau, b/tau],
                      [0.0, 1.0]])

        # Store true parameters
        self.true_params = {
            'transition_matrices': A,
            'transition_covariance': Q,
            'observation_matrices': C,
            'observation_covariance': R,
            'initial_state_mean': x0,
            'initial_state_covariance': np.eye(2) * 0.1
        }

        # Generate states manually using the exact dynamics
        # x_{t+1} = x_t + (b - x_t)/tau + noise
        states = np.zeros((T, 2))
        observations = np.zeros((T, 1))

        # Initial state
        states[0] = x0

        # Generate process and observation noise
        if sigma_q > 0:
            process_noise = np.random.normal(0, sigma_q, T)
        else:
            process_noise = np.zeros(T)

        if sigma_r > 0:
            observation_noise = np.random.normal(0, sigma_r, T)
        else:
            observation_noise = np.zeros(T)

        # Generate states using exact dynamics
        for t in range(T - 1):
            # x_{t+1} = x_t + (b - x_t)/tau + process_noise
            states[t + 1, 0] = states[t, 0] + (b - states[t, 0]) / tau + process_noise[t + 1]
            states[t + 1, 1] = 1.0  # Keep second component constant

        # Generate observations: y_t = x_t + observation_noise
        for t in range(T):
            observations[t, 0] = states[t, 0] + observation_noise[t]

        return states, observations, sigma_q, sigma_r

    def estimate_parameters(self, observations, n_iter=10, tau_init=None, b_init=None,
                          true_sigma_q=None, true_sigma_r=None):
        """Use PyKalman's built-in EM algorithm for parameter estimation
        Uses ground truth noise covariances (fixed, not estimated)"""

        # Use reasonable or provided initial guesses
        if tau_init is None:
            tau_init = 3.0  # Reasonable default initial guess for tau
        if b_init is None:
            # Estimate b from final observations (rough steady state estimate)
            if len(observations) > 20:
                b_init = np.mean(observations[-20:])  # Use mean of final observations
            else:
                b_init = np.mean(observations[-len(observations)//2:])  # Use latter half

        # Use provided or default ground truth noise parameters
        if true_sigma_q is None:
            true_sigma_q = 0.1  # Default process noise
        if true_sigma_r is None:
            true_sigma_r = 0.2  # Default observation noise

        # print(true_sigma_q)    

        # Convert to A matrix format: x_t = x_t-1 + (b - x_t-1)/tau
        A_init = np.array([[1.0 - 1.0/tau_init, b_init/tau_init], [0.0, 1.0]])

        # Ground truth noise covariances (FIXED, not estimated)
        Q_true = np.array([[true_sigma_q**2, 0.0], [0.0, 0.0]])  # Only first state has noise
        R_true = np.array([[true_sigma_r**2]])  # 1D observation noise

        kf = KalmanFilter(
            transition_matrices=A_init,  # Start with random initial guess
            observation_matrices=np.array([[1.0, 0.0]]),  # Fixed - observe only x, not intercept
            transition_covariance=Q_true,  # Ground truth (FIXED)
            observation_covariance=R_true,  # Ground truth (FIXED)
            initial_state_mean=np.array([0.0, 1.0]),  # Second component should be 1
            initial_state_covariance=np.array([[1.0, 0.0], [0.0, 0.01]]),  # Small variance on intercept
            n_dim_state=2,
            n_dim_obs=1  # 1D observations
        )

        # Fit parameters using EM algorithm - only estimate transition matrix and initial conditions
        kf_fitted = kf.em(observations, n_iter=n_iter,
                          em_vars=['transition_matrices', 'initial_state_mean', 'initial_state_covariance'])

        return kf_fitted

    def extract_tau_b_from_A(self, A_matrix):
        """Extract tau and b parameters from estimated A matrix
        A = [[1-1/tau, b/tau],
             [0,       1    ]]
        """
        tau_est = 1.0 / (1.0 - A_matrix[0, 0])
        b_est = A_matrix[0, 1] * tau_est
        return tau_est, b_est

    def filter_and_smooth(self, kf_fitted, observations):
        """Apply Kalman filtering and smoothing with fitted parameters"""

        # Kalman filtering
        state_means_filt, state_covariances_filt = kf_fitted.filter(observations)

        # Kalman smoothing
        state_means_smooth, state_covariances_smooth = kf_fitted.smooth(observations)

        return state_means_filt, state_covariances_filt, state_means_smooth, state_covariances_smooth

    def plot_results(self, true_states, observations, estimated_states, kf_fitted):
        """Plot comparison of true vs estimated states and parameters"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        T = len(observations)
        time = range(T)

        # Plot state 1
        ax1.plot(time, true_states[:, 0], 'g-', label='True State 1', linewidth=2)
        ax1.plot(time, estimated_states[:, 0], 'b--', label='Estimated State 1', linewidth=2)
        ax1.plot(time, observations.flatten(), 'r.', label='Observations', markersize=4, alpha=0.7)
        ax1.set_title('State 1 Estimation')
        ax1.legend()
        ax1.grid(True)

        # Plot state 2
        ax2.plot(time, true_states[:, 1], 'g-', label='True State 2', linewidth=2)
        ax2.plot(time, estimated_states[:, 1], 'b--', label='Estimated State 2', linewidth=2)
        ax2.set_title('State 2 Estimation (Unobserved)')
        ax2.legend()
        ax2.grid(True)

        # Plot innovation/residuals for state 1
        residuals = observations.flatten() - estimated_states[:, 0]
        ax3.plot(time, residuals, 'r-', linewidth=1)
        ax3.set_title('Residuals (Obs - Est) for State 1')
        ax3.grid(True)

        # Plot state covariances
        state_vars = [kf_fitted.filter(observations)[1][t][0,0] for t in range(T)]
        ax4.plot(time, np.sqrt(state_vars), 'b-', label='State 1 Std Dev')
        ax4.set_title('Estimation Uncertainty')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

        # Extract tau and b from estimated A matrix
        tau_est, b_est = self.extract_tau_b_from_A(kf_fitted.transition_matrices)

        # Print parameter comparison
        print("\n=== Parameter Comparison ===")
        print(f"True tau: {self.true_tau:.4f}")
        print(f"Estimated tau: {tau_est:.4f}")
        print(f"True b: {self.true_b:.4f}")
        print(f"Estimated b: {b_est:.4f}")

        print(f"\nTrue A:\n{self.true_params['transition_matrices']}")
        print(f"Estimated A:\n{kf_fitted.transition_matrices}")
        print(f"\nTrue Q:\n{self.true_params['transition_covariance']}")
        print(f"Estimated Q:\n{kf_fitted.transition_covariance}")
        print(f"\nTrue R:\n{self.true_params['observation_covariance']}")
        print(f"Estimated R:\n{kf_fitted.observation_covariance}")

        # Compute estimation errors
        tau_error = abs(self.true_tau - tau_est)
        b_error = abs(self.true_b - b_est)
        A_error = np.linalg.norm(self.true_params['transition_matrices'] - kf_fitted.transition_matrices)
        Q_error = np.linalg.norm(self.true_params['transition_covariance'] - kf_fitted.transition_covariance)
        print(f"\nParameter estimation errors:")
        print(f"|tau_true - tau_est|: {tau_error:.4f}")
        print(f"|b_true - b_est|: {b_error:.4f}")
        print(f"||A_true - A_est||: {A_error:.4f}")
        print(f"||Q_true - Q_est||: {Q_error:.4f}")

    def parameter_estimation_analysis(self, sigma_r, n_b_samples=100):
        """
        Comprehensive analysis of parameter estimation across different N and tau values
        with varying sigma_q values

        Parameters:
        - sigma_r: Observation noise standard deviation
        - n_b_samples: Number of b values to sample for each (N, tau) pair
        """
        # Define parameter ranges
        N_values = [8, 16, 24, 32, 40, 48, 56] # multiples of trial lengths
        tau_values = np.array([16, 40, 160, 240]) # use tau values currently implemented
        sigma_q_values = [None]  # see function above
        b_range = (300, 1500)  # Hz range for sampling b values

        # Define colors for different sigma_q values
        colors = ['blue', 'green', 'orange', 'red']

        # Create figure with subplots
        fig, axes = plt.subplots(len(N_values), len(tau_values),
                                figsize=(len(tau_values)*3, len(N_values)*3))

        # Ensure axes is always 2D
        if len(N_values) == 1:
            axes = axes.reshape(1, -1)
        if len(tau_values) == 1:
            axes = axes.reshape(-1, 1)

        print(f"Running analysis with sigma_q={sigma_q_values}, sigma_r={sigma_r}")
        print(f"N values: {N_values}")
        print(f"Tau values: {tau_values}")

        for i, N in enumerate(N_values):
            for j, tau in enumerate(tau_values):
                print(f"Processing N={N}, tau={tau:.2f}")

                ax = axes[i, j]
                legend_labels = []

                # Iterate over different sigma_q values
                for k, sigma_q in enumerate(sigma_q_values):
                    # Sample b values using linspace
                    b_true_values = np.linspace(b_range[0], b_range[1], n_b_samples)
                    b_estimated_values = []

                    for b_true in b_true_values:
                        try:
                            # Generate observations
                            true_states, observations, sigma_q_out, sigma_r_out = self.generate_observations(
                                T=N, tau=tau, b=b_true, sigma_q=sigma_q, sigma_r=sigma_r
                            )

                            # Estimate parameters using ground truth noise covariances
                            kf_fitted = self.estimate_parameters(observations, n_iter=30,
                                                              true_sigma_q=sigma_q_out, true_sigma_r=sigma_r_out)

                            # Extract estimated b
                            tau_est, b_est = self.extract_tau_b_from_A(kf_fitted.transition_matrices)
                            # print(f"estimated tau = {tau_est}")
                            b_estimated_values.append(b_est)

                        except Exception as e:
                            print(f"Error for N={N}, tau={tau:.2f}, b={b_true:.2f}, sigma_q={sigma_q}: {e}")
                            b_estimated_values.append(np.nan)

                    # Create scatter plot for this sigma_q
                    valid_mask = ~np.isnan(b_estimated_values)
                    if np.sum(valid_mask) > 0:
                        b_est_array = np.array(b_estimated_values)[valid_mask]
                        b_true_array = b_true_values[valid_mask]

                        # Count points outside the (-2, 2) range
                        out_of_range = np.sum((b_est_array < 300) | (b_est_array > 1500))
                        out_ratio = out_of_range / len(b_est_array) if len(b_est_array) > 0 else 0

                        # Plot with more transparency and specific color
                        ax.scatter(b_true_array, b_est_array, alpha=0.4, s=30,
                                  color=colors[k], facecolors=colors[k])

                        # Add to legend
                        legend_labels.append(f'σ_q={sigma_q_out:.2f} (out_rat={out_ratio:.2f})')

                # Add identity line within the (-2, 2) range
                ax.plot([300, 1500], [300, 1500], 'k--', alpha=0.5)

                # Add legend in top-left corner
                if legend_labels:
                    ax.legend(legend_labels, loc='upper left', fontsize=6, framealpha=0.8)

                # Set axis limits to (-2, 2)
                ax.set_xlim(300, 1500)
                ax.set_ylim(300, 1500)

                ax.set_title(f'N={N}, τ={tau:.1f}', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Set labels only for edge subplots
                if i == len(N_values) - 1:  # Bottom row
                    ax.set_xlabel('True b', fontsize=9)
                if j == 0:  # Left column
                    ax.set_ylabel('Estimated b', fontsize=9)

        # Add overall labels
        fig.text(0.5, 0.04, 'True b', ha='center', fontsize=12, weight='bold')
        fig.text(0.04, 0.5, 'Estimated b', va='center', rotation='vertical', fontsize=12, weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.08, bottom=0.08)

        # Save the plot
        filename = f'kalman_b_estimation_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.close()  # Close the figure to free memory

        return fig


if __name__ == "__main__":
    # Create instance
    kem = KalmanEM()

    # Run comprehensive parameter estimation analysis
    print("=== Parameter Estimation Analysis ===")
    print("Running comprehensive analysis across N and tau values...")
    print("This may take a few minutes...")

    # Run analysis with specific noise parameters
    kem.parameter_estimation_analysis(sigma_r=10, n_b_samples=100)