import numpy as np
import scipy.stats as ss

class effect_sizes:

    '''
    Aim: determine mu deviant from mu standard and desired effect size
    Problem: effect size would need to be calculated on a logarithmic (log2) scale since pitch perception is logarithmic

    Step 1: compares d computed using delta rule and simulated data
    --> the results show that the computed standard deviations, means, and d's are relatively similar
    --> NOTE: delta method should probably get less precise the larger the standard deviation compared to the mean

    Step 2: compute deviant mu from standard mu and desired d (on log2 scale) using the delta rule and quick & dirty grid search
    
    '''

    def __init__(self):
        self.params = {
            "mu_standard": 700, # mu standard in Hz
            "mu_deviant": 750, # mu deviant in Hz (only for method comparison)
            "stationary_std_lin": 25, # stationary std in Hz
            "d_aim": -2 # desired effect size on log2 scale to compute mu deviant from mu standard
        }

    def get_log_std(self, mu, lin_std):
        '''search terms: delta method approximate standard deviation on log2 scale'''
        '''see also https://en.wikipedia.org/wiki/Variance-stabilizing_transformation'''
        '''takes mu and std on the linear scale and approximates standard deviation on the log scale'''

        log_std = lin_std/(mu * np.log(2))

        return log_std

    def get_d(self, mu_1, mu_2, std_1, std_2):
        '''compute cohens d from means and standard deviations of two variables'''

        d = (mu_1 - mu_2)/(np.sqrt((std_1**2 + std_2**2)/2))

        return d

    def get_mu_sigma(self, vals):
        '''compute mean and standard deviation'''

        mu = np.mean(vals)
        sigma = np.std(vals, ddof=1)

        return mu, sigma

    def get_mu_dev_from_d(self):
        '''computes a mean for the deviant sound based on mu_standard, the stationary standard deviation and a desired d (from log scale)'''

        # compute log std for standard
        log_std_standard = self.get_log_std(self.params['mu_standard'], self.params['stationary_std_lin'])

        # rough grid search for deviant mu from a range of mu_stdandard to mu_standard + mu_standard/2
        mu_dev_guess_list = np.linspace(self.params['mu_standard'], self.params['mu_standard'] + self.params['mu_standard']/2, 1000)

        for mu_dev_guess in mu_dev_guess_list:
            
            # compute standard deviation of deviant on log scale using the delta rule (and the shared stationary std on linear scale)
            log_std_dev = self.get_log_std(mu_dev_guess, self.params['stationary_std_lin'])

            # compute d on log scale and compare to desired d (leave a error tolerance of 0.005)
            d_now = self.get_d(np.log2(self.params['mu_standard']), np.log2(mu_dev_guess), log_std_standard, log_std_dev)

            if (self.params['d_aim'] - 0.005 <= d_now < self.params['d_aim']  + 0.005):

                return mu_dev_guess, d_now

    def compare_delta_sim(self):
        '''compare cohens d computed by delta method and simulated data'''

        # use simulated data from stationary normal distibution
        sim_1 = ss.norm.rvs(self.params['mu_standard'], self.params['stationary_std_lin'], 10000)
        sim_2 = ss.norm.rvs(self.params['mu_deviant'], self.params['stationary_std_lin'], 10000)

        # approximate d from linear scale using delta rule
        log_std_1 = self.get_log_std(self.params['mu_standard'], self.params['stationary_std_lin'])
        log_std_2 = self.get_log_std(self.params['mu_deviant'], self.params['stationary_std_lin'])

        cohens_d = self.get_d(np.log2(self.params['mu_standard']), np.log2(self.params['mu_deviant']), log_std_1, log_std_2)

        # approximate d from simulated data
        sim_1_log = np.log2(sim_1)
        sim_2_log = np.log2(sim_2)

        mu_sim_1, sigma_sim_1 = self.get_mu_sigma(sim_1_log)
        mu_sim_2, sigma_sim_2 = self.get_mu_sigma(sim_2_log)

        cohens_d_sim = self.get_d(mu_sim_1, mu_sim_2, sigma_sim_1, sigma_sim_2)

        # compare both approximations methods
        '''NOTE: for larger number of data points in simulation relatively close'''
        print(f"Cohen's d (approx via delta method): {cohens_d:.4f}") 
        print(f"Cohen's d (approx from simulation): {cohens_d_sim:.4f}")


if __name__ == "__main__":

    eff = effect_sizes()

    print("=== Comparing effect sizes from delta method and simulation: ===")
    eff.compare_delta_sim()

    print("=== Computing deviant mu from standard mu and effect size ===")
    mu_dev, d = eff.get_mu_dev_from_d()
    print(f"Mu deviant: {mu_dev:.4f}")
    print(f"Effect size: {d:.4f}")