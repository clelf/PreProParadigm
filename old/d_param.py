import numpy as np
import scipy.stats as ss

class effect_size:

    def __init__(self):
        self.params = {
            "mu_std": -1, # mu standard
            "mu_dev": 1, # mu deviant
            "sigma_stat": 0.3, # stationary std
            "sigma_r": 0.15, # observation noise
            "d": 0.7, # observation noise
            "fmin": 20,
            "fmax": 20000
        }

    def compute_d(self):

        sigma_total = self.params["sigma_stat"] + self.params["sigma_r"]
        d = (self.params["mu_dev"]-self.params["mu_std"])/sigma_total

        return d

    def compute_mu_dev_from_d(self):

        sigma_total = self.params["sigma_stat"] + self.params["sigma_r"]
        mu_dev_comp = (self.params["d"]*sigma_total) + self.params["mu_std"]

        return mu_dev_comp



