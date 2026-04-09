import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob



#from audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM

# Check which folder exists and append the correct path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
if os.path.exists(os.path.join(base_path, 'Kalman')):
    from Kalman.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext, likelihood_observation, contexts_to_probabilities
elif os.path.exists(os.path.join(base_path, 'KalmanFilterViz1D')):
    from KalmanFilterViz1D.kalman import MIN_OBS_FOR_EM, kalman_online_fit_predict_multicontext, likelihood_observation, contexts_to_probabilities
else:
    raise ImportError("Neither 'Kalman' nor 'KalmanFilterViz1D' folder found.")

from model_RTs import compute_likelihoods_at_deviants


from tqdm import tqdm



if __name__ == "__main__":

    subs = ['01','02','03','04','05','06','07']

    for sub in tqdm(subs[::-1], desc="Subject"):

        n_sessions = len(os.listdir(os.path.join(os.path.dirname(__file__), f"triallists/sub-{sub}/")))
        print(n_sessions)

        for sess in tqdm(range(0,n_sessions), desc="Session", leave=False):

            trials_path = glob.glob(os.path.join(os.path.dirname(__file__), f"triallists/sub-{sub}/sub-{sub}_ses-{sess+1}*.csv"))
            print(trials_path)

            results_save_path = os.path.join(os.path.dirname(__file__), 'kalman_predictions_new')

            if os.path.exists(os.path.join(results_save_path, f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv')) and os.path.exists(os.path.join(results_save_path, f'kalman_predictions_and_likelihoods_at_deviants_sub-{sub}_ses-{sess+1}.csv')):
                print('Results already exist, skipping computation.')
                continue
            else:
                compute_likelihoods_at_deviants(trials_path[0],sub,sess, results_save_path)
