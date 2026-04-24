"""
Visualize RT vs Kalman Likelihoods for each observation noise level.
Produces one figure per observation noise level found in kalman_predictions_noise_comparison.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as ss
from matplotlib.lines import Line2D
from tqdm import tqdm
from model_RTs import compare_likelihoods_with_RTs_global


def extract_observation_noise_levels(noise_dir):
    """
    Extract all unique observation noise levels from filenames in the directory.
    
    Args:
        noise_dir: Directory containing kalman prediction files with noise levels
        
    Returns:
        List of float observation noise levels sorted in ascending order
    """
    files = glob.glob(os.path.join(noise_dir, "kalman_predictions_and_likelihoods_at_deviants_*.csv"))
    noise_levels = set()
    
    pattern = r'_obs_noise_([0-9.]+)\.csv$'
    for filepath in files:
        match = re.search(pattern, filepath)
        if match:
            noise_levels.add(float(match.group(1)))
    
    return sorted(list(noise_levels))




if __name__ == "__main__":
    
    # Paths
    trials_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/'
    preds_liks_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/kalman_predictions_noise_comparison_corr'
    comparison_save_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/RT_noise_comparison'
    logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles'
    
    # Ensure output directory exists
    os.makedirs(comparison_save_path, exist_ok=True)
    
    # Rule transition matrix
    pi_rule = np.array([
        [0.8, 0.1, 0.1], 
        [0.1, 0.8, 0.1], 
        [0.5, 0.5, 0.0]  
    ])
    
    subjects = ['04', '05', '06']
    
    # Extract unique observation noise levels from the directory
    observation_noise_levels = extract_observation_noise_levels(preds_liks_path)
    
    print(f"Found observation noise levels: {observation_noise_levels}")
    
    # Generate visualization for each noise level
    for obs_noise in observation_noise_levels:
        print(f"\nProcessing observation noise level: {obs_noise}")
        compare_likelihoods_with_RTs_global(subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule, take_dpos=True, take_rules=True, obs_noise=obs_noise)
    
    print(f"\nAll visualizations completed, results saved to {comparison_save_path}")
