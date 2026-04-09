from model_RTs import compare_likelihoods_with_RTs_global
import numpy as np

if __name__ == "__main__":


    trials_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/'
            
    # Path to save Kalman predictions and likelihoods
    # results_save_path = os.path.join(os.path.dirname(__file__), 'results')
    results_save_path = f'/home/clevyfidel/Documents/Workspace/PreProParadigm/kalman_predictions_new' #os.path.join(os.path.dirname(__file__), 'results')

    # Path to save RT vs likelihood comparison results
    RT_results_path =  f'/home/clevyfidel/Documents/Workspace/PreProParadigm/RT_comparisons'
    
    # Logfiles path
    logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles' #sub-{sub}-ses-{sess+1}*run{run+1}*.tsv'


    # Rule transition matrix
    pi_rule = np.array([ # actual values of pilots so far
        [0.8, 0.1, 0.1], 
        [0.1, 0.8, 0.1], 
        [0.5, 0.5, 0.0]  
    ])

    compare_likelihoods_with_RTs_global(trials_path, results_save_path, logfiles_path, RT_results_path, pi_rule, take_dpos=True, take_rules=True)