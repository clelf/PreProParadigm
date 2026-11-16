import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model_RTs import aggregate_data, get_session_types, _prep_rt_frame, PER_RUN_MODELS
from figure_metadoc import log_figure_metadata


def get_loglik_per_session_type(subjects, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                                pi_rule, correcttrials):
    """Log-likelihood (natural log) of every model of PER_RUN_MODELS at every valid (optionally correct)
    deviant trial, with subject, session type and model labels.

    These are the values on the x-axis of the RT-vs-likelihood figures with loglik=True: same KF predictions
    (each session at its own true si_r), same LI predictions, same trial selection. Non-positive values have
    no log and are left out, as in those figures; their number per model is printed.
    """
    session_types, session_type_to_params = get_session_types(trials_path)
    dfs = []
    for session_type in session_types:
        d, si_stat, si_r = session_type
        label = f"d={d:g}, si_stat={si_stat:g}, si_r={si_r:g}"
        for sub in subjects:
            sess_num = session_type_to_params[session_type][sub]
            df_kf = aggregate_data(trials_path, logfiles_path, sub, sess_num, preds_liks_path, pi_rule,
                                   sigma_r=si_r)
            df_li = aggregate_data(trials_path, logfiles_path, sub, sess_num, leaky_preds_path, pi_rule,
                                   file_prefix='leaky_predictions')
            df_kf = _prep_rt_frame(df_kf, logrt=True, correcttrials=correcttrials)
            df_li = _prep_rt_frame(df_li, logrt=True, correcttrials=correcttrials)
            for model in PER_RUN_MODELS:
                values = (df_kf if model['src'] == 'kf' else df_li)[model['col']].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                n_non_positive = int(np.sum(values <= 0))
                if n_non_positive:
                    print(f"sub {sub}, {label}, {model['name']}: {n_non_positive} non-positive values left out")
                dfs.append(pd.DataFrame({
                    'sub': sub,
                    'session_type': label,
                    'model': model['name'],
                    'loglik': np.log(values[values > 0]),
                }))
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    # Subject 07 has trial lists but no KF / LI prediction files, so it cannot be included here.
    SUBJECTS = ['04', '05', '06']
    CORRECTTRIALS = True

    trials_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/'
    logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles'
    preds_liks_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/kalman_pred'
    leaky_preds_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/leaky_predictions'
    save_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/RT_analysis_figures'
    os.makedirs(save_path, exist_ok=True)

    # 3-rule transition matrix, last row the null (no-deviant) rule -- as in RT_analysis_figures_persubject.py.
    pi_rule = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.5, 0.5, 0.0]
    ])

    df = get_loglik_per_session_type(SUBJECTS, trials_path, logfiles_path, preds_liks_path, leaky_preds_path,
                                     pi_rule, CORRECTTRIALS)
    print(df.groupby(['model', 'session_type'])['loglik'].agg(['count', 'mean', 'std']))

    session_type_order = list(dict.fromkeys(df['session_type']))  # get_session_types order: sorted by (d, si_stat, si_r)
    model_names = [model['name'] for model in PER_RUN_MODELS]

    # One panel per model, each with its own y-axis: the bare KF / LI log-likelihoods span a much wider
    # range than the priors and the prior-weighted likelihoods.
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))
    for ax, model_name in zip(axs.flat, model_names):
        sns.violinplot(data=df[df['model'] == model_name], x='session_type', y='loglik',
                       order=session_type_order, ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel('session type')
        ax.set_ylabel('log likelihood')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    fig.suptitle('log likelihood distributions' + (' (correct trials)' if CORRECTTRIALS else ' (valid trials)'))
    fig.tight_layout()

    fig_path = os.path.join(
        save_path, f"loglik_distributions_per_session_type{'_correcttrials' if CORRECTTRIALS else '_validtrials'}_sub-{'-'.join(SUBJECTS)}.png")
    fig.savefig(fig_path, dpi=200)
    log_figure_metadata(fig_path, __file__)
    print(f"Saved {fig_path}")
