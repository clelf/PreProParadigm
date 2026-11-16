import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model_RTs import load_logs, _prep_rt_frame
from figure_metadoc import log_figure_metadata


def get_sessions_of_subject(trials_path, sub):
    """(session number, session type label, sort key) of each session the subject has, read from the trials filenames."""
    trial_files = (glob.glob(f"{trials_path}/sub-{sub}/sub-{sub}_ses-*-*trials.csv")
                   or glob.glob(f"{trials_path}/sub-{sub}_ses-*-*trials.csv"))
    sessions = []
    for trial_file in trial_files:
        name = os.path.basename(trial_file)
        sess_num = int(re.search(r'_ses-(\d+)-', name).group(1))
        d, si_stat, si_r = [float(v) for v in re.search(r'-d([\d.]+)-si_stat([\d.]+)-si_r([\d.]+)', name).groups()]
        sessions.append((sess_num, f"d={d:g}, si_stat={si_stat:g}, si_r={si_r:g}", (d, si_stat, si_r)))
    return sessions


def get_logrt_per_session_type(subjects, trials_path, logfiles_path, correcttrials):
    """Valid (optionally correct) log-RT of every deviant trial, with subject and session type labels."""
    dfs = []
    for sub in subjects:
        for sess_num, label, sort_key in get_sessions_of_subject(trials_path, sub):
            logs = load_logs(logfiles_path, sub, sess_num, filter=False)
            logs = _prep_rt_frame(logs, logrt=True, correcttrials=correcttrials)
            logs['sub'] = sub
            logs['session_type'] = label
            logs['sort_key'] = [sort_key] * len(logs)
            dfs.append(logs[['sub', 'session_type', 'sort_key', 'rt']].rename(columns={'rt': 'log_rt'}))
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    SUBJECTS = ['04', '05', '06', '07']
    CORRECTTRIALS = True

    trials_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/triallists/'
    logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/logfiles'
    save_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/RT_analysis_figures'
    os.makedirs(save_path, exist_ok=True)

    df = get_logrt_per_session_type(SUBJECTS, trials_path, logfiles_path, CORRECTTRIALS)
    print(df.groupby(['session_type', 'sub'])['log_rt'].agg(['count', 'mean', 'std']))

    session_type_order = df.drop_duplicates('session_type').sort_values('sort_key')['session_type'].tolist()
    fig, ax = plt.subplots(figsize=(3 * len(session_type_order) + 2, 5))
    sns.violinplot(data=df, x='session_type', y='log_rt', hue='sub', order=session_type_order, ax=ax)
    ax.set_xlabel('session type')
    ax.set_ylabel('log RT')
    ax.set_title('log RT distributions' + (' (correct trials)' if CORRECTTRIALS else ' (valid trials)'))
    ax.legend(title='subject')
    fig.tight_layout()

    fig_path = os.path.join(
        save_path, f"logRT_distributions_per_session_type{'_correcttrials' if CORRECTTRIALS else '_validtrials'}_sub-{'-'.join(SUBJECTS)}.png")
    fig.savefig(fig_path, dpi=200)
    log_figure_metadata(fig_path, __file__)
    print(f"Saved {fig_path}")
