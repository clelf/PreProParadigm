"""Plot, for each model, condition and within-trial position k, the distribution over the sequences of the
difference between the MSE of the two batches of sequences of generate_task_sequences_gm_test.py,
MSE_k^original - MSE_k^corrected. The difference is computed between the two sequences with the same name
(e.g. 'sub-seq10') and condition, which share their random seed.

Reads the CSVs written by compute_mse_by_position_gm_test.py in OUTPUT_ROOT/mse_by_position/<batch>/<model>.csv
(one row per sequence). The models are those with a CSV in both batch folders.

As a baseline to compare the differences with, each figure also shows the variance sigma_q**2 + sigma_r**2 of a
condition at +/- its value (red dotted lines): the MSE of a prediction that would know the current standard state
exactly, left with the process noise of one step and the observation noise only. sigma_q and sigma_r are read from
the trials CSVs of the sequences of the original batch, in SEQUENCES_ROOT/<batch>/.

Outputs, in OUTPUT_ROOT/mse_by_position/<model>/, two figures where each dot is a pair of sequences and a
gray dashed line marks a difference of 0 (no effect of the sampler):
  mse_difference_by_position.png    one panel per position k, with one violin per condition
  mse_difference_by_condition.png   one panel per condition, with one violin per position k; the dots of a
                                    sequence are connected by a thin black line from violin to violin
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PREPROPARADIGM_DIR = Path(__file__).resolve().parent

MSE_DIFFERENCE_LABEL = r'$MSE_k^{original} - MSE_k^{corrected}$'


def get_mse_difference_df(mse_dir, original_batch_name, corrected_batch_name):
    """DataFrame of MSE_k^original - MSE_k^corrected for each pair of sequences, in long format, with the
    columns model, sequence, condition, position, mse_difference. Only the models with a CSV in both batch
    folders are kept.
    """
    original_csvs = {csv.stem: csv for csv in (mse_dir / original_batch_name).glob('*.csv')}
    corrected_csvs = {csv.stem: csv for csv in (mse_dir / corrected_batch_name).glob('*.csv')}
    model_names = sorted(set(original_csvs) & set(corrected_csvs))
    if len(model_names) == 0:
        raise FileNotFoundError(f"No model with a CSV in both {original_batch_name} and {corrected_batch_name}")

    model_dfs = []
    for model_name in model_names:
        original_df = pd.read_csv(original_csvs[model_name], index_col=['sequence', 'condition'])
        corrected_df = pd.read_csv(corrected_csvs[model_name], index_col=['sequence', 'condition'])
        if not original_df.index.sort_values().equals(corrected_df.index.sort_values()):
            raise ValueError(f"The sequences of {model_name} differ between the two batches")
        mse_columns = [column for column in original_df.columns if column.startswith('MSE_')]
        difference_df = original_df[mse_columns] - corrected_df.loc[original_df.index, mse_columns]

        model_df = difference_df.reset_index().melt(id_vars=['sequence', 'condition'], var_name='position',
                                                    value_name='mse_difference')
        model_df['position'] = model_df['position'].str.removeprefix('MSE_').astype(int)
        model_df.insert(0, 'model', model_name)
        model_dfs.append(model_df)
    return pd.concat(model_dfs, ignore_index=True)


def get_baseline_variance_by_condition(sequences_dir, condition_order):
    """Series of the baseline variance sigma_q_std**2 + sigma_r**2 of each condition (index: condition), read
    from the trials CSV of the first sequence of the condition. sigma_q_std and sigma_r are set by the
    parameters of the condition only, so they are the same for all its sequences.
    """
    baseline_variances = pd.Series(index=condition_order, dtype=float)
    for condition in condition_order:
        trials_csv = sorted(sequences_dir.glob(f'sub-seq*_{condition}_ses-1_trials.csv'))[0]
        trials_df = pd.read_csv(trials_csv)
        baseline_variances[condition] = trials_df['sigma_q_std'].iloc[0] ** 2 + trials_df['sigma_r'].iloc[0] ** 2
    return baseline_variances


def plot_mse_difference_by_position(model_mse_difference_df, condition_order, baseline_variances, model_name,
                                    figure_path):
    """Plot, for one model, one panel per position, one violin per condition, one dot per pair of sequences,
    with the baseline variance of each condition at +/- its value over the violin of the condition."""
    positions = sorted(model_mse_difference_df['position'].unique())
    n_columns = 2
    n_rows = int(np.ceil(len(positions) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(16, 3.5 * n_rows), sharex=True, squeeze=False, sharey=True)

    for ax, position in zip(axes.flat, positions):
        position_df = model_mse_difference_df[model_mse_difference_df['position'] == position]
        sns.violinplot(data=position_df, x='condition', y='mse_difference', order=condition_order,
                       inner=None, cut=0, ax=ax)
        sns.stripplot(data=position_df, x='condition', y='mse_difference', order=condition_order,
                      color='black', size=3, jitter=0.1, ax=ax)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        condition_x = np.arange(len(condition_order))
        for sign in [1, -1]:
            ax.hlines(sign * baseline_variances[condition_order].to_numpy(), condition_x - 0.4, condition_x + 0.4,
                      color='tab:red', linestyle=':', linewidth=1)

        ax.set_title(f'position k = {position}')
        ax.set_xlabel('')
        ax.set_ylabel(MSE_DIFFERENCE_LABEL)

    for ax in axes.flat[len(positions):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('condition')

    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_mse_difference_by_condition(model_mse_difference_df, condition_order, baseline_variances, model_name,
                                     figure_path):
    """Plot, for one model, one panel per condition, one violin per position, one dot per pair of sequences,
    with the dots of a pair of sequences connected across positions, and the baseline variance of the condition
    at +/- its value."""
    positions = sorted(model_mse_difference_df['position'].unique())
    n_columns = 4
    n_rows = int(np.ceil(len(condition_order) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(16, 3.5 * n_rows), sharex=True, squeeze=False, sharey=True)

    for ax, condition in zip(axes.flat, condition_order):
        condition_df = model_mse_difference_df[model_mse_difference_df['condition'] == condition]
        sns.violinplot(data=condition_df, x='position', y='mse_difference', order=positions,
                       inner=None, cut=0, color='lightgray', ax=ax)

        position_x = np.arange(len(positions))
        for sequence_name, sequence_df in condition_df.groupby('sequence'):
            sequence_differences = sequence_df.set_index('position').loc[positions, 'mse_difference']
            ax.plot(position_x, sequence_differences.to_numpy(), color='gray', linewidth=0.5, alpha=0.5)
            ax.scatter(position_x, sequence_differences.to_numpy(), color='black', s=8, zorder=3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        for sign in [1, -1]:
            ax.axhline(sign * baseline_variances[condition], color='tab:red', linestyle=':', linewidth=1)

        ax.set_title(condition, fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax in axes.flat[len(condition_order):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(MSE_DIFFERENCE_LABEL)
    for ax in axes[-1]:
        ax.set_xlabel('position k')

    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":

    OUTPUT_ROOT = PREPROPARADIGM_DIR / 'predictions_gm_test'
    MSE_DIR = OUTPUT_ROOT / 'mse_by_position'
    SEQUENCES_ROOT = PREPROPARADIGM_DIR / 'sequences_gm_test'
    ORIGINAL_BATCH_NAME = 'gm_original'
    CORRECTED_BATCH_NAME = 'gm_corrected'

    mse_difference_df = get_mse_difference_df(MSE_DIR, ORIGINAL_BATCH_NAME, CORRECTED_BATCH_NAME)

    # Conditions in the order of their number (cond1, cond2, ..., cond16), as in the MSE CSVs.
    condition_order = list(mse_difference_df['condition'].unique())

    baseline_variances = get_baseline_variance_by_condition(SEQUENCES_ROOT / ORIGINAL_BATCH_NAME, condition_order)
    print(f"Baseline variance sigma_q**2 + sigma_r**2 of each condition:\n{baseline_variances.to_string()}")

    for model_name, model_mse_difference_df in mse_difference_df.groupby('model'):

        model_figure_dir = MSE_DIR / model_name
        model_figure_dir.mkdir(exist_ok=True)

        figure_path = model_figure_dir / 'mse_difference_by_position.png'
        plot_mse_difference_by_position(model_mse_difference_df, condition_order, baseline_variances, model_name,
                                        figure_path)
        print(f"Figure saved to {figure_path}")

        figure_path = model_figure_dir / 'mse_difference_by_condition.png'
        plot_mse_difference_by_condition(model_mse_difference_df, condition_order, baseline_variances, model_name,
                                         figure_path)
        print(f"Figure saved to {figure_path}")
