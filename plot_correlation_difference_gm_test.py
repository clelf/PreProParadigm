"""Plot, for each model and condition, the distribution over the sequences of the difference between the
squared correlations of the two batches of sequences of generate_task_sequences_gm_test.py,
r_original**2 - r_corrected**2.
The difference is computed between the two sequences with the same name (e.g. 'sub-seq10') and condition, which
share their random seed.

Reads the CSVs written by compute_correlation_gm_test.py in OUTPUT_ROOT/correlation/<batch>/<model>.csv
(one row per sequence). The models are those with a CSV in both batch folders.

Outputs OUTPUT_ROOT/correlation/<model>/squared_correlation_difference_by_condition.png: one violin per
condition, where each dot is a pair of sequences and a dashed line marks a difference of 0 (no effect of the
sampler). An asterisk above a violin marks a mean difference significantly different from 0 (one-sample t-test over
the pairs of sequences of the condition, i.e. a paired t-test between r_original**2 and r_corrected**2,
p < SIGNIFICANCE_LEVEL, not corrected for the number of conditions).
Also outputs OUTPUT_ROOT/correlation/squared_correlation_difference_by_condition_all_models.png: the same plot for
all the models, one panel per model, stacked vertically.
And outputs, for all the models, with one row of panels per model, stacked vertically:
  correlation_original_vs_corrected_all_models.png   one panel per condition, r_original (x) against
                                                     r_corrected (y), one dot per pair of sequences, with the
                                                     identity line
  squared_correlation_by_condition_all_models.png    one panel per model, paired bars of the mean of r_original**2
                                                     and r_corrected**2 for each condition (95% confidence
                                                     intervals over the sequences)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

PREPROPARADIGM_DIR = Path(__file__).resolve().parent

SQUARED_CORRELATION_DIFFERENCE_LABEL = r'$r_{original}^2 - r_{corrected}^2$'
SIGNIFICANCE_LEVEL = 0.05


def get_squared_correlation_difference_df(correlation_dir, original_batch_name, corrected_batch_name):
    """DataFrame of r_original, r_corrected and r_original**2 - r_corrected**2 for each pair of sequences, with the
    columns model, sequence, condition, r_original, r_corrected, squared_correlation_difference. Only the models with
    a CSV in both batch folders are kept.
    """
    original_csvs = {csv.stem: csv for csv in (correlation_dir / original_batch_name).glob('*.csv')}
    corrected_csvs = {csv.stem: csv for csv in (correlation_dir / corrected_batch_name).glob('*.csv')}
    model_names = sorted(set(original_csvs) & set(corrected_csvs))
    if len(model_names) == 0:
        raise FileNotFoundError(f"No model with a CSV in both {original_batch_name} and {corrected_batch_name}")

    model_dfs = []
    for model_name in model_names:
        original_df = pd.read_csv(original_csvs[model_name], index_col=['sequence', 'condition'])
        corrected_df = pd.read_csv(corrected_csvs[model_name], index_col=['sequence', 'condition'])
        if not original_df.index.sort_values().equals(corrected_df.index.sort_values()):
            raise ValueError(f"The sequences of {model_name} differ between the two batches")
        model_df = pd.DataFrame({'r_original': original_df['correlation'],
                                 'r_corrected': corrected_df.loc[original_df.index, 'correlation']})
        model_df['squared_correlation_difference'] = model_df['r_original'] ** 2 - model_df['r_corrected'] ** 2
        model_df = model_df.reset_index()
        model_df.insert(0, 'model', model_name)
        model_dfs.append(model_df)
    return pd.concat(model_dfs, ignore_index=True)


def get_squared_correlation_difference_p_values(model_squared_correlation_difference_df, condition_order):
    """p-value of a one-sample t-test of the mean of r_original**2 - r_corrected**2 against 0, for each condition, as a
    Series indexed by condition."""
    p_values = {}
    for condition in condition_order:
        condition_differences = model_squared_correlation_difference_df.loc[
            model_squared_correlation_difference_df['condition'] == condition, 'squared_correlation_difference']
        p_values[condition] = stats.ttest_1samp(condition_differences, 0).pvalue
    return pd.Series(p_values)


def plot_model_squared_correlation_difference_panel(ax, model_squared_correlation_difference_df, condition_order, p_values,
                                            model_name):
    """Plot, in one panel, for one model, one violin per condition, one dot per pair of sequences, and an asterisk
    above the conditions whose p-value (in p_values, indexed by condition) is below SIGNIFICANCE_LEVEL."""
    sns.violinplot(data=model_squared_correlation_difference_df, x='condition', y='squared_correlation_difference',
                   order=condition_order, inner=None, cut=0, ax=ax)
    sns.stripplot(data=model_squared_correlation_difference_df, x='condition', y='squared_correlation_difference',
                  order=condition_order, color='black', size=3, jitter=0.1, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # Asterisks at the top of the frame, above the data: the y-axis is extended by 10% to leave room for them.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.1 * (y_max - y_min))
    for x, condition in enumerate(condition_order):
        if p_values[condition] < SIGNIFICANCE_LEVEL:
            ax.text(x, 0.98, '*', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=14)

    ax.set_xlabel('')
    ax.set_ylabel(SQUARED_CORRELATION_DIFFERENCE_LABEL)
    ax.set_title(model_name)


def plot_squared_correlation_difference_by_condition(model_squared_correlation_difference_df, condition_order, p_values, model_name,
                                             figure_path):
    """Plot, for one model, the panel of plot_model_squared_correlation_difference_panel in its own figure."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_model_squared_correlation_difference_panel(ax, model_squared_correlation_difference_df, condition_order, p_values,
                                            model_name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('condition')
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_squared_correlation_difference_by_condition_all_models(squared_correlation_difference_df, condition_order,
                                                        p_values_by_model, figure_path):
    """Plot the panels of plot_model_squared_correlation_difference_panel of all the models in one figure, stacked
    vertically, with p_values_by_model a dict of the p-values of each model (Series indexed by condition)."""
    model_names = list(p_values_by_model)
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 3.5 * len(model_names)), sharex=True,
                             squeeze=False)
    for ax, model_name in zip(axes[:, 0], model_names):
        model_squared_correlation_difference_df = squared_correlation_difference_df[squared_correlation_difference_df['model'] == model_name]
        plot_model_squared_correlation_difference_panel(ax, model_squared_correlation_difference_df, condition_order,
                                                p_values_by_model[model_name], model_name)
    last_ax = axes[-1, 0]
    plt.setp(last_ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    last_ax.set_xlabel('condition')
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_correlation_original_vs_corrected_all_models(squared_correlation_difference_df, condition_order, figure_path):
    """Plot one row of panels per model and one panel per condition, with r_original (x) against r_corrected (y),
    one dot per pair of sequences, and the identity line (no effect of the sampler)."""
    model_names = sorted(squared_correlation_difference_df['model'].unique())
    fig = plt.figure(figsize=(1.6 * len(condition_order), 2.4 * len(model_names)), layout='constrained')
    # One sub-figure per model, titled with the name of the model, holding its row of panels.
    model_subfigures = fig.subfigures(len(model_names), 1)
    for row, (model_subfigure, model_name) in enumerate(zip(model_subfigures, model_names)):
        model_subfigure.suptitle(model_name, fontsize=9)
        axes = model_subfigure.subplots(1, len(condition_order))
        for column, (ax, condition) in enumerate(zip(axes, condition_order)):
            is_model_and_condition = ((squared_correlation_difference_df['model'] == model_name)
                                      & (squared_correlation_difference_df['condition'] == condition))
            condition_df = squared_correlation_difference_df[is_model_and_condition]
            sns.scatterplot(data=condition_df, x='r_original', y='r_corrected', color='black', s=6, ax=ax)

            # Same limits on both axes, so that the identity line is the diagonal of the panel.
            r_min = min(condition_df['r_original'].min(), condition_df['r_corrected'].min())
            r_max = max(condition_df['r_original'].max(), condition_df['r_corrected'].max())
            ax.plot([r_min, r_max], [r_min, r_max], color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlim(r_min, r_max)
            ax.set_ylim(r_min, r_max)
            ax.tick_params(labelsize=6)

            ax.set_xlabel(r'$r_{original}$' if row == len(model_names) - 1 else '')
            ax.set_ylabel(r'$r_{corrected}$' if column == 0 else '')
            ax.set_title(condition, fontsize=6)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_squared_correlation_by_condition_all_models(squared_correlation_difference_df, condition_order, figure_path):
    """Plot one panel per model, stacked vertically, with paired bars of the mean of r_original**2 and
    r_corrected**2 for each condition, with 95% confidence intervals over the sequences."""
    squared_correlation_df = squared_correlation_difference_df.melt(
        id_vars=['model', 'sequence', 'condition'], value_vars=['r_original', 'r_corrected'], var_name='batch',
        value_name='correlation')
    squared_correlation_df['squared_correlation'] = squared_correlation_df['correlation'] ** 2
    squared_correlation_df['batch'] = squared_correlation_df['batch'].str.removeprefix('r_')

    model_names = sorted(squared_correlation_difference_df['model'].unique())
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 3.5 * len(model_names)), sharex=True,
                             squeeze=False)
    for ax, model_name in zip(axes[:, 0], model_names):
        model_df = squared_correlation_df[squared_correlation_df['model'] == model_name]
        sns.barplot(data=model_df, x='condition', y='squared_correlation', hue='batch', order=condition_order,
                    hue_order=['original', 'corrected'], ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(r'$r^2$')
        ax.set_title(model_name)
        ax.get_legend().remove()
    # One legend for all the panels, outside of the first panel so that it does not hide the bars.
    axes[0, 0].legend(*axes[0, 0].get_legend_handles_labels(), title='batch', loc='upper left',
                      bbox_to_anchor=(1.01, 1))
    last_ax = axes[-1, 0]
    plt.setp(last_ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    last_ax.set_xlabel('condition')
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":

    OUTPUT_ROOT = PREPROPARADIGM_DIR / 'predictions_gm_test'
    CORRELATION_DIR = OUTPUT_ROOT / 'correlation'
    ORIGINAL_BATCH_NAME = 'gm_original'
    CORRECTED_BATCH_NAME = 'gm_corrected'

    squared_correlation_difference_df = get_squared_correlation_difference_df(CORRELATION_DIR, ORIGINAL_BATCH_NAME,
                                                              CORRECTED_BATCH_NAME)

    # Conditions in the order of their number (cond1, cond2, ..., cond16), as in the correlation CSVs.
    condition_order = list(squared_correlation_difference_df['condition'].unique())

    p_values_by_model = {}
    for model_name, model_squared_correlation_difference_df in squared_correlation_difference_df.groupby('model'):

        model_figure_dir = CORRELATION_DIR / model_name
        model_figure_dir.mkdir(exist_ok=True)

        p_values = get_squared_correlation_difference_p_values(model_squared_correlation_difference_df, condition_order)
        print(f"\n{model_name}, p-value of the t-test of r_original**2 - r_corrected**2 against 0")
        print(p_values.to_string(float_format=lambda value: f'{value:.4f}'))
        p_values_by_model[model_name] = p_values

        figure_path = model_figure_dir / 'squared_correlation_difference_by_condition.png'
        plot_squared_correlation_difference_by_condition(model_squared_correlation_difference_df, condition_order, p_values,
                                                 model_name, figure_path)
        print(f"Figure saved to {figure_path}")

    figure_path = CORRELATION_DIR / 'squared_correlation_difference_by_condition_all_models.png'
    plot_squared_correlation_difference_by_condition_all_models(squared_correlation_difference_df, condition_order, p_values_by_model,
                                                        figure_path)
    print(f"Figure saved to {figure_path}")

    figure_path = CORRELATION_DIR / 'correlation_original_vs_corrected_all_models.png'
    plot_correlation_original_vs_corrected_all_models(squared_correlation_difference_df, condition_order, figure_path)
    print(f"Figure saved to {figure_path}")

    figure_path = CORRELATION_DIR / 'squared_correlation_by_condition_all_models.png'
    plot_squared_correlation_by_condition_all_models(squared_correlation_difference_df, condition_order, figure_path)
    print(f"Figure saved to {figure_path}")
