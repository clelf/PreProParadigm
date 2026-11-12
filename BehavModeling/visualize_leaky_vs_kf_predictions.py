"""
Plot Kalman-filter predictions, optionally overlaid with the leaky integrator, per
subject/session/run. Covers subjects 04/05/06, sessions 1-4, all runs.

The `MODELS` flag below chooses whether the leaky integrator is included: 'KF' plots the
Kalman filter alone, while 'KF_LI' additionally overlays the leaky integrator whenever its
prediction file exists (a missing leaky file is ignored and the figure falls back to KF-only).
Orthogonally, the `KF_VARIANT` flag chooses which KF(s) are drawn: the true-sigma_r fit, the
EM-fitted fit, or 'both' overlaid together in different colours.

This is a copy of `visualize_leaky_predictions.py` that overlays a second model:
  - LEAKY integrator  -> black  (the "model" channel of plot_sample: mu_estim/sigma_estim)
  - KALMAN filter     -> purple (the "kalman" overlay channel: kalman_mu/kalman_sigma)

Both are the STANDARD-context prediction (`mu_pred_std` +/- `sigma_pred_std`). The leaky
predictions come from `PreProParadigm/leaky_predictions/`; the KF predictions come from
`BehavModeling/kalman_pred/` (written by `compute_kf_predictions.py`).

The two bands are only comparable because of two fixes; before them this figure was
misleading in the KF's disfavour, which is what motivated the rewrite:

  1. The KF used to be run with an observation noise it was never given. R was neither
     supplied nor EM-estimated and silently sat at pykalman's default of 1.0, against a true
     R = sigma_r**2 ~ 1e-4. That drove the Kalman gain to ~0, so the filter over-smoothed
     (mean would not track the observations) and its band could never be narrower than
     sqrt(R). Both variants we can read now actually set R; see `KF_VARIANT` below to pick
     one. 'true_sigma_r' fixes R at each session's TRUE sigma_r, which is also what the leaky
     integrator is handed via its `s_floor` -- so neither model gets an information advantage
     over the other, and that is the like-for-like figure. 'em_fitted' instead lets the KF
     estimate R itself; it is the fairer test of the KF as a standalone model but not a fair
     leaky-vs-KF comparison, since only one model is then told the true noise. Note its band
     is currently much too wide because EM (n_iter=5, from pykalman's default R=1.0) does not
     converge, NOT because the KF cannot represent the process -- see compute_kf_predictions.py.
     'both' overlays the two KF bands on one figure (true-sigma_r vs EM-fitted, distinct colours)
     for a side-by-side read.
  2. The KF's `sigma_*` columns used to hold variances while the leaky's held STANDARD
     DEVIATIONS, and plot_sample draws mu +/- sigma for both. The two bands were in different
     units. Both are standard deviations now.

The old `PreProParadigm/kalman_predictions_new/` files are from March, predate both fixes, and
use a different schema (`per_ctx_mu_pred_std`); pointing this script back at them would compare
a March KF against a July leaky integrator.

Plotting is done by `plot_obs_and_models` below rather than by pipeline_core_v2's
`plot_sample`. It keeps only that function's top panel (observations + a caller-supplied list
of model overlays) and drops the context panel, so the legend can name each model instead of
using plot_sample's generic 'model'/'KF' entries.

Figures are saved into a per-variant `exp_runs_viz_likelihoods*` folder next to this script
(see `KF_VARIANT_FOLDERS`).
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

# Reuse the workspace's data prep.
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace')
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace/PreProParadigm')

from model_RTs import prepare_trials_data

# Index of each process (context) along the N_ctx axes of HAudIn's inference outputs
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavExpDataGen')
from kalman_full import STD_CONTEXT, DEV_CONTEXT

# Where the leaky-integrator inputs (triallists) and both models' predictions live.
PREPRO_DIR = '/home/clevyfidel/Documents/Workspace/PreProParadigm'
LEAKY_PREDICTIONS_DIR = os.path.join(PREPRO_DIR, 'leaky_predictions')
KF_PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), 'kalman_pred')

# Which models to overlay.
#   'KF'    -> Kalman filter alone. Leaky prediction files are NEVER read, even if they exist.
#   'KF_LI' -> also overlay the leaky integrator, but only WHEN its prediction file is present;
#              a missing leaky file is silently ignored and the figure falls back to KF-only.
# So anything leaky-related (band, legend entry, likelihood bars) appears iff MODELS == 'KF_LI'
# and the leaky file for that session exists.
MODELS = 'KF'
MODELS_CHOICES = ('KF', 'KF_LI')

# Colours of the model overlays -- THE single place to change or swap them. The caller pairs
# each colour with its data when it assembles the `models` list handed to plot_obs_and_models,
# so changing a value here just recolours that model's line + band:
#   LI     = leaky integrator (black)
#   KF     = Kalman filter with true sigma_r, and the EM-fitted KF when it is shown on its own
#   KF_EM  = EM-fitted KF, but only when shown ALONGSIDE the true-sigma_r one (KF_VARIANT='both'),
#            so the two KF bands stay distinguishable
LI_COLOR = 'black'
KF_COLOR = 'purple'
KF_EM_COLOR = 'darkorange'

# Which KF variant(s) to overlay. All share the leaky file's schema, so only the filename and
# the output folder change.
#   'true_sigma_r' -> R fixed at each session's true observation noise
#                     (kalman_predictions_sub-XX_ses-Y_sigma_r_Z.csv)
#   'em_fitted'    -> R estimated by the filter itself, alongside the other parameters
#                     (kalman_predictions_sub-XX_ses-Y.csv)
#   'both'         -> BOTH of the above on one figure (true-sigma_r in KF_COLOR, EM-fitted in
#                     KF_EM_COLOR), on top of the LI when MODELS == 'KF_LI'. Both KF files must
#                     exist or the session is skipped.
# TODO: modify here to switch variants.
KF_VARIANT = 'true_sigma_r'

KF_VARIANT_FOLDERS = {
    'true_sigma_r': 'exp_runs_viz_likelihoods',
    'em_fitted': 'exp_runs_viz_likelihoods_sirfitted',
    'both': 'exp_runs_viz_likelihoods_bothkf',
}

# Optional twin-axis overlay quantifying how surprising each DEVIANT observation is under
# each model's standard-context predictive Gaussian. See plot_obs_and_models' `show_lik`.
#   'show_lik'    -> Gaussian likelihood      p(y | mu_pred_std, sigma_pred_std)
#   'show_loglik' -> Gaussian log-likelihood  log p(y | mu_pred_std, sigma_pred_std)
#   None          -> no twin axis (figure unchanged)
SHOW_LIK = 'show_lik'

# When True, title each figure with the run's generative parameters -- d, sigma_r, tau_std,
# tau_dev, sigma_stat, sigma_q_std, sigma_q_dev. All but sigma_stat are triallist columns
# (tau_* and sigma_q_* vary per run, d/sigma_r are session-constant); sigma_stat is a
# per-session parameter read from the triallist filename (…-si_statZ-…). False leaves the
# axes untitled, as before.
SHOW_PARAMS = True


# Colours and heights of the optional panels showing the hierarchical inference of HAudIn (kalman_full.py):
#   CONTEXT_COLORS               = std / dev (lines, error bars, background of the true context)
#   DPOS_COLORS                  = one per possible deviant position, in the order of HAudIn's dpos_set
#   RULE_COLORS                  = rule 0 / rule 1
#   PREDICTED_DISTRIBUTION_COLOR = density of the predicted distribution of y_t
#   PREDICTED_MEAN_COLOR         = mean of the predicted distribution of y_t
CONTEXT_COLORS = {STD_CONTEXT: 'tab:blue', DEV_CONTEXT: 'tab:red'}
DPOS_COLORS = ['tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
RULE_COLORS = {0: 'tab:purple', 1: 'tab:orange'}
PREDICTED_DISTRIBUTION_COLOR = 'tab:purple'
PREDICTED_MEAN_COLOR = 'tab:cyan'

# Height of each panel of plot_obs_and_models, in inches (multiplied by its panel_height_scale argument), and gaps
# between panels, in inches: below the observation panel (which carries the x tick labels), and between the others
PANEL_HEIGHTS = {
    'observations': 2.3,
    'likelihood': 0.77,
    'context_probabilities': 0.8,
    'dpos_probabilities': 0.8,
    'rule_probabilities': 0.8,
    'context_predictions': 1.5,
    'predicted_distribution': 1.5,
    'tau': 0.8,
    'b': 0.8,
    'process_noise': 0.8,
}
GAP_BELOW_FIRST_PANEL = 0.6
GAP_BETWEEN_PANELS = 0.2


def plot_true_category_background(ax, true_categories, category_colors):
    """Colour the background of ax, at each timestep, with the colour of the true category of that timestep, as
    audit_gm's plot_combined_with_matrix colours each trial with its rule. Consecutive timesteps of the same category
    share one span; a category without a colour in category_colors (e.g. dpos 0, an omission) leaves a white background.

    Args:
        ax: axes to colour
        true_categories: true category of each timestep, shape (T,)
        category_colors: dict {category value: colour}
    """
    true_categories = np.asarray(true_categories)
    category_changes = np.flatnonzero(np.diff(true_categories)) + 1
    span_starts = np.concatenate([[0], category_changes])
    span_ends = np.concatenate([category_changes, [len(true_categories)]])
    for span_start, span_end in zip(span_starts, span_ends):
        category = true_categories[span_start]
        if category in category_colors:
            ax.axvspan(span_start - 0.5, span_end - 0.5, facecolor=category_colors[category], alpha=0.25, linewidth=0)


def plot_category_probabilities(ax, probabilities, category_values, category_labels, category_colors,
                                true_categories, ylabel):
    """Plot the estimated probability of each category of one categorical variable (context, deviant position or rule)
    at each timestep, one line per category, over a background coloured with the true category of each timestep.

    Args:
        ax: axes to plot on
        probabilities: estimated probability of each category, shape (T, number of categories), columns in the order
            of category_values
        category_values: value of each category (e.g. STD_CONTEXT / DEV_CONTEXT, the deviant positions, the rules)
        category_labels: legend entry of each category, in the order of category_values
        category_colors: dict {category value: colour}, used for the lines and the background
        true_categories: true category of each timestep, shape (T,)
        ylabel: label of the y axis
    """
    plot_true_category_background(ax, true_categories, category_colors)
    x = np.arange(probabilities.shape[0])
    for i_category, category in enumerate(category_values):
        ax.plot(x, probabilities[:, i_category], color=category_colors[category], linewidth=1,
                label=category_labels[i_category])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)


def plot_context_predictions(ax, context_means, context_variances, context_probabilities, y_limits):
    """Plot the predicted distribution of y_t by each process (context), given that the tone comes from it,
    p(y_t | c_t = c, H_t) = N(y; mean_c, variance_c), as a background in the colour of the process (blue: std, red: dev).
    The opacity at each point is p(c_t = c | H_t) x density(y) / maximum of the density at that timestep (its value at
    the mean, 1 / sqrt(2 pi variance_c)). The two quantities are therefore read separately: the opacity at the mean is
    the probability of the context, and the vertical fading follows the shape (spread) of the density. NaN predictions
    (process that cannot be heard at that timestep) are not plotted. The dev density is drawn over the std one.

    Args:
        ax: axes to plot on
        context_means: mean of the prediction of y_t by each process, shape (T, N_ctx)
        context_variances: variance of the same prediction, shape (T, N_ctx)
        context_probabilities: estimated probability of each context, p(c_t = c | H_t), shape (T, N_ctx)
        y_limits: (lowest, highest) y value over which the densities are computed and plotted
    """
    n_timesteps = context_means.shape[0]
    y_grid = np.linspace(y_limits[0], y_limits[1], 400)
    legend_handles = []
    for i_ctx, context_label in [(STD_CONTEXT, 'std'), (DEV_CONTEXT, 'dev')]:
        is_predicted = ~np.isnan(context_means[:, i_ctx])
        means = context_means[is_predicted, i_ctx]
        variances = context_variances[is_predicted, i_ctx]
        # Density divided by its maximum (its value at the mean): exp(-(y - mean)^2 / (2 variance)), between 0 and 1;
        # computed from the mean rather than from the plotted y range, so that a mean outside the panel stays at its
        # true scale. Timesteps without prediction stay 0
        scaled_density = np.zeros((len(y_grid), n_timesteps))
        scaled_density[:, is_predicted] = np.exp(-0.5 * (y_grid[:, np.newaxis] - means) ** 2 / variances)
        density_image = np.zeros((len(y_grid), n_timesteps, 4))
        density_image[:, :, :3] = mcolors.to_rgb(CONTEXT_COLORS[i_ctx])
        density_image[:, :, 3] = context_probabilities[:, i_ctx] * scaled_density
        ax.imshow(density_image, origin='lower', aspect='auto', interpolation='nearest',
                  extent=(-0.5, n_timesteps - 0.5, y_limits[0], y_limits[1]))
        legend_handles.append(Patch(facecolor=CONTEXT_COLORS[i_ctx], label=f'{context_label} density'))
    ax.set_ylim(y_limits)
    ax.set_ylabel(r'$p(y_t = y \mid c_t = c, H_t)$', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)


def plot_predicted_distribution(ax, predicted_mean, context_means, context_variances, context_probabilities,
                                y_limits):
    """Plot the predicted distribution of y_t at each timestep as a purple background, with its mean as a cyan line.

    The predicted distribution is the mixture of the processes' Gaussian predictions (plot_context_predictions),
    weighted by the estimated probability of each context: sum_c p(c_t = c | H_t) N(y; mean_c, variance_c). A process
    whose prediction is NaN at t has probability 0 at t and is left out. The purple opacity is the density, linearly
    scaled at each timestep between 0 and the maximum of that timestep's density on the plotted y range.

    Args:
        ax: axes to plot on
        predicted_mean: mean of the predicted distribution of y_t, shape (T,)
        context_means, context_variances: as in plot_context_predictions, shape (T, N_ctx)
        context_probabilities: estimated probability of each context, p(c_t = c | H_t), shape (T, N_ctx)
        y_limits: (lowest, highest) y value over which the density is computed and plotted
    """
    n_timesteps = context_means.shape[0]
    y_grid = np.linspace(y_limits[0], y_limits[1], 400)
    predicted_density = np.zeros((len(y_grid), n_timesteps))
    for i_ctx in [STD_CONTEXT, DEV_CONTEXT]:
        is_predicted = ~np.isnan(context_means[:, i_ctx])
        means = context_means[is_predicted, i_ctx]
        variances = context_variances[is_predicted, i_ctx]
        context_density = np.exp(-0.5 * (y_grid[:, np.newaxis] - means) ** 2 / variances) / np.sqrt(2 * np.pi * variances)
        predicted_density[:, is_predicted] += context_probabilities[is_predicted, i_ctx] * context_density

    # Linear scaling between 0 and the maximum of each timestep's density; a density that is 0 everywhere on the plotted
    # y range (all its mass outside of it) stays 0
    density_maxima = np.max(predicted_density, axis=0)
    scaled_density = np.divide(predicted_density, density_maxima, out=np.zeros_like(predicted_density),
                               where=density_maxima > 0)
    density_image = np.zeros((len(y_grid), n_timesteps, 4))
    density_image[:, :, :3] = mcolors.to_rgb(PREDICTED_DISTRIBUTION_COLOR)
    density_image[:, :, 3] = scaled_density
    ax.imshow(density_image, origin='lower', aspect='auto', interpolation='nearest',
              extent=(-0.5, n_timesteps - 0.5, y_limits[0], y_limits[1]))

    ax.plot(np.arange(n_timesteps), predicted_mean, color=PREDICTED_MEAN_COLOR, linewidth=1, label='predicted mean')
    ax.set_ylim(y_limits)
    ax.set_ylabel(r'$p(y_t = y \mid H_t)$', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_estimate_use_start_timesteps(inference_model, inference_states):
    """Timestep after which the estimates of each process are computed from the data rather than taken from their
    start values: the first tone after whose update the number of transitions of the process in the running sums
    reaches n_transitions_before_using_estimates, under every hypothesis m about the deviant position. The predictions
    of the process are made with its estimates from the next tone on. The dev process only moves at the deviant
    position of hypothesis m, so it reaches the threshold at the tone of the latest such position in that trial.

    Args:
        inference_model: HAudIn instance that produced inference_states
        inference_states: output of inference_model.inference()

    Returns:
        dict {STD_CONTEXT: timestep, DEV_CONTEXT: timestep}; None for a process that never reaches the threshold
    """
    thresholds = np.broadcast_to(inference_model.n_transitions_before_using_estimates, (inference_model.N_ctx,))
    # (N_tones, N_trials, N_ctx, N_dpos) -> (N_trials * N_tones, N_ctx, N_dpos), flat tone order t = N_tones * k + j
    n_transitions = inference_states['n_transitions_ctx_trials_m'].transpose(1, 0, 2, 3).reshape(
        -1, inference_model.N_ctx, inference_model.N_dpos)
    estimate_use_start_timesteps = {}
    for i_ctx in [STD_CONTEXT, DEV_CONTEXT]:
        threshold_reached = np.all(n_transitions[:, i_ctx, :] >= thresholds[i_ctx], axis=1)
        if np.any(threshold_reached):
            estimate_use_start_timesteps[i_ctx] = np.argmax(threshold_reached)
        else:
            estimate_use_start_timesteps[i_ctx] = None
    return estimate_use_start_timesteps


def plot_parameter_estimates(ax, inference_model, inference_states, parameter_name, true_values,
                             comparison_estimates, ylabel, show_legend):
    """Plot the estimates of one parameter (tau, b or process_noise) of each process, after the merge over the
    hypotheses m at the end of each trial (inference_states['kf_states_end_trials']): one value per trial, placed at
    the last tone of the trial so that the x axis stays the timestep axis of the other panels. The true value of the
    parameter of each process is a dashed horizontal line of the same colour.

    Args:
        ax: axes to plot on
        inference_model: HAudIn instance that produced inference_states
        inference_states: output of inference_model.inference()
        parameter_name: key of the parameter in each merged predictor: 'tau', 'b' or 'process_noise' (a variance)
        true_values: true value of the parameter of each process, shape (N_ctx,), index STD_CONTEXT / DEV_CONTEXT
        comparison_estimates: list of (estimates, color, label) of the same parameter by other models (e.g. the old KF),
            estimates of shape (T,), one per timestep; plotted at the last tone of each trial, like the HAudIn estimates.
            Empty list: none
        ylabel: label of the y axis
        show_legend: whether to add the std / dev legend
    """
    estimates = np.array([kf_state[parameter_name] for kf_state in inference_states['kf_states_end_trials']])
    trial_end_timesteps = np.arange(inference_model.N_trials) * inference_model.N_tones + inference_model.N_tones - 1
    for i_ctx, context_label in [(STD_CONTEXT, 'std'), (DEV_CONTEXT, 'dev')]:
        ax.plot(trial_end_timesteps, estimates[:, i_ctx], color=CONTEXT_COLORS[i_ctx], linewidth=1,
                marker='o', markersize=2, label=context_label)
        ax.axhline(true_values[i_ctx], color=CONTEXT_COLORS[i_ctx], linestyle='--', linewidth=1, alpha=0.6,
                   label=f'{context_label} true value')
    for estimates_per_timestep, color, label in comparison_estimates:
        ax.plot(trial_end_timesteps, np.asarray(estimates_per_timestep, dtype=float)[trial_end_timesteps], color=color,
                linewidth=1, marker='o', markersize=2, label=label)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)


def create_stacked_panels(panel_names, fig_width, panel_height_scale):
    """Create a figure with one panel per name, stacked vertically and sharing the x axis of the first one. Each panel
    has its own height in inches (PANEL_HEIGHTS, times panel_height_scale), so that adding a panel extends the figure
    instead of compressing the others. The gaps between panels are fixed in inches (they hold text): a larger one below
    the first panel, which carries the x tick labels, a small one elsewhere.

    Args:
        panel_names: names of the panels, from top to bottom; keys of PANEL_HEIGHTS
        fig_width: width of the figure, in inches
        panel_height_scale: factor applied to the height of every panel

    Returns:
        fig, dict {panel name: axes}
    """
    # Grid rows: the panels, with an empty row as gap between two consecutive panels (hspace = 0 between rows)
    row_heights = []
    for i_panel, panel_name in enumerate(panel_names):
        if i_panel == 1:
            row_heights.append(GAP_BELOW_FIRST_PANEL)
        elif i_panel > 1:
            row_heights.append(GAP_BETWEEN_PANELS)
        row_heights.append(PANEL_HEIGHTS[panel_name] * panel_height_scale)

    # The panels fill the whole figure (top=1, bottom=0): the labels and titles outside of it are kept by the
    # bbox_inches='tight' saving
    fig = plt.figure(figsize=(fig_width, np.sum(row_heights)))
    grid = fig.add_gridspec(len(row_heights), 1, height_ratios=row_heights, hspace=0, top=1, bottom=0)
    axes_by_panel = {panel_names[0]: fig.add_subplot(grid[0])}
    for i_panel, panel_name in enumerate(panel_names[1:], start=1):
        axes_by_panel[panel_name] = fig.add_subplot(grid[2 * i_panel], sharex=axes_by_panel[panel_names[0]])
    return fig, axes_by_panel


def plot_obs_and_models(obs, contexts, models, save_path, title=None, show_lik=None,
                        fig_width=10, panel_height_scale=1,
                        inference_model=None, inference_states=None, true_dpos=None, true_rules=None,
                        true_parameters=None, comparison_parameter_estimates=None,
                        show_category_probabilities=False, show_predicted_distributions=False,
                        show_estimate_use_start=False, show_parameter_estimates=False):
    """Plot one run: the observation sequence with one or more model predictions +/- 1 SD.

    This is pipeline_core_v2.plot_sample's top panel only -- same marks (obs in blue with
    deviant trials in red, each model a coloured line over a matching +/-1 SD band) -- minus
    the context panel, and with a legend that names each model instead of plot_sample's generic
    'model'/'KF' entries. Each model gets a single legend entry whose swatch shows its line over
    its band, via HandlerTuple.

    `models` is a list of (mu, sigma, color, label) overlays drawn in the given order (the first
    underneath). The caller decides what goes in it -- the leaky integrator, one KF variant, or
    both KF variants -- so this function is agnostic to which models it is showing and needs no
    per-model keyword arguments. Predictions are assumed aligned with the observations (one per
    trial, which is how the prediction files are written), so every series shares the same x.

    Optional panels below the main one show the hierarchical inference of a HAudIn instance
    (kalman_full.py); they need inference_model and inference_states, and all share the x axis
    (timesteps) of the main panel. Each panel has its own height, so adding panels extends the
    figure downwards instead of compressing the others.

    Args:
        obs: observation sequence for this run, shape (T,)
        contexts: context id per trial, shape (T,); nonzero = deviant, marked in red
        models: list of (mu, sigma, color, label); mu/sigma are that model's mean and standard
            deviation, shape (T,), color/label its plotting colour and legend entry. Drawn in
            order (first underneath); its length also sets the grouped-bar layout below.
        save_path: full path of the .png to write
        title: axes title
        show_lik: if 'show_lik' or 'show_loglik', add a shorter panel below the main one
            (shared trial axis, a third its height) with, per DEVIANT observation, one bar per
            model showing how (log-)likely it is under that model's standard-context predictive
            Gaussian N(mu_pred_std, sigma_pred_std). None (the default) leaves the figure as a
            single panel.
        fig_width: width of the figure, in inches
        panel_height_scale: factor applied to the height of every panel (PANEL_HEIGHTS)
        inference_model: HAudIn instance (kalman_full.py) that produced inference_states; needed by
            the four options below
        inference_states: output of inference_model.inference()
        true_dpos: true deviant position of each timestep (constant within a trial), shape (T,);
            needed by show_category_probabilities
        true_rules: true rule of each timestep (constant within a trial), shape (T,); needed by
            show_category_probabilities
        true_parameters: dict {'tau', 'b', 'process_noise'} of true parameter values, one per process, shape
            (N_ctx,), index STD_CONTEXT / DEV_CONTEXT; process_noise is a variance (sigma_q**2); needed by
            show_parameter_estimates
        comparison_parameter_estimates: dict {parameter name ('tau', 'b' or 'process_noise'): list of
            (estimates, color, label)} of the estimates of other models (e.g. the old KF), estimates of shape
            (T,), one per timestep; a parameter left out of the dict gets no comparison line. None: no
            comparison lines. Used by show_parameter_estimates
        show_category_probabilities: add one panel per categorical variable -- context, deviant
            position, rule -- with the estimated probability of each category at each timestep,
            over a background coloured with the true category
        show_predicted_distributions: add two panels: the predicted distribution of y_t by each
            context (opacity: probability of the context x density scaled to its maximum), and the predicted
            distribution of y_t (mixture of the contexts' predictions) with its mean
        show_estimate_use_start: add one vertical line per process, on every panel, after the
            timestep at which its estimates start being used (n_transitions_before_using_estimates)
        show_parameter_estimates: add one panel per parameter (tau, b, process noise variance) with
            the estimates of each process after the merge at the end of each trial
    """
    obs = np.asarray(obs, dtype=float)
    contexts = np.asarray(contexts)
    models = [(np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float), color, label)
              for mu, sigma, color, label in models]

    # Small (nonzero) save pad: we save with bbox_inches='tight' below, and pad=0 leaves no
    # margin around the tight bbox, so 'tight' shaves the top spine / edge markers -- that is
    # the top-border crop. A tenth of an inch of margin removes it.
    mpl.rcParams['savefig.pad_inches'] = 0.1
    show_lik_on = show_lik in ('show_lik', 'show_loglik')

    # Panels, from top to bottom
    panel_names = ['observations']
    if show_lik_on:
        panel_names.append('likelihood')
    if show_category_probabilities:
        panel_names += ['context_probabilities', 'dpos_probabilities', 'rule_probabilities']
    if show_predicted_distributions:
        panel_names += ['context_predictions', 'predicted_distribution']
    if show_parameter_estimates:
        panel_names += ['tau', 'b', 'process_noise']
    fig, axes_by_panel = create_stacked_panels(panel_names, fig_width, panel_height_scale)
    inference_panel_names = [name for name in panel_names if name not in ('observations', 'likelihood')]
    ax = axes_by_panel['observations']
    if show_lik_on:
        ax_lik = axes_by_panel['likelihood']

    x = np.arange(len(obs))
    obs_line, = ax.plot(x, obs, color='tab:blue', alpha=0.8,
                        linestyle='-', markersize=5) # marker='o'
    handles = [obs_line]
    labels = [r'$y_{std}$']

    # Same marker over the deviant trials, in red, so context switches stay readable
    # without the separate context panel.
    deviant = np.where(contexts)[0]
    if len(deviant) > 0:
        dev_pts, = ax.plot(x[deviant], obs[deviant], color='tab:red',
                           linestyle='None', marker='o', markersize=5)
        handles.append(dev_pts)
        labels.append(r'$y_{dev}$')

    # Draw every overlay the caller assembled, in order: the first is underneath (the LI, when
    # present), later ones on top (the KF variant(s)). Everything downstream -- bands, legend,
    # likelihood bars -- iterates this same list, so the figure shows exactly these models.
    for mu, sigma, color, label in models:
        mx = np.arange(len(mu))
        line, = ax.plot(mx, mu, color=color, alpha=0.8,
                        linestyle='-', markersize=5, markerfacecolor='None')
        band = ax.fill_between(mx, mu - sigma, mu + sigma, color=color, alpha=0.2)
        handles.append((band, line))  # one entry per model: line drawn over its band
        labels.append(f'{label}')

    # Start of the use of each process's estimates: one vertical line per process, after the timestep at which its
    # number of transitions reaches n_transitions_before_using_estimates; drawn on every panel, keyed in this legend
    if show_estimate_use_start:
        estimate_use_start_timesteps = get_estimate_use_start_timesteps(inference_model, inference_states)
        thresholds = np.broadcast_to(inference_model.n_transitions_before_using_estimates, (inference_model.N_ctx,))
        for i_ctx, context_label in [(STD_CONTEXT, 'std'), (DEV_CONTEXT, 'dev')]:
            if estimate_use_start_timesteps[i_ctx] is not None:
                for panel_ax in axes_by_panel.values():
                    panel_ax.axvline(estimate_use_start_timesteps[i_ctx] + 0.5, color=CONTEXT_COLORS[i_ctx],
                                     linestyle=':', linewidth=1.2)
                handles.append(plt.Line2D([0], [0], color=CONTEXT_COLORS[i_ctx], linestyle=':', linewidth=1.2))
                labels.append(context_label + r' $\hat{\theta}$ start')

    ax.margins(x=0)
    ax.set_ylabel('observation process')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if title is not None:
        # Small font: the parameter title is long, and this keeps it on one line within the
        # 10-inch figure width instead of overflowing (which bbox='tight' would widen to fit).
        ax.set_title(title, fontsize=8)

    # Legend inside the top panel (upper-right, usually clear of the near-zero observations and
    # sparse deviant spikes). frameon=False makes it frameless: no border and a transparent
    # background, so whatever sits behind it shows through.
    # With the optional inference panels, the legend goes outside on the right, like the legends of those panels
    # (it also holds more entries then, which would cover the data inside the panel).
    if len(inference_panel_names) > 0 or show_estimate_use_start:
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7,
                  handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        ax.legend(handles, labels, loc='best', frameon=False,
                  handler_map={tuple: HandlerTuple(ndivide=None)})

    # Optional bottom panel: how surprising each deviant observation is under each model's
    # STANDARD-context predictive Gaussian N(mu_pred_std, sigma_pred_std). A deviant trial is
    # drawn from the OTHER context, so a well-calibrated standard model should give it a low
    # likelihood; the bars put that surprise on a trial-aligned scale below the bands. Only
    # deviant trials get a bar (a std-obs likelihood under the std model is uninteresting).
    if show_lik_on:
        as_log = show_lik == 'show_loglik'
        # One bar per model per deviant trial, seaborn-"hue" style: the n bars are centred as a
        # group on the trial and touch. Each bar's width is set PER group from the local spacing
        # to the neighbouring deviants, so a single tight group no longer thins every bar --
        # well-separated deviants get thick bars -- and the group is capped so a lone deviant's
        # bars stay legible. No bottom-panel legend on purpose: the top panel's legend already
        # keys the colours.
        n = len(models)
        if len(deviant) > 0:
            d = deviant.astype(float)
            left = np.diff(d, prepend=d[0] - 8)   # spacing to previous deviant (edge: assume 8)
            right = np.diff(d, append=d[-1] + 8)  # spacing to next deviant
            bw = np.minimum(2.0, (0.9 / n) * np.minimum(left, right))  # per-bar width; group spans n*bw
            for i, (mu, sigma, color, label) in enumerate(models):
                offset = (i - (n - 1) / 2) * bw  # centre the group of n bars on the trial
                yd, md, sd = obs[deviant], mu[deviant], sigma[deviant]
                # Gaussian log-likelihood of each deviant obs under this model's std prediction.
                loglik = -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((yd - md) / sd) ** 2
                ax_lik.bar(x[deviant] + offset, loglik if as_log else np.exp(loglik),
                           width=bw, color=color, alpha=0.85)
        ax_lik.axhline(0, color='k', linewidth=0.6)
        # Compact label: the panel is only a third of the height, so the full word would run
        # past its top/bottom and get clipped.
        ax_lik.set_ylabel(f'{"log-" if as_log else ""}likelihood')
        ax_lik.spines['top'].set_visible(False)
        ax_lik.spines['right'].set_visible(False)

    # Optional panels: estimated probabilities of the categories of each categorical variable, over the true category
    if show_category_probabilities:
        true_contexts = np.where(contexts, DEV_CONTEXT, STD_CONTEXT)
        plot_category_probabilities(
            axes_by_panel['context_probabilities'], inference_states['hat_ck'], [STD_CONTEXT, DEV_CONTEXT],
            ['std', 'dev'], CONTEXT_COLORS, true_contexts, r'$p(c_t = c \mid H_t)$')
        # p(d_k = m | H_t): (N_tones, N_trials, N_dpos) -> (N_trials * N_tones, N_dpos), flat tone order t = N_tones * k + j
        dpos_probabilities = inference_states['hat_dk_trials'].transpose(1, 0, 2).reshape(-1, inference_model.N_dpos)
        dpos_colors = dict(zip(inference_model.dpos_set, DPOS_COLORS))
        plot_category_probabilities(
            axes_by_panel['dpos_probabilities'], dpos_probabilities, inference_model.dpos_set,
            [f'dpos {m}' for m in inference_model.dpos_set], dpos_colors, true_dpos, r'$p(d_k = m \mid H_t)$')
        rule_values = np.arange(inference_model.N_rules)
        plot_category_probabilities(
            axes_by_panel['rule_probabilities'], inference_states['hat_rk'], rule_values,
            [f'rule {r}' for r in rule_values], RULE_COLORS, true_rules, r'$p(r_k = r \mid H_t)$')

    # Optional pair of panels: prediction of y_t by each context, and their mixture, the predicted distribution of y_t;
    # same y range as the observation panel
    if show_predicted_distributions:
        observation_y_limits = ax.get_ylim()
        context_probabilities = inference_states['hat_ck']
        plot_context_predictions(
            axes_by_panel['context_predictions'], inference_states['hat_y_mean_ctx'],
            inference_states['hat_y_var_ctx'], context_probabilities, observation_y_limits)
        plot_predicted_distribution(
            axes_by_panel['predicted_distribution'], inference_states['hat_y_mean'],
            inference_states['hat_y_mean_ctx'], inference_states['hat_y_var_ctx'], context_probabilities,
            observation_y_limits)

    # Optional panels: estimates of each parameter after the merge at the end of each trial
    if show_parameter_estimates:
        if comparison_parameter_estimates is None:
            comparison_parameter_estimates = {}
        comparison_estimates_by_parameter = {
            parameter_name: (comparison_parameter_estimates[parameter_name]
                             if parameter_name in comparison_parameter_estimates else [])
            for parameter_name in ['tau', 'b', 'process_noise']}
        plot_parameter_estimates(axes_by_panel['tau'], inference_model, inference_states, 'tau', true_parameters['tau'],
                                 comparison_estimates_by_parameter['tau'],
                                 'tau estimate', show_legend=True)
        plot_parameter_estimates(axes_by_panel['b'], inference_model, inference_states, 'b', true_parameters['b'],
                                 comparison_estimates_by_parameter['b'],
                                 'b estimate', show_legend=False)
        plot_parameter_estimates(axes_by_panel['process_noise'], inference_model, inference_states, 'process_noise',
                                 true_parameters['process_noise'], comparison_estimates_by_parameter['process_noise'],
                                 r'$\sigma_q$ estimate', show_legend=False)

    # The trial axis lives on the top panel's bottom spine (its natural place). sharex propagates
    # the limits, so the panels below stay aligned while carrying no x tick labels of their own --
    # except the last of the optional inference panels, which repeats the trial axis at the bottom
    # of the figure. For the log-likelihood view (bars hanging from the 0 line) the likelihood panel
    # has no bottom spine either, which would sit far below the data and only clutter.
    ax.set_xlim(-0.5, len(obs) - 0.5)
    ax.set_xlabel('trial')
    if show_lik_on:
        ax_lik.tick_params(axis='x', bottom=False, labelbottom=False)
        if as_log:
            ax_lik.spines['bottom'].set_visible(False)
    for panel_name in inference_panel_names[:-1]:
        axes_by_panel[panel_name].tick_params(axis='x', labelbottom=False)
    if len(inference_panel_names) > 0:
        axes_by_panel[inference_panel_names[-1]].set_xlabel('trial')

    fig.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close(fig)


if __name__ == "__main__":

    # figure_example.py plots subject 04 only; we cover every subject the RT comparisons use,
    # since the KF predictions now exist for all of them.
    subs = ['04', '05', '06']
    # subs = ['11', '12', '13']

    n_sessions = 4

    if MODELS not in MODELS_CHOICES:
        raise ValueError(f"MODELS must be one of {MODELS_CHOICES}, got {MODELS!r}")

    if KF_VARIANT not in KF_VARIANT_FOLDERS:
        raise ValueError(f"KF_VARIANT must be one of {sorted(KF_VARIANT_FOLDERS)}, got {KF_VARIANT!r}")

    if SHOW_LIK not in (None, 'show_lik', 'show_loglik'):
        raise ValueError(f"SHOW_LIK must be None, 'show_lik' or 'show_loglik', got {SHOW_LIK!r}")

    if not isinstance(SHOW_PARAMS, bool):
        raise ValueError(f"SHOW_PARAMS must be True or False, got {SHOW_PARAMS!r}")

    results_save_folder = os.path.join(os.path.dirname(__file__), KF_VARIANT_FOLDERS[KF_VARIANT])
    os.makedirs(results_save_folder, exist_ok=True)

    # Tag filenames with the likelihood mode when it is on, so the twin-axis figures don't
    # overwrite the plain ones: 'show_lik' -> '_lik', 'show_loglik' -> '_loglik', None -> ''.
    lik_suffix = SHOW_LIK[len('show'):] if SHOW_LIK else ''

    for sub, sess in ((s, e) for s in subs for e in range(n_sessions)):

        # Load trials for this session (glob resolves the parameter-tagged filename).
        trials_path = glob.glob(
            os.path.join(PREPRO_DIR, f"triallists/sub-{sub}/sub-{sub}_ses-{sess+1}*.csv"))
        if not trials_path:
            print(f"No trials found for sub-{sub}, ses-{sess+1}")
            continue

        trials = prepare_trials_data(trials_path[0])

        # sigma_stat (the stationary std) is a per-session parameter encoded in the triallist
        # filename (…-si_stat0.1-…), not a column, so parse it once here for the title. None if
        # the pattern is absent, in which case the title shows sigma_stat=NA.
        si_stat_match = re.search(r'si_stat([0-9.]+)', os.path.basename(trials_path[0]))
        sigma_stat = float(si_stat_match.group(1)) if si_stat_match else None

        # Assemble the KF overlay(s) for this session as (filename, colour, label). 'both' returns
        # two -- the true-sigma_r KF and the EM-fitted KF -- so they can be read side by side on
        # one figure; the single-variant modes return one.
        #   true_sigma_r: R fixed at THIS session's true observation noise (0.005 or 0.01), which
        #     is also what the leaky integrator gets via its s_floor -- a like-for-like band.
        #   em_fitted:    the filter estimates its own R, so unlike the leaky model it is never
        #     handed the true sigma_r. Its band is much wider (~0.36 vs ~0.02) because EM
        #     overestimates R (~0.06 vs a true ~1e-4) -- an optimization artifact (n_iter=5 from
        #     pykalman's default R=1.0 does not converge, see compute_kf_predictions.py), NOT
        #     model misspecification.
        sigma_r = float(trials['sigma_r'].iloc[0])
        true_name = f'kalman_predictions_sub-{sub}_ses-{sess+1}_sigma_r_{sigma_r}.csv'
        em_name = f'kalman_predictions_sub-{sub}_ses-{sess+1}.csv'
        if KF_VARIANT == 'true_sigma_r':
            kf_variants = [(true_name, KF_COLOR, 'KF')]
        elif KF_VARIANT == 'em_fitted':
            kf_variants = [(em_name, KF_COLOR, 'KF with EM-fitted sigma_r')]
        else:  # 'both'
            kf_variants = [(true_name, KF_COLOR, 'KF (true sigma_r)'),
                           (em_name, KF_EM_COLOR, 'KF (EM-fitted sigma_r)')]

        # Every selected KF file is required; if any is missing there is nothing to plot for this
        # session. Load them into (df, colour, label) specs the run loop reads per run_id.
        kf_specs = []
        for kf_name, kf_color, kf_label in kf_variants:
            kf_file = os.path.join(KF_PREDICTIONS_DIR, kf_name)
            if not os.path.exists(kf_file):
                print(f"KF predictions not found: {kf_file}\n"
                      f"  Run: python3 BehavModeling/compute_kf_predictions.py")
                break
            kf_specs.append((pd.read_csv(kf_file, index_col=False), kf_color, kf_label))
        if len(kf_specs) != len(kf_variants):
            continue

        # The leaky prediction is optional: read it only in 'KF_LI' mode, and only if the file
        # is actually there. In 'KF' mode we never touch it (even if present); in 'KF_LI' mode a
        # missing file is not an error -- we just fall back to a KF-only figure for this session.
        leaky_df = None
        if MODELS == 'KF_LI':
            leaky_file = os.path.join(
                LEAKY_PREDICTIONS_DIR, f'leaky_predictions_sub-{sub}_ses-{sess+1}.csv')
            if os.path.exists(leaky_file):
                leaky_df = pd.read_csv(leaky_file, index_col=False)
            else:
                print(f"Leaky predictions not found, plotting KF only: {leaky_file}")

        # One figure per run.
        for run_id in trials['run_n'].unique():
            model = 'KF' if MODELS == 'KF' else 'KF_LI'
            save_path = os.path.join(
                results_save_folder,
                f'likelihoods_sub-{sub}_ses-{sess+1}_run_{run_id}{lik_suffix}_{model}.png')

            obs = trials.loc[trials['run_n'] == run_id, 'observation'].reset_index(drop=True)
            ctx = trials.loc[trials['run_n'] == run_id, 'contexts'].reset_index(drop=True)

            # Optional parameter title. tau_* and sigma_q_* vary per run, so read them from THIS
            # run's rows (they are constant within a run, hence .iloc[0]); d/sigma_r are session-
            # constant and sigma_stat came from the filename above. :g trims float noise so e.g.
            # 0.034800000000000005 prints as 0.0348.
            title = None
            if SHOW_PARAMS:
                p = trials.loc[trials['run_n'] == run_id].iloc[0]
                ss = f"{sigma_stat:g}" if sigma_stat is not None else "NA"
                title = (f"sub-{sub}, ses-{sess+1}, run {run_id} -- "
                         f"d={p['d']:g}, sigma_r={p['sigma_r']:g}, "
                         f"tau_std={p['tau_std']:g}, tau_dev={p['tau_dev']:g}, sigma_stat={ss}, "
                         f"sigma_q_std={p['sigma_q_std']:g}, sigma_q_dev={p['sigma_q_dev']:g}")

            # Assemble this run's overlays for plot_obs_and_models, drawn in order: the LI first
            # (black, underneath) when available, then the selected KF variant(s). All use the
            # standard-context prediction (mu_pred_std +/- sigma_pred_std), a standard deviation
            # in every file thanks to the shared schema.
            run_models = []
            if leaky_df is not None:
                sel = leaky_df['run_id'] == run_id
                run_models.append((leaky_df.loc[sel, 'mu_pred_std'].reset_index(drop=True),
                                   leaky_df.loc[sel, 'sigma_pred_std'].reset_index(drop=True),
                                   LI_COLOR, 'LI'))
            for kf_df, kf_color, kf_label in kf_specs:
                sel = kf_df['run_id'] == run_id
                run_models.append((kf_df.loc[sel, 'mu_pred_std'].reset_index(drop=True),
                                   kf_df.loc[sel, 'sigma_pred_std'].reset_index(drop=True),
                                   kf_color, kf_label))

            plot_obs_and_models(
                obs=obs, contexts=ctx, models=run_models, save_path=save_path,
                title=title,
                show_lik=SHOW_LIK,
            )
