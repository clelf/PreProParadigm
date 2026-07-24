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
  2. The KF's `sigma_*` columns used to hold VARIANCES while the leaky's held STANDARD
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
from matplotlib.legend_handler import HandlerTuple

# Reuse the workspace's data prep.
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace')
sys.path.insert(0, '/home/clevyfidel/Documents/Workspace/PreProParadigm')

from model_RTs import prepare_trials_data

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


def plot_obs_and_models(obs, contexts, models, save_path, title=None, show_lik=None):
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
    if show_lik_on:
        # Likelihoods go in a shorter panel below the main one, sharing the trial axis;
        # height_ratios [3, 1] makes it a third of the top panel's height. hspace leaves room
        # under the top panel for its trial axis (ticks + label), which sits between the panels.
        fig, (ax, ax_lik) = plt.subplots(
            2, 1, figsize=(10, 4), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.45})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax_lik = None

    x = np.arange(len(obs))
    obs_line, = ax.plot(x, obs, color='tab:blue', alpha=0.8,
                        linestyle='-', marker='o', markersize=5)
    handles = [obs_line]
    labels = [r'$y$ (std)']

    # Same marker over the deviant trials, in red, so context switches stay readable
    # without the separate context panel.
    deviant = np.where(contexts)[0]
    if len(deviant) > 0:
        dev_pts, = ax.plot(x[deviant], obs[deviant], color='tab:red',
                           linestyle='None', marker='o', markersize=5)
        handles.append(dev_pts)
        labels.append(rf'$y$ (dev)')

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

    ax.margins(x=0)
    ax.set_ylabel('observation process')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if show_lik_on:
        # The trial axis stays in its natural place -- the top panel's own bottom spine. sharex
        # would hide these tick labels as an "inner" axis, so re-enable them; the widened hspace
        # seats them (and the 'trial' label) in the gap above the likelihood panel.
        ax.tick_params(axis='x', labelbottom=True)
    if title is not None:
        # Small font: the parameter title is long, and this keeps it on one line within the
        # 10-inch figure width instead of overflowing (which bbox='tight' would widen to fit).
        ax.set_title(title, fontsize=8)

    # Legend inside the top panel (upper-right, usually clear of the near-zero observations and
    # sparse deviant spikes). frameon=False makes it frameless: no border and a transparent
    # background, so whatever sits behind it shows through.
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

    # The trial axis lives on the top panel's bottom spine (its natural place). sharex propagates
    # the limits, so the likelihood panel stays aligned while carrying no x-axis of its own: no
    # ticks/labels, and for the log-likelihood view (bars hanging from the 0 line) not even a
    # bottom spine, which would sit far below the data and only clutter.
    ax.set_xlim(-0.5, len(obs) - 0.5)
    ax.set_xlabel('trial')
    if show_lik_on:
        ax_lik.tick_params(axis='x', bottom=False, labelbottom=False)
        if as_log:
            ax_lik.spines['bottom'].set_visible(False)

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
