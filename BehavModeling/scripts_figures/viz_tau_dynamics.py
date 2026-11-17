"""Standard-process dynamics for two experimental runs, side by side.

Strips ax1 of `HierarchicalAuditGM.plot_combined_with_matrix` down to the standard
process only (no deviant trace, no observations, no dpos/transition-matrix panels),
and lays out the selected runs in one wide figure (width grows with each run). Panels
carry no ticks and no tick labels; the only annotation is the run's tau_std as x-label.

Runs plotted here contrast the fastest and slowest standard process of the session
(sub-04, session 4 on disk = "sess3" in the figure naming of figure_example.py):
run4 has tau_std = 16, run1 has tau_std = 240.
"""

import os
import glob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model_RTs import prepare_trials_data


# --- What to plot ----------------------------------------------------------
SUB = '04'
# (session index, run index), both 0-based as in figure_example.py, where the
# figure names are built as sess{sess}_run{run_id+1}.
RUNS = [(3, 3), (3, 0)]  # sub04_sess3_run4, sub04_sess3_run1

# --- Appearance ------------------------------------------------------------
N_TONES = 8
RULES_CMAP = {0: sns.color_palette("Paired")[7], 1: sns.color_palette("Paired")[9], 2: "#ffffff"}
SHOW_RULE_SHADING = False  # coloured band per trial, by rule
SHOW_TRIAL_LINES = False   # vertical line at each trial onset
SHOW_TITLES = False        # per-panel title with the run parameters
# Common y-SPAN across panels, each centred on its own lim_std: the runs sit at
# different stationary means, so a literally shared y-range would leave each trace
# squashed into half its panel. With a shared span, amplitudes stay comparable.
SHARE_Y = True
PANEL_SIZE = (8, 1)       # per-run size in inches; total width = n_runs * width
SAVE_FORMATS = ("png",)


def plot_std_process(ax, x_stds, rules, lim_std, si_stat, tau_std):
    """Draw the standard process of one run on `ax` (ax1 of plot_combined_with_matrix,
    standard context only): no ticks, no tick labels, no top/right spines, and the
    run's tau_std as the only annotation."""
    n_blocks = len(x_stds) // N_TONES

    ax.set_xlim(0, len(x_stds) - 1)
    ax.plot(x_stds, color="tab:blue", linestyle="-", linewidth=2, alpha=0.8)

    if SHOW_RULE_SHADING:
        for i, rule in enumerate(rules):
            ax.axvspan(i * N_TONES, (i + 1) * N_TONES, facecolor=RULES_CMAP[rule], alpha=0.25)
    if SHOW_TRIAL_LINES:
        for i in range(n_blocks):
            ax.axvline(i * N_TONES, color="tab:gray", linewidth=0.9)

    # Stationary mean and its ± si_stat envelope
    ax.hlines(lim_std, xmin=0, xmax=len(x_stds) - 1, color="tab:blue", linestyle="--", linewidth=2, alpha=0.5)
    ax.fill_between(range(len(x_stds)), lim_std - si_stat, lim_std + si_stat, color="tab:blue", alpha=0.2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    tau_txt = f"{tau_std:g}"
    ax.set_xlabel(rf"$\tau^{{\mathrm{{std}}}} = {tau_txt}$", labelpad=8, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def run_pars(run_df):
    """Per-run generative parameters, read off the first tone of the run."""
    first = run_df.iloc[0]
    return {
        "lim": [first['lim_std'], first['lim_dev']],
        "tau": [first['tau_std'], first['tau_dev']],
        "si_q": [first['sigma_q_std'], first['sigma_q_dev']],
        "si_stat": first['sigma_r'] * 10,
        "si_r": first['sigma_r'],
        "d": first['d'],
    }


if __name__ == "__main__":

    trial_list_dir = os.path.join(os.path.dirname(__file__), 'triallists')
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Load first, so the panels can be given a common y-span (see SHARE_Y)
    runs = []
    for sess, run_id in RUNS:
        trial_list_path = glob.glob(f"{trial_list_dir}/sub-{SUB}/sub-{SUB}_ses-{sess + 1}*.csv")[0]
        trials_df = prepare_trials_data(trial_list_path)
        run_df = trials_df[trials_df['run_n'] == run_id]
        runs.append((sess, run_id, run_df, run_pars(run_df)))

    # Largest excursion from the stationary mean, over all runs
    half_span = 1.05 * max(
        np.abs(run_df['state_std'].to_numpy() - pars['lim'][0]).max()
        for _, _, run_df, pars in runs
    )

    fig, axes = plt.subplots(
        1, len(RUNS),
        figsize=(PANEL_SIZE[0] * len(RUNS), PANEL_SIZE[1]),
    )
    axes = np.atleast_1d(axes)

    for ax, (sess, run_id, run_df, pars) in zip(axes, runs):
        plot_std_process(
            ax,
            run_df['state_std'].reset_index(drop=True),
            run_df.drop_duplicates('trial_no')['rule'].reset_index(drop=True),
            lim_std=pars['lim'][0],
            si_stat=pars['si_stat'],
            tau_std=pars['tau'][0],
        )
        if SHARE_Y:
            ax.set_ylim(pars['lim'][0] - half_span, pars['lim'][0] + half_span)

        if SHOW_TITLES:
            ax.set_title(rf"sess{sess}, run{run_id + 1}: $\tau$_std = {pars['tau'][0]}, "
                         rf"$\sigma_q$_std = {pars['si_q'][0]}, $\sigma$_stat = {pars['si_stat']}")

    fig.tight_layout()

    runs_tag = "_".join(f"sess{sess}_run{run_id + 1}" for sess, run_id in RUNS)
    for ext in SAVE_FORMATS:
        save_path = os.path.join(fig_dir, f"tau_dynamics_sub{SUB}_{runs_tag}.{ext}")
        fig.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")
