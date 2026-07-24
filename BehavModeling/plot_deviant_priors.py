"""Bar plots of the prior probability that a given timestep in a trial is the deviant.

Visualises the three priors that `PreProParadigm/model_RTs.py` multiplies into the deviant
likelihoods, as a function of the within-trial position d (0..7). All three are *hazards*: they
already condition on "no deviant has occurred yet" (S_{<d}), which is why they rise to 1 at the
last position a rule allows.

  row 0        global prior     `prior_dpos_given_prev_stds(d, p_no_dev)`
                                MARGINALIZES OVER the rules rather than ignoring them: the deviant
                                is uniform over the pooled MULTISET of slots {2,3,4} U {4,5,6} =
                                {2,3,4,4,5,6}, so position 4 -- owned by both rules -- carries
                                double weight. What it is blind to is the rules' DYNAMICS (no
                                transition matrix, no previous rule) and which rule owns which
                                slot. Since rule structure is represented, a null rule must be too:
                                `p_no_dev` (from `null_rule_marginal(pi_rule)`) reserves mass for
                                no-deviant trials, so the last slot never reaches certainty.

  rows 1..N    rule-aware prior `pior_dpos_given_prev_rule_and_stds(d, pi_rule, r_prev)`
                                One row per PREVIOUS rule (the rows of `pi_rule`), since the prior
                                is a transition-weighted mixture over the current rule.

  rows 1..N    cue-aware prior  `pior_dpos_given_prev_rule_cue_and_stds(d, pi_rule, r_prev, cue)`
               (optional)       REPLACES the row above when `cue_rule_lik` is passed: the rule
                                weights become the cue-informed posterior (transition prior x cue
                                likelihood, renormalized). Cues are drawn as grouped bars inside
                                each previous-rule panel so the cue's effect is read off directly.

Run as a script it produces both variants:

  deviant_priors_pi3x3_nocue.png   3x3 pi_rule of subjects 04/05/06 (BehavModeling/compare_RT_3subj_4sess.py)
  deviant_priors_pi2x2_cue.png     2x2 no-null-rule pi_rule + 0.8/0.2 cue association of
                                   subjects 11/12/13 (BehavModeling/compare_RT_sub11-12-13.py)

`plot_deviant_priors` is the reusable entry point; it returns the figure so it can be embedded
elsewhere without saving.
"""

import os
import sys
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import (
    DEFAULT_CUE_RULE_LIK,
    get_valid_positions_per_rule,
    null_rule_marginal,
    pior_dpos_given_prev_rule_and_stds,
    pior_dpos_given_prev_rule_cue_and_stds,
    prior_dpos_given_prev_stds,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'prior_viz')

N_POSITIONS = 8  # within-trial timesteps 0..7, as keyed by prior_dpos_given_prev_stds

# Every panel plots the same quantity against the same positions, so the axis labels belong to the
# figure rather than to any one panel (fig.supxlabel/supylabel).
XLABEL = "Position within trial"
YLABEL = r"$P(t \in \mathrm{dev} \mid \{k \, : \, k \in \mathrm{std}, \, k<t\})$"

# Categorical slots 1 and 2 of the validated reference palette, used in fixed order (never cycled):
# slot 1 carries the single-series panels, slots 1-2 the two cues.
SERIES = ['tab:blue', 'tab:orange']
INK, MUTED = '#1a1a19', '#6b6b68'

# Fraction of the unit spacing between positions taken up by a bar (or by a whole cue group).
GROUP_WIDTH = 0.8

# Bar-label type size. Grouped panels get the smaller one so the horizontal labels of side-by-side
# cue bars have a chance of staying clear of each other; the figure is also widened for those panels
# (FIG_W / EXTRA_W_PER_CUE below) because the smaller size alone does not open up enough room.
LABEL_FS, LABEL_FS_GROUPED = 7, 5.5

# Figure width. The single-series (no-cue) figure is spaced just right at FIG_W; each extra cue bar
# sharing a position needs a little more room for its own label, so grouped figures grow by
# EXTRA_W_PER_CUE per cue beyond the first. The tightest pair -- the two equal-height '0.50' bars at
# position 5 -- only clears the eye at ~0.7: the label gap grows slowly with width (labels are fixed
# point size), so it merely stops overlapping at 0.45 and does not open a visible gap until here.
FIG_W = 3.8
EXTRA_W_PER_CUE = 0.7

# Points between a panel and its title (matplotlib's default is 6). Tightened because the panels
# above have no x tick labels to clear -- sharex hides them everywhere but the bottom row.
TITLE_PAD = 1.5

# Every prior here is a ratio of small counts of rule-legal slots, so each bar height is an exact
# rational -- 1/6 rather than 0.17. Denominators stay well under this cap (51 is the largest the
# shipped pi_rule/cue settings produce); anything that misses an exact hit falls back to decimals.
MAX_DENOM = 200


def _frac_label(y):
    """Bar height as the fraction it actually is: '1/6', not '0.17'.

    Floating-point noise (0.6249999999999999) is absorbed by `limit_denominator`, but only a
    match that round-trips to within float precision is trusted -- an irrational-looking value
    keeps its decimal rendering rather than being silently rewritten as a nearby fraction.
    """
    f = Fraction(y).limit_denominator(MAX_DENOM)
    if abs(float(f) - y) > 1e-9:
        return _dec_label(y)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _dec_label(y):
    """'0.' and '1.' for the certainties, two decimals for everything in between.

    The endpoints of a probability are exact, not rounded, so padding them out to 0.00/1.00
    only implies a precision they don't need. The tolerance keeps that short form for the
    genuine endpoints alone -- 0.997 still prints as 1.00, not as a bare 1.
    """
    if abs(y) < 1e-9 or abs(y - 1.0) < 1e-9:
        return f"{y:.0f}."
    return f"{y:.2f}"


def _bars(ax, x, y, color, width, label=None, fontsize=LABEL_FS, frac_labels=False):
    """One bar series with a direct label on every position.

    Labels are decimals unless `frac_labels`, which switches them to the exact fractions the
    priors really are (see `_frac_label`).

    Zero-height bars are labelled too ("0."), so a flat position reads as a hard zero rather
    than as missing data -- including position 7, which no rule owns and which therefore carries
    zero deviant mass exactly as positions 0 and 1 do. Should a prior ever come back UNDEFINED
    (`model_RTs._hazard_without_rule_posterior`: a degenerate pi_rule that excludes every rule
    able to reach d), it is marked 'n/a' in grey rather than drawn as a zero.
    """
    ax.bar(x, y, width=width, color=color, label=label, zorder=3)
    for xi, yi in zip(x, y):
        defined = np.isfinite(yi)
        ax.text(xi, (yi if defined else 0.0) + 0.03,
                (_frac_label(yi) if frac_labels else _dec_label(yi)) if defined else "n/a",
                ha='center', va='bottom',
                fontsize=fontsize, color=INK if defined else MUTED, zorder=4)


def _style(ax, positions, headroom=1.18):
    ax.set_ylim(0, headroom)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(list(positions))
    ax.tick_params(labelsize=8, colors=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK)
    # The headroom above y=1 exists for the labels, not for data. Stopping the y-spine at 1 keeps
    # the drawn axis from advertising a range the probabilities can never reach.
    ax.spines['left'].set_bounds(0, 1)


def plot_deviant_priors(pi_rule, cue_rule_lik=None, valid_positions=None, prev_rules=None,
                        n_positions=N_POSITIONS, frac_labels=False, save_path=None):
    """Vertically stacked bar plots of P(deviant at position d) under each prior.

    Parameters
    ----------
    pi_rule : (R, R) array
        Rule transition matrix, `pi_rule[r_prev, r]`. Its number of ROWS sets how many
        previous-rule panels are drawn (a 3x3 includes the null previous rule).
    cue_rule_lik : dict, optional
        {cue -> {rule -> P(cue | rule)}}, e.g. `model_RTs.DEFAULT_CUE_RULE_LIK`. When given, the
        CUE-AWARE prior replaces the transition-only one and each panel shows one bar per cue.
    valid_positions : dict, optional
        {rule -> valid deviant positions}; defaults to `get_valid_positions_per_rule()`
        ({0: [2,3,4], 1: [4,5,6]}). Note this keys the CURRENT rules summed over, which for a
        3x3 `pi_rule` is a strict subset of its rows.
    prev_rules : sequence, optional
        Which previous rules to draw; defaults to every row of `pi_rule`.
    n_positions : int
        Number of within-trial timesteps on the x-axis (0..n_positions-1).
    frac_labels : bool
        Label each bar with the exact fraction it is (1/6) instead of a decimal (0.17).
    save_path : str, optional
        If given, the figure is written here at 600 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pi_rule = np.asarray(pi_rule, dtype=float)
    if valid_positions is None:
        valid_positions = get_valid_positions_per_rule()
    if prev_rules is None:
        prev_rules = range(pi_rule.shape[0])
    prev_rules = list(prev_rules)
    cues = list(cue_rule_lik) if cue_rule_lik else [None]

    positions = np.arange(n_positions)
    nrows = 1 + len(prev_rules)
    # Constrained (not tight) layout: only it reserves room for the figure-level axis labels,
    # which tight_layout sizes around the axes alone and would let the panels overlap.
    figwidth = FIG_W + EXTRA_W_PER_CUE * (len(cues) - 1)
    fig, axs = plt.subplots(nrows, 1, figsize=(figwidth, 0.9 * nrows + 0.4), sharex=True,
                            layout='constrained')
    # Rows still need a little air between them, but less now that each title sits close to its
    # own panel: the gap reads as separating panels rather than orphaning the title above.
    fig.get_layout_engine().set(hspace=0.04)

    # Row 0: the global prior. Single series -> no legend, the panel title names it.
    # It marginalizes OVER the rules, so a null rule in pi_rule must be represented here too:
    # with one, surviving to the last slot no longer makes a deviant certain.
    p_no_dev = null_rule_marginal(pi_rule, valid_positions)
    _bars(axs[0], positions,
          [prior_dpos_given_prev_stds(d, p_no_dev=p_no_dev) for d in positions],
          SERIES[0], width=GROUP_WIDTH, frac_labels=frac_labels)
    axs[0].set_title("Global prior (marginalized over rules)",
                    #  + (f"   P(no-deviant trial)={p_no_dev:.3f}" if p_no_dev else ""),
                     fontsize=9, color=INK, pad=TITLE_PAD)
    _style(axs[0], positions)

    # Rows 1..N: the rule-aware prior, one panel per PREVIOUS rule; cues become grouped bars.
    width = GROUP_WIDTH / len(cues)
    for ax, r_prev in zip(axs[1:], prev_rules):
        for k, (cue, color) in enumerate(zip(cues, SERIES)):
            if cue is None:
                y = [pior_dpos_given_prev_rule_and_stds(d, pi_rule, r_prev, valid_positions)
                     for d in positions]
            else:
                y = [pior_dpos_given_prev_rule_cue_and_stds(d, pi_rule, r_prev, cue,
                                                            cue_rule_lik, valid_positions)
                     for d in positions]
            # Offsets centre the group on the tick; bars within a group touch, so the only gaps
            # on the axis are the ones between positions.
            offset = (k - (len(cues) - 1) / 2) * width
            _bars(ax, positions + offset, y, color, width=width, label=cue,
                  fontsize=LABEL_FS_GROUPED if len(cues) > 1 else LABEL_FS,
                  frac_labels=frac_labels)

        kind = "Cue-rule-modulated prior" if cue_rule_lik else "Rule-modulated prior"
        row = np.array2string(pi_rule[r_prev], precision=2, separator=', ')
        ax.set_title(rf"{kind}  |  previous rule = {r_prev}", #    ($\pi$ row {row})
                     fontsize=9, color=INK, pad=TITLE_PAD)
        _style(ax, positions)
        if len(cues) > 1:
            # Default vertical stacking (one cue per row), auto-sized by matplotlib. Sits over the
            # empty upper-left corner, where the early positions carry little or no deviant mass.
            leg = ax.legend(fontsize=7, loc='upper left', frameon=False, framealpha=0,
                            labelcolor=INK)
            leg.set_zorder(5)

    # One label per figure rather than per panel: every panel shares both axes.
    fig.supxlabel(XLABEL, fontsize=9, color=INK)
    fig.supylabel(YLABEL, fontsize=9, color=INK)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, facecolor='white')
        print(f"Saved {save_path}")
    return fig


# 3x3 transition matrix of subjects 04/05/06 (row 2 = the null previous rule).
PI_RULE_3X3 = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.5, 0.5, 0.0],
])

# 2x2 NO-NULL-RULE matrix of subjects 11/12/13, who additionally see a cue every trial.
PI_RULE_2X2 = np.array([
    [0.8, 0.2],
    [0.2, 0.8],
])


def main():
    plot_deviant_priors(
        PI_RULE_3X3,
        save_path=os.path.join(OUT, 'deviant_priors_pi3x3_nocue.png'),
    )
    plot_deviant_priors(
        PI_RULE_2X2,
        cue_rule_lik=DEFAULT_CUE_RULE_LIK,
        save_path=os.path.join(OUT, 'deviant_priors_pi2x2_cue.png'),
    )


if __name__ == '__main__':
    main()
