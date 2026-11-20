"""One driver for the five per-run / per-subject RT-vs-likelihood figures.

This merges the previously separate drivers into a single script so that the subject cohort and
every optional flag are declared ONCE and applied UNIFORMLY to all five figures. Edit the config
block at the top of `__main__` and every figure below picks the change up.

Figures produced (all written to BehavModeling/RT_comparisons/):

  1. Per-run x per-model grid  (was compare_RT_3sub_4sess_perrun_models.py)
     8 model rows x (subject x session x run) columns; each cell a trial-level RT-vs-likelihood
     scatter for one run (~55 deviant trials) with a regression line and rho + stars.

  2. Per-subject (pooled) x per-model grid  (was compare_RT_3sub_pooled_persubject_models.py)
     Same 8 model rows, but every run/session a subject contributed is pooled into ONE column, so
     each cell scatters ALL of that subject's deviant trials (hundreds) with rho, stars and n.

  3. Per-subject R2 violins  (was summary_R2_violins_perrun_models_persubject.py)
     One stacked panel per subject; one violin per model summarising that subject's per-run R2s.

  4. Per-subject Cohen's d heatmaps  (was summary_cohensd_heatmap_perrun_models_persubject.py)
     One pairwise-model Cohen's d heatmap per subject, laid out horizontally on a shared scale.

  5. Combined pooled + per-run grid  (mix of figures 1 and 2)
     The figure-1 per-run grid with figure-2's three pooled per-subject columns PREPENDED, divided
     from the per-run columns by a plain line twice as thick as the per-run dotted separators.

Each session's KF likelihoods are computed with that session's TRUE observation noise (its si_r).

Two cohorts are supported, selected by SUBJECTS alone -- the paths AND the collection route follow
from it (see the cohort switch in `__main__`):

  04/05/06 (session-type cohort)      11/12/13 (logfiles2clem cohort)
  trials : PreProParadigm/triallists  trials : BehavModeling/triallists_logfiles2clem
  logs   : PreProParadigm/logfiles    logs   : BehavModeling/logfiles2clem
  KF     : BehavModeling/kalman_pred  KF     : BehavModeling/kalman_pred
  leaky  : PreProParadigm/leaky_predictions   leaky : BehavModeling/leaky_pred
  4 sessions x 4 runs, per-session si_r       1 session x 4 runs, fixed si_r = 0.02, cue-aware
                                              rule priors
  figures out (both) : BehavModeling/RT_analysis_figures/

All five figures exist in both forms; the cohort selects which one each call below dispatches to.

Note on flag unification: the four original drivers did NOT all use identical flags -- the per-run
grid used loglik=False, and the R2 violins used correcttrials=False, while the others used
loglik=True / correcttrials=True. Per the request, this script applies a single choice of each flag
to every figure. Change LOGLIK / CORRECTTRIALS below to reproduce a particular original if needed.

CORR ('pearson' | 'spearman') picks the correlation behind the rho annotated on the SCATTER figures
(1/2/2b/5) and nothing else. Every R2 in this script -- figures 3/4/4b, and the collection pass they
share -- is always the OLS one, because a squared Spearman rho is the R2 of a fit on RANKS and not a
coefficient of determination. So CORR only moves the figures that report a rho without an R2:
Spearman versions of 1/2/2b/5 are written as '..._spearman.png' beside the Pearson ones, while
3/4/4b keep one unsuffixed name whatever CORR is. See the flag's comment below.
"""

import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import (
    compare_likelihoods_with_RTs_per_run_models,
    compare_likelihoods_with_RTs_per_run_models_single_session,
    compare_likelihoods_with_RTs_per_subject_models,
    compare_likelihoods_with_RTs_per_subject_models_single_session,
    compare_likelihoods_with_RTs_per_subject_runmeans_models,
    compare_likelihoods_with_RTs_per_subject_runmeans_models_single_session,
    compare_likelihoods_with_RTs_per_run_and_subject_models,
    compare_likelihoods_with_RTs_per_run_and_subject_models_single_session,
    get_perrun_models_R2,
    get_perrun_models_R2_single_session,
    plot_perrun_models_R2_violins_per_subject,
    plot_model_R2_cohensd_heatmap_per_subject,
    plot_model_R2_cohensd_heatmap_and_violins_per_subject,
)


if __name__ == "__main__":

    # ------------------------------------------------------------------ shared config (edit here)
    # Subject cohort -- set once, used by all five figures.
    SUBJECTS = ['04', '05', '06']
    # SUBJECTS = ['11', '12', '13']

    # Optional flags -- set once, applied uniformly to all five figures.
    LOGRT = True          # regress log(RT) instead of raw RT
    LOGLIK = True         # use log-likelihood instead of raw likelihood on the x-axis
    # Trial selection / RT reference -- a single switch between two mutually exclusive modes:
    #   False -> CORRECT-ONLY (the previous `correcttrials=True` behaviour): keep only trials the
    #            subject answered at the right position (dpos == key_pressed, i.e. correct==1), with
    #            RT and likelihoods measured at the true deviant. Figure names carry '_correcttrials'.
    #   True  -> RESPONSE-LOCKED / ALL VALID: keep every valid-RT trial, but re-reference the RT
    #            (`pressed_tone_rt`) and every model likelihood/prior to the tone the key actually
    #            NAMED (see `_response_locked_columns` in model_RTs). Incorrect-position trials are
    #            then kept and meaningfully paired instead of dropped; for correct trials the pressed
    #            tone IS the deviant, so their values are unchanged. Figure names carry '_allvalidtrials'.
    # The two versions coexist in the output. Mapped to the shared functions' correcttrials/
    # allvalidtrials pair below (exactly one is ever True, so they never conflict).
    ALLVALIDTRIALS = True
    # Correlation ANNOTATED on the scatter figures (1/2/2b/5) -- the rho + stars in each cell:
    #   'pearson'  -> the linear coupling the drawn regression line describes. Computed on the
    #                 transformed values, so LOGRT/LOGLIK change it. Figure names are unchanged
    #                 (the historical default).
    #   'spearman' -> the monotonic coupling, on ranks. Invariant to LOGRT/LOGLIK (up to the
    #                 non-positive likelihoods LOGLIK drops), so it reads a run's RT-vs-likelihood
    #                 ordering without assuming the relation is a straight line. Figures are named
    #                 '..._spearman.png', so both versions coexist in the output.
    CORR = 'spearman'
    # Uniform size multiplier for the two Cohen's d figures (4 and 4b). Font sizes are set in POINTS
    # and do NOT follow it, so raising this enlarges the figure at constant text size -- which is
    # what gives the 8 x 8 heatmap's per-cell labels (the +d value and its stars) room to breathe.
    # Proportions are untouched: both figsize dimensions are multiplied, and the combined figure's
    # violin/heatmap split and row heights are ratios. 1.0 reproduces the previous size.
    HEATMAP_SCALE = 1.35
    # ---------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------- cohort switch
    # The two cohorts live in DIFFERENT trees and have different session structures, so the subject
    # list alone must select the paths AND the data-collection route -- setting SUBJECTS without
    # switching these is what silently produced an empty R2 table ("No data for subject 11 in
    # session type ...", then a bare KeyError: 'sub' in the plotter).
    #   04/05/06 : 4 sessions each, one per (d, si_stat, si_r) session TYPE encoded in the trials
    #              filenames; every session uses its own true si_r.
    #   11/12/13 : ONE session each (logfiles2clem, fMRI event logs), 4 runs, a single fixed
    #              si_r = 0.02, and cue-aware rule priors.
    SINGLE_SESSION = set(SUBJECTS) <= {'11', '12', '13'}

    if SINGLE_SESSION:
        trials_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/triallists_logfiles2clem'
        logfiles_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/logfiles2clem'
        preds_liks_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/kalman_pred'
        leaky_preds_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/leaky_pred'
        SESS_NUM = 1      # the only session these subjects have
        SIGMA_R = 0.02    # this cohort's true observation noise, fixed across runs
        # 2-rule transition matrix (no null rule for this cohort).
        pi_rule = np.array([
            [0.8, 0.2],
            [0.2, 0.8]
        ])
    else:
        trials_path = os.path.join(WORKSPACE, 'PreProParadigm', 'triallists') + '/'
        logfiles_path = os.path.join(WORKSPACE, 'PreProParadigm', 'logfiles')
        # Regenerated KF predictions (written by BehavModeling/compute_kf_predictions.py).
        preds_liks_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/kalman_pred'
        leaky_preds_path = os.path.join(WORKSPACE, 'PreProParadigm', 'leaky_predictions')
        # 3-rule transition matrix, last row the null (no-deviant) rule.
        pi_rule = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.5, 0.5, 0.0]
        ])

    comparison_save_path = '/home/clevyfidel/Documents/Workspace/PreProParadigm/BehavModeling/RT_analysis_figures'
    os.makedirs(comparison_save_path, exist_ok=True)

    # Positional args and shared flags common to all five figure functions.
    common_args = (
        SUBJECTS, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
    )
    # ALLVALIDTRIALS=False falls back to the old correct-only selection; =True is response-locked.
    trial_kwargs = dict(
        logrt=LOGRT,
        loglik=LOGLIK,
        correcttrials=not ALLVALIDTRIALS,
        allvalidtrials=ALLVALIDTRIALS,
    )
    # `corr` deliberately does NOT ride in `shared_kwargs`: only the four scatter-grid figures
    # annotate a rho and so accept it (it joins `grid_kwargs` below). Figures 3/4/4b and the R2
    # functions take no `corr` at all -- their R2 is the OLS one by construction -- so passing it
    # here would be a TypeError, which is the point: the flag cannot silently redefine an R2.
    shared_kwargs = dict(leaky_preds_path=leaky_preds_path, **trial_kwargs)
    # The session-type route reads si_r per session; the single-session route is given the
    # cohort's one session and its sigma_r, which only the four grid figures take
    if SINGLE_SESSION:
        grid_kwargs = dict(corr=CORR, sess_num=SESS_NUM, sigma_r=SIGMA_R)
    else:
        grid_kwargs = dict(corr=CORR)

    # ---- The R2 table behind figures 3, 4 and 4b ------------------------------------------------
    # Computed once here, with the R2 function matching the cohort, then given to all three figures
    # via their `df=` argument -- previously each figure recomputed it, and each computed it with
    # the 04/05/06-only `get_perrun_models_R2`, which yields zero rows for 11/12/13.
    # Independent of CORR: every R2 here is the OLS one (Pearson rho**2 = the coefficient of
    # determination of the line the grids draw), so this table is the same whichever rho they report.
    if SINGLE_SESSION:
        r2_df = get_perrun_models_R2_single_session(
            SUBJECTS, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
            sess_num=SESS_NUM, sigma_r=SIGMA_R, **trial_kwargs)
    else:
        r2_df = get_perrun_models_R2(
            SUBJECTS, trials_path, preds_liks_path, logfiles_path, pi_rule, leaky_preds_path,
            **trial_kwargs)

    # Each figure exists in a session-type form and a single-session form; both take the same
    # `common_args`/`shared_kwargs` and differ only by `grid_kwargs` above.
    perrun_models = (compare_likelihoods_with_RTs_per_run_models_single_session if SINGLE_SESSION
                     else compare_likelihoods_with_RTs_per_run_models)
    persubject_models = (compare_likelihoods_with_RTs_per_subject_models_single_session
                         if SINGLE_SESSION else compare_likelihoods_with_RTs_per_subject_models)
    # persubject_runmeans = (compare_likelihoods_with_RTs_per_subject_runmeans_models_single_session
    #                        if SINGLE_SESSION
    #                        else compare_likelihoods_with_RTs_per_subject_runmeans_models)
    perrun_and_subject = (compare_likelihoods_with_RTs_per_run_and_subject_models_single_session
                          if SINGLE_SESSION
                          else compare_likelihoods_with_RTs_per_run_and_subject_models)

    # ---- Figure 1: per-run x per-model RT-vs-likelihood grid -------------------------------------
    print("[1/5] Per-run x per-model RT-vs-likelihood grid ...")
    perrun_models(*common_args, **shared_kwargs, **grid_kwargs)

    # ---- Figure 2: per-subject (pooled) x per-model RT-vs-likelihood grid ------------------------
    print("\n[2/5] Per-subject (pooled) x per-model RT-vs-likelihood grid ...")
    persubject_models(*common_args, **shared_kwargs, **grid_kwargs)

    # ---- Figure 2b: per-subject x per-model, ONE run-mean dot per run (RT_persubject_perrun_linrel) ----------
    # DISCARDED: not meaningful
    # Same 8 x 3 layout as Figure 2, but each run is collapsed to one dot (mean likelihood, mean RT)
    # with SEM brackets on both axes, dots linked by a plain line (ascending likelihood) and fitted
    # by a dashed OLS trend + shaded CI -- so each subject column shows one dot per run (16 for
    # 04/05/06, 4 for 11/12/13), not every trial.
    # print("\n[2b/5] Per-subject x per-model run-mean RT-vs-likelihood grid (RT_persubject_perrun_linrel) ...")
    # persubject_runmeans(*common_args, **shared_kwargs, **grid_kwargs)

    # # ---- Figure 3: per-subject R2 violins --------------------------------------------------------
    print("[3/5] Per-subject R2 violins ...")
    plot_perrun_models_R2_violins_per_subject(*common_args, **shared_kwargs, df=r2_df)

    # # ---- Figure 4: per-subject Cohen's d heatmaps ------------------------------------------------
    print("\n[4/5] Per-subject Cohen's d heatmaps ...")
    plot_model_R2_cohensd_heatmap_per_subject(*common_args, **shared_kwargs, df=r2_df,
                                              scale=HEATMAP_SCALE)

    # ---- Figure 4b: figures 3 + 4 merged, heatmap LEFT / violins RIGHT, one row per subject ------
    # Same content as figures 3 and 4, laid out on one grid so both halves come out at matching
    # element sizes. Reuses the R2 table figure 3 already collected (no second collection pass).
    print("\n[4b/5] Per-subject Cohen's d heatmap + R2 violins (combined) ...")
    plot_model_R2_cohensd_heatmap_and_violins_per_subject(*common_args, **shared_kwargs, df=r2_df,
                                                          scale=HEATMAP_SCALE)

    # ---- Figure 5: combined pooled + per-run grid (mix of figures 1 and 2) -----------------------
    print("\n[5/5] Combined pooled per-subject + per-run x per-model grid ...")
    perrun_and_subject(*common_args, **shared_kwargs, **grid_kwargs)

    print(f"\nVisualization completed, all figures saved to {comparison_save_path}")
