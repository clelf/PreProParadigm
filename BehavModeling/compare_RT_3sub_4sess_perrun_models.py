"""Per-run x per-model RT-vs-likelihood grid, on REGENERATED predictions.

Complements `compare_RT_3sub_4sess_noise_levels_indiv.py`. Where that figure is a
subjects x session-types grid that POOLS all runs and overlays models as colored lines, this
one produces a single grid with:

  - one ROW per model (8 models: KF alone, LI alone, global-prior alone, rule-prior alone,
    KF+global, LI+global, KF+rule, LI+rule), and
  - one COLUMN per (subject, session, run) combination -- no pooling across runs.

With 3 subjects x 4 session-types x 4 runs, that is an 8 x 48 grid. Each subplot is a
trial-level RT-vs-likelihood scatter (~55 deviant trials for that run) with a regression line
and a Pearson rho + significance-stars readout.

Like the `_indiv` figure, each session uses the KF likelihoods computed with that session's TRUE
observation noise (its si_r). Same paths and inputs as that driver.

  predictions in : BehavModeling/kalman_pred/
  leaky preds in : PreProParadigm/leaky_predictions/
  figures out    : BehavModeling/RT_comparisons/   (suffix '_indivobsnoise')
"""

import os
import sys
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import compare_likelihoods_with_RTs_per_run_models


if __name__ == "__main__":

    trials_path = os.path.join(WORKSPACE, 'PreProParadigm', 'triallists') + '/'

    # Regenerated KF predictions (written by BehavModeling/compute_kf_predictions.py).
    preds_liks_path = os.path.join(os.path.dirname(__file__), 'kalman_pred')

    leaky_preds_path = os.path.join(WORKSPACE, 'PreProParadigm', 'leaky_predictions')

    comparison_save_path = os.path.join(os.path.dirname(__file__), 'RT_comparisons')
    os.makedirs(comparison_save_path, exist_ok=True)

    logfiles_path = os.path.join(WORKSPACE, 'PreProParadigm', 'logfiles')

    # Rule transition matrix
    pi_rule = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.5, 0.5, 0.0]
    ])

    subjects = ['04', '05', '06']

    # Trial selection. False (default, and what every RT analysis here has used so far) keeps
    # every trial with a valid RT; True additionally requires that the subject identified the
    # deviant's POSITION correctly. Switching it on only renames the figure ('_correcttrials'
    # suffix), so the two versions coexist in RT_comparisons/ rather than overwriting each other.
    correcttrials = True

    compare_likelihoods_with_RTs_per_run_models(
        subjects, trials_path, preds_liks_path, logfiles_path, comparison_save_path, pi_rule,
        leaky_preds_path=leaky_preds_path,
        logrt=True, loglik=False,
        sigma_r_files=True,  # kalman_pred uses the current `_sigma_r_{si_r}` naming
        correcttrials=correcttrials,
    )

    print(f"\nVisualization completed, result saved to {comparison_save_path}")
