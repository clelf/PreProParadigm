"""Add cue-aware rule-weighted likelihoods to the logfiles2clem KF/leaky outputs.

Subjects 11/12/13 have a cue on every trial, with a known cue<->rule association
[[0.8, 0.2], [0.2, 0.8]] and no null rule. The cue is a second, independent observation of the
latent rule, so it enters the rule prior by Bayes' rule -- transition-prior x cue-likelihood,
renormalized (a PRODUCT, not a mixture). `compute_dev_likelihoods_over_rules` now detects the
`cue` column and swaps `pior_dpos_given_prev_rule_and_stds` for its cue-aware counterpart
`pior_dpos_given_prev_rule_cue_and_stds` automatically (see PreProParadigm/model_RTs.py).

This driver just applies that step to the already-computed deviant-likelihood CSVs, adding the
`prior_dpos_rules_{std,dev}` and `likelihood_obs_std_at_dev_over_rules` columns. It needs no
reaction times and does not re-run the Kalman filter, so it is cheap.

For reference/QC it also writes the transition-ONLY rule-weighted likelihood
(`..._over_rules_nocue`) so the effect of adding the cue is visible in the saved files.
"""

import os
import sys

import numpy as np
import pandas as pd

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(WORKSPACE, 'PreProParadigm'))

from model_RTs import compute_dev_likelihoods_over_rules, DEFAULT_CUE_RULE_LIK

HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED_TRIALS_DIR = os.path.join(HERE, 'triallists_logfiles2clem')
KF_OUT_DIR = os.path.join(HERE, 'kalman_pred')
LEAKY_OUT_DIR = os.path.join(HERE, 'leaky_pred')

SUBS = ['11', '12', '13']
SESS = 1
SIGMA_R = 0.02

# 2x2 NO-NULL-RULE transition matrix -- these subjects only ever see rules {0, 1}. This must NOT be
# the 04/05/06 3x3: although the cue-aware prior renormalizes over rules {0, 1} (so only the 0<->1
# transition RATIO survives), that ratio is 0.8:0.1 = 8:1 under the 3x3 but 0.8:0.2 = 4:1 here --
# a different prior, not an equivalent one. The transition-only (`_nocue`) prior is worse off still:
# it does not renormalize, so a 3x3 leaves the rule weights summing to 0.9 instead of 1.0.
PI_RULE = np.array([
    [0.8, 0.2],
    [0.2, 0.8],
])

# The KF/leaky deviant-likelihood files, per subject.
MODELS = {
    'KF':    (KF_OUT_DIR,    f'kalman_predictions_and_likelihoods_at_deviants_sub-{{sub}}_ses-{SESS}_sigma_r_{SIGMA_R}.csv'),
    'leaky': (LEAKY_OUT_DIR, f'leaky_predictions_and_likelihoods_at_deviants_sub-{{sub}}_ses-{SESS}.csv'),
}


def _augment(sub, trials_path, results_path):
    """Add cue-aware (and, for QC, transition-only) rule-weighted likelihood columns; save in place."""
    results_df = pd.read_csv(results_path)

    # ONE call now yields both the cue-aware rule prior AND its transition-only (`_nocue`) QC
    # counterpart: compute_dev_likelihoods_over_rules computes the latter from the SAME PI_RULE
    # whenever the trials carry a `cue` column. That guarantees the two share one matrix, and makes
    # the cue-stripped temp-file round-trip this used to do unnecessary.
    cue_df = compute_dev_likelihoods_over_rules(
        trials_path, results_df.copy(), sub, SESS, PI_RULE, cue_rule_lik=DEFAULT_CUE_RULE_LIK)

    out = results_df.copy()
    for c in ('rule', 'prev_rule', 'prior_dpos_rules_std', 'prior_dpos_rules_dev',
              'prior_dpos_rules_std_nocue', 'likelihood_obs_std_at_dev_over_rules',
              'likelihood_obs_std_at_dev_over_rules_nocue'):
        out[c] = cue_df[c]

    out.to_csv(results_path, index=False)
    return out


def main():
    for sub in SUBS:
        trials_path = os.path.join(COMBINED_TRIALS_DIR, f'sub-{sub}_ses-{SESS}_trials.csv')
        for label, (out_dir, fname) in MODELS.items():
            results_path = os.path.join(out_dir, fname.format(sub=sub))
            if not os.path.exists(results_path):
                print(f'MISS  sub-{sub} {label}: {results_path}')
                continue
            df = _augment(sub, trials_path, results_path)
            col, ncol = 'likelihood_obs_std_at_dev_over_rules', 'likelihood_obs_std_at_dev_over_rules_nocue'
            print(f'OK    sub-{sub} {label}: rows={len(df)} '
                  f'mean(cue)={np.nanmean(df[col]):.4f} mean(nocue)={np.nanmean(df[ncol]):.4f} '
                  f'-> {results_path}')
    print('done')


if __name__ == '__main__':
    main()
