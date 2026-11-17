"""Get the mean squared error (MSE) between the observations and the predictions of the Kalman filter
(KF) and of the pRNN models, for each within-trial position and each condition, on the two batches of
sequences of generate_task_sequences_gm_test.py (gm_original, gm_corrected).

Reads the CSVs written by compute_predictions_gm_test.py in OUTPUT_ROOT/<batch>/<model>/probabilities/.

For one sequence, the MSE at position p (p = 0 ... 7, the position of the tone within its trial, equal to
timestep % PERIOD) is the mean over the standard tones at position p, in the 60 trials, of
(observation - predicted mean)**2. The tones at the deviant position of their trial are left out, and so
are the tones without a prediction (NaN, the KF does not predict the first tones; the pRNN models do not
predict timestep 0). The predicted mean is
  KF:    mu_pred_std, the mean predicted by the filter of the standard state
  pRNN:  obs_mean, the mean predicted by the observation module

The MSE is not averaged over the sequences of a condition, so that the sequences of the two batches can be
compared pair by pair: a sequence of gm_original and the sequence of gm_corrected with the same name share
their random seed.

Outputs, for each batch and model, OUTPUT_ROOT/mse_by_position/<batch>/<model>.csv:
one row per sequence, with the columns
  sequence    name of the sequence, e.g. 'sub-seq10'
  condition   name of its condition, e.g. 'cond1_tau16_mu-0.6_dev1'
  MSE_0 ... MSE_7   MSE at each within-trial position
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREPROPARADIGM_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PREPROPARADIGM_DIR.parent
sys.path.append(str(WORKSPACE_ROOT / 'RNN_paradigm' / 'RNN' / 'model_analysis'))

import analysis_config as cfg


# Name of the folder of the KF outputs, as in compute_predictions_gm_test.py.
KALMAN_FILTER_NAME = 'kalman_filter'


def get_condition_name(probabilities_csv):
    """Name of the condition of a sequence, read from the name of its CSV, e.g. 'cond1_tau16_mu-0.6_dev1'."""
    match = re.search(r'cond\d+_tau\d+_mu-?[0-9.]+_dev-?1', probabilities_csv.name)
    if match is None:
        raise ValueError(f"No condition name in the file name {probabilities_csv.name}")
    return match.group(0)


def get_sequence_name(probabilities_csv):
    """Name of a sequence, read from the name of its CSV, e.g. 'sub-seq10'."""
    match = re.match(r'sub-seq\d+', probabilities_csv.name)
    if match is None:
        raise ValueError(f"No sequence name in the file name {probabilities_csv.name}")
    return match.group(0)


def get_sequence_mse_by_position(probabilities_df, prediction_column, period):
    """MSE of one sequence at each within-trial position, over its standard tones with a prediction.

    Returns an array of length `period`, NaN at a position without any such tone.
    """
    squared_errors = (probabilities_df['observation'] - probabilities_df[prediction_column]) ** 2
    position = probabilities_df['timestep'] % period
    is_standard_tone_with_prediction = (probabilities_df['ctx'] == 0) & probabilities_df[prediction_column].notna()

    mse_by_position = np.full(period, np.nan)
    for p in range(period):
        squared_errors_at_position = squared_errors[is_standard_tone_with_prediction & (position == p)]
        if len(squared_errors_at_position) > 0:
            mse_by_position[p] = squared_errors_at_position.mean()
    return mse_by_position


def get_mse_by_position_per_sequence(probabilities_dir, prediction_column, period):
    """DataFrame of the MSE of a model at each within-trial position, for each sequence.
    Columns: sequence, condition, MSE_0 ... MSE_{period-1}. Rows sorted by condition number, then sequence number.
    """
    probabilities_csvs = sorted(probabilities_dir.glob('*_probabilities.csv'))
    if len(probabilities_csvs) == 0:
        raise FileNotFoundError(f"No *_probabilities.csv in {probabilities_dir}")

    mse_columns = [f'MSE_{p}' for p in range(period)]
    mse_df = pd.DataFrame(index=range(len(probabilities_csvs)), columns=['sequence', 'condition'] + mse_columns)
    for i, probabilities_csv in enumerate(probabilities_csvs):
        probabilities_df = pd.read_csv(probabilities_csv)
        mse_df.loc[i, 'sequence'] = get_sequence_name(probabilities_csv)
        mse_df.loc[i, 'condition'] = get_condition_name(probabilities_csv)
        mse_df.loc[i, mse_columns] = get_sequence_mse_by_position(probabilities_df, prediction_column, period)
    mse_df[mse_columns] = mse_df[mse_columns].astype(float)

    # Conditions in the order of their number (cond1, cond2, ..., cond16), not alphabetical; same for sequences.
    mse_df['condition_number'] = mse_df['condition'].str.extract(r'cond(\d+)_', expand=False).astype(int)
    mse_df['sequence_number'] = mse_df['sequence'].str.removeprefix('sub-seq').astype(int)
    mse_df = mse_df.sort_values(['condition_number', 'sequence_number'])
    return mse_df.drop(columns=['condition_number', 'sequence_number']).reset_index(drop=True)


if __name__ == "__main__":

    OUTPUT_ROOT = PREPROPARADIGM_DIR / 'predictions_gm_test'
    BATCH_FOLDER_NAMES = ['gm_original', 'gm_corrected']

    # Models of compute_predictions_gm_test.py: the KF and the pRNN models.
    # MODEL_NAMES = [KALMAN_FILTER_NAME] + cfg.EVALUATION_MODELS
    MODEL_NAMES = cfg.EVALUATION_MODELS

    PERIOD = cfg.PERIOD             # timesteps per trial

    for batch_folder_name in BATCH_FOLDER_NAMES:

        mse_dir = OUTPUT_ROOT / 'mse_by_position' / batch_folder_name
        mse_dir.mkdir(parents=True, exist_ok=True)

        for model_name in MODEL_NAMES:

            if model_name == KALMAN_FILTER_NAME:
                prediction_column = 'mu_pred_std'
            else:
                prediction_column = 'obs_mean'

            probabilities_dir = OUTPUT_ROOT / batch_folder_name / model_name / 'probabilities'
            mse_df = get_mse_by_position_per_sequence(probabilities_dir, prediction_column, PERIOD)
            mse_df.to_csv(mse_dir / f'{model_name}.csv', index=False)

            print(f"\n{batch_folder_name}, {model_name}, MSE averaged over the sequences of each condition")
            print(mse_df.groupby('condition', sort=False)[mse_df.columns[2:]].mean()
                  .to_string(float_format=lambda value: f'{value:.5f}'))
