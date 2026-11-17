"""Get the Pearson correlation between the observations and the predictions of the Kalman filter (KF) and of the
pRNN models, for each sequence, on the two batches of sequences of generate_task_sequences_gm_test.py
(gm_original, gm_corrected).

Reads the CSVs written by compute_predictions_gm_test.py in OUTPUT_ROOT/<batch>/<model>/probabilities/.

For one sequence, the correlation is computed over all its standard tones with a prediction, as in
compute_mse_by_position_gm_test.py: the tones at the deviant position of their trial are left out, and so are the
tones without a prediction (NaN). The predicted mean is
  KF:    mu_pred_std, the mean predicted by the filter of the standard state
  pRNN:  obs_mean, the mean predicted by the observation module

The correlation is not averaged over the sequences of a condition, so that the sequences of the two batches can be
compared pair by pair: a sequence of gm_original and the sequence of gm_corrected with the same name share their
random seed.

Outputs, for each batch and model, OUTPUT_ROOT/correlation/<batch>/<model>.csv:
one row per sequence, with the columns
  sequence      name of the sequence, e.g. 'sub-seq10'
  condition     name of its condition, e.g. 'cond1_tau16_mu-0.6_dev1'
  correlation   Pearson correlation between the observations and the predicted means
"""

import sys
from pathlib import Path

import pandas as pd

PREPROPARADIGM_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PREPROPARADIGM_DIR.parent
sys.path.append(str(WORKSPACE_ROOT / 'RNN_paradigm' / 'RNN' / 'model_analysis'))

import analysis_config as cfg
from test_sample_states_version.compute_mse_by_position_gm_test import KALMAN_FILTER_NAME, get_condition_name, get_sequence_name


def get_sequence_correlation(probabilities_df, prediction_column):
    """Pearson correlation of one sequence between the observations and the predicted means, over its standard
    tones with a prediction."""
    is_standard_tone_with_prediction = (probabilities_df['ctx'] == 0) & probabilities_df[prediction_column].notna()
    tones_df = probabilities_df[is_standard_tone_with_prediction]
    return tones_df['observation'].corr(tones_df[prediction_column])


def get_correlation_per_sequence(probabilities_dir, prediction_column):
    """DataFrame of the correlation of a model for each sequence.
    Columns: sequence, condition, correlation. Rows sorted by condition number, then sequence number.
    """
    probabilities_csvs = sorted(probabilities_dir.glob('*_probabilities.csv'))
    if len(probabilities_csvs) == 0:
        raise FileNotFoundError(f"No *_probabilities.csv in {probabilities_dir}")

    rows = []
    for probabilities_csv in probabilities_csvs:
        probabilities_df = pd.read_csv(probabilities_csv)
        rows.append({'sequence': get_sequence_name(probabilities_csv),
                     'condition': get_condition_name(probabilities_csv),
                     'correlation': get_sequence_correlation(probabilities_df, prediction_column)})
    correlation_df = pd.DataFrame(rows)

    # Conditions in the order of their number (cond1, cond2, ..., cond16), not alphabetical; same for sequences.
    correlation_df['condition_number'] = correlation_df['condition'].str.extract(r'cond(\d+)_', expand=False).astype(int)
    correlation_df['sequence_number'] = correlation_df['sequence'].str.removeprefix('sub-seq').astype(int)
    correlation_df = correlation_df.sort_values(['condition_number', 'sequence_number'])
    return correlation_df.drop(columns=['condition_number', 'sequence_number']).reset_index(drop=True)


if __name__ == "__main__":

    OUTPUT_ROOT = PREPROPARADIGM_DIR / 'predictions_gm_test'
    BATCH_FOLDER_NAMES = ['gm_original', 'gm_corrected']

    # Models of compute_predictions_gm_test.py: the KF and the pRNN models.
    # MODEL_NAMES = [KALMAN_FILTER_NAME] + cfg.EVALUATION_MODELS
    MODEL_NAMES = cfg.EVALUATION_MODELS

    for batch_folder_name in BATCH_FOLDER_NAMES:

        correlation_dir = OUTPUT_ROOT / 'correlation' / batch_folder_name
        correlation_dir.mkdir(parents=True, exist_ok=True)

        for model_name in MODEL_NAMES:

            if model_name == KALMAN_FILTER_NAME:
                prediction_column = 'mu_pred_std'
            else:
                prediction_column = 'obs_mean'

            probabilities_dir = OUTPUT_ROOT / batch_folder_name / model_name / 'probabilities'
            correlation_df = get_correlation_per_sequence(probabilities_dir, prediction_column)
            correlation_df.to_csv(correlation_dir / f'{model_name}.csv', index=False)

            print(f"\n{batch_folder_name}, {model_name}, correlation averaged over the sequences of each condition")
            print(correlation_df.groupby('condition', sort=False)['correlation'].mean()
                  .to_string(float_format=lambda value: f'{value:.5f}'))
