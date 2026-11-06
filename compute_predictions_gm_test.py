"""Get the predicted probabilities of the Kalman filter (KF) and of trained pRNN models on the two
batches of sequences of generate_task_sequences_gm_test.py, at every timestep of every sequence.

The batches (sequences_gm_test/gm_original and sequences_gm_test/gm_corrected) hold the same
sequences of the 16 conditions, sampled with the two versions of the state sampling of the
generative model (audit_gm.py: AuditGenerativeModel.sample_states and sample_states_corr). They
share their rules, deviant positions, cues and random draws, and differ by their states, so a
sequence file has the same name in both batches.

Outputs, for each batch, in OUTPUT_ROOT/<batch folder name>/:

  kalman_filter/probabilities/*_probabilities.csv
      One row per timestep t = 0 ... T-1. The observation of a row is y[timestep], and its
      predictions are made from y[0:timestep]. Columns:
        observation, ctx, dpos, rule, trial_n   from the sequence file
        mu_pred_std, sigma_pred_std        mean and STANDARD DEVIATION of the Gaussian predicted by
        mu_pred_dev, sigma_pred_dev        the filter of the standard / of the deviant state
        lik_obs_std, lik_obs_dev           probability density of the observation under each of
                                           these Gaussians
        A_{state}_{ij}, Q_{state}_{ij}, H_{state}_{j}, R_{state}
                                           the fitted matrices of the filter of each state that
                                           made the prediction of the row (state = std or dev).
                                           R is a VARIANCE: compare it with sigma_r squared.
        tau_hat_{state}                    time constant implied by the A of the state (its
                                           non-unit eigenvalue, see model_RTs._tau_from_A); weakly
                                           identified when tau is long relative to the sequence.
                                           There is no estimate of the limit value: it lives in the
                                           state of the filter, not in A, Q, H and R.
        lim_std, d, tau_std, sigma_r       parameters of the sequence
      The values are NaN while a state has too few observations to be fitted (see
      Kalman.kalman.MIN_OBS_FOR_EM); they are not filled in.
      NOTE the pRNN 'obs_var' column is a VARIANCE, the KF 'sigma_pred_*' columns are STANDARD
      DEVIATIONS.

  <pRNN model name>/probabilities/*_probabilities.csv
  <pRNN model name>/probabilities_deviant/*_probabilities_deviant.csv
      Written by run_exp_trials_pipeline.write_proba_and_activ_csvs (RNN_paradigm), see there for
      the columns. One row per timestep t = 1 ... T-1, with 'timestep' meaning the same as above.

The KF is run as in model_RTs.compute_likelihoods_at_deviants and compute_kf_predictions.py: one
filter per state, fitted by EM on the observations of its state, with the true state of each tone
as input, and its observation noise R fixed at the sequence's own sigma_r squared. The predictions
of every timestep are kept, instead of only the likelihoods at the deviants.

Cost: the KF is O(T**2) (EM is refitted at every timestep), about 65 s per sequence of T = 480
on one core, so it is the slow part (320 sequences for the two batches). The pRNN models take
seconds per batch. PREDICTION_MODELS (in __main__) chooses which of them are run.

On the SLURM server (slurm_scripts/batch_predictions_gm_test.slurm), the models are given on the
command line instead, which replaces PREDICTION_MODELS:
    python compute_predictions_gm_test.py --models KF
and the KF runs on as many workers as the CPUs given to the job (SLURM_CPUS_PER_TASK).
"""

# Pin BLAS to one thread per KF worker, BEFORE numpy is imported (below). The KF is parallelized
# over sequences, and BLAS threads spawned inside each worker spin-wait on the contended cores
# instead of doing work (see BehavModeling/compute_kf_predictions.py). The pRNN forward passes in
# the main process run on one thread too, which is fast enough for these small networks.
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# The RNN analysis modules are in RNN_paradigm, which is not a package; Kalman/ is found from the workspace root.
PREPROPARADIGM_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PREPROPARADIGM_DIR.parent
# On the SLURM server (/scratch/clevyfidel/Workspace), RNN_paradigm is cloned under the name of
# its repository, RNN_HierarchicalAuditory.
RNN_REPOSITORY_DIR = WORKSPACE_ROOT / 'RNN_paradigm'
if not RNN_REPOSITORY_DIR.exists():
    RNN_REPOSITORY_DIR = WORKSPACE_ROOT / 'RNN_HierarchicalAuditory'
sys.path.append(str(PREPROPARADIGM_DIR))
sys.path.append(str(RNN_REPOSITORY_DIR / 'RNN' / 'model_analysis'))
sys.path.append(str(WORKSPACE_ROOT))

import analysis_config as cfg
from analysis_core import ModelInfo, find_trial_files, load_model, outputs_up_to_date, sequence_csv_path
from run_exp_trials_pipeline import analysis_output_kinds_to_write, write_proba_and_activ_csvs
from model_RTs import prepare_trials_data, _kf_param_columns
from Kalman.kalman import contexts_to_probabilities, kalman_online_fit_predict_multicontext, likelihood_observation


# Name of the folder of the KF outputs, next to the folders of the pRNN models.
KALMAN_FILTER_NAME = 'kalman_filter'

# Number of EM iterations of the KF at each timestep; the value of model_RTs.compute_likelihoods_at_deviants.
KALMAN_FILTER_N_EM_ITERATIONS = 5


def write_kalman_filter_probabilities(task):
    """Pool worker: run the KF on one sequence file and write its CSV of probabilities.

    `task` is the list [trial_file, output_root, overwrite]. The CSV is not rewritten when it is
    newer than the sequence file, unless `overwrite`. Returns a line for the log.
    """
    trial_file, output_root, overwrite = task

    probabilities_csv = sequence_csv_path(output_root, KALMAN_FILTER_NAME, 'probabilities',
                                          trial_file.stem)
    if not overwrite and outputs_up_to_date([probabilities_csv], [trial_file]):
        return f'up to date, not rerun: {output_root.name}/{trial_file.name}'
    probabilities_csv.parent.mkdir(parents=True, exist_ok=True)

    trials_df = prepare_trials_data(str(trial_file))
    if trials_df['run_n'].nunique() != 1:
        raise ValueError(f"{trial_file.name}: expected a single run, got {trials_df['run_n'].unique()}")

    # The observation noise the KF is given is the true one of the sequence (R = sigma_r squared).
    sigma_r_values = trials_df['sigma_r'].unique()
    if len(sigma_r_values) != 1:
        raise ValueError(f"{trial_file.name}: expected a single sigma_r, got {sigma_r_values}")
    sigma_r = float(sigma_r_values[0])

    observations = trials_df['observation']
    ctx_probabilities = contexts_to_probabilities(trials_df['contexts'], n_ctx=2)

    # pykalman is noisy about masked arrays and EM convergence, and importing analysis_core makes
    # warnings errors (pipeline_core_v2): ignore them for the KF run only.
    # per_ctx_mu_pred: (T, 2) means; per_ctx_var_pred: (T, 2) VARIANCES; columns are (std, dev).
    # kf_params: fitted A, Q, H, R of each state at each timestep.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, _, per_ctx_mu_pred, per_ctx_var_pred, kf_params = kalman_online_fit_predict_multicontext(
            observations, ctx_probabilities, n_iter=KALMAN_FILTER_N_EM_ITERATIONS,
            return_per_ctx=True, return_params=True, observation_noise=sigma_r ** 2)

    probabilities_df = pd.DataFrame({
        'timestep':    np.arange(len(trials_df)),
        'observation': observations.to_numpy(),
        'ctx':         trials_df['contexts'].astype(int).to_numpy(),
        'dpos':        trials_df['dpos'].to_numpy(),
        'rule':        trials_df['rule'].to_numpy(),
        'trial_n':     trials_df['trial_no'].to_numpy(),
    })
    for state_index, state in enumerate(['std', 'dev']):
        probabilities_df[f'mu_pred_{state}'] = per_ctx_mu_pred[:, state_index]
        probabilities_df[f'sigma_pred_{state}'] = np.sqrt(per_ctx_var_pred[:, state_index])
    for state_index, state in enumerate(['std', 'dev']):
        # likelihood_observation is parameterized by the VARIANCE, not by the standard deviation.
        probabilities_df[f'lik_obs_{state}'] = likelihood_observation(
            y=observations.to_numpy(), mu=per_ctx_mu_pred[:, state_index],
            sigma=per_ctx_var_pred[:, state_index])
    for column_name, values in _kf_param_columns(kf_params).items():
        probabilities_df[column_name] = values
    probabilities_df['lim_std'] = trials_df['lim_std'].iloc[0]
    probabilities_df['d'] = trials_df['d'].iloc[0]
    probabilities_df['tau_std'] = trials_df['tau_std'].iloc[0]
    probabilities_df['sigma_r'] = sigma_r

    probabilities_df.to_csv(probabilities_csv, index=False)
    return f'written: {output_root.name}/{trial_file.name}'


if __name__ == '__main__':

    # ------------------------- SETTINGS (edit these) -------------------------
    # The batches, as written by generate_task_sequences_gm_test.py (output_dir and
    # batch_folder_names there). The name of a batch folder names the batch in the outputs.
    SEQUENCES_DIR = PREPROPARADIGM_DIR / 'sequences_gm_test'
    BATCH_FOLDER_NAMES = ['gm_original', 'gm_corrected']

    # Each batch writes to OUTPUT_ROOT/<batch folder name>/.
    OUTPUT_ROOT = PREPROPARADIGM_DIR / 'predictions_gm_test'

    # Models to compute predictions from: ['KF', 'pRNN'], ['KF'] or ['pRNN'].
    # Replaced by the models given after --models on the command line (SLURM job).
    # PREDICTION_MODELS = ['KF', 'pRNN']
    PREDICTION_MODELS = ['pRNN']


    # pRNN models run on every batch: names of the folders of MODEL_DIR.
    MODEL_NAMES = cfg.EVALUATION_MODELS
    MODEL_DIR = cfg.TRAINING_RESULTS_DIR

    # Existing CSVs are reused when they are newer than what they come from (the sequence file,
    # and the model's weights for the pRNN CSVs). Set True to rewrite them all.
    OVERWRITE = False

    # Number of KF runs in parallel. The workstation has 8 physical cores (16 CPUs with
    # hyperthreading): more workers than cores makes every run slower.
    # In a SLURM job, it is replaced by the number of CPUs of the job (--cpus-per-task).
    N_JOBS = 8

    CUE_SEED = cfg.CUE_SEED         # cue re-encoding; keep it fixed across runs
    CHUNK_SIZE = cfg.CHUNK_SIZE     # sequences per batched forward pass
    PERIOD = cfg.PERIOD             # timesteps per trial

    # Deviant-only probabilities file: deviant rows plus the immediate next stimulus (True),
    # or the deviant rows alone (False).
    INCLUDE_NEXT_STIMULUS = True
    # Also write the prior (first-call) readout of every module to the probability CSVs, as
    # prior_-prefixed columns. Not needed for the predictions themselves.
    INCLUDE_PRIOR = False
    # -------------------------------------------------------------------------

    if '--models' in sys.argv:
        PREDICTION_MODELS = sys.argv[sys.argv.index('--models') + 1:]
    for model_type in PREDICTION_MODELS:
        if model_type not in ['KF', 'pRNN']:
            raise ValueError(f"Unknown model '{model_type}' in PREDICTION_MODELS: use 'KF' and/or 'pRNN'")

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        N_JOBS = int(os.environ['SLURM_CPUS_PER_TASK'])

    trial_files_per_batch = []
    for batch_folder_name in BATCH_FOLDER_NAMES:
        trial_files = find_trial_files(SEQUENCES_DIR / batch_folder_name)
        if not trial_files:
            raise FileNotFoundError(f"No .csv or .txt sequence files in {SEQUENCES_DIR / batch_folder_name}")
        print(f"Found {len(trial_files)} sequence files in {SEQUENCES_DIR / batch_folder_name}")
        trial_files_per_batch.append(trial_files)

    # The KF runs first: the worker processes are forked before the main process runs any
    # torch forward pass.
    if 'KF' in PREDICTION_MODELS:
        kalman_filter_tasks = []
        for batch_folder_name, trial_files in zip(BATCH_FOLDER_NAMES, trial_files_per_batch):
            for trial_file in trial_files:
                kalman_filter_tasks.append([trial_file, OUTPUT_ROOT / batch_folder_name, OVERWRITE])

        print(f"\nKalman filter: {len(kalman_filter_tasks)} sequences on {N_JOBS} workers")
        with Pool(N_JOBS) as pool:
            for i, status in enumerate(pool.imap_unordered(write_kalman_filter_probabilities,
                                                           kalman_filter_tasks), start=1):
                print(f"[{i}/{len(kalman_filter_tasks)}] {status}", flush=True)

    if 'pRNN' in PREDICTION_MODELS:
        # run_exp_trials_pipeline.write_proba_and_activ_csvs writes the two probabilities CSVs together.
        analysis_output_kinds = ['probabilities', 'probabilities_deviant']

        for batch_folder_name, trial_files in zip(BATCH_FOLDER_NAMES, trial_files_per_batch):
            batch_output_root = OUTPUT_ROOT / batch_folder_name
            for i, model_name in enumerate(MODEL_NAMES, start=1):
                print(f"\n{'=' * 79}\n{batch_folder_name}: [{i}/{len(MODEL_NAMES)}] {model_name}\n{'=' * 79}")
                # Only the config is read here; the weights are loaded when a forward pass is due.
                info = ModelInfo.from_path(MODEL_DIR / model_name)

                files_missing_csvs = [f for f in trial_files
                                      if analysis_output_kinds_to_write(f, analysis_output_kinds,
                                                                        batch_output_root, model_name,
                                                                        info.weights_path, OVERWRITE)]
                print(f"{len(files_missing_csvs)}/{len(trial_files)} sequence file(s) need CSVs written")
                if files_missing_csvs:
                    model = load_model(info)
                    model.eval()
                    print(f"Loaded model: {model_name}")
                    write_proba_and_activ_csvs(model, info, files_missing_csvs, analysis_output_kinds,
                                               batch_output_root, model_name, PERIOD, CUE_SEED,
                                               CHUNK_SIZE, INCLUDE_PRIOR, INCLUDE_NEXT_STIMULUS,
                                               OVERWRITE)
