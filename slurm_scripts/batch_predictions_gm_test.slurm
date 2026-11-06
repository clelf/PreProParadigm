#!/bin/bash
#SBATCH --qos=xlong
#SBATCH --job-name=gm_test_kf
#SBATCH --nodes=1              # Use single node to avoid distributed complexity
#SBATCH --ntasks=1             # Single task (your Python script)
#SBATCH --cpus-per-task=32     # Number of KF runs in parallel (one sequence per CPU)
#SBATCH --output=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/gm_test_kf_%j.out
#SBATCH --error=/scratch/clevyfidel/Workspace/RNN_HierarchicalAuditory/RNN/outputs_terminal/gm_test_kf_%j.err
#SBATCH --mail-type=END,FAIL           # Notifications for job done or fail
#SBATCH --mail-user=c.levyfidel@bcbl.eu  # Email address for notifications
#SBATCH --mem=32G


# Load modules needed
module load Miniforge3

# Activate conda environment
conda activate /scratch/clevyfidel/conda_env/rnn2

# ============ CONFIGURABLE PARAMETERS ============
# Computing the Kalman filter (KF) predictions on the two batches of sequences of
# generate_task_sequences_gm_test.py (sequences_gm_test/gm_original and gm_corrected).
#
# The script needs, under /scratch/clevyfidel/Workspace:
# - PreProParadigm, with sequences_gm_test/
# - Kalman
# - RNN_HierarchicalAuditory (RNN_paradigm locally)
#
# Outputs are written to PreProParadigm/predictions_gm_test/<batch>/kalman_filter/.
# Existing CSVs newer than their sequence file are not recomputed.
#
# Cost: about 65 s per sequence on one core, 320 sequences: about 6 h on one CPU,
# divided by --cpus-per-task above.
# =================================================

# Navigate to the PreProParadigm directory
cd /scratch/clevyfidel/Workspace/PreProParadigm

# KF only: the pRNN predictions are computed locally (--models KF pRNN to run both).
/scratch/clevyfidel/conda_env/rnn2/bin/python compute_predictions_gm_test.py --models KF


# Verify Python version and activated environment
echo "Python version: $(python --version)"
echo "Which Python: $(which python)"
