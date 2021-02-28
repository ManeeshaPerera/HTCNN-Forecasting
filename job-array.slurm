#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="lstm-train"

# Partition for the job:
#SBATCH --partition gpgpu
#SBATCH --qos gpgpumse

# Number of GPUs requested per node:
#SBATCH --gres=gpu:2

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-12:00:0

#Job array
#SBATCH --array=0-38

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
# module load Python/3.6.10-intel-2017.u2-GCC-6.2.0-CUDA10.1
# module load Tensorflow/2.1.0-intel-2017.u2-GCC-6.2.0-CUDA10.1-Python-3.6.10-GPU

source /usr/local/module/spartan_new.sh
module load fosscuda/2019b
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load python/3.7.4
module load tensorflow/2.3.1-python-3.7.4

# The job command(s):
python tune_and_run.py 3 ${SLURM_ARRAY_TASK_ID} > fc_results/fc_${SLURM_ARRAY_TASK_ID}.txt