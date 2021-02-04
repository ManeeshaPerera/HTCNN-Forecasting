#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="lstm-train"

# Partition for the job:
#SBATCH --partition gpgpu
#SBATCH --qos gpgpumse

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-12:00:0

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
module load Python/3.6.10-intel-2017.u2-GCC-6.2.0-CUDA10.1
module load Tensorflow/2.1.0-intel-2017.u2-GCC-6.2.0-CUDA10.1-Python-3.6.10-GPU

# The job command(s):
python run.py 'stackedRNN_region' 3 > results/stackedRNN_region.txt