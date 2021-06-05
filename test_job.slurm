#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="test_model"

# Partition for the job:
#SBATCH --partition gpgpu
#SBATCH --qos gpgpumse

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-01:00:0

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

source /usr/local/module/spartan_new.sh
module load fosscuda/2019b
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load python/3.7.4

# The job command(s):
python test_slurm1.py > test_slurm1.txt
python test_slurm2.py > test_slurm2.txt