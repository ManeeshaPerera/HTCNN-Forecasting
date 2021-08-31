#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="ClusterModel"

# Partition for the job:
#SBATCH --partition gpgpu
#SBATCH --qos gpgpumse

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=05-00:00:0

#Job array
#SBATCH --array=29,4,0,21,20,14,27,18,28,10,3,26

# Send yourself an email when the job:
#SBATCH --mail-user=pereramg@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END

source /usr/local/module/spartan_new.sh
module load fosscuda/2019b
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load python/3.7.4
module load tensorflow/2.3.1-python-3.7.4

# The job command(s):
python run_global_approach_for_swis_clustering_approach.py 1 0 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 1 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 2 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 3 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 4 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 5 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 6 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 7 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 8 ${SLURM_ARRAY_TASK_ID}
python run_global_approach_for_swis_clustering_approach.py 1 9 ${SLURM_ARRAY_TASK_ID}

