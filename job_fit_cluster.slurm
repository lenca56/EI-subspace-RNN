#!/bin/bash
#SBATCH --job-name=fit_EI_subspace_RNN_fixed_N_no_Dale     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=07:00:59         # total run time limit (HH:MM:SS)
#SBATCH --array=0-149             # job array with index values 0, 1, ..., 10
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=cuturela@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2023.3
conda activate dglmhmm

python fit_cluster_RNN_fixed_N_no_Dale.py