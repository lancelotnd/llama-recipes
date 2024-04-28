#!/bin/bash
#Setup for Graham with 8 A100 GPU 2TB ram, max time is 3 hours
#SBATCH --account=def-kemme
#SBATCH --time=0-0:10
#SBATCH --job-name=test_finetune_llama3
#SBATCH --nodes=1
#SBATCH --mem=2048000M
#SBATCH --gpus-per-node=8
#SBATCH --output=/home/lancelot/logs/%N-%j.out


#Sending email when job status changes
#SBATCH --mail-user=lancelot.normand@mail.mcgill.ca
#SBATCH --mail-type=ALL

#Getting node list and IP of master node (head node)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

#Load the necessary modules
module load cuda python/3.10 arrow/14.0.0

export OMP_NUM_THREADS=4
srun ./run_graham1.sh $head_node_ip
