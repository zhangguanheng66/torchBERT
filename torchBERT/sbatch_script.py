#!/bin/bash
## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=sample
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/private/home/zhangguanheng/sample-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/private/home/zhangguanheng/sample-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes

## number of tasks per node

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean

# Load what we need
module purge
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
source activate /private/home/zhangguanheng/anaconda3/envs/slurm_envir


### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each node. However I am using
### the wrapper.sh example I showed before so that each task only
### sees one GPU

srun --label --ntasks-per-node=1 --time=600 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python distributed_mlm_task.py --world_size 8 --seed 2514683 --epochs 100 --emsize 128 --nhid 512  --nlayers 2 --nhead 8 --save-vocab squad_30k_vocab_cls_sep.pt --dataset EnWik9  --lr 2  --bptt 128  --log-interval 600  --batch_size 56
