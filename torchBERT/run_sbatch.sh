#!/bin/bash
## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=mlm_task
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=./mlm-task-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=./mlm-task-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2500
#SBATCH --mem-per-cpu=5120
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task 80

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

#srun --label --nodelist=learnfair[0721] --ntasks-per-node=1 --time=2500 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python distributed_mlm_task.py --world_size 8 --parallel DDP --seed 5431916812 --epochs 100 --emsize 768 --nhid 3072  --nlayers 12 --nhead 12 --save-vocab squad_30k_vocab_cls_sep.pt --dataset EnWik9 --lr 6  --bptt 128  --batch_size 56 --clip 0.1 --log-interval 600
#srun --label --nodelist=learnfair[0721] --ntasks-per-node=1 --time=2500 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python ns_task.py --world_size 8 --parallel DDP --bert-model bert_model.pt --epochs 100 --save-vocab squad_30k_vocab_cls_sep.pt --seed 312216194  --lr 6.0 --bptt 128 --batch_size 72 --clip 0.1 --log-interval 600

#python distributed_mlm_task.py --world_size 8 --seed 2514683 --epochs 100 --emsize 128 --nhid 512  --nlayers 2 --nhead 8 --save-vocab squad_30k_vocab_cls_sep.pt --dataset EnWik9  --lr 2  --bptt 128  --log-interval 600  --batch_size 56

#learnfair0778 learnfair1757 learnfair1614

srun --label --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python distributed_mlm_task.py --world_size 8 --parallel DDP --seed 5431916812 --epochs 100 --emsize 768 --nhid 3072  --nlayers 12 --nhead 12 --save-vocab squad_30k_vocab_cls_sep.pt --dataset EnWik9 --lr 6  --bptt 128  --batch_size 56 --clip 0.1 --log-interval 600 
#srun --label --nodelist=learnfair[0721] --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty ./run_bert.sh 
