#!/bin/bash
#SBATCH -p cs
#SBATCH -N 1
#SBATCH --mem=200000
#SBATCH --account cs
#SBATCH --qos csstaff 
#SBATCH -t 20:50:00
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/mygcd/slurm_output/run_aircraft_baseline_seed0.txt
#SBATCH -c 5


module load gcc/gcc-10.2.0
module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'aircraft' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name 'aircraft_simgcd' \
    --exp_root 'uon_hpc' \
    --seed_num  0