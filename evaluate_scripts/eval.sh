#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -o <error_file_name.out>
#SBATCH -e <error_file_name.err>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=1T
#SBATCH --time=6:00:00

conda activate newenv

CUDA_VISIBLE_DEVICES=0 nohup python -u evaluateBias_dexperts.py --expert_model eliolio/gpt2-finetuned-reddit-antibias --prompt_dir ../prompts/ --out_dir results/dexperts_gpt2_temp1_alpha1 > 0.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u evaluateBias_dexperts.py --base_model gpt2-medium --expert_model eliolio/gpt2-finetuned-reddit-antibias --prompt_dir ../prompts/ --out_dir results/dexperts_gpt2_med_temp1_alpha1 > 1.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u evaluateBias_dexperts.py --base_model gpt2 --prompt_dir ../prompts/ --out_dir results/dexperts_gpt2_antionly_temp1_alpha1 > 2.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u evaluateBias_dexperts.py --base_model gpt2-medium --prompt_dir ../prompts/ --out_dir results/dexperts_gpt2_med_antionly_temp1_alpha1 > 3.log &
