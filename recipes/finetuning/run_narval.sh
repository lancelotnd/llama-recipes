#!/bin/bash
module load cuda python/3.10 arrow/14.0.0
export OMP_NUM_THREADS=4
source ~/llama_env/bin/activate
export TOKENIZERS_PARALLELISM=true
torchrun --nnodes=1 --nproc_per_node 4 --rdzv_id 69 --rdzv_backend c10d --rdzv_endpoint $1:29500 ./finetuning.py --model_name /home/lancelot/scratch/Meta-Llama-3-8B-Instruct --output_dir /home/lancelot/scratch/peft --dataset alpaca_dataset --enable_fsdp --use_peft --peft_method lora
