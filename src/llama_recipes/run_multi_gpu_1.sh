export OMP_NUM_THREADS=40
source /home/lancelot/torchenv/bin/activate
torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp --model_name /home/lancelot/scratch/Meta-Llama-3-8B-Instruct --use_peft --peft_method lora --output_dir /home/lancelot/scratch/peft --use_fast_kernels

