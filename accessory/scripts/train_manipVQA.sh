#!/bin/bash

export PATH=/mnt/lustre/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-11.8/lib64:$LD_LIBRARY_PATH

# setting up GCC environment
export PATH=/mnt/lustre/share/gcc/gcc-7.3.0/bin:/mnt/lustre/share/gcc/gcc-7.3.0/lib64:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gcc-7.3.0/lib/:$LD_LIBRARY_PATH


pretrained_path='/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/params/dialog_llava1.5noboxRef4Cwbflicker30k_llamaEns_13B'
pretrained_type=consolidated

llama_path="/mnt/petrelfs/share_data/gaopeng/llama-accessory-shikra/params/llama2"
llama_config="$llama_path"/13B/params.json

# llama_config="$2"
# tokenizer_path="$3"
tokenizer_path=/mnt/petrelfs/share_data/llm_llama/tokenizer.model
data_config=configs/data/finetune/mm/manip.yaml

data_parallel=sdp
model_parallel=2

exp_name=finetune/mm/manip_all_part_all_tasks
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

srun -p alpha_vl --gres=gpu:8 --cpus-per-task 12 -n8 --ntasks-per-node=8 --quotatype=spot \
python -u main_finetune.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.03 \
--batch_size 4 --accum_iter 4 --num_workers 4 \
--max_words 2048 \
--lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_ens --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config --dialog \
--image_transform padded_resize \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"