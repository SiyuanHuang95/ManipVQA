remove_space=True
pretrained_path=/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/output/finetune/mm/mix_all_ens5v2_r2_long_local/epoch0
addition_flag=test

PORT=6066
# Generate a random number between 10 and 100
random_number=$((RANDOM % 91 + 10))

# Add the random number to the original PORT value
new_port=$((PORT + random_number))

MODEL=llama_ens5 
config=resample_args_query.json

llama_path="/mnt/petrelfs/huangsiyuan/data/llama2"
llama_config="$llama_path"/13B/params.json
tokenizer_path=/mnt/petrelfs/share_data/llm_llama/tokenizer.model

export OMP_NUM_THREADS=8
export NCCL_LL_THRESHOLD=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

vqa_data=demo
# vqa_data=handal_rec
# for vqa_data in DeepForm InfographicsVQA KleisterCharity TabFact WikiTableQuestions
srun -p llmeval2 -N 1 --gres=gpu:2 --quotatype=auto  --job-name=ade \
torchrun --nproc-per-node=2 --master_port=${new_port} eval_manip.py \
    --llama_type ${MODEL} \
    --llama_config ${llama_config} \
    --tokenizer_path ${tokenizer_path} \
    --pretrained_path ${pretrained_path} \
    --dataset ${vqa_data} \
    --batch_size 16 --input_size 448 \
    --model_parallel_size 2 --addition_flag ${addition_flag} --remove_space \

