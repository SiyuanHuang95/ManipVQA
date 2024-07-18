# ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Modal Large Language Models

While the integration of Multimodal Large Language Models (MLLMs) with robotic systems has significantly improved robots' ability to understand and execute natural language instructions, however, their performance in manipulation tasks remains hampered by a lack of robotics-specific knowledge. Conventional MLLMs are typically trained on generic image-text pairs, leaving them deficient in understanding affordances and physical concepts crucial for manipulation. To address this gap, we propose ManipVQA, a novel framework that infuses MLLMs with manipulation-centric knowledge through a Visual Question-Answering (VQA) format. This approach encompasses tool detection, affordance recognition, and a broader understanding of physical concepts. We curated a diverse dataset of images depicting interactive objects, to challenge robotic understanding in tool detection, affordance prediction, and physical concept comprehension. To seamlessly integrate this robotics-specific knowledge with the inherent vision-reasoning capabilities of MLLMs, we leverage a unified VQA format and devise a fine-tuning strategy. This strategy preserves the original vision-reasoning abilities while incorporating the newly acquired robotic insights. Empirical evaluations conducted in robotic simulators and across various vision task benchmarks demonstrate the robust performance of ManipVQA.

## An overview of ManipVQA
![ManipVQA - Method](figures/Method-Figure.png)
We created an extensive vision-language dataset by combining existing resources and expanding affordance grounding tasks using ChatGPT. To maintain consistency with existing VQA datasets, we structured our dataset in a similar VQA format. Utilizing this curated dataset, we then fine-tuned an MLLM. Once integrated with a heuristic policy, the enhanced MLLM is capable of performing a broad array of tasks, notably including complex manipulation tasks.

## News

We have extended the 2D ManipVQA to the 3D ones, with more powerful ability on the articulated joints understanding and action grounding. Check it on [GitHub](https://github.com/changhaonan/A3VLM)

## How to use

### Dataset Creation

We provide the sample code to convert the original HANDAL dataset to the VQA-format, please refer to the *dataset/handal_process.py*. The samples in *dataset/handal_grounding_tasks.py* are only the demostrations which are obtained from ChatGPT, the prompts we used could be found in our paper, you are free to add more.

Please:

1. Use the visualization tool to check the requried format.

2. Update the dataset path in *accessory/configs/data/finetune/mm/manip_mix_all_dataset_r2_with_complete.yaml*


### Finetuning

1. Environment setup is following the [Llama2-Accesory](https://github.com/Alpha-VLLM/LLaMA2-Accessory)

2. Obtained the pre-trained model from [HuggingFace](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-1k)

3. Use the script in *accessory/scripts/train_manipVQA.sh*, the default setting is using 8 A100 for the finetuning.

### Evaluation
1. We provide the basic evaluation script in *accessory/eval_manip.py* and *accessory/get_ap_tool.py*

### Inference with 7B
When you have limited GPU (even with quant, still cannot afford), you can check our 7B model with InternLM as the language backbone Huggingface[7B-Ckpts](https://huggingface.co/SiyuanH/ManipVQA-7B). Remember set the "model_type" to be "internlm_ems5_light" in the script. And modify the lines 41 with 47 in accessory/model/LLM/inmternlm_ems5_light.py and point to the actual path.

Also, due to the original internlm loading problem, you need the access to the InternLM ckpt when inference.


### Demo data

We provide some demo-usage data in the folder in the path "accessory/demo_data", you can create a similar json as the file in folder and used your data for training.


## Links
Paper: https://arxiv.org/abs/2403.11289

HuggingFace Ckpt: [Cktps](https://huggingface.co/SiyuanH/ManipVQA/tree/main)

HuggingFace 7B-Ckpt [7B-Ckpts](https://huggingface.co/SiyuanH/ManipVQA-7B)

Dataset: [ToUpdate]


## Acknowledgement

Llama2-Accesory, SAM-HQ, Affordance-LLM, etc
