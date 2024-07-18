import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from model.meta import MetaModel
from data.conversation.lib import conv_templates
from accessory.data.conversation import default_conversation


import argparse
import torch
import torch.distributed as dist
import gradio as gr

from PIL import Image
import PIL.ImageFile as ImageFile

# Increase the limit for decompression
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb limit

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from util.tensor_parallel import load_tensor_parallel_model_list
# from util.quant import quantize
from data.bbox_util import Expand2square, denorm_bboxes
import re
import json
import numpy as np
import cv2

global_config = {
    'temperature': 0.1,
    'top_p': 0.75
}

partnet_dataset_root = '/mnt/petrelfs/huangsiyuan/data/ManipVQA2/jsons_vqa_tasks_fix_angle/'
HANDAL_DATASET = ["handal_compelet_rec", "sapien_action_1225_996", "sapien_action_1119", "sapien_action_1116", "handal_grounding", "handal_rec", 'handal_grounding_test', 'handal_rec_test', 'genome_rec_new', 'coco_rec_new', 'part_afford_rec', 'part_afford_grounding', 'ego_object_transpancy', 'ego_object_liquid', 'ego_object_closed', 'ego_object_action', 'sapien_action']
PARTNET_DATASET = ['partnet_det_parts', 'partnet_grounding', "partnet_joint_rec", 'partnet_link_rec',  "partnet_joint_reg"]
ADE_DATASET = ["ade_unseen_general", "ade_unseen_robotic", "ade_seen_general", "ade_seen_robotic"]

ds_collections = {    
    # ADE20K
    "ade_unseen_general": {
        "test": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/unseen_general_affordance_rec.json",
        "train": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/unseen_general_affordance_rec.json",
        "max_new_tokens": 32,
        "use_answer_extractor": True,
    },
    "ade_unseen_robotic" : {
        "test": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/unseen_robotic_affordance_rec.json",
        "train": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/unseen_robotic_affordance_rec.json",
        "max_new_tokens": 32,
        "use_answer_extractor": True,
    }, 

    "ade_seen_general": {
        "test": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/seen_general_affordance_rec.json",
        "train": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/seen_general_affordance_rec.json",
        "max_new_tokens": 32,
        "use_answer_extractor": True,
    },
    "ade_seen_robotic" : {
        "test": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/seen_robotic_affordance_rec.json",
        "train": "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/util/seen_robotic_affordance_rec.json",
        "max_new_tokens": 32,
        "use_answer_extractor": True,
    }, 
    # HANDAL
    'handal_grounding': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/train_grounding_HANDAL.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/test_grounding_normal_HANDAL.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    'handal_rec': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/train_rec_HANDAL.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/test_rec_normal_HANDAL.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    'handal_compelet_rec': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/train_complete_obj_rec_normal_HANDAL.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/HANDAL/dataset_round_2/test_complete_obj_rec_normal_HANDAL.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    }, 
    'ego_object_transpancy': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_transpancy_EgoObjects_round_2.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_transpancy_EgoObjects_round_2.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    'ego_object_liquid': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_liquid_EgoObjects_round_2.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_liquid_EgoObjects_round_2.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    'ego_object_closed': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_closed_EgoObjects_round_2.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_reg_closed_EgoObjects_round_2.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    'ego_object_action': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_rec_EgoObject_action_round_2.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/EgoObjects/test/test_rec_EgoObject_action_round_2.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    'sapien_action': {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    "sapien_action_1116" :{
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1116.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1116.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    "sapien_action_1119" : {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1119.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1119.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    },
    
    "sapien_action_1225_996" : {
        'train': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1225.json',
        'test': '/mnt/petrelfs/huangsiyuan/data/ManipVQA/sapien/test_rec_sapien_1225.json',
        'max_new_tokens': 30,
        'use_answer_extractor': True,
    }
}

def collate_fn(batches, tokenizer):
    questions = [_['question'] for _ in batches]
    if 'question_id' in batches[0]:
        question_ids = [_['question_id'] for _ in batches]
    else:
        question_ids = [idx for idx, _ in enumerate(questions)]
        
    annotations = [_['annotation'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    image_paths = [_['image_path'] for _ in batches]

    # input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

    return input_image, question_ids, questions, annotations, image_paths


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:"
    ),
}

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x * 255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result


def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t
import torchvision.transforms as transforms


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, img_root, img_size=224, remove_space=False, sampled_num=5000):

        with open(test, "r") as f:
            self.test = json.load(f)
        if "train" in test or len(self.test) > sampled_num:
            # first shuffle, then sample
            random.shuffle(self.test)
            sampled_num = min(len(self.test), sampled_num)
            self.test = self.test[:sampled_num]
        # self.test = open(test).readlines()
        self.prompt = prompt
        self.img_root = img_root
        self.print = False
        self.transform_val = T_padded_resize(img_size)
        self.remove_space = remove_space

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        # print(self.test[idx].strip())
        data = self.test[idx]
        # print(data)
        # image_path, question, question_id, annotation, ocr_token = data['image'], data[
        #     'question'], data['question_id'], data.get('answer', None), data.get('ocr_tokens', '')
        
        image_path = data["image"]
        question = data["conversations"][0]["value"]
        question_id = idx
        annotation = data["conversations"][1]["value"]
        ocr_token = ""
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError as e:
            tmp_idx = random.randint(0, len(self.test) - 1)
            print(f"opening {image_path} failed with error {e} and randomly sample a new one")
            data = json.loads(self.test[tmp_idx].strip())
            image_path, _, _, _, _ = data['image'], data[
                'question'], data['question_id'], data.get('answer', None), data.get('ocr_tokens', '')
            image = Image.open(image_path).convert('RGB')
            question_id = 99999 # fake question id to indicate this is a fake image
  
        image = self.transform_val(image).unsqueeze(0)
        if ocr_token:
            question = question + '\nReference OCR token: ' + ' '.join(ocr_token[:20])
            if not self.print:
                print(f'Using OCR tokens example:{question}')
                self.print = True
        
        conv = default_conversation()
        conv.load_qas([[question, None]])
        prompt = conv.get_prompt()

        # conv = conv_templates["v1"].copy()
        # conv.append_message(conv.roles[0],
        #                     question + f"{self.prompt}\n")
        # # conv.append_message(conv.roles[0], prompt)
        # conv.append_message(conv.roles[1], "")
        # question = conv.get_prompt()
        
        if self.remove_space:
            question = question.replace("###Assistant: ", "###Assistant:")

        return {
            'question': prompt,
            'question_id': question_id,
            'annotation': annotation,
            'image': image,
            "image_path": image_path
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def normalize_number(x):
    if x > 100:
        return x / 1000
    elif x > 10:
        return x / 100
    elif x >= 1:
        return x / 10
    else:
        return x

def format_bounding_box(answer):
    # Remove any non-numeric and non-comma characters, clean extra whitespace
    cleaned_answer = re.sub(r'[^\d,]', '', answer.replace(" ", ""))

    # Function to insert dot before the last three digits of a number
    def insert_dot(match):
        number = match.group(0)
        return number[:-3] + '.' + number[-3:]
    
    # Apply the function to all numbers in the string
    formatted_answer = re.sub(r'\d{4,}', insert_dot, cleaned_answer)
    
    # Split into individual numbers and convert to float, assuming they are now correctly formatted
    bbox = [float(n) for n in formatted_answer.split(',') if n]
    bbox = [normalize_number(x) for x in bbox]
    return bbox

if __name__ == '__main__':

    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_type', default='llama_ens5', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="./data/nocaps/images",
                            help='path to tokenizer.model')
        parser.add_argument('--annotation_path', type=str, default="./data/nocaps/nocap_val.json",
                            help='path to tokenizer.model')

        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')

        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        # parser.add_argument('--quant', action="store_true", default=False,
        #                     help="enable quantization")
        parser.add_argument('--dataset', default='vqav2_val', type=str)
        parser.add_argument("--input_size", type=int, default=224)
        parser.add_argument('--ocr_question', action='store_true')
        parser.add_argument('--prompt', default='vqav2_val', type=str)
        parser.add_argument('--addition_flag', default=None, type=str)
        parser.add_argument("--remove_space", action="store_true", default=False)

        return parser


    args = get_args_parser().parse_args()
    add_flag = args.addition_flag
    remove_space = args.remove_space
    
    # define the model
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
    # if args.input_size != 224:
    #     print('rescale PE')
    #     model.llma.scale_pos_embed()
    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)
    tokenizer = model.tokenizer
    # print("Model = %s" % str(model))
    model.bfloat16().cuda()
    # model enabled done
    
    
    dataset_name = args.dataset
    if dataset_name == "all":
        dataset_names = list(ds_collections.keys())
    elif dataset_name == "PARTNET_DATASET":
        dataset_names = PARTNET_DATASET
    elif dataset_name == "ADE_DATASET":
        dataset_names = ADE_DATASET
    elif dataset_name == "demo":
        dataset_names = ["demo1", "demo2", "demo3"]
    else:
        assert dataset_name in HANDAL_DATASET
        dataset_names = [dataset_name]
    
    for dataset_name in dataset_names:
        save_path = f'vqa_logs/{add_flag}'
        os.makedirs(save_path, exist_ok=True)
        results_file = f'{save_path}/{dataset_name}.json'
        if os.path.exists(results_file):
            print(f"skip {dataset_name} since already exists")
            continue
        
        
        print(f"evaluating on {dataset_name}")
        log_path = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
            if f'dataset config: {dataset_name}' in content:
                exit(0)
        prompt = ''

        if 'prompt' in ds_collections[dataset_name]:
            prompt = ds_collections[dataset_name]['prompt']
        else:
            prompt = prompt
        random.seed(args.seed)
        dataset = VQADataset(
            train=ds_collections[dataset_name]['train'],
            test=ds_collections[dataset_name]['test'],
            img_root=args.img_root,
            # tokenizer=tokenizer,
            prompt=prompt,
            img_size=args.input_size,
            remove_space=remove_space,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            # sampler=InferenceSampler(len(dataset)),
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        max_gen_len = ds_collections[dataset_name]['max_new_tokens']
        gen_t = global_config['temperature']
        top_p = global_config['top_p']
        answer_extractor = ds_collections[dataset_name].get('use_answer_extractor', False)
        failed_tasks = []
        with torch.no_grad():
            for image, question_ids, _prompt, annotations, image_paths in tqdm(dataloader):
                if dist.get_rank() == 0:
                    dist.barrier()
                    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])

                    image = image.cuda()
                    # print(f'\ninput: {_prompt[0]}\n')
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

                    for question_id, answer, annotation, question, image_path in zip(question_ids, results,
                                                            annotations, _prompt, image_paths):
                        if answer_extractor:
                            # remove symbol
                            answer = answer.split('###')[0]
                            answer = answer.replace('.', '').strip()
                            if len(answer.strip().split(' ')) > 0:
                                ans_pattern = ['answer is']
                                for a_p in ans_pattern:
                                    if a_p in answer:
                                        try:
                                            answer_extracted = re.findall(f'{a_p}[ ]*[a-zA-Z0-9.]+', answer)[0]
                                            answer_extracted = re.sub(a_p, '', answer_extracted)
                                            answer = answer_extracted.strip()
                                        except Exception as e:
                                            print(e)
                                            print(answer)
                                            answer = answer.strip()

                        failed_flag = False
                        dt_bbox = format_bounding_box(answer)
                        if len(dt_bbox) != 4:
                            failed_flag = True
                        elif dt_bbox[0] > dt_bbox[2] or dt_bbox[1] > dt_bbox[3]:
                            failed_flag = True
                        outputs.append({
                            'answer': answer,
                            "format_answer": dt_bbox,
                            'annotation': annotation,
                            "question": question,
                            "image": image_path,
                            "fail": failed_flag
                        })
                        if failed_flag:
                            failed_tasks.append(
                                [image, question_ids, _prompt, annotations, image_paths]
                            )
                else:
                    dist.barrier()
                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data)
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        
        if torch.distributed.get_rank() == 0:

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            time_prefix_2 = time.strftime('%y%m%d%H', time.localtime())
            save_path = f'vqa_logs/{add_flag}'
            os.makedirs(save_path, exist_ok=True)
            
            results_file = f'{save_path}/{dataset_name}.json'
            json.dump(merged_outputs, open(results_file, 'w'),
                    ensure_ascii=False)  # save to results
        