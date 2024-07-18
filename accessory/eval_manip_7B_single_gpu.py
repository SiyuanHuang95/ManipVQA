import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from tqdm import tqdm
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from model.meta import MetaModel

from accessory.data.conversation import default_conversation

import argparse
import torch
import torch.distributed as dist

from PIL import Image
import PIL.ImageFile as ImageFile

# Increase the limit for decompression
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb limit

from util import misc

import re
import json


global_config = {
    'temperature': 0.1,
    'top_p': 0.75
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

    return input_image, question_ids, questions, annotations, image_paths

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

    def __init__(self, test, prompt, img_size=224, remove_space=False, sampled_num=5000, result=None):

        with open(test, "r") as f:
            self.test = json.load(f)
        if len(self.test) > sampled_num:
            # first shuffle, then sample
            random.shuffle(self.test)
            sampled_num = min(len(self.test), sampled_num)
            self.test = self.test[:sampled_num]
        
        if result is not None:
            # when image_path and question is the same, contine
            print(f"before remove, test length: {len(self.test)}")
            for test_item in self.test:
                img_path = test_item["image"]                    
                for result_item in result:
                    if img_path == result_item["image"]:
                        self.test.remove(test_item)
                        break    
            print(f"after remove, test length: {len(self.test)}")
        
        self.prompt = prompt
        self.print = False
        self.transform_val = T_padded_resize(img_size)
        self.remove_space = remove_space

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        image_path = data["image"]
        question = data["conversations"][0]["value"]
        question_id = idx
        annotation = data["conversations"][1]["value"]        
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
        
        conv = default_conversation()
        conv.load_qas([[question, None]])
        prompt = conv.get_prompt()

        if self.remove_space:
            question = question.replace("###Assistant: ", "###Assistant:")
            
        item_dict =  {
            'question': prompt,
            'question_id': question_id,
            'annotation': annotation,
            'image': image,
            "image_path": image_path
        }
        if idx < 2:
            print(f"question: {question}")
            print(f"prompt: {prompt}")
            print(f"annotation: {annotation}")
            print(f"image_path: {image_path}")
            print(f"image: {image.shape}")
            
        return item_dict
    
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
        parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', type=str, default=None)
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')
        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dataset', default='path_to_eval_json', type=str)
        parser.add_argument("--input_size", type=int, default=224)
        parser.add_argument('--addition_flag', default=None, type=str)
        parser.add_argument("--remove_space", action="store_true", default=False)
        parser.add_argument("--sampled_num", type=int, default=200)
        parser.add_argument("--max_gen_len", type=int, default=2048)
        return parser

    args = get_args_parser().parse_args()
    add_flag = args.addition_flag
    remove_space = args.remove_space
    print(f"load pretrained from {args.pretrained_path}")
    
    if type(args.pretrained_path) == list:
        pretrained_path = args.pretrained_path[0]
    else:
        pretrained_path = args.pretrained_path
    ckpt_path = os.path.join(pretrained_path, "consolidated.00-of-01.model.pth")

    print(f"load ckpt from {ckpt_path}")
    
    # define the model
    # misc.init_distributed_mode(args)
    # fs_init.initialize_model_parallel(args.model_parallel_size)
    
    if args.llama_config:
        model = MetaModel(args.llama_type, llama_config=[args.llama_config], tokenizer_path=args.tokenizer_path,
                            with_visual=True, max_seq_len=4096,
                            )
    else:
        model = MetaModel(
            args.llama_type, llama_config=[], tokenizer_path=args.tokenizer_path,
            with_visual=True, max_seq_len=4096,
        )    
    shard = torch.load(ckpt_path, map_location="cpu")
    if shard["model"] is not None:
        shard = shard["model"]
    load_result = model.load_state_dict(shard, strict=False)
    print(load_result)
    
    tokenizer = model.tokenizer
    # print("Model = %s" % str(model))
    model.bfloat16().cuda()
    # model enabled done
    
    dataset_name = args.dataset.rsplit("/")[-1].split(".")[0]    
    result = None
    save_path = f'vqa_logs/{add_flag}'
    os.makedirs(save_path, exist_ok=True)
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{save_path}/{dataset_name}_{time_prefix}.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            result = json.load(f)
    
    print(f"evaluating on {dataset_name}")
    log_path = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}.txt'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
        if f'dataset config: {dataset_name}' in content:
            exit(0)
            
    # use the default prompt
    prompt = ''    
    random.seed(args.seed)
    dataset = VQADataset(
        test=args.dataset,
        prompt=prompt,
        img_size=args.input_size,
        remove_space=remove_space,
        sampled_num=args.sampled_num,
        result=result
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    max_gen_len = args.max_gen_len
    gen_t = global_config['temperature']
    top_p = global_config['top_p']
    failed_tasks = []
    with torch.no_grad():
        for image, question_ids, _prompt, annotations, image_paths in tqdm(dataloader):

            image = image.cuda()
            # print(f'\ninput: {_prompt[0]}\n')
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                results = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            for question_id, answer, annotation, question, image_path in zip(question_ids, results,
                                                    annotations, _prompt, image_paths):
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

                dt_bbox = format_bounding_box(answer)
                outputs.append({
                    'answer': answer,
                    "format_answer": dt_bbox,
                    'annotation': annotation,
                    "question": question,
                    "image": image_path,
                })


    merged_outputs = outputs
    json.dump(merged_outputs, open(results_file, 'w'),
            ensure_ascii=False)  # save to results
    