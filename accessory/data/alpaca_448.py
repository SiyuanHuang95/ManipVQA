from typing import List, Dict
import random
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd
from model.tokenizer import Tokenizer
import copy
import torchvision.transforms as transforms
import numpy as np
import os
from data.oss_io import read_img_general
from data.bbox_util import Expand2square, BoxFormatProcess, PlainBoxFormatter
import re

try:
    from torchvision.transforms import InterpolationMode, Compose

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def format_prompt(instruction, input=None, template=None, task_type=None):
    rng = np.random.default_rng(0)
    question_placeholder = '<question>'
    dataset_placeholder_map = {
        'Caption': '<caption_question>',
        'TextContent': '<text_content_question>',
        'REG': '<replace with REG.json>.',
        "GC": '<replace with GC.json>.',
        'flickr30k': '<replace with flickr30k.json templates>.'
    }
    if task_type is not None:
        use_alpaca = False
        if '_Alpaca' in task_type:
            task_type = task_type.replace('_Alpaca', '')
            use_alpaca = True

        if task_type in ['ShortVQA', 'VQA', 'TextVQA', 'REC', 'VQA_BCoT']:
            result = rng.choice(template[task_type]).replace(question_placeholder, instruction)
        elif task_type in ['Caption', 'TextContent', 'GC', 'REG', 'flickr30k']:
            placeholder = dataset_placeholder_map[task_type]
            result = instruction.replace(placeholder, rng.choice(template[task_type]))
        elif task_type in ['Alpaca', 'VisualQuestionGeneration']:
            result = rng.choice(template[task_type]).format(instruction)
        elif task_type in ['Blank']:
            result = '{}'.format(instruction)
        else:
            raise NotImplementedError

        if use_alpaca and task_type != 'Alpaca':
            result = rng.choice(template['Alpaca']).format(result)

        return result
    return '{}'.format(instruction)


# create data
transform_train = transforms.Compose([
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
    #                              antialias=None),  # 3 is bicubic
    transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

get_transform_train = lambda size: transforms.Compose([
    # transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
    #                              antialias=None),  # 3 is bicubic
    transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

transform_val: Compose = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform=transform_train, max_words=30, image_words=257, tokenizer_path=None, seed=42):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.template = None
        data_info = {}
        group_ann = {}
        # self.templete_map = {}
        for i in self.config['META']:
            task_type = ''
            # type_name = ''
            if len(i) == 2:
                ratio = 1.0
                meta_path, meta_type = i
            elif len(i) == 3:
                meta_path, meta_type, ratio = i
            elif len(i) == 4:
                meta_path, meta_type, ratio, task_type = i
            # elif len(i) == 5:
            #     meta_path, meta_type, ratio, task_type, template_file = i
            #     type_name = os.path.splitext(template_file)[-2].split('/')[-1]
            #     if type_name not in self.templete_map.keys():
            #         print(f'load shikra template: {type_name}')
            #         templates = json.load(open(template_file, 'r', encoding='utf8'))
            #         self.templete_map[type_name] = templates
            else:
                raise NotImplementedError

            if meta_type == 'template':
                with open(meta_path) as f:
                    self.template = json.load(f)
                continue

            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                with open(meta_path) as f:
                    meta_l = json.load(f)
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                            raise e
            else:
                raise NotImplementedError(
                    f"Unknown meta file extension: \"{meta_ext}\". Currently, .json and .jsonl files are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )

            for data_line in meta_l:
                if task_type:
                    data_line['task_type'] = task_type
                elif 'task_type' not in data_line:
                    data_line['task_type'] = 'Blank'

            print(
                f"{meta_path}, type{meta_type}: len {len(meta_l)} portion: {ratio} used: {int(len(meta_l) * ratio)}")
            if task_type not in data_info:
                data_info[task_type] = int(len(meta_l) * ratio)
            else:
                data_info[task_type] += int(len(meta_l) * ratio)

            # random.shuffle(meta_l)
            meta_l = meta_l[:int(len(meta_l) * ratio)]

            if meta_type not in group_ann:
                group_ann[meta_type] = []

            # if type_name:
            #     for idx in range(len(meta_l)):
            #         meta_l[idx]['template'] = type_name

            group_ann[meta_type] += meta_l
        if self.template is None:
            raise ValueError("No template is provided!")
        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        total_data = sum(data_info.values())
        for n, d_l in data_info.items():
            print(f'{n}: {d_l} portion: {round(d_l / total_data, 4)}')

        print(f"total length: {len(self)}")
        self.transform = transform
        print(f'Used transform: {transform}')
        self.max_words = max_words
        self.image_words = image_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        # for bbox processing
        self.expand_transformer = Expand2square()
        self.target_transform = BoxFormatProcess(PlainBoxFormatter())

        self.printed_count = {}
        self.printed_limit = 2

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ann)

    def pretty_print(self, data, input1, input2):
        print(f"{'-' * 20} Data Sample Start {'-' * 20}")
        print(f"Data type: {data.get('task_type', None)} Image: {data.get('image', None)}")
        print(f"Input1: {input1}")
        print(f"Input2: {input2}")
        print(f"{'-' * 20} Data Sample End {'-' * 20}")

    def check_img_valid(self, filename):
        try:
            image_raw = read_img_general(filename)
            image, _ = self.expand_transformer(image_raw, None)
            self.transform(image)
        except:
            print(f'Broken image file: {filename}')
            while True:
                try:
                    data_item = self.ann[random.randint(0, len(self))]
                    filename = data_item['image']
                    image_raw = read_img_general(filename)
                    image, _ = self.expand_transformer(image_raw, None)
                    self.transform(image)
                    break
                except:
                    continue
        return image, image_raw

    def convert_bbox(self, data_item, image, question, answer, image_raw):
        if 'boxes' in data_item and 'boxes_seq' in data_item:
            # process bbox
            boxes = data_item['boxes']
            boxes_seq = data_item['boxes_seq']
            tgt = {'boxes': boxes}
            _, tgt = self.expand_transformer(image_raw, tgt)
            tgt['width'], tgt['height'] = image.width, image.height
            if isinstance(boxes_seq, dict):
                # convert <boxes> to sequence in question
                if '{' in question or '}' in question:
                    question = re.sub('\{', '&(', question)
                    question = re.sub('}', '&)', question)
                res, tgt = self.target_transform({'value': question, 'boxes_seq': boxes_seq['question']}, tgt)
                question = res['value']
                question = re.sub('&\(', '{', question)
                question = re.sub('&\)', '}', question)

                # convert <boxes> to sequence in answer
                if '{' in answer or '}' in answer:
                    answer = re.sub('\{', '&(', answer)
                    answer = re.sub('}', '&)', answer)
                res, tgt = self.target_transform({'value': answer, 'boxes_seq': boxes_seq['answer']}, tgt)
                answer = res['value']
                answer = re.sub('&\(', '{', answer)
                answer = re.sub('&\)', '}', answer)
            else:
                assert ValueError('Data of wrong format.')

        # process points
        if 'points' in data_item and 'points_seq' in data_item:
            points = data_item['points']
            points_seq = data_item['points_seq']
            tgt = {'points': points}
            _, tgt = self.expand_transformer(image_raw, tgt)
            tgt['width'], tgt['height'] = image.width, image.height
            if isinstance(points_seq, dict):
                # convert <points> to sequence in question
                res, tgt = self.target_transform({'value': question, 'points_seq': points_seq['question']}, tgt)
                question = res['value']

                # convert <points> to sequence in answer
                res, tgt = self.target_transform({'value': answer, 'points_seq': points_seq['answer']}, tgt)
                answer = res['value']
            else:
                assert ValueError('Data of wrong format.')

        return question, answer

    # def get_template(self, templates):
    #     return self.rng.choice(templates)

    def __getitem__(self, index):
        data_item: dict = self.ann[index]
        image = data_item.get("image", None)
        if image is not None:
            image, image_raw = self.check_img_valid(image)

            if 'conversations' in data_item:
                question = data_item['conversations'][0]['value']
                answer = data_item['conversations'][1]['value']
            elif 'question' in data_item and 'answer' in data_item:
                question = data_item['question']
                answer = data_item['answer']
            else:
                raise NotImplementedError

            # combining codes
            # if 'template' in data_item:
            #     template_key = data_item['template']
            #     split_question = question.split('.')
            #     if "QUESTION:" in question:
            #         new_question = question.lower()
            #     elif template_key and 'VQA' in template_key:
            #         new_question = question.lower()
            #     else:
            #         new_question = '.'.join([sentence.lower() for sentence in split_question[:-1]])
            #     # print('new',new_question)
            #     if template_key is not None:
            #         templte = self.get_template(self.templete_map[template_key])
            #         # templte = self.templete_map[template_key][0]
            #         if "<replace with " in question:
            #             question = templte
            #         else:
            #             question = templte.replace('<question>', new_question)
            format_instruction = question
            format_input = None
            input1 = format_prompt(format_instruction, format_input, self.template, data_item.get('task_type', None))

            input1, answer = self.convert_bbox(data_item, image, input1, answer, image_raw)

            image = self.transform(image)

        else:
            image = None
            if 'instruction' in data_item:
                format_instruction = data_item['instruction'],
                format_input = data_item['input']
                answer = data_item['output']
            else:
                format_instruction = data_item['conversations'][0]['value']
                format_input = ''
                answer = data_item['conversations'][1]['value']
            input1 = format_prompt(format_instruction, format_input, self.template, data_item.get('task_type', None))

        # input1 = format_prompt(format_instruction, format_input, self.template, data_item.get('task_type', None))
        input2 = input1 + answer

        if str(data_item.get('task_type', None)) not in self.printed_count:
            self.printed_count[str(data_item.get('task_type', None))] = 0
        if self.printed_count[str(data_item.get('task_type', None))] < self.printed_limit:
            self.pretty_print(data_item, input1, input2)
            self.printed_count[str(data_item.get('task_type', None))] += 1

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)

        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words

        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            print(f'Warning for truncation input!\nData item: {data_item}')
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        if image is None:
            return input2, labels, input2_mask
        else:
            return input2, labels, input2_mask, image

    def groups(self):
        return list(self.group_indices.values())


class MetaPreprocessor:
    def __init__(self):
        self.routing = {
            "single_turn_llava": self._preprocess_single_turn_llava,
            "caption": self._preprocess_caption
        }

    def preprocess(self, meta_l: List[Dict], recipe: str):
        return self.routing[recipe](meta_l)

    @staticmethod
    def _preprocess_single_turn_llava(meta_l: List[Dict]):
        new_meta = []
        for data_item in meta_l:
            new_meta.append({
                "image": data_item['image'],
                "instruction": data_item['conversations'][0]['value'],
                "output": data_item['conversations'][1]['value']
            })
        return new_meta

    @staticmethod
    def _preprocess_caption(meta_l: List[Dict]):
        new_meta = []
        for data_item in meta_l:
            caption = data_item['caption']
            if isinstance(caption, list):
                caption = random.choice(caption)
            new_meta.append({
                "image": data_item['url'],
                "output": caption
            })

        return new_meta


import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset


class FinetuneDistSampler(Sampler):
    #   Distrubuted Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
    def __init__(self, dataset: FinetuneDataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size=None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_) // batch_size for _ in group_indices]
        assert all([_ % num_replicas == 0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)

        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            for _ in group_indices_shuffle:
                rng.shuffle(_)
            global_batched_group_indices = [
                [_[i:i + self.batch_size * self.num_replicas * self.acc_grad]
                 for i in range(0, len(_), self.batch_size * self.num_replicas * self.acc_grad)]
                for _ in group_indices_shuffle]
            global_batched_indices = sum(global_batched_group_indices, start=[])
            rng.shuffle(global_batched_indices)

            indices = []
            for i in global_batched_indices:
                indices.extend(i)
            # indices = sum(global_batched_indices, start=[])
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = []
            for i in group_indices:
                indices.extend(i)
            # indices = sum(group_indices, start=[])

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
