import json
import random
import re
import os
from tqdm import tqdm


# global config
data_root = './shikra_data'
image_root = './data/images'



def clean(s_i):
    s_i = re.sub('[ ]+', ' ', s_i)
    s_i = re.sub('\n', '', s_i)
    s_i = s_i.replace('image image', 'image')
    for i in re.findall('[a-zA-Z>]+ [.?,]', s_i):
        s_i = s_i.replace(i, i.replace(' ', ''))
    if s_i[-1] != '?' and s_i[-1] != '.':
        s_i += '.'
    return s_i

def vqav2(item, img_root=None):
    img_path = f"{image_root}/{img_root}/{item['image_path']}"
    question = item['question']

    if 'annotation' in item:
        final_answer = item['annotation']['multiple_choice_answer']
    else:
        final_answer = 'UNKNOWN'

    final_answer = clean(final_answer)
    final_question = clean(question)
    if not isinstance(final_answer, str) or not isinstance(final_question, str) or not final_answer or not final_question:
        return None
    return {
        'image': img_path,
        'conversations': [
            {
                'from': 'human',
                'value': final_question,
            },
            {
                'from': 'gpt',
                'value': final_answer,
            }
        ],
    }

if __name__ == '__main__':
    data_pair = {
        'vqav2': {
            'source': 'v2_OpenEnded_mscoco_train2014_questions.jsonl',
            'func': vqav2,
            'img_root': 'coco'
        }
    }

    process_list = ['vqav2']
    for name in process_list:
        ds = data_pair[name]

        data_lines = []
        with open(os.path.join(data_root, ds['source']), 'r', encoding="utf-8") as f:
            for line in f:
                data_lines.append(json.loads(line))

        data_list = []
        for sample in tqdm(data_lines, desc=f"{name}"):
            res = ds['func'](sample, ds['img_root'])
            data_list.append(res)

        with open(f'./{name}.json', 'w') as f:
            f.write(json.dumps(data_list))
