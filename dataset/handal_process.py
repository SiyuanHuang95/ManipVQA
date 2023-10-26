import os
import json
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from handal_grounding_tasks import *

handal_dataset_path = '/mnt/petrelfs/huangsiyuan/data/HANDAL/HANDAL_DATASET_NEW'
json_save_path = '/mnt/petrelfs/huangsiyuan/data/HANDAL/json'
if not os.path.exists(json_save_path):
    os.makedirs(json_save_path)


class_names = grounding_tasks.keys()

def find_bounding_box(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    height, width = mask.shape[:2]
    
    if mask.shape[-1] == 2:
        mask, alpha = cv2.split(mask)
    elif mask.shape[-1] == 4:
        mask, alpha = cv2.split(mask[:, :, 0:3])
    else:
        alpha = None

    white_pixels = np.argwhere(mask > 0)

    if white_pixels.size > 0:
        y1, x1 = white_pixels.min(axis=0)
        y2, x2 = white_pixels.max(axis=0)
        
        if height > width:
            pad_x0 = int((height - width) / 2)
            pad_y0 = 0
            width = height
        else:
            pad_x0 = 0
            pad_y0 = int((width - height) / 2)
            height = width
            
        x1 = x1 + pad_x0
        x2 = x2 + pad_x0
        y1 = y1 + pad_y0
        y2 = y2 + pad_y0
        
        x1 = x1 / width
        x2 = x2 / width
        y1 = y1 / height
        y2 = y2 / height
        
        return float(x1), float(y1), float(x2), float(y2)
    else:
        return None, None, None, None

def find_minimum_rotated_bounding_box(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    height, width = mask.shape[:2]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rotated_rect = cv2.minAreaRect(largest_contour)

        center, size, angle = rotated_rect
        box_points = cv2.boxPoints(rotated_rect).astype(int)
        box_points = [(int(point[0]), int(point[1])) for point in box_points]

        x1 = float(box_points[0][0])
        y1 = float(box_points[0][1])
        x2 = float(box_points[1][0])
        y2 = float(box_points[1][1])
        x3 = float(box_points[2][0])
        y3 = float(box_points[2][1])
        x4 = float(box_points[3][0])
        y4 = float(box_points[3][1])
        
        if height > width:
            pad_x0 = int((height - width) / 2)
            pad_y0 = 0
            width = height
        else:
            pad_x0 = 0
            pad_y0 = int((width - height) / 2)
            height = width
            
        for x in [x1, x2, x3, x4]:
            x = x + pad_x0
            x = x / width
        
        for y in [y1, y2, y3, y4]:
            y = y + pad_y0
            y = y / height
        
        return x1, y1, x2, y2, x3, y3, x4, y4, center, size, angle
    else:
        return None, None, None, None, None, None, None, None, (None, None), (None, None), None


def data_tuples_for_json(path, sample_rate=5):
    tuples_list = []
    scenes_count = 0
    
    object_category_name = path.rsplit('/')[-2]
    category_name = object_category_name.replace('handal_dataset_', '')
    assert category_name in class_names, f"{category_name} not in {class_names}"
    
    for object_id in os.listdir(path):

        object_id_path = os.path.join(path, object_id)
        rgb_path = os.path.join(object_id_path, 'rgb')
        mask_path = os.path.join(object_id_path, 'mask')
        mask_parts_path = os.path.join(object_id_path, 'mask_parts')
        
        if not os.path.exists(rgb_path):
            continue
        if not os.path.exists(mask_path):
            continue
        if not os.path.exists(mask_parts_path):
            continue
        
        rgb_images = os.listdir(rgb_path) # 0000002.jpg
        mask_images = os.listdir(mask_path) # 000002_000000.png
        mask_parts_images = os.listdir(mask_parts_path) # 000002_000000_handle.png
        
        # sort images
        rgb_images = sorted(rgb_images, key=lambda x: int(x.split('.')[0]))
        mask_images = sorted(mask_images, key=lambda x: int(x.split('_')[0]))
        mask_parts_images = sorted(mask_parts_images, key=lambda x: int(x.split('_')[0]))
        part_name = mask_parts_images[0].split('_')[-1].split('.')[0]
        
        assert len(rgb_images) == len(mask_images) == len(mask_parts_images)
        scene_rgb_paths = []
        objects_masks_paths = []
        objects_parts_masks_paths = []
        
        for i in range(len(rgb_images)):
            if i % sample_rate == 0:
                scene_rgb_paths.append(os.path.join(rgb_path, rgb_images[i]))
                objects_masks_paths.append(os.path.join(mask_path, mask_images[i]))
                objects_parts_masks_paths.append(os.path.join(mask_parts_path, mask_parts_images[i]))
                
        for scene_rgb_path, object_mask_path, object_part_mask_path in zip(scene_rgb_paths, objects_masks_paths, objects_parts_masks_paths):
            # object_category_name = os.path.basename(category_path)
            # BUGFIX: When use multiple-process, the category_path is always the last one.
            
            object_part_name = os.path.basename(object_part_mask_path).split('_')[
                2].replace('.png', '')

            x1, y1, x2, y2 = find_bounding_box(object_mask_path)
            if x1 is None:
                continue
            object_bbox = [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]

            x1, y1, x2, y2 = find_bounding_box(object_part_mask_path)
            if x1 is None:
                continue
            object_part_bbox = [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]

            x1, y1, x2, y2, x3, y3, x4, y4, object_minimum_rotated_bbox_center, object_minimum_rotated_bbox_size, object_minimum_rotated_bbox_angle = find_minimum_rotated_bounding_box(object_mask_path)
            object_minimum_rotated_bbox = [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2), round(float(x3), 2), round(float(y3), 2), round(float(x4), 2), round(float(y4), 2)]
            object_minimum_rotated_bbox_center = (round(float(object_minimum_rotated_bbox_center[0]), 2), round(float(object_minimum_rotated_bbox_center[1]), 2))
            object_minimum_rotated_bbox_size = (round(float(object_minimum_rotated_bbox_size[0]), 2), round(float(object_minimum_rotated_bbox_size[1]), 2))
            object_minimum_rotated_bbox_angle = round(float(object_minimum_rotated_bbox_angle), 2)
            
            x1, y1, x2, y2, x3, y3, x4, y4, object_minimum_rotated_part_bbox_center, object_minimum_rotated_part_bbox_size, object_minimum_rotated_part_bbox_angle = find_minimum_rotated_bounding_box(object_part_mask_path)
            object_minimum_rotated_part_bbox = [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2), round(float(x3), 2), round(float(y3), 2), round(float(x4), 2), round(float(y4), 2)]
            object_minimum_rotated_part_bbox_center = (round(float(object_minimum_rotated_part_bbox_center[0]), 2), round(float(object_minimum_rotated_part_bbox_center[1]), 2))
            object_minimum_rotated_part_bbox_size = (round(float(object_minimum_rotated_part_bbox_size[0]), 2), round(float(object_minimum_rotated_part_bbox_size[1]), 2))
            object_minimum_rotated_part_bbox_angle = round(float(object_minimum_rotated_part_bbox_angle), 2)
            
            tuples_list.append(
                (
                    object_id, # object id
                    object_category_name.replace('handal_dataset_', ''), # object_category_name
                    object_part_name, # object_part_name
                    object_bbox, # object_bbox
                    object_part_bbox, # object_part_bbox
                    object_minimum_rotated_bbox, # object_minimum_rotated_bbox
                    object_minimum_rotated_bbox_center, # object_minimum_rotated_bbox_center
                    object_minimum_rotated_bbox_size, # object_minimum_rotated_bbox_size
                    object_minimum_rotated_bbox_angle, # object_minimum_rotated_bbox_angle
                    object_minimum_rotated_part_bbox, # object_minimum_rotated_part_bbox
                    object_minimum_rotated_part_bbox_center, # object_minimum_rotated_part_bbox_center
                    object_minimum_rotated_part_bbox_size, # object_minimum_rotated_part_bbox_size
                    object_minimum_rotated_part_bbox_angle, # object_minimum_rotated_part_bbox_angle
                    category_path, # category_path
                    scene_rgb_path, # scene_rgb_path
                    object_mask_path, # object_mask_path
                    object_part_mask_path # object_part_mask_path
                    )
            )
            scenes_count += 1
            
    return tuples_list

def create_tasks(tuples_list, train_split=True):
    whole_object_reg_tasks = []
    whole_object_rec_tasks = []
    whole_object_grounding_rec_tasks = []
    
    for object_id, object_category_name, object_part_name, object_bbox, object_part_bbox, object_minimum_rotated_bbox, object_minimum_rotated_bbox_center, object_minimum_rotated_bbox_size, object_minimum_rotated_bbox_angle, object_minimum_rotated_part_bbox, object_minimum_rotated_part_bbox_center, object_minimum_rotated_part_bbox_size, object_minimum_rotated_part_bbox_angle, category_path, scene_rgb_path, object_mask_path, object_part_mask_path in tuples_list:
        ## Whole Object
        # REG
        reg_question_template = "Please provide a short description of this region: "
        reg_question = reg_question_template + str(object_part_bbox)
        
        reg_answer = object_part_name
        whole_object_reg_task = {
            "image" : scene_rgb_path,
            "conversations": [
                {
                    "from": "human",
                    "value": reg_question
                },
                {
                    "from": "gpt",
                    "value": reg_answer
                }
            ]
        }
        whole_object_reg_tasks.append(whole_object_reg_task)
        # print(whole_object_reg_task)
        
        # REC
        rec_question_template = "Please provide the bounding box coordinate of the region this sentence describes: "
        rec_question = rec_question_template + object_part_name
        whole_object_rec_task = {
            "image" : scene_rgb_path,
            "conversations": [
                {
                    "from": "human",
                    "value": rec_question
                },
                {
                    "from": "gpt",
                    "value": str(object_part_bbox)
                }
            ]
        }
        whole_object_rec_tasks.append(whole_object_rec_task)
        # print(whole_object_rec_task)
        
        # Grounding REC: task
        grounding_rec_question_template = "Please provide the bounding box coordinate of the region this sentence describes: grasp for "
        possible_tasks = grounding_tasks[object_category_name]
        slected_task = np.random.choice(possible_tasks)
        grounding_rec_question = grounding_rec_question_template + slected_task
        whole_object_grounding_rec_task = {
            "image" : scene_rgb_path,
            "conversations": [
                {
                    "from": "human",
                    "value": grounding_rec_question,
                },
                {
                    "from": "gpt",
                    "value": str(object_part_bbox)
                }
                
            ]
        }
        whole_object_grounding_rec_tasks.append(whole_object_grounding_rec_task)
        # print(whole_object_grounding_rec_task)
            
    if train_split: 
        reg_tasks_json = f"train_reg_tasks_{object_category_name}.json"
        rec_tasks_json = f"train_rec_tasks_{object_category_name}.json"
        grounding_rec_tasks_json = f"train_grounding_tasks_{object_category_name}.json"
    else:
        reg_tasks_json = f"test_reg_tasks_{object_category_name}.json"
        rec_tasks_json = f"test_rec_tasks_{object_category_name}.json"
        grounding_rec_tasks_json = f"test_grounding_tasks_{object_category_name}.json"
        
    reg_tasks_json = os.path.join(json_save_path, reg_tasks_json)
    rec_tasks_json = os.path.join(json_save_path, rec_tasks_json)
    grounding_rec_tasks_json = os.path.join(json_save_path, grounding_rec_tasks_json)
    
    print(f"For train: {train_split} {object_category_name} has {len(whole_object_reg_tasks)} original pairs.")
    with open(reg_tasks_json, "w") as json_file:
        json.dump(whole_object_reg_tasks, json_file, indent=4)
    # print(f"Save to {reg_tasks_json}")
    
    # print(f"For train: {train_split} {object_category_name} has {len(whole_object_rec_tasks)} original pairs.")
    with open(rec_tasks_json, "w") as json_file:
        json.dump(whole_object_rec_tasks, json_file, indent=4)
    # print(f"Save to {rec_tasks_json}")
    
    # print(f"For train: {train_split} {object_category_name} has {len(whole_object_grounding_rec_tasks)} original pairs.")
    with open(grounding_rec_tasks_json, "w") as json_file:
        json.dump(whole_object_grounding_rec_tasks, json_file, indent=4)
    # print(f"Save to {grounding_rec_tasks_json}")
    
    return True

def handal_to_vqa(folder_path, sample_rate=5, train_split=True):
    try: 
        data_tuple = data_tuples_for_json(folder_path, sample_rate=sample_rate)
        print("DEBUG: data_tuples_for_json is successful")
        if create_tasks(data_tuple, train_split):
            print(f"create tasks successfully for {folder_path}!")
            return True
        else:
            print(f"create tasks failed for {folder_path}!")
            return False
    except Exception as e:
        print(f"create tasks failed for {folder_path}!")
        print(e)
        return False

    
if __name__ == "__main__":
    sample_rate = 5

    handal_objects_categories_paths = []
    
    tasks = []
    for category_subfolder in os.listdir(handal_dataset_path):
        category_path = os.path.join(handal_dataset_path, category_subfolder)
        category_name = category_subfolder.replace('handal_dataset_', '')
        
        train_reg_tasks_json = f"train_reg_tasks_{category_name}.json"
        test_reg_tasks_json = f"test_reg_tasks_{category_name}.json"
        
        if not os.path.isdir(category_path):
            continue
        
        if os.path.exists(os.path.join(category_path, 'test')):
            if os.path.exists(test_reg_tasks_json):
                continue
            tasks.append(
                [os.path.join(category_path, 'test'), sample_rate, False]
            )
            
        if os.path.exists(os.path.join(category_path, 'train')):
            if os.path.exists(train_reg_tasks_json):
                continue
            tasks.append(
                [os.path.join(category_path, 'train'), sample_rate, True]
            )
    
    print(f"Dataset Path to Process: {tasks}")
    print(f"Dataset Path number to Process: {len(tasks)}")

    worker = cpu_count()
    with Pool(processes=worker) as pool:  # You can adjust the number of processes based on your needs
        status = pool.starmap(handal_to_vqa, tasks)
        
    # for task in tasks:
    #     handal_to_vqa(task[0], task[1], task[2])
        
    print(status)
    