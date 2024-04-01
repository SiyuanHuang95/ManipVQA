import re
import json
import os
import numpy as np

import logging
from PIL import Image
from PIL import ImageDraw, ImageFont

def extract_bounding_box(string):
    # Extract numbers from the string using regex
    numbers = re.findall(r"(\d+\.\d+|\.\d+|\d+)", string)
    # Convert extracted strings to float
    return [float(num) for num in numbers]

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

def bb_intersection_over_union(boxA, boxB):
    # determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the area of the union
    union_area = boxA_area + boxB_area - intersection_area

    # compute the IoU by dividing the intersection area by the union area
    iou = intersection_area / union_area

    # return the IoU value
    return iou

def get_iou_score(ground_truth, prediction):
    # print(ground_truth, prediction)
    # Extract the bounding boxes from the strings
    gt_box = extract_bounding_box(ground_truth)
    pred_box = format_bounding_box(prediction)
    # Calculate the IoU score
    # print(gt_box, pred_box)
    iou = bb_intersection_over_union(gt_box, pred_box)

    return iou


def draw_box_on_image(img_path, l_name_box_color):
    img = Image.open(img_path).convert("RGB")
    max_edge = max((img.width, img.height))

    if img.width < img.height:
        x_origin = (img.height - img.width) // 2
        y_origin = 0
    else:
        x_origin = 0
        y_origin = (img.width - img.height) // 2

    draw = ImageDraw.Draw(img)
    for name, points_in_square, color in l_name_box_color:
        for per_box_start in range(0, len(points_in_square), 4):
            x1, y1, x2, y2 = points_in_square[per_box_start:per_box_start+4]
            x1 = x1 * max_edge - x_origin
            y1 = y1 * max_edge - y_origin
            x2 = x2 * max_edge - x_origin
            y2 = y2 * max_edge - y_origin

            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            draw.text((x1 + 3, y1 + 3), name, fill=color)

    return img

def normalize_number(x):
    if x > 100:
        return x / 1000
    elif x > 10:
        return x / 100
    elif x >= 1:
        return x / 10
    else:
        return x
    
def evaluate_rec(json_path, skp_false_iou=False):
    with open(json_path) as f:
        data = json.load(f)
    print(f"for {json_path}, totally have {len(data)}")
    ious = {}
    for element in data:
        gt_string = element["annotation"]
        gt_box = extract_bounding_box(gt_string)
        
        object_name = element["image"].split("/")[-5].split("handal_dataset_")[-1]
        
        dt_box = element["format_answer"]
        dt_box = [normalize_number(x) for x in dt_box]
        if len(dt_box) == 4:
            iou = bb_intersection_over_union(gt_box, dt_box)
        else:
            if not skp_false_iou:
                iou = 0
            else:
                continue
            # continue
        # ious[object_name].append(iou)
        if object_name in ious:
            ious[object_name].append(iou)
        else:
            ious[object_name] = []
            ious[object_name].append(iou)
            
    return ious

def get_ap(ious, iou_threshold=0.75):
    total_ap = 0
    class_ap = {}
    for object_name in ious:
        iou = ious[object_name]
        tp = 0
        fp = 0
        total_gt = len(iou)
        preceision_at_recall = []
        for iou_score in iou:
            if iou_score > iou_threshold:
                tp += 1
            else:
                fp += 1
            recall = tp / total_gt
            precision = tp / (tp + fp)
            preceision_at_recall.append((recall, precision))
        
        ap = 0 
        for recall, precision in preceision_at_recall:
            ap += precision / total_gt
        
        # print(f"for {object_name}, ap is {ap}")
        class_ap[object_name] = ap
        total_ap += ap
    if len(ious) == 0:
        map_score = 0
        class_ap = {}
    else:
        map_score = total_ap / len(ious)
    # print(f"mAP is {map_score}")
    return map_score, class_ap

if __name__ == "__main__":
    logs_folder = "../vqa_logs"
    evaluation_model_folder = os.listdir(logs_folder)
    evaluation_model_folder = [x for x in evaluation_model_folder if os.path.isdir(os.path.join(logs_folder, x))]
    handel_rec_evaluation = {}
    class_name_order = ["hammers", "slip_joint_pliers", "fixed_joint_pliers", "locking_pliers", "power_drills", "ratchets", "screwdrivers", "adjustable_wrenches", "combinational_wrenches", "ladles", "measuring_cups", "mugs", "pots_pans", "spatulas", "strainers", "utensils", "whisks"]
    for model_folder in evaluation_model_folder:
        json_path = os.path.join(logs_folder, model_folder, "handal_rec.json") 
        # handal_compelet_rec.json
        json_path = os.path.join(logs_folder, model_folder, "handal_compelet_rec.json") 
        if not os.path.exists(json_path):
            continue
        ious_all = evaluate_rec(json_path, skp_false_iou=False)
        eval_dict = {}
        for threshold in np.arange(0.5, 1, 0.05):
            map_score_all, class_ap_all = get_ap(ious_all, iou_threshold=threshold)
            
            eval_dict[threshold] = {
                "mAP_all": map_score_all,
                "class_ap_all": class_ap_all,
            }
            
        mAP_average_all = np.mean([eval_dict[x]["mAP_all"] for x in eval_dict])
        class_ap_average_all = {}
        for threshold in eval_dict:
            class_ap = eval_dict[threshold]["class_ap_all"]
            for cls in class_ap:
                class_ap_average_all[cls] = class_ap_average_all.get(cls, 0) + class_ap[cls]
        class_ap_average_all = {x: class_ap_average_all[x] / len(eval_dict) for x in class_ap_average_all}
        try:
            class_ap_average_all = {x: class_ap_average_all[x] for x in class_name_order}
        except:
            class_ap_average_all = class_ap_average_all

        eval_dict["average"] = {
            "mAP_all": mAP_average_all,
            "class_ap_all": class_ap_average_all,
        }        
        handel_rec_evaluation[model_folder] = eval_dict
        
    handel_rec_evaluation = sorted(handel_rec_evaluation.items(), key=lambda x: x[1]["average"]["mAP"], reverse=True)
    
    with open("handel_complete_rec_evaluation_2.json", 'w') as f:
        json.dump(handel_rec_evaluation, f, indent=4)
    