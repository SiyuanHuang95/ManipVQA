import json
import os

def merge_json_files(json_file_list, output_file):
    merged_data = {}
    for json_file in json_file_list:
        with open(json_file, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
            
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
    print(f"Merged JSON files saved to {output_file}")
    

def get_json_file_list(json_dir, keyword):
    json_file_list = []
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json") and keyword in file:
                json_file_list.append(os.path.join(root, file))
    return json_file_list


if __name__ == "__main__":
    json_path = "/mnt/petrelfs/huangsiyuan/data/HANDAL/json"
    split = ["train", "test"]
    task = ["rec", "reg", "grounding"]
    key_words = []
    for s in split:
        for t in task:
            key_words.append(f"{s}_{t}")
            
    for keyword in key_words:
        json_file_list = get_json_file_list(json_path, keyword)
        output_file = os.path.join(json_path, f"{keyword}_HANDAL.json")
        merge_json_files(json_file_list, output_file)