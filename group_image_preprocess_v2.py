import os
from tqdm import tqdm
import json
from collections import defaultdict
from statistics import mode
import pickle
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

# dataset 과 model 로드
ds = load_dataset("Hongik-ML-2024/processed-dataset")
ds = ds['train']
model = YOLO("./yolov9c_custom.pt")

# index 리스트 전달 받아 해당 image list 반환
def make_img_list(index_list):
    image_list = []
    i = 0
    for index in index_list:
        binary_data = ds[index]['image']
        with open(f'./image/image{i}.jpg', 'wb') as file:
            file.write(binary_data)
        image = Image.open(f'./image/image{i}.jpg')
        image_list.append(image)
        i += 1
    return image_list

# 리스트 내 최다로 나온 요소로 전환
def convert_most_common(cls_name_list, dict):
    for key in dict:
        if dict[key]:
            most_common = mode(dict[key])
            dict[key] = cls_name_list[most_common]
    return dict

with open("grouped_images_index.pkl","rb") as fr:
    data = pickle.load(fr)

# main
result_list = []   

for index_list in tqdm(data):
    track_history = []
    image_list = make_img_list(index_list)
    id_class_history = defaultdict(lambda: [])
    frame_num = 1

    for image in image_list:
        frame_info = {}
        # conf = 0.5 이상인 객체만 탐지
        results = model.track(image, persist=True)
        boxes = results[0].boxes.xywh

        if results[0].boxes.id == None:
            pass
        else:
            track_ids = results[0].boxes.id.int().tolist()
            ids_labels = results[0].boxes.cls.int().tolist()
            
            frame_info['frame_number'] = frame_num
            # track history 저장
            for box, track_id, object_label in zip(boxes, track_ids, ids_labels):
                object_info = {}
                x, y, w, h = box
                object_info['coordinate'] = (float(x), float(y))
                frame_info[f'{track_id}'] = object_info
                label = id_class_history[track_id]
                label.append(object_label)
        if frame_info != {}:
            track_history.append(frame_info)
        frame_num += 1
    
    # 추적된 id 내에서 최다로 인식된 개체로
    cls_name_list = results[0].names
    id_class_history = convert_most_common(cls_name_list, id_class_history)

    # Update track_history to include 'object name'
    for frame_data in track_history:
        for obj_id, obj_data in frame_data.items():
            if obj_id != 'frame_number':  # Skip 'frame_number' key
                obj_id_int = int(obj_id)
                obj_data['object name'] = id_class_history[obj_id_int]
    
    # 1번만 tracking된 개체 삭제
    # Counting occurrences of each object ID
    occurrences = {}
    for frame_data in track_history:
        for obj_id in frame_data.keys():
            if obj_id != 'frame_number':
                occurrences[obj_id] = occurrences.get(obj_id, 0) + 1

    # Remove entries with object IDs that appear only once
    for frame_data in track_history:
        to_remove = [obj_id for obj_id, count in occurrences.items() if count == 1 and obj_id in frame_data]
        for obj_id in to_remove:
            del frame_data[obj_id]

    # Remove entries with only 'frame_number' field
    track_history = [frame_data for frame_data in track_history if len(frame_data) > 1]
    
    # make dict_data
    if track_history != []:
        dict_data = {
            'size' : results[0].orig_shape,
            'total_frame_count' : len(image_list),
            'track_history' : track_history
        }
    
        result_list.append(dict_data)

with open(
    os.path.join('data', f"grouped_images_tracking_v2.json"),
        "w",
         encoding="utf-8",
    ) as output_file:
        json.dump(result_list, output_file, ensure_ascii=False)

