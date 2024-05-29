import os
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
model = YOLO("./yolov8n_custom/weights/best.pt")

# index 리스트 전달 받아 해당 image list 반환
def make_img_list(index_list):
    image_list = []
    i = 0
    for index in index_list:
        binary_data = ds[index]['image']
        with open(f'image{i}.jpg', 'wb') as file:
            file.write(binary_data)
        image = Image.open(f'./image{i}.jpg')
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

# track_history의 인덱스명을 객체 이름으로 변환 (객체가 중복 시 넘버링)
def convert_track_history(track_history, id_class_history):
    new_track_history = {}
    object_count = defaultdict(int)
    for index in track_history:
        object_name = id_class_history[index]
        object_count[object_name] += 1
        new_object_name = f"{object_name} {object_count[object_name]}"  # 객체 이름에 순차적인 숫자를 추가
        coordinates = track_history[index]
        new_track_history[new_object_name] = coordinates
    return new_track_history

with open("grouped_images_index.pkl","rb") as fr:
    data = pickle.load(fr)

result_list = []   

for index_list in data:
    image_list = make_img_list(index_list)
    
    track_history = defaultdict(lambda: [])
    id_class_history = defaultdict(lambda: [])

    for image in image_list:
        # conf = 0.5 이상인 객체만 탐지
        results = model.track(image, persist=True)
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id == None:
            track_ids = [0]
        else:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            ids_labels = results[0].boxes.cls.int().cpu().tolist()

        # track history 저장
        for box, track_id, object_label in zip(boxes, track_ids, ids_labels):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            label = id_class_history[track_id]
            label.append(object_label)

    # 추적 안된 개체 history 제거
    if 0 in track_history:
        del track_history[0]
        del id_class_history[0]
    
    # 추적된 id 내에서 최다로 인식된 개체로
    cls_name_list = results[0].names
    id_class_history = convert_most_common(cls_name_list, id_class_history)

    # 1번만 tracking된 개체 삭제
    keys_to_remove = [key for key, value in track_history.items() if len(value) == 1]

    for key in keys_to_remove:
        del track_history[key]
        del id_class_history[key]

    track_history = convert_track_history(track_history, id_class_history)

    # dict_data 만들기
    dict_data = {
        'size' : results[0].orig_shape,
        'track_history' : track_history,
    }
    result_list.append(dict_data)

with open(
    os.path.join('data', f"grouped_images_tracking.json"),
        "w",
         encoding="utf-8",
    ) as output_file:
        json.dump(result_list, output_file, ensure_ascii=False)

