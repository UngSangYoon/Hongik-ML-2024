from datasets import load_dataset
import json
from tqdm import tqdm
import os

def get_index(item, list):
    if item in list:
        return list.index(item)
    
# box[0], box[1], box[2], box[3] : xmin, ymin, xmax, ymax
# 좌표변환 (xyxy -> xywhn) 
def convert_box(box):
          dw, dh = 1. / 1024, 1. / 576
          x, y, w, h = (box[0] + box[2]) / 2.0 - 1, (box[1] + box[3]) / 2.0 - 1, box[2] - box[0], box[3] - box[1]
          return[x * dw, y * dh, w * dw, h * dh]

# 소수 6자리로 변환
def format_to_6_decimal_places(number_list):
    output_list = []
    for number in number_list:
        formatted_number = round(number, 6)
        output_list.append(str(formatted_number))
    return output_list

ds = load_dataset("Hongik-ML-2024/processed-dataset")

# 미리 생성해둔 object_list.json 로드 (After make_object_list.py run)
with open('object_list.json', "r") as f:
    object_list = json.load(f)

train = ds['train']
val = ds['validation']

train_data_dir = './data/train'
val_data_dir = './data/validation'

os.makedirs(train_data_dir  + '/images', exist_ok=True)
os.makedirs(train_data_dir  + '/labels', exist_ok=True)
os.makedirs(val_data_dir + '/images', exist_ok=True)
os.makedirs(val_data_dir + '/labels', exist_ok=True)

# train data 저장
index = 0

for train_data in tqdm(train):
    # if index == 10000:
    #      break
    # image 저장
    binary_data = train_data['image']
    with open(os.path.join(train_data_dir + f'/images/{index}.jpg'), 'wb') as file:
        file.write(binary_data)

    # label text 생성 후 저장 
    label_text = ''
    label = train_data['label']
    Annotations = label['Learning_Data_Info.']['Annotations']
    for annotation in Annotations:
        # 사물 index 알아내기
        object_num = get_index(annotation['Class_ID'], object_list)
        object_num = str(object_num)

        # 좌표 변환 (xyxy -> xywhn)
        box = annotation['Type_value']
        box = convert_box(box)
        box = format_to_6_decimal_places(box)
        coordinate = ' '.join(box)

        # 한줄 추가
        text = object_num + ' ' + coordinate
        label_text += text + '\n'
    label_text = label_text[:-1]

    with open(os.path.join(train_data_dir + f'/labels/{index}.txt'), "w") as file:
        file.write(label_text)
    index += 1
    
# validation 저장
index = 0
for val_data in tqdm(val):
    # image 저장
    binary_data = val_data['image']
    with open(os.path.join(val_data_dir + f'/images/{index}.jpg'), 'wb') as file:
        file.write(binary_data)
    
    # label text 생성 후 저장 
    label_text = ''
    label = val_data['label']
    Annotations = label['Learning_Data_Info.']['Annotations']
    for annotation in Annotations:
        # 사물 index 알아내기
        object_num = get_index(annotation['Class_ID'], object_list)
        object_num = str(object_num)

        # 좌표 변환 (xyxy -> xywhn)
        box = annotation['Type_value']
        box = convert_box(box)
        box = format_to_6_decimal_places(box)
        coordinate = ' '.join(box)

        # 한줄 추가
        text = object_num + ' ' + coordinate
        label_text += text + '\n'
    label_text = label_text[:-1]

    with open(os.path.join(val_data_dir + f'/labels/{index}.txt'), "w") as file:
        file.write(label_text)
    index += 1
