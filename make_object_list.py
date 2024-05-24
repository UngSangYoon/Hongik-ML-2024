from datasets import load_dataset
import json

ds = load_dataset("Hongik-ML-2024/processed-dataset")
count = 0

val = ds['validation']
object_list = []

for data in val:
    label = data['label']
    Annotations = label['Learning_Data_Info.']['Annotations']
    for annotation in Annotations:
        if annotation['Type'] != 'Bounding_box':
            count += 1
            continue
        object = annotation['Class_ID']
        if object not in object_list:
            object_list.append(object)

print(object_list)
print('\n')
object_list.sort()
print(object_list)

with open('object_list.json', 'w', encoding='utf-8') as f:
    json.dump(object_list, f, ensure_ascii = False)





