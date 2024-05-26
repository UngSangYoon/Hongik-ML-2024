from datasets import load_dataset
import io
from PIL import Image
from tqdm import tqdm
import re

def split_last_number(s):
    match = re.search(r'\d+$', s)
    if match:
        return s[:match.start()], int(s[match.start():])
    else:
        return s, None

def is_equal_2_labellist(labellist1, labellist2):
    if len(labellist1) != len(labellist2):
        return False
    for i in range(len(labellist1)):
        for j in range(len(labellist1[i])):
            if labellist1[i][j] != labellist2[i][j]:
                return False
    return True

ds = load_dataset("Hongik-ML-2024/processed-dataset")

fileinfo_list = []
for i in tqdm(range(len(ds['train']))):
    index = i
    info = ds['train'][i]['label']['Raw_Data_Info.']
    labels = ds['train'][i]['label']['Learning_Data_Info.']['Annotations']
    labels = [[annotation['Class_ID'], int(annotation['Object_num'])] for annotation in labels]
    labels = sorted(labels, key=lambda x: (x[0], x[1]))
    fileinfo_list.append([index, info, labels]) # 1 file info has 3 elements: index, info, labels

# sort by filename
fileinfo_list = sorted(fileinfo_list, key=lambda x: x[1]['Raw_Data_ID'])

# if the image filenames are sequential, consider the group as a single video
grouped_images = []
group = []
for i in tqdm(range(len(fileinfo_list) - 1)):
    background, num = split_last_number(fileinfo_list[i][1]['Raw_Data_ID'])
    next_background, next_num = split_last_number(fileinfo_list[i + 1][1]['Raw_Data_ID'])
    if background == next_background and num + 1 == next_num and is_equal_2_labellist(fileinfo_list[i][2], fileinfo_list[i + 1][2]):
        group.append(fileinfo_list[i])
    else:
        group.append(fileinfo_list[i])
        grouped_images.append(group)
        group = []

# if there is only one image, remove it
grouped_images = [group for group in grouped_images if len(group) >= 10]

# show the length of the grouped images
import matplotlib.pyplot as plt
lengths = [len(group) for group in grouped_images]
plt.hist(lengths, bins=100)
plt.show()

# show the grouped images with length 10 with PIL
for group in grouped_images:
    if len(group) == 10:
        print(group)
        for index, info, labels in group:
            image = Image.open(io.BytesIO(ds['train'][index]['image']))
            image.show()
        break

# save the grouped images index in a pickle file
import pickle
grouped_images_index = [[index for index, _, _ in group] for group in grouped_images]
with open('grouped_images_index.pkl', 'wb') as f:
    pickle.dump(grouped_images_index, f)