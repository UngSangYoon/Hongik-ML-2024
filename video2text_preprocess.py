from collections import defaultdict
import cv2
import numpy as np
from statistics import mode
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("./yolov9c_custom.pt")

# Open the video file
video_path = "./test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = []
id_class_history = defaultdict(lambda: [])

# 리스트 내 최다로 나온 요소로 전환
def convert_most_common(cls_name_list, dict):
    for key in dict:
        if dict[key]:
            most_common = mode(dict[key])
            dict[key] = cls_name_list[most_common]
    return dict

# main
frame_num = 0

# Loop through the video frames
while cap.isOpened():
    frame_num += 1
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    if frame_num % 15 == 0:
        frame_info = {}
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, show=True)
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is not None:
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

# 추적된 id 내에서 최다로 인식된 개체로
cls_name_list = results[0].names
id_class_history = convert_most_common(cls_name_list, id_class_history)

# 1번만 tracking된 개체 삭제
# Update track_history to include 'object name'
occurrences = {}
for frame_data in track_history:
    for obj_id, obj_data in frame_data.items():
        if obj_id != 'frame_number':  # Skip 'frame_number' key
            obj_id_int = int(obj_id)
            obj_data['object name'] = id_class_history[obj_id_int]

# Remove entries with object IDs that appear only once
for frame_data in track_history:
    to_remove = [obj_id for obj_id, count in occurrences.items() if count == 1 and obj_id in frame_data]
    for obj_id in to_remove:
        del frame_data[obj_id]

# Remove entries with only 'frame_number' field
track_history = [frame_data for frame_data in track_history if len(frame_data) > 1]

if track_history != []:
    dict_data = {
        'size': results[0].orig_shape,
        'total_frame_count': frame_num,
        'track_history': track_history
    }

# Release the video capture object
cap.release()

# Print or save the dict_data as needed
print(dict_data)
