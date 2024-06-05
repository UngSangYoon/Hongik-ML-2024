import cv2
from collections import defaultdict
from statistics import mode
from ultralytics import YOLO
import threading
from multiprocessing import Manager
import time
from LLMSummarizer import LLMSummarizer

class VideoTracker:
    def __init__(self, model_path, video_source, shared_data):
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.track_history = []
        self.id_class_history = {}
        self.cls_name_list = []
        self.frame_num = 0
        self.fps = None
        self.shared_data = shared_data

    def process_video_stream(self):
        cap = cv2.VideoCapture(self.video_source)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_frames_per_segment = 10 * self.fps  # Total frames to process for each 10-second segment

        while cap.isOpened():
            frame_count = 0
            while frame_count < total_frames_per_segment:
                success, frame = cap.read()
                if not success:
                    break

                self.frame_num += 1
                frame_count += 1

                # Process frame every second
                if self.frame_num % self.fps == 0:
                    self.process_frame(frame)

            if frame_count == 0:
                break  # Exit the loop if no frames were read

            self.finalize_processing()
            self.output_result()

        cap.release()

    def process_frame(self, frame):
        frame_info = {}
        results = self.model.track(frame, persist=True, show=True)
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
            ids_labels = results[0].boxes.cls.int().tolist()

            frame_info['frame_number'] = self.frame_num
            for box, track_id, object_label in zip(boxes, track_ids, ids_labels):
                object_info = {}
                x, y, w, h = box
                object_info['coordinate'] = (float(x), float(y))
                frame_info[f'{track_id}'] = object_info
                if track_id not in self.id_class_history:
                    self.id_class_history[track_id] = []
                self.id_class_history[track_id].append(object_label)
            if frame_info:
                self.track_history.append(frame_info)

    def finalize_processing(self):
        self.cls_name_list = self.model.names
        self.add_object_name()

    def add_object_name(self):
        for frame_data in self.track_history:
            for obj_id, obj_data in frame_data.items():
                if obj_id != 'frame_number':
                    obj_id_int = int(obj_id)
                    if obj_id_int in self.id_class_history:
                        most_common_label = mode(self.id_class_history[obj_id_int])
                        obj_data['object name'] = self.cls_name_list[most_common_label]

    def output_result(self):
        if self.track_history:
            dict_data = {
                'size': (self.fps, self.frame_num),
                'total_frame_count': self.frame_num,
                'track_history': self.track_history
            }
            self.shared_data['tracking_data'] = dict_data
            print("Data sent to shared memory:", dict_data)

# Function to simulate the LLM reading from shared memory and generating a description
# def llm_reader(shared_data):
#     while True:
#         if 'tracking_data' in shared_data:
#             tracking_data = shared_data['tracking_data']
#             print("LLM is processing the tracking data...")
#             # Simulate LLM processing
#             time.sleep(5)  # Simulate some processing time
#             print("LLM generated description for the tracking data:", tracking_data)
#             del shared_data['tracking_data']  # Clear shared data after processing

# Example usage
if __name__ == "__main__":
    model_path = "./yolov9c.pt"
    video_source = './test_video.mp4'  # Use 0 for webcam, or replace with video file path

    manager = Manager()
    shared_data = manager.dict()

    llm_summarizer = LLMSummarizer(fps=30, size_x=640, size_y=360)
    tracker = VideoTracker(model_path, video_source, shared_data)
    
    tracking_thread = threading.Thread(target=tracker.process_video_stream)
    llm_summarizer = threading.Thread(target=llm_summarizer.read, args=(shared_data,))

    tracking_thread.start()
    llm_summarizer.start()

    tracking_thread.join()
    llm_summarizer.join()