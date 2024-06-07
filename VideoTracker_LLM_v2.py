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
        self.width = None
        self.height = None
        self.shared_data = shared_data
        self.frame_buffer = []

    def get_video_info(self):
        cap = cv2.VideoCapture(self.video_source)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return self.fps, self.width, self.height

    def process_video_stream(self):
        cap = cv2.VideoCapture(self.video_source)
        interval = self.fps // 5  # 0.2초마다 처리
        total_frames_per_segment = 5 * self.fps  # Total frames to process for each 5-second segment

        while cap.isOpened():
            frame_count = 0
            self.frame_buffer = []

            while frame_count < total_frames_per_segment:
                success, frame = cap.read()
                if not success:
                    break

                self.frame_num += 1
                frame_count += 1

                if self.frame_num % interval == 0:
                    self.process_frame(frame)

            if frame_count == 0:
                break  # Exit the loop if no frames were read

            self.finalize_processing()
            self.output_result()

        cap.release()

    def process_frame(self, frame):
        frame_info = {}
        results = self.model.track(frame, persist=True, show=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
            ids_labels = results[0].boxes.cls.int().tolist()

            frame_info['frame_number'] = self.frame_num
            frame_info['track_count'] = len(track_ids)
            for box, track_id, object_label in zip(boxes, track_ids, ids_labels):
                object_info = {}
                x, y, w, h = box
                object_info['coordinate'] = (float(x), float(y))
                frame_info[f'{track_id}'] = object_info
                if track_id not in self.id_class_history:
                    self.id_class_history[track_id] = []
                self.id_class_history[track_id].append(object_label)
            if frame_info:
                self.frame_buffer.append(frame_info)
            if len(self.frame_buffer) == 5:
                frame_number = self.frame_buffer[-1]['frame_number']
                best_frame = max(self.frame_buffer, key=lambda x: x['track_count'])
                best_frame['frame_number'] = frame_number
                del best_frame['track_count']
                self.track_history.append(best_frame)
                self.frame_buffer = []

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
                'total_frame_count': self.frame_num,
                'track_history': self.track_history
            }
            self.shared_data['tracking_data'] = dict_data
            # print("Data sent to shared memory:", dict_data)

# Example usage
if __name__ == "__main__":
    model_path = "./yolov8n_nobox.pt"
    video_source = './nightvideo.mp4' # Use 0 for webcam, or replace with video file path

    manager = Manager()
    shared_data = manager.dict()

    tracker = VideoTracker(model_path, video_source, shared_data)
    fps, width, height = tracker.get_video_info()
    llm_summarizer = LLMSummarizer(fps=fps, size_x=width, size_y=height)
    
    tracking_thread = threading.Thread(target=tracker.process_video_stream)
    llm_summarizer = threading.Thread(target=llm_summarizer.read, args=(shared_data,))

    tracking_thread.start()
    llm_summarizer.start()

    llm_summarizer.join()
    tracking_thread.join()
