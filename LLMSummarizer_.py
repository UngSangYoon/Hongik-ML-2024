import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMSummarizer:
    def __init__(self, fps, size_x, size_y):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("Hongik-ML-2024/cctv-llm")
        self.model = AutoModelForCausalLM.from_pretrained("Hongik-ML-2024/cctv-llm", trust_remote_code=True, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.current_tracking_data = 0
        self.interval_number = 1
        self.long_interval_number = 1
        self.frame_summaries = []
        self.interval_summaries = []
        self.fps = fps
        self.size_x = size_x
        self.size_y = size_y
        self.short_interval_output_file = 'short_interval_output.txt' # short interval summaries will be written to this file
        self.long_interval_output_file = 'long_interval_output.txt' # long interval summary will be written to this file

        self.SYSTEM_MESSAGE = """You are a highly advanced AI model specialized in summarizing CCTV footage. You receive a list of detected objects with their positions at different timestamps, and your task is to generate a concise, coherent, and comprehensive summary of the entire scene. The summary should accurately describe the movements and changes in the positions of objects.\n

When summarizing:
1. Provide a comprehensive overview of the entire scene.
2. Mention significant movements and changes in object positions, including their names, IDs and coordinates.
3. If an object does not move significantly or stays in a similar position, describe its movement briefly or provide an average position.
4. Use clear and concise language.
5. Highlight notable events or actions.
6. Keep the summary to 1~2 sentences for brevity.
7. Do not include any information about the specific internal behavior of the objects in the scene.\n\n"""

    def time_to_text(self, time: int) -> str:
        if time < 60:
            return f"{time} sec"
        minutes = time // 60
        seconds = time % 60
        return f"{minutes} min {seconds} sec"

    def frames_to_text(self, framelist):
        text = ""
        for frame in framelist:
            text += f"TIME: +{self.time_to_text(int(frame['frame_number']) // self.fps)}\n"
            for obj_id, obj_data in frame.items():
                if obj_id != 'frame_number':
                    coordinate = (int(obj_data['coordinate'][0]/self.size_x*1024), int(obj_data['coordinate'][1]/self.size_y*576))
                    text += f"OBJECT_NAME: {obj_data['object name']}, OBJECT_ID: {obj_id}, POSITION: {coordinate}\n"
            text += "\n"
        return text

    def intervals_to_text(self, interval_list):
        text = "## Interval Summaries:\n\n"
        for interval in interval_list:
            text += f"**INTERVAL {interval['interval_number']}: ({self.time_to_text(interval['start_time'])} to {self.time_to_text(interval['end_time'])})**\n\n"
            text += interval['text'] + "\n\n"
        return text

    def llm_generate(self, text, max_length=2048):
        model_input = self.SYSTEM_MESSAGE + text
        decoded_output = ""
        input_length = len(self.tokenizer.encode(model_input))
        
        if input_length > max_length:
            split_texts = self.split_text(text, max_length - len(self.SYSTEM_MESSAGE))
            summaries = []
            for split_text in split_texts:
                summaries.append(self.llm_generate(split_text, max_length))
            return " ".join(summaries)
        
        while len(decoded_output[len(model_input):].strip()) < 50 or len(decoded_output[len(model_input):].strip()) > 500: # Ensure the output is not too short or long
            inputs = self.tokenizer(model_input, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=2048)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(decoded_output)
        return decoded_output[len(model_input):].strip()
    
    def split_text(self, text, max_length):
        words = text.split('\n\n')
        splits = []
        current_split = ""
        for word in words:
            if len(self.tokenizer.encode(current_split + word)) > max_length:
                splits.append(current_split)
                current_split = word + "\n\n"
            else:
                current_split += word + "\n\n"
        if current_split:
            splits.append(current_split)
        return splits

    def read(self, shared_data):
        while True:
            # 10 frames -> 1 framelist summary
            if 'tracking_data' in shared_data and self.current_tracking_data < len(shared_data['tracking_data']['track_history']) - 10: # Ensure there are at least 10 new frames
                print("LLM is processing the tracking data...")
                track_history = shared_data['tracking_data']['track_history'][self.current_tracking_data:]
                self.current_tracking_data = len(shared_data['tracking_data']['track_history'])
                text = self.frames_to_text(track_history)
                summary = self.llm_generate(text)
                print("LLM Summary:", summary)
                self.frame_summaries.append({'interval_number': self.interval_number, 
                                             'start_time': int(track_history[0]['frame_number'])//self.fps, 
                                             'end_time': int(track_history[-1]['frame_number'])//self.fps, 
                                             'text': summary})
                self.interval_number += 1

            # 10 framelist summaries -> 1 interval summary
            if len(self.frame_summaries) >= 10: # Process interval summaries every 10 tracked frame summaries
                print("LLM is processing the interval summary...")
                intervals = self.intervals_to_text(self.frame_summaries)
                interval_summary = self.llm_generate(intervals)
                print("Interval Summary:", interval_summary)
                starttime = int(self.frame_summaries[0]['start_time'])
                endtime = int(self.frame_summaries[-1]['end_time'])
                self.interval_summaries.append({'interval_number': self.long_interval_number, 
                                                'start_time': starttime,
                                                'end_time': endtime,
                                                'text': interval_summary})
                self.write(self.short_interval_output_file, interval_summary, starttime, endtime)
                self.frame_summaries = []
                self.long_interval_number += 1
                self.interval_number = 1

            # 10 interval summaries -> 1 long interval summary
            if len(self.interval_summaries) >= 10: # Process long interval summaries every 10 interval summaries
                print("LLM is processing the long interval summary...")
                long_intervals = self.intervals_to_text(self.interval_summaries)
                long_interval_summary = self.llm_generate(long_intervals)
                print("Long Interval Summary:", long_interval_summary)
                starttime = int(self.interval_summaries[0]['start_time'])
                endtime = int(self.interval_summaries[-1]['end_time'])
                self.write(self.long_interval_output_file, long_interval_summary, starttime, endtime)
                self.interval_summaries = []
                
    def write(self, output_file, summary, starttime, endtime):
        with open(output_file, 'a') as f:
            f.write(f"TIME: ({self.time_to_text(starttime)} to {self.time_to_text(endtime)})\n")
            f.write(summary)
            f.write("\n\n")
