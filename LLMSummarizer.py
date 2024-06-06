import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMSummarizer:
    def __init__(self, fps=30, size_x=640, size_y=360):
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

    def llm_generate(self, text):
        model_input = self.SYSTEM_MESSAGE + text
        decoded_output = ""
        while len(decoded_output[len(model_input):].strip()) < 50 or len(decoded_output[len(model_input):].strip()) > 500: # Ensure the output is not too short or long
            inputs = self.tokenizer(model_input, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=2048)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output[len(model_input):].strip()

    def read(self, shared_data):
        # 5 frame (second) -> 1 framelist summary
        # 5 framelist summaries -> 1 interval summary
        # 5 interval summaries -> 1 long interval summary
        while True:
            # 5 frames -> 1 framelist summary
            if 'tracking_data' in shared_data and self.current_tracking_data < len(shared_data['tracking_data']['track_history']) - 5: # Ensure there are at least 5 new frames
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

            # 5 framelist summaries -> 1 interval summary
            if len(self.frame_summaries) >= 5: # Process interval summaries every 5 tracked frame summaries
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

            # 5 interval summaries -> 1 long interval summary
            if len(self.interval_summaries) >= 5: # Process long interval summaries every 5 interval summaries
                print("LLM is processing the long interval summary...")
                long_intervals = self.intervals_to_text(self.interval_summaries)
                long_interval_summary = self.llm_generate(long_intervals)
                print("Long Interval Summary:", long_interval_summary)
                starttime = int(self.interval_summaries[0]['start_time'])
                endtime = int(self.interval_summaries[-1]['end_time'])
                self.write(self.long_interval_output_file, long_interval_summary, starttime, endtime)
                self.interval_summaries = []
                
    def write(self, output_file, summary, starttime, endtime):
        with open(output_file, 'wa') as f:
            f.write(f"TIME: ({self.time_to_text(starttime)} to {self.time_to_text(endtime)})\n")
            f.write(summary)
            f.write("\n\n")


#[{'frame_number': 180, '2': {'coordinate': (376.98193359375, 108.3979721069336), 'object name': 'car'}}, {'frame_number': 210, '2': {'coordinate': (417.44854736328125, 153.99610900878906), 'object name': 't name': 'car'}}, {'frame_number': 750, '6': {'coordinate': (121.0830078125, 37.1324577331543), 'object name': 'car'}}, {'frame_number': 780, '6': {'coordinate': (120.88444519042969, 35.3698844909668), 'object name': 'car'}, '7': {'coordinate': (358.8294982910156, 53.35453796386719), 'object name': 'car'}}, {'frame_number': 810, '7': {'coordinate': (324.2286682128906, 66.62930297851562), 'object name': 'car'}}, {'frame_number': 840, '7': {'coordinate': (309.5027770996094, 73.25189971923828), 'object name': 'car'}}, {'frame_number': 870, '7': {'coordinate': (315.6584167480469, 71.94894409179688), 'object name': 'car'}}, {'frame_number': 900, '7': {'coordinate': (353.6012878417969, 65.21642303466797), 'object name': 'car'}}, {'frame_number': 930, '7': {'coordinate': (369.57928466796875, 58.2105598449707), 'object name': 'car'}}, {'frame_number': 960, '7': {'coordinate': (367.5044250488281, 47.27363204956055), 'object name': 'car'}, '6': {'coordinate': (119.79318237304688, 33.673858642578125), 'object name': 'car'}}, {'frame_number': 990, '7': {'coordinate': (358.7887268066406, 43.240020751953125), 'object name': 'car'}, '6': {'coordinate': (120.547119140625, 34.52531814575195), 'object name': 'car'}}, {'frame_number': 1020, '7': {'coordinate': (343.2637634277344, 40.044708251953125), 'object name': 'car'}, '6': {'coordinate': (120.82643127441406, 34.888484954833984), 'object name': 'car'}}, {'frame_number': 1050, '7': {'coordinate': (336.27740478515625, 38.794830322265625), 'object name': 'car'}, '6': {'coordinate': (120.67593383789062, 35.29459762573242), 'object name': 'car'}}, {'frame_number': 1080, '7': {'coordinate': (372.3018798828125, 39.06236267089844), 'object name': 'car'}, '6': {'coordinate': (120.97589111328125, 36.428218841552734), 'object name': 'car'}}, {'frame_number': 1110, '7': {'coordinate': (435.17401123046875, 37.20831298828125), 'object name': 'car'}, '6': {'coordinate': (121.2057876586914, 35.73623275756836), 'object name': 'car'}}, {'frame_number': 1140, '6': {'coordinate': (115.82227325439453, 35.98075485229492), 'object name': 'car'}}]

text = """SCREEN_SIZE: (1024, 576), DURATION: 12 seconds

TIME: +0 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (444, 339)

TIME: +1 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (474, 332)

TIME: +2 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (517, 342)

TIME: +3 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (533, 347)

TIME: +4 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (561, 366)

TIME: +5 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (567, 364)

TIME: +6 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (557, 363)

TIME: +7 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (561, 353)

TIME: +8 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (554, 350)

TIME: +9 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (534, 355)

TIME: +10 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (508, 356)

TIME: +11 sec
OBJECT_NAME: Table, OBJECT_ID: 4523, POSITION: (435, 333)"""

text2 = """## Interval Summaries:

**INTERVAL 1: (+2 min 50 sec to +5 min 10 sec)**

The scene shows a human (ID: 12345) entering from the left, moving to the center, and standing still. A bicycle (ID: 16747) enters and exits to the right, followed by a second bicycle (ID: 91011) traveling in the opposite direction. A car (ID: 98765) remains stationary at the center of the scene.

**INTERVAL 2: (+5 min 10 sec to +7 min 30 sec)**

A human (ID: 78901) enters from the right and walks towards the center, where they briefly interact with the first human (ID: 12345) before exiting leftward. A car (ID: 98765) exits the scene rightward, followed by a bus (ID: 34567) entering from the left and traveling to the center.

**INTERVAL 3: (+7 min 30 sec to +9 min 50 sec)**

The bus (ID: 34567) remains stationary in the center of the scene while a human (ID: 12345) re-enters from the left and moves to the right side of the scene. A bicycle (ID: 91011) enters from the left and exits rightward, followed by a second bicycle (ID: 87654) traveling in the opposite direction.

**INTERVAL 4: (+9 min 50 sec to +12 min 10 sec)**

A human (ID: 12345) exits the scene rightward. The bus (ID: 34567) exits rightward. A car (ID: 23456) enters from the left and travels across the scene, followed by a dog (ID: 24680) entering from the right and walking towards the center.

**INTERVAL 5: (+12 min 10 sec to +14 min 30 sec)**

The car (ID: 23456) exits the scene rightward. The dog (ID: 24680) remains in the center, wagging its tail. A human (ID: 78901) enters from the left and walks towards the dog.

**INTERVAL 6: (+14 min 30 sec to +16 min 50 sec)**

The human (ID: 78901) interacts with the dog (ID: 24680) and then exits leftward. A human (ID: 56789) enters from the right and moves towards the center, followed by a car (ID: 98765) entering from the left.

**INTERVAL 7: (+16 min 50 sec to +19 min 10 sec)**

The human (ID: 56789) interacts with the car (ID: 98765) and then exits leftward. The car (ID: 98765) remains in the center of the scene. A bicycle (ID: 16747) enters from the left and exits rightward, followed by a motorcycle (ID: 43210) entering from the left and traveling to the center.

**INTERVAL 8: (+19 min 10 sec to +21 min 30 sec)**

The motorcycle (ID: 43210) remains in the center of the scene. A human (ID: 12345) enters from the left and walks towards the motorcycle.

**INTERVAL 9: (+21 min 30 sec to +23 min 50 sec)**

The human (ID: 12345) interacts with the motorcycle (ID: 43210) and then exits leftward. The motorcycle (ID: 43210) exits rightward. A car (ID: 23456) enters from the left and travels across the scene.

**INTERVAL 10: (+23 min 50 sec to +26 min 10 sec)**

The car (ID: 23456) exits the scene rightward. A human (ID: 78901) enters from the right and walks towards the center, followed by a dog (ID: 24680) entering from the left and walking towards the center."""
