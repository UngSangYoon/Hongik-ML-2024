from dotenv import load_dotenv
import json
import google.generativeai as genai
import os
import time
from tqdm import tqdm
import yaml

load_dotenv('.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

with open('data/grouped_images_tracking_v2.json', 'r') as f:
    data = json.load(f)

def footage_to_text(footage, frame_interval_sec=1, screen_size=(1024, 576)):
    size = footage['size']
    total_frame_count = footage['total_frame_count']
    track_history = footage['track_history']
    text = f'SCREEN_SIZE: {screen_size}, DURATION: {int(total_frame_count * frame_interval_sec)} seconds\n\n'
    for frame_data in track_history:
        frame_number = frame_data['frame_number']
        text += f"TIME: +{frame_number * frame_interval_sec - 1} sec\n"
        for id in frame_data.keys():
            if id != 'frame_number':
                object_name = frame_data[id]['object name']
                coordinate = frame_data[id]['coordinate']
                x, y = coordinate
                x = int(x)
                y = int(y)
                text += f"OBJECT_NAME: {object_name}, OBJECT_ID: {id}, POSITION: ({x}, {y})\n"
        text += '\n'
    return text[:-1]

SYSTEM_MESSAGE = """You are a highly advanced AI model specialized in summarizing CCTV footage. You receive a list of detected objects with their positions at different timestamps, and your task is to generate a concise, coherent, and comprehensive summary of the entire scene. The summary should accurately describe the movements and changes in the positions of objects.\n

When summarizing:
1. Provide a comprehensive overview of the entire scene.
2. Mention significant movements and changes in object positions, including their names, IDs and coordinates.
3. If an object does not move significantly or stays in a similar position, describe its movement briefly or provide an average position.
4. Use clear and concise language.
5. Highlight notable events or actions.
6. Keep the summary to 1~2 sentences for brevity.
7. Do not include any information about the specific internal behavior of the objects in the scene."""

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              system_instruction=[SYSTEM_MESSAGE])

def request_gemini_response(model, text):
    response = model.generate_content(text)
    while not response or not response.parts:
        time.sleep(5)
        print('Retrying...')
        response = model.generate_content(text)
    return response.text.strip()

print(request_gemini_response(model, footage_to_text(data[13])))

new_data = []
for d in tqdm(data):
    text = footage_to_text(d)
    response = request_gemini_response(model, text)
    new_data.append({'json': d, 'text': text, 'response': response})

with open('data/gemini_response.json', 'w') as f:
    json.dump(new_data, f, indent=4)

with open('data/gemini_response.json', 'r') as f:
    new_data = json.load(f)

# read yaml
with open('data/data.yaml', 'r') as f:
    conf = yaml.safe_load(f)
names = [conf['names'][name] for name in conf['names']]
names = ', '.join(names)

SYSTEM_MESSAGE_FOR_INTERVAL = SYSTEM_MESSAGE
SYSTEM_MESSAGE_FOR_INTERVAL += "\n\n"
SYSTEM_MESSAGE_FOR_INTERVAL += f"""The CCTV footage specifications are as follows:
- Resolution: 1024x576 pixels.
- Objects that can appear: {names}.
- Maximum number of objects in the frame at any time: 10."""

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                               system_instruction=[SYSTEM_MESSAGE_FOR_INTERVAL])

def generate_random_interval_text(model, data_idx):
    interval_sec = new_data[data_idx]['json']['total_frame_count']
    interval_example = new_data[data_idx]['response']
    text = "Example of Interval Summary:\n\n"
    text += f"INTERVAL 0: (+0 to +{interval_sec} sec)\n"
    text += interval_example + "\n\n"
    text += "Generate the summaries for the next (at least 10) intervals as you imagine:"
    response = request_gemini_response(model, text)
    return response

interval_data = []
for i in tqdm(range(len(new_data))):
    response = generate_random_interval_text(model, i)
    print(response)
    interval_data.append({'json': new_data[i]['json'], 'text': new_data[i]['text'], 'response': new_data[i]['response'], 'interval_response': response})

with open('data/gemini_response_interval.json', 'w') as f:
    json.dump(interval_data, f, indent=4)

with open('data/gemini_response_interval.json', 'r') as f:
    interval_data = json.load(f)

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                system_instruction=[SYSTEM_MESSAGE])

def generate_interval_summary_text(model, data_idx):
    interval_example = interval_data[data_idx]['interval_response']
    text = interval_example + "\n\n"
    text += "Generate a comprehensive summary in 1~2 sentences for the entire footage based on the given interval summaries:"
    response = request_gemini_response(model, text)
    return response

final_data = []
for i in tqdm(range(len(interval_data))):
    response = generate_interval_summary_text(model, i)
    print(response)
    final_data.append({'json': interval_data[i]['json'], 
                       'text': interval_data[i]['text'], 
                       'response': interval_data[i]['response'], 
                       'interval_response': interval_data[i]['interval_response'], 
                       'interval_summary': response})
    
with open('data/gemini_response_interval_summary.json', 'w') as f:
    json.dump(final_data, f, indent=4)

with open('data/gemini_response_interval_summary.json', 'r') as f:
    final_data = json.load(f)

def generate_random_long_interval_text(model, data_idx):
    interval_sec = final_data[data_idx]['json']['total_frame_count'] * 10
    interval_min = int(interval_sec / 60)
    interval_example = final_data[data_idx]['interval_summary']
    text = "Example of Interval Summary:\n\n"
    text += f"INTERVAL 0: (+0 to +{interval_min} min {interval_sec % 60} sec)\n"
    text += interval_example + "\n\n"
    text += "Generate the summaries for the next (at least 10) intervals as you imagine:"
    response = request_gemini_response(model, text)
    return response

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                system_instruction=[SYSTEM_MESSAGE_FOR_INTERVAL])

print(generate_random_long_interval_text(model, 0))

long_interval_data = []
for i in tqdm(range(len(final_data))):
    response = generate_random_long_interval_text(model, i)
    print(response)
    long_interval_data.append({'json': final_data[i]['json'], 
                               'text': final_data[i]['text'], 
                               'response': final_data[i]['response'], 
                               'interval_response': final_data[i]['interval_response'], 
                               'interval_summary': final_data[i]['interval_summary'], 
                               'long_interval_response': response})

with open('data/gemini_response_long_interval.json', 'w') as f:
    json.dump(long_interval_data, f, indent=4)

with open('data/gemini_response_long_interval.json', 'r') as f:
    long_interval_data = json.load(f)

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                system_instruction=[SYSTEM_MESSAGE])

def generate_long_interval_summary_text(model, data_idx):
    interval_example = long_interval_data[data_idx]['long_interval_response']
    text = interval_example + "\n\n"
    text += "Generate a comprehensive summary in 1~2 sentences for the entire footage based on the given interval summaries:"
    response = request_gemini_response(model, text)
    return response

final_long_data = []
for i in tqdm(range(len(long_interval_data))):
    response = generate_long_interval_summary_text(model, i)
    print(response)
    final_long_data.append({'json': long_interval_data[i]['json'], 
                            'text': long_interval_data[i]['text'], 
                            'response': long_interval_data[i]['response'], 
                            'interval_response': long_interval_data[i]['interval_response'], 
                            'interval_summary': long_interval_data[i]['interval_summary'], 
                            'long_interval_response': long_interval_data[i]['long_interval_response'], 
                            'long_interval_summary': response})
    
with open('data/gemini_response_long_interval_summary.json', 'w') as f:
    json.dump(final_long_data, f, indent=4)
