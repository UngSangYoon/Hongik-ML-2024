from datasets import Dataset, DatasetDict
import json

with open("data/gemini_response_long_interval_summary.json", "r") as f:
    data = json.load(f)

SYSTEM_MESSAGE = """You are a highly advanced AI model specialized in summarizing CCTV footage. You receive a list of detected objects with their positions at different timestamps, and your task is to generate a concise, coherent, and comprehensive summary of the entire scene. The summary should accurately describe the movements and changes in the positions of objects.\n

When summarizing:
1. Provide a comprehensive overview of the entire scene.
2. Mention significant movements and changes in object positions, including their names, IDs and coordinates.
3. If an object does not move significantly or stays in a similar position, describe its movement briefly or provide an average position.
4. Use clear and concise language.
5. Highlight notable events or actions.
6. Keep the summary to 1~2 sentences for brevity.
7. Do not include any information about the specific internal behavior of the objects in the scene.\n\n"""

training_dataset = []
for d in data:
    text1_instruction = SYSTEM_MESSAGE + d['text']
    text1_output = d['response']
    text2_instruction = SYSTEM_MESSAGE + d['interval_response']
    text2_output = d['interval_summary']
    text3_instruction = SYSTEM_MESSAGE + d['long_interval_response']
    text3_output = d['long_interval_summary']
    training_dataset.append({'instruction': text1_instruction, 'output': text1_output})
    training_dataset.append({'instruction': text2_instruction, 'output': text2_output})
    training_dataset.append({'instruction': text3_instruction, 'output': text3_output})

training_dataset = Dataset.from_list(training_dataset)
training_dataset = training_dataset.shuffle(seed=42)
training_dataset = DatasetDict({'train': training_dataset})
training_dataset.save_to_disk("data/llm-training-dataset", num_shards={'train': 2})

from datasets import load_dataset

ds = load_dataset("data/llm-training-dataset")