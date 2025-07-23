from huggingface_hub import login
import os
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
import json

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
print("Log into HuggingFace")

data = load_dataset("chengxuphd/liar2")
print("LIAR2 is loaded!")

liar_total_sentence_lengths = []

for split in data.keys():
    # Get the statement column
    statement = data[split]["statement"]
    word_counts = [len(str(text).split()) for text in statement]
    liar_total_sentence_lengths.extend(word_counts)

# Calculate the mean, standard deviation and median of the LIAR2 dataset samples
mean_length_liar2 = np.mean(liar_total_sentence_lengths)
sd_liar = np.std(liar_total_sentence_lengths)
median_length_liar2 = np.median(liar_total_sentence_lengths)

argilla_total_sentence_lengths = []
with open('./data/datasets/argilla/argilla_processed.json', 'r') as f:
    json_data = json.load(f)
for item in json_data:
    text = item["text"]
    word_counts = [len(str(text).split())]
    argilla_total_sentence_lengths.extend(word_counts)

# Calculate the mean, standard deviation and median of the Fake News dataset samples
mean_length_argilla = np.mean(argilla_total_sentence_lengths)
sd_argilla = np.std(argilla_total_sentence_lengths)
median_length_argilla = np.median(argilla_total_sentence_lengths)


print(f"LIAR2 Dataset Mean Statement Length:{mean_length_liar2} ± {sd_liar}\nMedian Statement Length:{median_length_liar2}")
print(f"Fake News Dataset Mean Statement Length:{mean_length_argilla} ± {sd_argilla} \nMedian Statement Length:{median_length_argilla}")
