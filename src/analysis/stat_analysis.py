from huggingface_hub import login
import os
from dotenv import load_dotenv
from datasets import load_dataset
import matplotlib.pyplot as plot
import numpy as np

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)

data = load_dataset("chengxuphd/liar2")

LIAR2_LABELS = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}

amount_labels = {}

for split in data:  
    amount_labels[split] = {}
    for label in LIAR2_LABELS:
        amount_labels[split][LIAR2_LABELS[label]] = (len([x for x in data[split] if x["label"] == label])/len(data[split])*100)

    # # Get values and labels from the dictionary for each split
    # labels = list(amount_labels[split].keys())
    # values = list(amount_labels[split].values())
    
    # # Create pie chart
    # plot.figure(figsize=(10, 8))
    # plot.pie(values, labels=labels, autopct='%1.1f%%')
    # plot.title(f'Distribution of Labels in {split} set')
    # plot.savefig(f"liar2_dist_{split}")

class_sample_count = np.unique(data["train"]["label"], return_counts=True)
import code; code.interact(local=locals())
