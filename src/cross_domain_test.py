from argparse import ArgumentParser
from model_selection import ModelSelection
from dataset_selection import DatasetSelection
from finetuner import Finetuner
from tester import Tester
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import wandb
import os

# Random seed between 1 and 10000
# Login
load_dotenv()
hf_key = os.getenv("HF_TOKEN")
wb_key = os.getenv("WB_TOKEN")

login(hf_key)
wandb.login(key=wb_key)


def set_seed(seed: int):
    """ A function to set seed for a run """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize the Parser
parser = ArgumentParser(description="A parser to get information from the job submitting script")

# Add arguments to parser
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)

# Get the arguments
args = parser.parse_args()

model_name = args.model_name
model_type = args.model_type
dataset_name = args.dataset_name
batch_size = args.batch_size
learning_rate = args.lr

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select dataset
data_selector = DatasetSelection(dataset_name, model_name, batch_size, model_type)
data, tokenizer, data_collator = data_selector.selectDataset()

# Select model
model_selector = ModelSelection(model_name, model_type, "liar2", tokenizer)
model = model_selector.selectLanguageModel()
print("Model is selected")

model.to(device)
print(f"Model is on {next(model.parameters()).device}\n")



for step in range(3):

    print(f"Testing {step+1}:")

    seed = random.randint(1, 10000)
    set_seed(seed=seed)

    # Set the DataLoaders for train and validation data
    train_dataloader = DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        data["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        data["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )

    wandb.init(
        entity="sbafsari-rug-university-of-groningen",
        project="disinformation-slm",
        name=f"{dataset_name}-{model_name}-test-{step+1}",
        config={
            "model": model_name,
            "dataset": dataset_name,
            "batch_size": batch_size,
            "seed": seed
        }
    )
    
    best_model_save_path = f"./data/models/roberta_liar2/roberta_32_2e-05-last-1.pt"

    model.load_state_dict(torch.load(best_model_save_path, map_location=device))

    wandb.watch(model, log="all", log_freq=1)

    tester = Tester(model=model, device=device, test_split=test_dataloader, logging_step=1, wandb_run=wandb)

    tester.testArgilla()

    wandb.finish()