import torch
import os
import sys
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
from finetuner import Finetuner
from dataset_loader import DatasetLoader
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import RobertaForSequenceClassification, DataCollatorWithPadding, RobertaConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


# Login to the HF
load_dotenv()
hf_key = os.getenv("HF_TOKEN")
wb_key = os.getenv("WB_TOKEN")
login(token=hf_key)

# Set the global variables
seed = 184

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

roberta = "FacebookAI/roberta-base"
dataset_name = "chengxuphd/liar2"
best_model_save_path = f"./data/models/roberta_liar2/best.pt" #TODO: Make it dynamic
save_raw_data_path = f"../data/datasets/liar2/"
save_tokenized_data_path = f"../data/tokenized_datasets/liar2_roberta/"

LIAR_LABELS = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}

batch_size = 32
# learning_rate = 1e-5
epochs = 5
logging_step = 1


# Initialize the model and set to GPU before data

# Initialize the WandB
wandb.login(key=wb_key)

# Initialize the DatasetLoader
datasetLoader = DatasetLoader()

# Load the dataset and save it in JSON format
datasetLoader.loadDatasetFromHuggingFace(dataset_name, 
                                         save_data=False, 
                                         save_data_path=save_raw_data_path,
                                         json=True)

# Tokenize the dataset and save
tokenized_dataset, tokenizer = datasetLoader.tokenizeDatasetAndSave(model_name=roberta,
                                     finetune_column="statement",
                                     batch_size=batch_size,
                                     save_data=False,
                                     save_data_path=save_tokenized_data_path)

# Data Collator: Objects that will form a batch by using a list of dataset elements as input.
# We used the DataCollatorWithPadding as we are working on classification task
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8
)

# classes, class_counts = np.unique(tokenized_dataset["train"]["labels"], return_counts=True)
# class_weights = 1.0 / class_counts.astype(np.float32)

# sample_weights = class_weights[np.array(tokenized_dataset["train"]["labels"])]

# sample_weights_tensor = torch.from_numpy(sample_weights).double()

# Create a sampler to balance the training data
# As we want to have a model that learn minority data labels too
# sampler = WeightedRandomSampler(
#     weights=sample_weights_tensor,
#     num_samples=len(sample_weights_tensor),
#     replacement=True
# )

# Set the DataLoaders for train and validation data
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=True
)

val_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True
)

test_dataloader = DataLoader(
    tokenized_dataset["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True
)

print("Fine tuning will start soon")

# Set a Config for Roberta Model
conf = RobertaConfig.from_pretrained(roberta)
conf.num_labels = len(LIAR_LABELS)

# # NOTE: SWEEPING EXPERIMENT
learning_rates = [5e-7]

for lr in learning_rates:

    # Initialize the model with config
    model = RobertaForSequenceClassification.from_pretrained(roberta, config=conf)

    print("Model is loaded\n")

    # Transfer model to GPU (or back to CPU)
    model.to(device)

    print(f"Model is on {next(model.parameters()).device}\n")

    # Initialize wandb
    wandb.init(
        entity="sbafsari-rug-university-of-groningen",
        project="disinformation-slm",
        name=f"roberta-{batch_size}-{lr}",
        config={
            "model": roberta,
            "dataset": "LIAR2",
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "seed": seed
        }
    )

    best_model_save_path = f"./data/models/roberta_liar2/roberta_{batch_size}_{lr}.pt"

    # Initialize a Finetuner
    finetuner = Finetuner(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        lr=lr
    )

    wandb.watch(model, log="all", log_freq=logging_step)

    # Finetune the model
    result = finetuner.train(device=device, epochs=epochs, logging_step=logging_step, best_model_path=best_model_save_path, wandb_run=wandb)

    # Print train losses
    print(f"Training Losses: {result['train_loss']} \n")
    print(f"Validation Losses: {result['val_loss']} \n")

    # Test the model
    finetuner.test_model(device,test_dataloader, best_model_save_path, logging_step, wandb, 2)

    wandb.finish()