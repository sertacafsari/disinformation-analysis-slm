from argparse import ArgumentParser
from model_selection import ModelSelection
from dataset_selection import DatasetSelection
from finetuner import Finetuner
from testing.tester import Tester
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import wandb
import os


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
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)

# Get the arguments
args = parser.parse_args()

seed = args.seed
epochs = args.epochs
model_name = args.model_name
model_type = args.model_type
dataset_name = args.dataset_name
batch_size = args.batch_size
learning_rate = args.lr

# Global variables
set_seed(seed=seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select dataset
data_selector = DatasetSelection(dataset_name, model_name, batch_size, model_type)
data, tokenizer, data_collator = data_selector.selectDataset()

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

# Select the model
model_selector = ModelSelection(model_name, model_type, dataset_name, tokenizer)
print("Model is selected")

learning_rates = [learning_rate]
for lr in learning_rates:
    model = model_selector.selectLanguageModel()
    model.to(device)
    print(f"Model is on {next(model.parameters()).device}\n")

    wandb.init(
        entity="sbafsari-rug-university-of-groningen",
        project="test-disinformation",
        name=f"{dataset_name}-{model_name}-{batch_size}-{lr}-test-2",
        config={
            "model": model_name,
            "dataset": dataset_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "seed": seed
        }
    )

    best_model_save_path = f"./data/models/{model_name}/{model_name}_{batch_size}_{lr}-test-1.pt"

    finetuner = Finetuner(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        lr=lr
    )
    print(f"Model is being trained with the lr: {lr}")

    wandb.watch(model, log="all", log_freq=1)
    
    model.gradient_checkpointing_enable()

    if model_type == "text":
        result = finetuner.train(device=device, epochs=epochs, logging_step=1, best_model_path=best_model_save_path, wandb_run=wandb)
    else:
        result = finetuner.vision_train(device=device, epochs=epochs, logging_step=1, gradient_acc_step=4, best_model_path=best_model_save_path, wandb_run=wandb)

    # Print train losses
    print(f"Training Losses: {result['train_loss']} \n")
    print(f"Validation Losses: {result['val_loss']} \n")

    # Load the best model's weights
    model.load_state_dict(torch.load(best_model_save_path, map_location=device))

    # Test the model
    tester = Tester(model=model,
        device=device,
        test_split=test_dataloader,
        logging_step=1,
        wandb_run=wandb
    )
    
    tester.testModel(k=2)

    wandb.finish()

