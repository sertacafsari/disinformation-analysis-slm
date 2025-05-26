import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
from finetuner import Finetuner
from dataset_loader import DatasetLoader
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import MistralForSequenceClassification, DataCollatorWithPadding, MistralConfig
from torch.utils.data import DataLoader


# Login to the HF
load_dotenv()
hf_key = os.getenv("HF_TOKEN")
wb_key = os.getenv("WB_TOKEN")
login(token=hf_key)

# Set the global variables
seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

mistral_model = "mistralai/Mistral-7B-v0.1"
dataset_name = "chengxuphd/liar2"
best_model_save_path = f"../data/models/mistral_liar2/best.pt" #TODO: Make it dynamic
save_raw_data_path = f"../data/datasets/liar2/"
save_tokenized_data_path = f"../data/tokenized_datasets/liar2_mistral/"

LIAR_LABELS = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}

batch_size = 256
learning_rate = 1e-3 
epochs = 4
logging_step = 1

# Initialize the WandB
wandb.login(key=wb_key)
wandb.init(
    entity="sbafsari-rug-university-of-groningen",
    project="mistral-liar2",
    name="initial-run1",
    config={
        "model": mistral_model,
        "dataset": "LIAR2",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed
    }
)

# Initialize the DatasetLoader
datasetLoader = DatasetLoader()

# Load the dataset and save it in JSON format
datasetLoader.loadDatasetFromHuggingFace(dataset_name, 
                                         save_data=False, 
                                         save_data_path=save_raw_data_path,
                                         json=True)

# Tokenize the dataset and save
tokenized_dataset, tokenizer = datasetLoader.tokenizeDatasetAndSave(model_name=mistral_model,
                                     finetune_column="statement",
                                     batch_size=batch_size,
                                     save_data=False,
                                     save_data_path=save_tokenized_data_path)


# Data Collator: Objects that will form a batch by using a list of dataset elements as input.
# We used the DataCollatorWithPadding as we are working on classification task
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=2
)

# Set the DataLoaders for train and validation data
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

val_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

# Set a Config for Mistral Model
conf = MistralConfig()
conf.vocab_size = len(tokenizer)
if (str(tokenizer.added_tokens_decoder[conf.vocab_size - 1]) == '<pad>'):
    conf.pad_token_id = conf.vocab_size - 1

conf.num_labels = len(LIAR_LABELS)

# Initialize the model with config
model = MistralForSequenceClassification.from_pretrained(mistral_model)

model.resize_token_embeddings(len(tokenizer))

print("Model is loaded, finetuning will be start soon...\n")

# Initialize a Finetuner
finetuner = Finetuner(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    model=model,
    seed=seed,
    lr=learning_rate
)

wandb.watch(model, log="all", log_freq=logging_step)

print("Fine tuning is starting...\n")

# Finetune the model
result = finetuner.train(device=device, epochs=epochs, logging_step=logging_step, best_model_path=best_model_save_path, wandb_run=wandb)

# Print train losses
print(f"Training Losses: {result['train_loss']} \n")
print(f"Validation Losses: {result['val_loss']} \n")