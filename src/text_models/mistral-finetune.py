import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, MistralForSequenceClassification, set_seed, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# LIAR LABELS = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}

## Login to the HF
load_dotenv()
hf_key = os.getenv("HF_TOKEN")
login(token=hf_key)
    
# Set the global variables
seed = 1234
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mistral_model = "mistralai/Mistral-7B-v0.1"
model_save_path = '../saved_models/'

set_seed(seed)   

# Set variables for training
batch_size = 100

# Load the dataset and tokenize it
liar_dataset = load_dataset("chengxuphd/liar2")

# Rename label column to labels as it is parameter of model class
for split in liar_dataset:
    liar_dataset[split] = liar_dataset[split].rename_column("label", "labels")

# Initialize a tokenizer
tokenizer = AutoTokenizer.from_pretrained(mistral_model)

# A function to tokenize the "statements" and "labels" from the LIAR2 dataset
# It applies padding, truncation.
def tokenize(batch:dict):
    return tokenizer(batch["statement"], padding=True, truncation=True)

# Tokenizing the dataset
tokenized_dataset = liar_dataset.map(tokenize, batched=True, batch_size=batch_size, remove_columns=[col for col in liar_dataset["train"].column_names if col != "labels"])

# tokenized_dataset.set_format('torch')

# Data Collator: Objects tht will form a batch by using a list of dataset elements as input.
# We used the DataCollatorWithPadding as we are working on classification task
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8
)

# train_dataloader = torch.utils.data.DataLoader(
#     tokenized_dataset["train"],
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=data_collator
# )

# val_dataloader = torch.utils.data.DataLoader(
#     tokenized_dataset["validation"],
#     batch_size=batch_size,
#     shuffle=False,
#     collate_fn=data_collator
# )

# Load Mistral with a classification head
# Used bfloat16 for high precision, fast training with medium memory usage
# Could be used float32 for higher precision with higher memory usage
# Setting device_map="auto" automatically fills all available space on the GPU(s) first, 
# then the CPU, and finally, the hard drive (the absolute slowest option) if there is still not enough memory.
mistral_model = MistralForSequenceClassification.from_pretrained(mistral_model, num_labels=6, torch_dtype=torch.bfloat16,device_map="auto")

# Apply LoRa 
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices                 
    lora_alpha=32, # Scaling factor
    lora_dropout=0.05, # dropout for regularization
    bias="all", # Allow training biases
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target modules for Transformer-based LM (self-attention)
    task_type=TaskType.SEQ_CLS
)

peft_mistral_model = get_peft_model(mistral_model, lora_config)

#peft_mistral_model.to(device)

# DEBUG: Print number of parameters will train
peft_mistral_model.print_trainable_parameters()

# NOTE: NEED TO WORK ON
def metric():
    return None

training_args = TrainingArguments(
    output_dir=model_save_path,
    auto_find_batch_size=True,
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=5,
    learning_rate=1e-3,
    logging_dir=f"{model_save_path}/logs",
    logging_strategy="steps",
    logging_steps=100,
)

# Create a Trainer instance
trainer = Trainer(
    peft_mistral_model, 
    training_args, 
    train_dataset=tokenized_dataset["train"], 
    eval=tokenized_dataset["validation"], 
    tokenizer=tokenizer, 
    data_collator=data_collator, 
    compute_metrics=metric
)

trainer.train()