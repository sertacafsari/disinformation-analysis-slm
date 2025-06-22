import os
import json
from datasets import load_dataset, load_from_disk, Image
from transformers import AutoTokenizer, AutoProcessor, DataCollatorWithPadding
from data_collator_vision import DataCollatorVision

def getTextModel(model_name:str):
    """ Returns the real name of the given model name"""
    if model_name == "roberta":
        return "FacebookAI/roberta-base"
    elif model_name == "smol":
        return "HuggingFaceTB/SmolLM2-360M"
    elif model_name == "qwen":
        return "Qwen/Qwen3-0.6B"
    raise ValueError("No model is provided!")
        
def getVisionModel(model_name:str):
    """ Returns the real name of the given model name"""
    if model_name == "qwen":
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    elif model_name == "microsoft":
        return "microsoft/Florence-2-base"
    elif model_name == "smol":
        return "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    raise ValueError("No model or wrong model is provided for vision models")


def changeLabelColumn(data):
    """ Changes the label column in the data to labels """
    # For each split in the dataset
    for split in data:
         # If there is "label" column
         if (data[split]["label"]):
              # Change the column name label to labels
              data[split] = data[split].rename_column("label", "labels")
              print(f"{split}'s label column changed to labels\n")
    return data

def saveHuggingFaceData(data, path:str):
    """ Saves the data from Huggingface to local directory """
    if not path:
        raise ValueError("A path must be provided!\n")
    # For each split in the dataset
    for split in data:
        if len(data[split]) > 0:
            path = f"{path}_{split}.json"
            data[split].to_json(path)
            print("Data is saved in JSON format")
        else:
            print(f"No {split} set is available")

def getData(dataset_name:str):
        """ Get the dataset name, if it is not saved then it downloads the dataset from the Huggingface. Otherwise, it just access dataset from the local folder """
        # If the name is liar2
        if dataset_name == "liar2":
            dataset_name = "chengxuphd/liar2"
            train_column = "statement"
            
            # Check whether the dataset is already saved 
            if os.path.exists("./data/datasets/liar2/liar2.json"):
                # Then load the data 
                with open("./data/datasets/liar2/liar2.json", "r", encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = load_dataset(dataset_name)
                print(f"Dataset {dataset_name} is loaded!")
                # Save the data in JSON format
                saveHuggingFaceData(data, "./data/datasets/liar2/liar2.json")
        # If the name is argilla
        elif dataset_name == "argilla":
            dataset_name = "argilla"
            train_column = "text"
            data = load_from_disk("./data/datasets/argilla")
        # If the name is faux
        elif dataset_name == "faux":
            dataset_name = "faux"
            train_column = "claim"
            data = load_from_disk("./data/datasets/faux")

             
        # Change the label column of data to labels for fine tuning process
        data = changeLabelColumn(data)

        # Return the data, its name, and train column
        return data, dataset_name, train_column
             

class DatasetSelection():

    def __init__(self, dataset_name:str, model_name:str, batch_size:int, model_type:str):
        self.data, self.dataset_name, self.train_column = getData(dataset_name)
        if model_type == "text":
            self.model_name = getTextModel(model_name)
        else:
            self.model_name = getVisionModel(model_name)
        self.batch_size = batch_size
    
    def __tokenizeTextData(self):
        """Loads the chosen model's tokenizer, tokenize the selected dataset and save it to the disk
        
            Parameters:
                model_name (str): The name of the model that will be used for Tokenizer
                finetune_column (str): The column of data that will be used in finetuning. For this project, the columns will contain fake news, mis/disinformation.
                save_data (bool (Optional)): A boolean value to decide whether the dataset will be saved or not.
                save_data_path (str (Optional)): A path to save the dataset.
            
            Return:
                dataset: The tokenized dataset
        """
        # Get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # If the tokenizer does not have <PAD> token
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        # A funciton to tokenize the given column in the batch
        def tokenize(batch):
            return tokenizer(batch[self.train_column], padding=True, truncation=True)
        
        # Set the tokenized dataset as self.dataset which will include input_ids, attention_mask and labels
        # Assuming each split has the same column names
        self.data = self.data.map(tokenize, batched=True, batch_size=self.batch_size, remove_columns=[col for col in self.data["train"].column_names if col != "labels"])

        print("The data is tokenized and replaced as main dataset")

        # Convert Python lists to Tensor objects
        self.data.set_format('torch')

        return self.data, tokenizer
    
    def __processVisualData(self):

        # Loads processor that includes a tokenizer for text and image processor
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        self.data = self.data.cast_column("img_main", Image())

        def process(batch):
            text_inputs = processor.tokenizer(batch["claim"], truncation=True)
            final_inputs = {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
                'pixel_values': batch["img_main"]
            }
            return final_inputs
        
        self.data = self.data.map(process,remove_columns=["img_main", "claim"])
        self.data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "pixel_values", "labels"],
        )

        print("Multimodal dataset is processed!")

        data_col = DataCollatorVision(processor)
        
        return self.data, processor, data_col
    
    def selectDataset(self):
        if self.dataset_name in ["chengxuphd/liar2", "argilla"]:
            data, tokenizer = self.__tokenizeTextData()
            # Data Collator: Objects that will form a batch by using a list of dataset elements as input.
            # We used the DataCollatorWithPadding as we are working on classification task
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)
            return data, tokenizer, data_collator
        else:
            return self.__processVisualData()