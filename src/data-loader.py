from datasets import  load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os


class DataLoader:
    """A class to load data from the HuggingFace, tokenize it and save to the data folder

        When the DataLoader class is initialized, it directly logins to your HuggingFace account.
        Then you can download raw data, tokenize it and save the tokenized version.
        The user can also save the raw data in JSON or CSV format.

    """

    def __init__(self):
        load_dotenv()
        huggingface_key = os.getenv("HF_TOKEN")
        if not huggingface_key:
            raise ValueError("HuggingFace token is not provided or wrong! Please check the documentation to do this!\n")
        login(token=huggingface_key)
        self.dataset = None


    def loadDatasetFromHuggingFace(self, dataset_name:str, save_data: bool=False, save_data_path:str=None, json:bool=False):
        """Loads the chosen dataset from the Huggingface, The user can save the raw data in CSV or JSON format.

            Args:
                dataset_name (str): The name of the dataset to download.
                save_data (bool (Optional)): A boolean value to decide whether the dataset will be saved or not.
                save_data_path (str (Optional)): A path to save the dataset.
                json (bool (Optional)): Save the dataset in JSON format if it is true; otherwise, save it in CSV format.
        """
        
        # Load the dataset
        self.dataset = load_dataset(dataset_name)

        print(f"Dataset {dataset_name} is loaded!\n")

        # For each split in the dataset
        for split in self.dataset:
            # If there is "label" column
            if (self.dataset[split]["label"]):
                # Change the column name label to labels
                self.dataset[split]  = self.dataset[split].rename_column("label", "labels")
                print(f"{split}'s label column changed to labels\n")
        

        # If the user wants to save the downloaded dataset
        if (save_data):
            if not save_data_path:
                raise ValueError("A path must be provided!\n")
            # For each split in the dataset
            for split in self.dataset:
                if len(self.dataset[split]) > 0:
                    path = f"{save_data_path}_{split}"
                    
                    # If preferred format is JSON
                    if json:
                        self.dataset[split].to_json(f"{path}.json")
                        print("Dataset is saved in JSON format.\n")
                    else:
                        self.dataset[split].to_csv(f"{path}.csv")
                        print("Dataset is saved in CSV format.\n")
                else:
                    print(f"No {split} set is available!\n")
        
                    
    def tokenizeDatasetAndSave(self, model_name:str, finetune_column:str, batch_size:int, save_data: bool = False, save_data_path:str = None):
        """Loads the chosen model's tokenizer, tokenize the selected dataset and save it to the disk
        
            Args:
                model_name (str): The name of the model that will be used for Tokenizer
                finetune_column (str): The column of data that will be used in finetuning. For this project, the columns will contain fake news, mis/disinformation.
                save_data (bool (Optional)): A boolean value to decide whether the dataset will be saved or not.
                save_data_path (str (Optional)): A path to save the dataset.
        """

        # Load the model' tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # If the tokenizer does not have PAD token
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

        # A function to tokenize the given column in the batch
        def tokenize(batch:str):
            return tokenizer(batch[finetune_column], padding=True, truncation=True)
        
        # Set the tokenized dataset as self.dataset which will include input_ids, attention_mask and labels
        self.dataset = self.dataset.map(tokenize, batched=True, batch_size=batch_size, remove_columns=[col for col in self.dataset["train"].column_names if col != "labels"])

        print("The data is tokenized and replaced as main dataset!\n")

        import code; code.interact(local=locals())

        if save_data:
            if not save_data_path:
                raise ValueError("A path must be provided!\n")
            # Save the tokenized dataset in to the disk
            self.dataset.save_to_disk(save_data_path)
            print("The data is saved!\n")


if __name__ == "__main__":

    # An example
    loadData = DataLoader()

    loadData.loadDatasetFromHuggingFace("chengxuphd/liar2", save_data=False, save_data_path="../data/deneme", json=True)

    loadData.tokenizeDatasetAndSave("mistralai/Mistral-7B-v0.1", "statement", 32, save_data=False, save_data_path="../data/tokenized")
    
