from datasets import  load_dataset
from transformers import AutoTokenizer


class DatasetLoader:
    """A class to load data from the HuggingFace, tokenize it and save to the data folder

        When the DatasetLoader class is initialized, it directly logins to your HuggingFace account.
        Then you can download raw data, tokenize it and save the tokenized version.
        The user can also save the raw data in JSON or CSV format.

    """

    def __init__(self):
        self.dataset = None

    def loadDatasetFromHuggingFace(self, dataset_name:str, save_data: bool=False, save_data_path:str=None, json:bool=False):
        """Loads the chosen dataset from the Huggingface, The user can save the raw data in CSV or JSON format.

            Parameters:
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
        
            Parameters:
                model_name (str): The name of the model that will be used for Tokenizer
                finetune_column (str): The column of data that will be used in finetuning. For this project, the columns will contain fake news, mis/disinformation.
                save_data (bool (Optional)): A boolean value to decide whether the dataset will be saved or not.
                save_data_path (str (Optional)): A path to save the dataset.
            
            Return:
                dataset: The tokenized dataset
        """

        # Load the model' tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # If the tokenizer does not have PAD token
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

        # A function to tokenize the given column in the batch
        def tokenize(batch:str):
            return tokenizer(batch[finetune_column], padding=True, truncation=True)

        # labels, input_ids attention_mask 

        # Set the tokenized dataset as self.dataset which will include input_ids, attention_mask and labels
        self.dataset = self.dataset.map(tokenize, batched=True, batch_size=batch_size, remove_columns=[col for col in self.dataset["train"].column_names if col != "labels"])

        import code; code.interact(local=locals())

        print("The data is tokenized and replaced as main dataset!\n")

        if save_data:
            if not save_data_path:
                raise ValueError("A path must be provided!\n")
            # Save the tokenized dataset in to the disk
            self.dataset.save_to_disk(save_data_path)
            print("The data is saved!\n")
        
        # Convert Python lists to Tensor objects
        self.dataset.set_format('torch')
        
        return self.dataset, tokenizer


if __name__ == "__main__":

    # An example
    loadData = DatasetLoader()

    loadData.loadDatasetFromHuggingFace("chengxuphd/liar2", save_data=True, save_data_path="../data/deneme", json=True)

    tokenized_dataset = loadData.tokenizeDatasetAndSave("FacebookAI/roberta-base", "statement", 32, save_data=True, save_data_path="../data/tokenized")