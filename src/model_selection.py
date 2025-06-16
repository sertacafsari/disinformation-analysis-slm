from argparse import ArgumentParser
from transformers import AutoConfig, AutoModelForSequenceClassification

def getTextModel(model_name:str):
    """ Returns the real name of the given model name"""
    if model_name == "roberta":
        model_name = "FacebookAI/roberta-base"
    elif model_name == "smol":
        model_name = "HuggingFaceTB/SmolLM2-360M"
    elif model_name == "qwen":
        model_name = "Qwen/Qwen3-0.6B"
    else:
        raise ValueError("No model is provided!")
    
    return model_name
    

class ModelSelection():

    def __init__(self, model_name:str, dataset_name:str, tokenizer):
        self.parser = ArgumentParser()
        self.text_model_name = getTextModel(model_name)    
        self.config = AutoConfig.from_pretrained(self.text_model_name)
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def configRoberta(self):
        """ Config the Roberta model to desired and return it """
        # Set the number of labels to 6 if the dataset is LIAR2
        if self.dataset_name == "liar2":
            self.config.num_labels = 6
        else:
            # Else set to 2 which is the default value to be sure it works
            self.config.num_labels = 2
        
        model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name, config=self.config)

        return model

    def configSmol(self):
        """ Config the SmolLM-360 model to desired and return it"""        
        # Set the number of labels to 6 if the dataset is LIAR2
        if self.dataset_name == "liar2":
            self.config.num_labels = 6
        else:
            # Else set to 2 which is the default value to be sure it works
            self.config.num_labels = 2

        # Set the <PAD> token
        self.config.pad_token_id = self.tokenizer.pad_token_id

        # Set the vocab size
        self.config.vocab_size = len(self.tokenizer)

        # Get the model
        model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name, config=self.config, ignore_mismatched_sizes=True)

        model.resize_token_embeddings(len(self.tokenizer))

        return model


    def configQwen(self):
        """ Config the Qwen-0.6B model to desired and return it"""        
        # Set the number of labels to 6 if the dataset is LIAR2
        if self.dataset_name == "liar2":
            self.config.num_labels = 6
        else:
            # Else set to 2 which is the default value to be sure it works
            self.config.num_labels = 2

        # Set the <PAD> token
        self.config.pad_token_id = self.tokenizer.pad_token_id

        # Get the model 
        model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name, config=self.config)

        # Resize the embedding layer
        model.resize_token_embeddings(len(self.tokenizer))

        # Resize the vocab size
        model.config.vocab_size = len(self.tokenizer)

        return model


    def selectTextModel(self):
        """ Returns the model with the selected config"""
        if "roberta" in self.text_model_name:
            return self.configRoberta()
        elif "Smol" in self.text_model_name:
            return self.configSmol()
        elif "qwen" in self.text_model_name:
            return self.configQwen()
        raise ValueError("No text model is inputted!")