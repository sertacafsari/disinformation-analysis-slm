from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
    
def getVisionModel(model_name:str):
    """ Returns the real name of the given model name"""
    if model_name == "qwen":
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif model_name == "microsoft":
        model_name = "microsoft/Florence-2-base"
    elif model_name == "aya":
        model_name = "CohereLabs/aya-vision-8b"
    else:
        raise ValueError("No model or wrong model is provided for vision models")
    return model_name

class ModelSelection():

    def __init__(self, model_name:str, model_type:str, dataset_name:str, tokenizer):
        
        self.type = model_type
        # Check the model type and get the original name of the model
        if model_type == "text":
            self.model_name = getTextModel(model_name)
        elif model_type == "vision":
            self.model_name = getVisionModel(model_name)
        else:
            raise ValueError("The model type is invalid!")
        
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    def __configRoberta(self):
        """ Config the Roberta model to desired and return it """
        # Set the number of labels to 6 if the dataset is LIAR2
        if self.dataset_name == "liar2":
            self.config.num_labels = 6
        else:
            # Else set to 2 which is the default value to be sure it works
            self.config.num_labels = 2
        
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        return model

    def __configSmol(self):
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
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config, ignore_mismatched_sizes=True)

        model.resize_token_embeddings(len(self.tokenizer))

        return model


    def __configQwen(self):
        """ Config the Qwen3-0.6B model to desired and return it"""        
        # Set the number of labels to 6 if the dataset is LIAR2
        if self.dataset_name == "liar2":
            self.config.num_labels = 6
        else:
            # Else set to 2 which is the default value to be sure it works
            self.config.num_labels = 2

        # Set the <PAD> token
        self.config.pad_token_id = self.tokenizer.pad_token_id

        # Get the model 
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        # Resize the embedding layer
        model.resize_token_embeddings(len(self.tokenizer))

        # Resize the vocab size
        model.config.vocab_size = len(self.tokenizer)

        return model
    
    # TODO
    def __configVisionQwen():
        """ Config the Qwen2.5-VL model to desired and return it""" 
        pass

    # TODO
    def __configFlorence():
        pass

    # TODO
    def __configAya():
        pass

    def selectLanguageModel(self):
        """ Returns the language model with the selected config"""
        if self.type == "text":
            if "roberta" in self.model_name:
                return self.__configRoberta()
            elif "Smol" in self.model_name:
                return self.__configSmol()
            elif "Qwen" in self.model_name:
                return self.__configQwen()
            raise ValueError("No text model is inputted!")
        elif self.type == "vision":
            if "Qwen" in self.model_name:
                return self.__configVisionQwen()
            elif "aya" in self.model_name:
                return self.__configAya()
            elif "Florence" in self.model_name:
                return self.__configFlorence
            raise ValueError("No vision model is inputted!")
        else:
            raise ValueError("The model type is invalid!")