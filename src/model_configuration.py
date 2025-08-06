from transformers import AutoConfig, AutoModelForSequenceClassification
from classification_head import SmolVisionModelForClassification, CLIPForClassification, LlavaForClassification
import torch

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
    if model_name == "clip":
        return "openai/clip-vit-base-patch32"
    elif model_name == "smol":
        return "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    elif model_name == "llava":
        return "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    raise ValueError("No model or wrong model is provided for vision models")

class ModelConfiguration():
    """
        A class to select models and change their configurations.
        Params:
            model_name(str): the name of the model.
            model_type(str): the type of the model, which is either text or vision.
            dataset_name(str): the name of the selected dataset.
            tokenizer: used tokenizer.
    """
    def __init__(self, model_name:str, model_type:str, dataset_name:str, tokenizer):
        
        self.type = model_type
        # Check the model type and get the original name of the model
        if model_type == "text":
            self.model_name = getTextModel(model_name)
        elif model_type == "vision":
            self.model_name = getVisionModel(model_name)
        else:
            raise ValueError("The model type is invalid!")
        
        self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
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
    
    def __configClip(self):
        """ Config the SmolLM2-VL model to desired and return it""" 
        # Config
        self.config.num_labels = 2
        self.config.id2label = {0: "true", 1: "false"}
        self.config.label2id = {"true":0, "false": 1}
        self.config.use_cache = False

        if self.tokenizer.tokenizer.pad_token_id is not None:
            self.config.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        
        model = CLIPForClassification.from_pretrained(self.model_name, config=self.config, torch_dtype=torch.float32, class_weight=[0.76491646778, 0.23508353222])

        model.config.vocab_size = len(self.tokenizer.tokenizer)

        return model



    def __configVisionSmol(self):
        """ Config the SmolLM2-VL model to desired and return it""" 
        # Config
        self.config.num_labels = 3
        self.config.id2label = {0: "true", 1: "false", 2: "mixed"}
        self.config.label2id = {"true":0, "false": 1, "mixed":2}
        self.config.use_cache = False

        if self.tokenizer.tokenizer.pad_token_id is not None:
            self.config.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        
        model = SmolVisionModelForClassification.from_pretrained(self.model_name, config=self.config, torch_dtype=torch.bfloat16)

        model.vision_language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))

        return model


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
            if "Smol" in self.model_name:
                return self.__configVisionSmol()
            raise ValueError("No vision model is inputted!")
        else:
            raise ValueError("The model type is invalid!")