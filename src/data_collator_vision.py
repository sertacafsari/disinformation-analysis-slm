import torch
from torch.nn.functional import pad


class DataCollatorVision:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):

        # Take out all images
        imgs = [f["pixel_values"] for f in features]

        # Processor will handle padding
        image_batch = self.processor.image_processor(images=imgs, return_tensors="pt")

        # Get pixel values
        pixel_values = image_batch["pixel_values"]

        # Get labels
        labels = torch.tensor([feature["labels"] for feature in features])

        # Get text input
        text_inputs = {
            "input_ids":      [feature["input_ids"]      for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features],
        }

        text_batch = self.processor.tokenizer.pad(text_inputs, padding="longest", return_tensors="pt")

        text_batch["pixel_values"] = pixel_values
        text_batch["labels"]       = labels
                
        return text_batch