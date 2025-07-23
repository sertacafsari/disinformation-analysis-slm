import torch

class DataCollatorVision:
    """ A custom Data Collator for VLMs to handle padding for images in Fauxtography dataset."""

    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):

        # Take out all images
        imgs = [f["pixel_values"] for f in features]

        sizes = []
        for img in imgs:
            if img.ndim == 3 and img.shape[0] in (1,3):
                height, width = img.shape[1], img.shape[2]
            else:
                height, width = img.shape[0], img.shape[1]
            sizes.append((height,width))
            
        img_sizes = torch.tensor(sizes, dtype=torch.long)

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
        text_batch["image_sizes"]  = img_sizes
                
        return text_batch