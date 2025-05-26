from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import AdamW
from transformers import set_seed
import torch
import tqdm

class Finetuner():
    """ 
    A class to fine tune the models with dis/misinformation datasets.
    
    Parameters:
        train_dataloader(DataLoader):
        val_dataloader(DataLoader):
    """

    def __init__(self, train_dataloader: DataLoader=None, val_dataloader:DataLoader=None,  model:Module=None, seed:int=1234, lr:float=0.0):
        self.seed = seed
        set_seed(self.seed)

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
    
    def train(self, device: torch.device, epochs:int=1, logging_step:int=10, best_model_path:str=None):
        # Transfer model to GPU (or back to CPU)
        self.model.to(device)
        
        # A variable to store the least validation loss (or best evaluation loss)
        # Set to infinity as we want to have the least one
        best_eval_loss = float('inf')

        train_losses, val_losses = [],[]
        
        for epoch in range(epochs):

            print(f"Epoch {epoch+1}/{epochs}")

            # Set the model's mode to "training"
            self.model.train()

            steps_in_epoch = len(self.train_dataloader)
            epoch_train_loss = 0.0

            for batch_idx, batch_dict in enumerate(self.train_dataloader):

                # Get labels, input_ids, and attention_masks
                labels = batch_dict["labels"]
                input_ids = batch_dict["input_ids"]
                attention_mask = batch_dict["attention_mask"]

                # Send them to the GPU
                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Get output
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Get the loss
                loss = output["loss"]

                # Empty the gradients accumulated in the optimizer
                self.optimizer.zero_grad()

                # Apply the backpropagation
                loss.backward()

                # Perform a single optimization step.
                self.optimizer.step()

                # Get the current loss
                loss = loss.item()
                epoch_train_loss += loss

                # Log the progress
                if ((batch_idx + 1) % logging_step == 0):
                    print(f" Epoch {epoch+1}, Step {batch_idx+1}/{steps_in_epoch}, Loss: {loss}")
                
            avg_train = epoch_train_loss / steps_in_epoch
            train_losses.append(avg_train)
            
            # Set the model's mode to "eval"
            self.model.eval()

            total_eval_loss = 0
            steps_in_eval = len(self.val_dataloader)

            # We wont use backprop and change weights during validation/inference
            with torch.no_grad():
                for batch_dict in self.val_dataloader:

                    labels = batch_dict["labels"]
                    input_ids = batch_dict["input_ids"]
                    attention_mask = batch_dict["attention_mask"]

                    labels = labels.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    loss = output["loss"]

                    total_eval_loss += loss.item()

            average_eval_loss = total_eval_loss/steps_in_eval

            val_losses.append(average_eval_loss)

            # Save the model 
            if best_model_path and average_eval_loss < best_eval_loss:
                best_eval_loss = average_eval_loss
                torch.save(self.model.state_dict(), best_model_path)
                print("The best model is saved to given path")
    
        return {"train_loss": train_losses,"val_loss": val_losses}