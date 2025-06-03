from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from transformers.optimization import get_linear_schedule_with_warmup

class Finetuner():
    """ 
    A class to fine tune the models with dis/misinformation datasets.
    
    Parameters:
        train_dataloader(DataLoader):
        val_dataloader(DataLoader):
    """

    def __init__(self, train_dataloader: DataLoader=None, val_dataloader:DataLoader=None,  model:Module=None, lr:float=0.0):
        # self.model = torch.compile(model)
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if model.base_model_prefix == "roberta":
            # Using weight decay to penalize large weights, standard 0.01 for AdamW in Roberta
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        

    
    def train(self, device: torch.device, epochs:int=1, logging_step:int=10, best_model_path:str=None, wandb_run=None, k_top:int=2):
        # A variable to store the least validation loss (or best evaluation loss)
        # Set to infinity as we want to have the least one
        best_eval_loss = float('inf')

        train_losses, val_losses = [],[]

        steps_in_epoch = len(self.train_dataloader)
        total_training_steps = epochs*steps_in_epoch
        warmup_steps = int(0.1*total_training_steps)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

        # For F1 metric        
        for epoch in range(epochs):

            train_preds, train_labels = [], []

            if wandb_run:
                wandb_run.log({"epoch": epoch+1})
            else:
                print(f"Epoch {epoch+1}/{epochs}")


            # Set the model's mode to "training"
            self.model.train()

            epoch_train_loss = 0.0

            # Set the accuracy
            top_k_accuracy = 0

            # acc = 0

            progress_traindataloader = tqdm(self.train_dataloader, desc="Training", total=len(self.train_dataloader))

            for batch_idx, batch_dict in enumerate(progress_traindataloader):

                # Get labels, input_ids, and attention_masks
                labels = batch_dict["labels"].to(device)
                input_ids = batch_dict["input_ids"].to(device)
                attention_mask = batch_dict["attention_mask"].to(device)
  
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Get the loss
                loss = output["loss"]

                # Accuracy calculation
                logits = output["logits"]

                # Top-1 Accuracy
                preds = logits.argmax(dim=1)
                # acc += (((preds == labels).float().sum()) / (labels.size(0))).item()

                # Top-k Accuracy
                _, top_k_indices = torch.topk(logits,k_top, dim=1)

                labels_expanded = labels.unsqueeze(1).expand_as(top_k_indices)

                top_k_correct = (top_k_indices == labels_expanded).any(dim=1).float().sum()
                batch_top_k_acc = top_k_correct / labels.size(0)
                top_k_accuracy += batch_top_k_acc.item()

                train_preds.extend(preds.cpu().tolist())
                train_labels.extend(labels.cpu().tolist())

                # Empty the gradients accumulated in the optimizer
                self.optimizer.zero_grad()

                # Apply the backpropagation
                loss.backward()

                # Apply the gradient norm clipping to prevent exploding gradients
                clip_grad_norm_(self.model.parameters(),1.0)

                # Perform a single optimization step.
                self.optimizer.step()
                scheduler.step()

                # Get the current loss
                loss = loss.item()
                epoch_train_loss += loss

                # Log train step
                if ((batch_idx + 1) % logging_step == 0):
                    if wandb_run:
                        wandb_run.log(
                            {
                                "training_loss": loss,
                            })
                    else:
                        print(f" Epoch {epoch+1}, Step {batch_idx+1}/{steps_in_epoch}, Loss: {loss}")

                
            avg_train = epoch_train_loss / steps_in_epoch
            train_losses.append(avg_train)

            train_macro_f1 = f1_score(train_labels, train_preds,average="macro")

            # Log train epoch => "train_accuracy": (acc * 100) / steps_in_epoch,
            if wandb_run:
                wandb_run.log({
                    "train_avg_loss": avg_train,
                    "train_macro_f1": train_macro_f1,
                    "train_top_k_acc": (top_k_accuracy * 100)/steps_in_epoch,
                    "epoch": epoch
                    })
            
            # Set the model's mode to "eval"
            self.model.eval()

            total_eval_loss = 0
            steps_in_eval = len(self.val_dataloader)

            val_top_k_accuracy = 0

            progress_val = tqdm(self.val_dataloader, desc="Validation", total=len(self.val_dataloader))

            val_preds, val_labels = [], []

            # We wont use backprop and change weights during validation/inference
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(progress_val):

                    labels = batch_dict["labels"]
                    input_ids = batch_dict["input_ids"]
                    attention_mask = batch_dict["attention_mask"]

                    labels = labels.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    loss = output["loss"]

                    # Accuracy Calculation
                    logits = output["logits"]
                    preds = logits.argmax(dim=1)

                    # acc += (((preds == labels).float().sum()) / (labels.size(0))).item()

                    # Validation top-k accuracy

                    _, top_k_indices = torch.topk(logits, k=k_top, dim=1)
                    labels_expanded = labels.unsqueeze(1).expand_as(top_k_indices)

                    top_k_correct = (top_k_indices == labels_expanded).any(dim=1).float().sum()
                    batch_top_k_acc = top_k_correct / labels.size(0)
                    val_top_k_accuracy += batch_top_k_acc.item()

                    val_preds.extend(preds.cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())
                    

                    total_eval_loss += loss.item()

                    # Log validation step
                    if ((batch_idx + 1) % logging_step == 0):
                        if wandb_run:
                            wandb_run.log(
                                {
                                    "validation_loss": loss,
                                })
                        else:
                            print(f" Epoch {epoch+1}, Step {batch_idx+1}/{steps_in_epoch}, Loss: {loss}")

                

            average_eval_loss = total_eval_loss/steps_in_eval

            val_losses.append(average_eval_loss)

            val_macro_f1 = f1_score(val_labels, val_preds, average="macro")

            # Log validation step loss => "val_accuracy": (acc * 100) / steps_in_eval
            if wandb_run:
                wandb_run.log({
                    "val_avg_loss": average_eval_loss,
                    "val_top_k_accuracy": (val_top_k_accuracy * 100) / steps_in_eval,
                    "val_macro_f1": val_macro_f1,
                    "epoch": epoch

                })

            # Save the model 
            if average_eval_loss < best_eval_loss:
                best_eval_loss = average_eval_loss
                if best_model_path:
                    save_dir = os.path.dirname(best_model_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                        print("The directory created")
                    torch.save(self.model.state_dict(), best_model_path)
                    print("The best model is saved to given path")
                else:
                    print("The best model is improved, but not saved")

            
        return {"train_loss": train_losses,"val_loss": val_losses}
    
    def test_model(self, device: torch.device, test_dataloader:DataLoader, model_path:str, logging_step:int=10, wandb_run=None, k_top:int=2):

        if model_path is None:
            print("Model path is wrong")
        
        # Load the saved model
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.to(device)

        self.model.eval()

        total_test_loss = 0
        test_preds = []
        test_labels = []
        test_top_k_acc = 0

        progress_testloader = tqdm(test_dataloader, desc="Testing", total=len(test_dataloader))

        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(progress_testloader):

                labels = batch_dict["labels"].to(device)
                input_ids = batch_dict["input_ids"].to(device)
                attention_mask = batch_dict["attention_mask"].to(device)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = output["loss"]

                total_test_loss += loss.item()

                # Accuracy Calculation
                logits = output["logits"]
                preds = logits.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

                # Top-k Accuracy
                _, top_k_indices_test = torch.topk(logits, k=k_top, dim=1)
                labels_expanded_test = labels.unsqueeze(1).expand_as(top_k_indices_test)

                top_k_correct_test = (top_k_indices_test == labels_expanded_test).any(dim=1).float().sum()
                batch_top_k_acc_test = top_k_correct_test / labels.size(0)
                test_top_k_acc += batch_top_k_acc_test.item()
                # Log validation step
                if ((batch_idx + 1) % logging_step == 0):
                    if wandb_run:
                        wandb_run.log(
                            {
                                "test_loss": loss,
                            })
                
            avg_test_loss = total_test_loss/len(test_dataloader)
            final_top_k_accuracy = (test_top_k_acc * 100) / len(test_dataloader)
            test_f1_macro = f1_score(test_labels, test_preds, average="macro")

            if wandb_run:
                wandb_run.log({
                    "test_avg_loss": avg_test_loss,
                    "test_top_k_accuracy": final_top_k_accuracy,
                    "test_macro_f1": test_f1_macro
                })






