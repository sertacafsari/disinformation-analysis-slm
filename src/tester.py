import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


class Tester:
    """
        A class for the testing the finetuned model through "test" splits and different domains.
        Args:
            model_path (str): The path of chosen model to load and test.
            device (torch.device): The device to run the model on.
            test_split (DataLoader): The test split of the chosen data.
            logging_step (int): The number of steps to log the progress.
            wandb_run (wandb): The Weights and Biases run instance to log the process.
    """

    def __init__(self, model:Module, device:torch.device, test_split: DataLoader, logging_step:int=10, wandb_run=None,):
        self.device = device
        # Load model from the given path
        self.model = model
        self.data = test_split
        self.logging_step = logging_step
        self.wandb = wandb_run
    
    def testModel(self, k:int):
        """
            A function to test the model with the test split of the dataset that is used for the training.
            Parameters:
                k (int): The k value for top-k accuracy
        """

        self.model.to(self.device)

        # Turn on the evaluation mode of the model
        self.model.eval()

        # Set the step size
        step_size = len(self.data)

        progress_tracker = tqdm(self.data, desc="Testing", total=step_size)

        # Set the variables for calculations
        total_loss = 0
        accuracy = 0
        top_k_accuracy = 0
        f1_macro = 0
        test_pred, test_labels = [], []

        # No gradient calculation
        with torch.no_grad():
            
            for batch_idx, batch_dict in enumerate(progress_tracker):

                labels = batch_dict["labels"].to(self.device)
                input_ids = batch_dict["input_ids"].to(self.device)
                attention_mask = batch_dict["attention_mask"].to(self.device)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Get the loss
                step_loss = output["loss"].item()

                # Get the accuracy which is the best logit (score)
                step_logits = output["logits"]
                prediction = step_logits.argmax(dim=1)

                # Get the accuracy
                accuracy += (((prediction == labels).float().sum()) / (labels.size(0))).item()

                # Get the top-k accuracy
                _, top_k_indices = torch.topk(step_logits, k=k, dim=1)
                labels_expanded = labels.unsqueeze(1).expand_as(top_k_indices)

                top_k_accuracy += (((top_k_indices == labels_expanded).any(dim=1).float().sum()) / labels.size(0)).item()

                # Get necessary values for F1 metric
                test_pred.extend(prediction.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())

                total_loss += step_loss

                if ((batch_idx + 1) % self.logging_step == 0):
                    self.wandb.log(
                        {
                            "test_step_loss": step_loss,
                    })
            
            # Calculate the average loss, accuracy, top k accuracy and f1
            avg_loss = total_loss / step_size
            avg_accuracy = accuracy / step_size
            avg_top_k = top_k_accuracy / step_size
            f1_macro = f1_score(test_labels, test_pred, average="macro")

            # Log results
            self.wandb.log({
                    "average_test_loss": avg_loss,
                    "average_test_accuracy": avg_accuracy,
                    f"average_top_{k}_accuracy": avg_top_k,
                    "test_f1_macro": f1_macro
                })
    
    #TODO: An additional function to cross domain test could be written
