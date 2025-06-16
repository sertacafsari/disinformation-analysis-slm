from argparse import ArgumentParser
from model_selection import ModelSelection
from dataset_selection import DatasetSelection


# Initialize the Parser
parser = ArgumentParser(description="A parser to get information from the job submitting script")

# Add arguments to parser
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--run_name", type=str, required=True)

# Get the arguments
args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name
batch_size = args.batch_size
learning_rate = args.lr
wandb_run = args.run_name


