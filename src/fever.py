from dataset_loader import DatasetLoader
from datasets import  load_dataset




loader = DatasetLoader()

data = load_dataset("fever/fever", 'v1.0')

import code; code.interact(local=locals())