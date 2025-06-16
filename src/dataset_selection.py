from argparse import ArgumentParser
from dataset_loader import DatasetLoader

class DatasetSelection():

    def __init__(self, dataset:str, model:str):
        self.parser = ArgumentParser()

        if dataset == 'liar2':
            self.dataset_name = "chengxuphd/liar2"
    

