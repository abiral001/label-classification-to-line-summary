import os
from .lc import TrainerModel
from .llama3 import Llama3

class LabelClassificationLlama:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config['ckpt_dir']
        self.tokenizer_path = config['tokenizer_path']

    def run(self, text): 
        embedding_inp = text # this is from intermediate output of the classification model
        labels = ["Cat1", "Cat2", "Cat3"] # this is generated by the label classification model
        length = 100
        llama = Llama3(self.ckpt_dir, self.tokenizer_path)
        summary = llama.generate(embedding_inp, labels, length)
        return summary

    def train(self, type):
        if type == "lc":
            print('Training Label Classification')
            args = {
                'train_data_path': './dataset/Reuters/train.json',
                'val_data_path': './dataset/Reuters/val.json',
                'test_data_path': './dataset/Reuters/test.json',
                'device': self.config['device'],
                'pretrained': 'roberta-base',
                'checkpoint': './modules/labels/ckpt/checkpoint_7.pt',
                'tokenizer_name': 'roberta-base',
                'sbert': 'paraphrase-distilroberta-base-v1',
                'mode': 'train'
            }
            trainer = TrainerModel(self.config, args)
        elif type == "both":
            print('Training both Label Classification and Llama3')