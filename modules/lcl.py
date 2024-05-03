import os
from .lc import TrainerModel

class LabelClassificationLlama:
    def __init__(self, config):
        self.config = config
        pass

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