import os
from .lc import TrainerModel
from .llama3 import Llama3
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LCLModel(nn.Module):
    def __init__(self):
        super(LCLModel, self).__init__()
        args = {
            'data_path': './dataset',
            'device': self.config['device'],
            'pretrained': 'roberta-large',
            'checkpoint': './models/labels-classification/checkpoint.pt',
            'tokenizer_name': "roberta-large",
            'sbert': 'paraphrase-distilroberta-base-v1',
            'mode': 'test'
        }
        self.label_classifier = TrainerModel(self.config, args)

    def forward(self, x):
        return x
class LabelClassificationLlama:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config['ckpt_dir']
        self.tokenizer_path = config['tokenizer_path']

    def run(self, text):
        print('Inferencing Label Classification')
        args = {
            'data_path': './dataset',
            'device': self.config['device'],
            'pretrained': 'roberta-base',
            'checkpoint': './modules/label/ckpt/checkpoint_7.pt',
            'tokenizer_name': 'roberta-large',
            'sbert': 'paraphrase-distilroberta-base-v1',
            'mode': 'infer'
        }
        trainer = TrainerModel(self.config, args)
        predicted_labels = trainer.infer(text)
        llama = Llama3(self.ckpt_dir, self.tokenizer_path)
        trainer = TrainerModel(self.config, args)
        summaries = []
        for (content, label) in zip(predicted_labels['content'], predicted_labels['labels']):
            length = 100
            summary = llama.generate(content, label, length)
            summaries.append(summary)
        # full_labels = set([lbl for lbl in predicted_labels['labels']])
        full_labels = list()
        for lbl in predicted_labels['labels']:
            full_labels.extend([l for l in lbl])
        full_labels = list(set(full_labels))
        while len(summaries) > 1:
            summaries = "".join(summaries)
            if len(summaries) > 4096:
                summaries = [summaries[i:i+4096] for i in range(0, len(summaries), 4096)]
            else:
                summaries = [summaries]
            new_summaries = []
            for summary in summaries:
                new_summaries.append(llama.generate(summary, full_labels, length))
            summaries = new_summaries.copy()
        return "".join(summaries)

    def train(self, type):
        if type == "lc":
            print('Training Label Classification')
            args = {
                'data_path': './dataset',
                'device': self.config['device'],
                'pretrained': 'roberta-base',
                'checkpoint': './modules/labels/ckpt/checkpoint_7.pt',
                'tokenizer_name': 'roberta-base',
                'sbert': 'paraphrase-distilroberta-base-v1',
                'mode': 'train'
            }
            trainer = TrainerModel(self.config, args)
            trainer.train()