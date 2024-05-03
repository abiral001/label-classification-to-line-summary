import os
import json
import math
import random
import re
import numpy as np
import torch
import torch.nn as nn
import wikipediaapi

from nltk.tokenize import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sentence_transformers import SentenceTransformer
from .label.bert_gcn import BertGCN
from sklearn import metrics
from torchmetrics import Precision

wiki_wiki = wikipediaapi.Wikipedia('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36')

class TrainerModel(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.sbert = SentenceTransformer(args['sbert'], device=args['device'])
        self.tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name'])

        self.dataset = Dataset(
            args['train_data_path'],
            args['val_data_path'],
            args['test_data_path'],
            self.tokenizer,
            config['batch_size'],
            config['max_length'],
            self.sbert
        )
        self.train_loader, self.val_loader, self.test_loader = self.dataset.train_loader, self.dataset.val_loader, self.dataset.test_loader

        if self.args['mode'] == "train":
            self.model = BertGCN(
                self.dataset.edges.to(args['device']),
                self.dataset.label_features.to(args['device']),
                config,
                args
            )
        # self.model.to(args['device'])
        self.optimizer, self.scheduler = self._get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
             'weight_decay': self.config['weight_decay']},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=self.config['n_epochs'] * len(self.train_loader),
                                                    num_warmup_steps=100)
        return optimizer, scheduler
    
    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predicted_labels, target_labels = list(), list()

            for i, batch in enumerate(tqdm(dataloader)):
                input_ids, attention_mask, y_true = tuple(t.to(self.config['device']) for t in batch)
                output = self.model.forward(input_ids, attention_mask)
                loss = self.loss_fn(output, y_true.float())

                total_loss += loss.item()

                target_labels.extend(y_true.cpu().detach().numpy())
                predicted_labels.extend(torch.sigmoid(output).cpu().detach().numpy())

            val_loss = total_loss/len(dataloader)

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
        micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
        macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')
        
        ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
        ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
        ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)
        
        n_classes = self.dataset.label_features.size(0)
        p1 = Precision(num_classes=n_classes, top_k=1)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p3 = Precision(num_classes=n_classes, top_k=3)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p5 = Precision(num_classes=n_classes, top_k=5)(torch.tensor(predicted_labels), torch.tensor(target_labels))

        return val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5
    
    def step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        input_ids, attention_mask, y_true = tuple(t.to(self.config['device']) for t in batch)
        output = self.model.forward(input_ids, attention_mask)
        loss = self.loss_fn(output, y_true.float())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def train(self):
        print("Training...")
        best_score = float("-inf")
        for epoch in range(self.config['n_epochs']):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                loss = self.step(batch)
                total_loss += loss 
                if (i + 1) % 50 == 0 or i == 0 or i == len(self.train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader), total_loss/(i + 1)))
                if i == len(self.train_loader) - 1:
                    print("Evaluating...")
                    val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.val_loader)
                    print("Val_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(val_loss, accuracy, micro_f1, macro_f1))
                    print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))

                    if best_score < micro_f1:
                        best_score = micro_f1
                        self.save(epoch)
    def test(self):
        print("Testing...")
        test_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.test_loader)
        print("Test_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(test_loss, accuracy, micro_f1, macro_f1))
        print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
        
        
    def save(self, epoch):
        torch.save(self.model, f'./modules/label/ckpt/checkpoint_{epoch}.pt')

class Dataset(object):
    def __init__(
            self, train_data_path, val_data_path, test_data_path, tokenizer, batch_size, max_length, sbert
        ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert
        self.train_loader, self.val_loader, self.test_loader, self.edges, self.label_features = self.load_dataset(train_data_path, val_data_path, test_data_path)

    def load_dataset(self, train_data_path, val_data_path, test_data_path):
        train = json.load(open(train_data_path))
        val = json.load(open(val_data_path))
        test = json.load(open(test_data_path))

        train_sents = [clean_string(text) for text in train['content']]
        val_sents = [clean_string(text) for text in val['content']]
        test_sents = [clean_string(text) for text in test['content']]

        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(train['labels'])
        print("Numbers of labels: ", len(mlb.classes_))
        val_labels = mlb.transform(val['labels'])
        test_labels = mlb.transform(test['labels'])

        edges, label_features = self.create_edges_and_features(train, mlb)
        train_loader = self.encode_data(train_sents, train_labels, shuffle=True)
        val_loader = self.encode_data(val_sents, val_labels, shuffle=False)
        test_loader = self.encode_data(test_sents, test_labels, shuffle=False)
        return train_loader, val_loader, test_loader, edges, label_features
    
    def create_edges_and_features(self, train_data, mlb):
        label2id = {v: k for k, v in enumerate(mlb.classes_)}
        edges = torch.zeros((len(label2id), len(label2id)))
        for label in train_data["labels"]:
            if len(label) >= 2:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        src, tgt = label2id[label[i]], label2id[label[j]]
                        edges[src][tgt] += 1
                        edges[tgt][src] += 1
        marginal_edges = torch.zeros((len(label2id)))
        for label in train_data['labels']:
            for i in range(len(label)):
                marginal_edges[label2id[label[i]]] += 1
        for i in range(edges.size(0)):
            for j in range(edges.size(1)):
                if marginal_edges[i] != 0 and marginal_edges[j] != 0:
                    edges[i][j] /= math.sqrt(marginal_edges[i] * marginal_edges[j])
        edges = normalizeAdjacency(edges+torch.diag(torch.ones(len(label2id))))
        features = torch.zeros(len(label2id), 768)
        for labels, i in label2id.items():
            features[i] = get_embedding_from_wiki(self.sbert, labels)
        return edges, features
    
    def encode_data(self, train_sents, train_labels, shuffle=False):
        X_train = self.tokenizer.batch_encode_plus(train_sents, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        y_train = torch.tensor(train_labels, dtype=torch.float32)
        train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=shuffle)

        return train_loader


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def get_text_from_wiki(text, n_sents=2):
    text = text.replace('-', ' ')
    page_py = wiki_wiki.page(text)
    paragraph = sent_tokenize(page_py.summary)
    if len(paragraph) == 0:
        return text
    elif len(paragraph) <= n_sents:
        return " ".join(paragraph)
    else:
        return " ".join(paragraph[:n_sents])

def normalizeAdjacencyv2(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/d
    D = torch.diag(d)
    return D @ W

def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D

def get_embedding_from_wiki(sbert, text, n_sent=1):
    text = get_text_from_wiki(text, n_sent)
    embedding = sbert.encode(text, convert_to_tensor=True)
    return embedding

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
