import os
import json
import math
import random
import re
import numpy as np
import torch
import torch.nn as nn
import wikipediaapi
import wandb

from nltk.tokenize import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from sklearn.model_selection import train_test_split
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if args['mode'] != "infer":
            self.run = wandb.init(
                project="label-classification-llama3",
                name=f"label-classification",
                config={
                    "learning_rate": config['learning_rate'],
                    "architecture": "BertGCN",
                    "dataset": "Custom",
                    "epochs": config['n_epochs'],
                }
            )
            self.dataset = Dataset(
                args['data_path'],
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
            elif self.args['mode'] == "test":
                self.model = torch.load(args['checkpoint'])
            self.optimizer, self.scheduler = self._get_optimizer()
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.model = torch.load(args['checkpoint'])
        self.model.to(args['device'])

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
        p1 = Precision(task='multiclass', num_classes=n_classes, top_k=1)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p3 = Precision(task='multiclass', num_classes=n_classes, top_k=3)(torch.tensor(predicted_labels), torch.tensor(target_labels))
        p5 = Precision(task='multiclass', num_classes=n_classes, top_k=5)(torch.tensor(predicted_labels), torch.tensor(target_labels))

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
                self.run.log({
                    "train_loss": loss,
                    "iteration":  i + len(self.train_loader)*self.config['n_epochs'],
                })
                if (i + 1) % 50 == 0 or i == 0 or i == len(self.train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader), total_loss/(i + 1)))
                if i == len(self.train_loader) - 1:
                    print("Evaluating...")
                    val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.val_loader)
                    print("Val_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(val_loss, accuracy, micro_f1, macro_f1))
                    print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
                    self.run.log({
                        "val_loss": val_loss,
                        "accuracy": accuracy,
                        "micro_f1": micro_f1,
                        "macro_f1": macro_f1,
                        "ndcg1": ndcg1,
                        "ndcg3": ndcg3,
                        "ndcg5": ndcg5,
                        "p1": p1,
                        "p3": p3,
                        "p5": p5,
                        "epoch": epoch,
                    })
                    if best_score < micro_f1:
                        best_score = micro_f1
                        self.save(epoch)
        self.run.finish()

    def test(self):
        print("Testing...")
        test_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = self.validate(self.test_loader)
        print("Test_loss: {} - Accuracy: {} - Micro-F1: {} - Macro-F1: {}".format(test_loss, accuracy, micro_f1, macro_f1))
        print("nDCG1: {} - nDCG@3: {} - nDCG@5: {} - P@1: {} - P@3: {} - P@5: {}".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
        
    def infer(self, text):
        final_labels = {
            "content": [],
            "labels": []
        }
        with open('./models/final/labels.txt', 'r') as f:
            labels = f.readlines()
        if len(text) > 4096:
            chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
            texts = chunks
        else:
            texts = [text]
        for t in texts:
            self.dataset = Dataset(
                t, 
                self.tokenizer,
                self.config['batch_size'],
                self.config['max_length'],
                self.sbert,
                True
            )
            self.infer_data = self.dataset.train_loader
            predicted_labels = self.get_infer_labels(self.infer_data)
            for predicted in predicted_labels:
                fit_labels = []
                for i in range(len(predicted)):
                    if predicted[i] >= 0.5:
                        fit_labels.append(labels[i].strip())
                final_labels['content'].append(t)
                final_labels['labels'].append(fit_labels)
        return final_labels

    def get_infer_labels(self, loader):
        self.model.eval()
        with torch.no_grad():
            predicted_labels = []
            for _, batch in enumerate(loader):
                input_ids, attention_mask = tuple(t.to(self.config['device']) for t in batch)
                output = self.model.forward(input_ids, attention_mask)
                predicted_labels.extend(torch.sigmoid(output).cpu().detach().numpy())
        return predicted_labels

        
    def save(self, epoch):
        torch.save(self.model, f'./modules/label/ckpt/checkpoint_{epoch}.pt')

class Dataset(object):
    def __init__(
            self, data_path, tokenizer, batch_size, max_length, sbert, custom = False
        ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert
        self.mlb = MultiLabelBinarizer()
        if not custom:
            self.train_loader, self.val_loader, self.test_loader, self.edges, self.label_features = self.load_dataset(data_path)
        else:
            self.train_loader = self.load_custom_dataset(data_path)
    
    def load_custom_dataset(self, text):
        data = {}
        if len(text) > 4096:
            chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
            data['content'] = chunks
        else:
            data['content'] = [text]
            self.batch_size = 1
        data_sents = [clean_string(x) for x in data['content']]
        data_loader = self.encode_data(data_sents, shuffle=False)
        return data_loader

    def load_dataset(self, data_path):
        trainval = None
        test = None
        for folder in sorted(os.listdir(data_path), reverse=True):
            train_filename = os.path.join(data_path, folder, 'train.json')
            test_filename = os.path.join(data_path, folder, 'test.json')
            if trainval is None:
                trainval = json.load(open(train_filename))
            else:
                temp = json.load(open(train_filename))
                trainval['content'].extend(temp['content'])
                trainval['labels'].extend(temp['labels'])
            if test is None:
                test = json.load(open(test_filename))
            else:
                temp = json.load(open(test_filename))
                test['content'].extend(temp['content'])
                test['labels'].extend(temp['labels'])
            break

        train = {
            "contents": [],
            "labels": []
        }
        val = train.copy()
        train['content'], val['content'], train['labels'], val['labels'] = train_test_split(trainval['content'], trainval['labels'], test_size=0.15, random_state=42)

        train_sents = [clean_string(text) for text in train['content']]
        test_sents = [clean_string(text) for text in test['content']]
        val_sents = [clean_string(text) for text in val['content']]

        train_labels = self.mlb.fit_transform(train['labels'])
        print("Numbers of labels: ", len(self.mlb.classes_))
        with open('./models/final/labels.txt', 'w') as f:
            for label in self.mlb.classes_:
                f.write(label + '\n')
        val_labels = self.mlb.transform(val['labels'])
        test_labels = self.mlb.transform(test['labels'])

        edges, label_features = self.create_edges_and_features(train, self.mlb)
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
    
    def encode_data(self, train_sents, train_labels=None, shuffle=False):
        X_train = self.tokenizer.batch_encode_plus(train_sents, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        if train_labels is not None:
            y_train = torch.tensor(train_labels, dtype=torch.float32)
            train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        else:
            train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'])
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
