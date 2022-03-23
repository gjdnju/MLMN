# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:train_tools.py
@time:2022/03/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup



class MyDataset(Dataset):
    def __init__(self, data, fact_len, article_len, article_dict):
        self.len = len(data)
        self.fact, self.article, self.label = [], [], []

        for item in data:
            fact, article_idx, label = item
            fact = fact[:fact_len] if len(fact) > fact_len else \
                [0] * int(math.floor((fact_len - len(fact)) / 2)) + fact + \
                [0] * int(math.ceil((fact_len - len(fact)) / 2))

            article = article_dict[article_idx]
            article = article[:article_len] if len(article) > article_len else \
                [0] * int(math.floor((article_len - len(article)) / 2)) + article + \
                [0] * int(math.ceil((article_len - len(article)) / 2))

            self.fact.append(fact)
            self.article.append(article)
            self.label.append(label)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return torch.tensor(self.fact[item]).long(), \
               torch.tensor(self.article[item]).long(), \
               torch.tensor(self.label[item]).long()


class MyDatasetWithElement(Dataset):
    def __init__(self, data, fact_len, article_len, article_dict):
        self.len = len(data)
        self.fact, self.article, self.article_label, self.label = [], [], [], []

        for item in data:
            fact, article_idx, label = item
            fact = fact[:fact_len] if len(fact) > fact_len else \
                [0] * int(math.floor((fact_len - len(fact)) / 2)) + fact + \
                [0] * int(math.ceil((fact_len - len(fact)) / 2))

            article, article_label = article_dict[article_idx]
            article = article[:article_len] if len(article) > article_len else \
                [0] * int(math.floor((article_len - len(article)) / 2)) + article + \
                [0] * int(math.ceil((article_len - len(article)) / 2))

            article_label = np.array(article_label[:article_len]) if len(article_label) > article_len else \
                np.vstack((np.zeros((int(math.floor((article_len - len(article_label)) / 2)), 2)),
                           np.array(article_label),
                           np.zeros((int(math.ceil((article_len - len(article_label)) / 2)), 2))))

            self.fact.append(fact)
            self.article.append(article)
            self.article_label.append(article_label)
            self.label.append(label)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return torch.tensor(self.fact[item]).long(), \
               torch.tensor(self.article[item]).long(), \
               torch.tensor(self.article_label[item]).float(), \
               torch.tensor(self.label[item]).long()


class MyDatasetForPretrain(Dataset):
    def __init__(self, data, fact_len, article_len, article_dict, tokenizer):
        self.len = len(data)
        self.fact, self.article, self.label = [], [], []

        for item in tqdm(data):
            fact, article_idx, label = item
            fact = tokenizer(fact, return_tensors='pt', padding='max_length',
                             truncation=True, max_length=fact_len+2)

            article = article_dict[article_idx]
            article = tokenizer(article, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=article_len+2)
            article['input_ids'] = torch.cat(
                [torch.full([article['input_ids'].size(0), 1], 2), article['input_ids'][:, 1:]], dim=-1)

            self.fact.append(fact)
            self.article.append(article)
            self.label.append(label)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.fact[item]['input_ids'].squeeze(), self.fact[item]['attention_mask'].squeeze(), \
               self.article[item]['input_ids'].squeeze(), self.article[item]['attention_mask'].squeeze(),\
               torch.tensor(self.label[item]).long()


class MyDatasetWithElementForPretrain(Dataset):
    def __init__(self, data, fact_len, article_len, article_dict, tokenizer):
        self.len = len(data)
        self.fact, self.article, self.article_label, self.label = [], [], [], []

        for item in data:
            fact, article_idx, label = item
            fact = tokenizer(fact, return_tensors='pt', padding='max_length',
                             truncation=True, max_length=fact_len + 2)

            article, article_label = article_dict[article_idx]
            article = tokenizer(article, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=article_len + 2)
            article['input_ids'] = torch.cat(
                [torch.full([article['input_ids'].size(0), 1], 2), article['input_ids'][:, 1:]], dim=-1)

            article_label = np.array(article_label[:article_len]) if len(article_label) > article_len else \
                np.vstack((np.array(article_label),
                           np.zeros((article_len - len(article_label), 2))))

            self.fact.append(fact)
            self.article.append(article)
            self.article_label.append(article_label)
            self.label.append(label)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.fact[item]['input_ids'].squeeze(), self.fact[item]['attention_mask'].squeeze(), \
               self.article[item]['input_ids'].squeeze(), self.article[item]['attention_mask'].squeeze(),\
               torch.tensor(self.article_label[item]).float(), \
               torch.tensor(self.label[item]).long()


def train_model(model, args, train_data, valid_data, test_data, article_dict):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name) if args.use_pretrain_model else None

    train_pos, train_neg = train_data
    valid_pos, valid_neg = valid_data
    test_pos, test_neg = test_data

    if 'WithElement' in args.model_name:
        pos_dataset = MyDatasetWithElementForPretrain(
            train_pos, args.fact_len, args.article_len, article_dict, tokenizer) if args.use_pretrain_model else \
            MyDatasetWithElement(train_pos, args.fact_len, args.article_len, article_dict)
        neg_dataset = MyDatasetWithElementForPretrain(
            train_neg, args.fact_len, args.article_len, article_dict, tokenizer) if args.use_pretrain_model else \
            MyDatasetWithElement(train_neg, args.fact_len, args.article_len, article_dict)
        valid_dataset = MyDatasetWithElementForPretrain(
            valid_pos + valid_neg, args.fact_len, args.article_len, article_dict, tokenizer) if args.use_pretrain_model else \
            MyDatasetWithElement(valid_pos + valid_neg, args.fact_len, args.article_len, article_dict)
    else:
        pos_dataset = MyDatasetForPretrain(train_pos, args.fact_len, args.article_len, article_dict, tokenizer) \
            if args.use_pretrain_model else MyDataset(train_pos, args.fact_len, args.article_len, article_dict)
        neg_dataset = MyDatasetForPretrain(train_neg, args.fact_len, args.article_len, article_dict, tokenizer) \
            if args.use_pretrain_model else MyDataset(train_neg, args.fact_len, args.article_len, article_dict)
        valid_dataset = MyDatasetForPretrain(valid_pos + valid_neg, args.fact_len, args.article_len, article_dict, tokenizer) \
            if args.use_pretrain_model else \
            MyDataset(valid_pos + valid_neg, args.fact_len, args.article_len, article_dict)

    pos_batch_size = math.ceil(args.batch_size / (args.negtive_multiple + 1))
    neg_batch_size = args.batch_size - pos_batch_size

    pos_dataloader = DataLoader(pos_dataset, batch_size=pos_batch_size, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, batch_size=neg_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if args.use_pretrain_model:
        optimizer = AdamW(model.parameters(), lr=1e-5)
        total_steps = len(pos_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06 * total_steps,
                                                    num_training_steps=total_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model.get_lr(), weight_decay=model.get_weight_decay())
        scheduler = None
    early_stopping = EarlyStopping(output_path=args.checkpoint_dir + args.crime + '/' + args.model_name,
                                   patience=args.earlystop_patience, verbose=True)

    print('Train model...')
    for epoch in range(args.epochs):
        print('[' + model.get_name() + '] Epoch ' + str(epoch + 1), ': ')
        start_time = time.time()

        model.train()
        train_loss = 0
        train_acc = 0
        for i, (pos_sample, neg_sample) in enumerate(zip(pos_dataloader, neg_dataloader)):
            batch_logits = []
            batch_labels = []
            optimizer.zero_grad()

            acc_denominator = 0
            for items in [pos_sample, neg_sample]:
                labels = items[-1].to(device)
                items = tuple([item.to(device) for i, item in enumerate(items) if i != len(items)-1])
                batch_logits.append(model(*items))
                batch_labels.append(labels)
                acc_denominator += items[0].size(0)

            batch_logits = torch.cat(batch_logits, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            loss = criterion(batch_logits, batch_labels)
            acc = torch.eq(batch_logits.argmax(dim=-1), batch_labels).sum().item()
            loss.backward()
            optimizer.step()
            if args.use_pretrain_model:
                scheduler.step()
            train_loss += loss.item()
            train_acc += acc

            if (i + 1) % 100 == 0:
                print('[ {}/{} ] loss: {:.3f} accuracy: {:.3f}'
                      .format(i + 1, len(pos_dataloader), loss.item(), acc / acc_denominator))

        secs = int(time.time() - start_time)
        print('\nTrain | Loss: {:.5f} | Accuracy: {:.3f} | Time: {:.1f}s'
              .format(train_loss / len(pos_dataloader),
                      train_acc / (len(pos_dataset) + len(pos_dataloader) * neg_batch_size),
                      secs))

        valid_loss = 0.0
        valid_acc = 0
        valid_preds = []
        valid_labels = []
        model.eval()
        with torch.no_grad():
            for items in valid_dataloader:
                labels = items[-1].to(device)
                items = tuple([item.to(device) for i, item in enumerate(items) if i != len(items)-1])
                preds = model(*items)
                loss = criterion(preds, labels)
                acc = torch.eq(preds.argmax(dim=-1), labels).sum().item()
                valid_loss += loss.item()
                valid_acc += acc
                valid_preds.append(torch.ge(F.softmax(preds, dim=-1)[:, 1], 0.6).int().cpu())
                valid_labels.append(labels.cpu())

        valid_loss = valid_loss / len(valid_dataloader)
        valid_acc = valid_acc / len(valid_dataset)
        valid_preds = torch.cat(valid_preds, dim=0)
        valid_labels = torch.cat(valid_labels, dim=0)
        precision = precision_score(valid_labels, valid_preds)
        recall = recall_score(valid_labels, valid_preds)
        f1 = f1_score(valid_labels, valid_preds)

        print('Valid | Loss: {:.5f} | Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'
              .format(valid_loss, valid_acc, precision, recall, f1))

        early_stopping(valid_loss, model)
        print('-----------------------------------------------')
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('\nTest model...')
    model.load_state_dict(torch.load(args.checkpoint_dir + args.crime + '/' + args.model_name))
    if 'WithElement' in args.model_name:
        test_dataset = MyDatasetWithElementForPretrain(
            test_pos + test_neg, args.fact_len, args.article_len, article_dict, tokenizer) if args.use_pretrain_model else \
            MyDatasetWithElement(test_pos + test_neg, args.fact_len, args.article_len, article_dict)
    else:
        test_dataset = MyDatasetForPretrain(test_pos + test_neg, args.fact_len, args.article_len, article_dict, tokenizer) \
            if args.use_pretrain_model else \
            MyDataset(test_pos + test_neg, args.fact_len, args.article_len, article_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for items in test_dataloader:
            labels = items[-1]
            items = tuple([item.to(device) for i, item in enumerate(items) if i != len(items) - 1])
            logits = F.softmax(model(*items), dim=-1)[:, 1]
            preds = torch.ge(logits, 0.6).int().cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print('Test | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'.format(precision, recall, f1))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, output_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output_path = output_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.output_path)
        self.val_loss_min = val_loss
