import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

import math
import time
import numpy as np

from utils import load_pkl, EarlyStopping
from preprocess import text2id
from random import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def process_data(data, stop_words, vocab):
    pos_res = []
    neg_res = []
    for line in data:
        items = line.strip().split('|')
        assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
        fact = text2id(items[1], stop_words, vocab)
        positive_samples = [int(num) for num in items[2].split(',')]
        negative_samples = [int(num) for num in items[3].split(',')]

        for sample in positive_samples:
            pos_res.append([fact, sample, 1])

        for sample in negative_samples:
            neg_res.append([fact, sample, 0])

    shuffle(pos_res)
    shuffle(neg_res)
    return pos_res, neg_res


class MyDataset(Dataset):
    def __init__(self, data, fact_len, article_len, article_dict):
        self.len = len(data)
        self.fact, self.article, self.label = [], [], []

        for item in data:
            fact, article_idx, label = item
            fact = fact[:fact_len] if len(fact) > fact_len else \
                [0] * int(math.floor((fact_len - len(fact)) / 2)) + fact + \
                [0] * int(math.ceil((fact_len - len(fact)) / 2))
            # fact = fact[:fact_len] if len(fact) > fact_len else \
            #     fact + [0] * (fact_len - len(fact))

            article = article_dict[article_idx]
            article = article[:article_len] if len(article) > article_len else \
                [0] * int(math.floor((article_len - len(article)) / 2)) + article + \
                [0] * int(math.ceil((article_len - len(article)) / 2))
            # article = article[:article_len] if len(article) > article_len else \
            #     article + [0] * (article_len - len(article))

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
            # fact = fact[:fact_len] if len(fact) > fact_len else \
            #     fact + [0] * (fact_len - len(fact))

            article, article_label = article_dict[article_idx]
            article = article[:article_len] if len(article) > article_len else \
                [0] * int(math.floor((article_len - len(article)) / 2)) + article + \
                [0] * int(math.ceil((article_len - len(article)) / 2))
            # article = article[:article_len] if len(article) > article_len else \
            #     article + [0] * (article_len - len(article))

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


def train_model(model, args, train_data, valid_data, test_data):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_pos, train_neg = train_data
    valid_pos, valid_neg = valid_data
    test_pos, test_neg = test_data

    article_dict = load_pkl(args.article_dict_path)
    pos_dataset = MyDataset(train_pos, args.fact_len, args.article_len, article_dict)
    neg_dataset = MyDataset(train_neg, args.fact_len, args.article_len, article_dict)
    valid_dataset = MyDataset(valid_pos + valid_neg, args.fact_len, args.article_len, article_dict)

    pos_batch_size = math.ceil(args.batch_size / (args.negtive_multiple + 1))
    neg_batch_size = args.batch_size - pos_batch_size

    pos_dataloader = DataLoader(pos_dataset, batch_size=pos_batch_size, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, batch_size=neg_batch_size, shuffle=True)
    print(len(pos_dataloader))
    print(len(neg_dataloader))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.get_lr(), weight_decay=model.get_weight_decay())
    early_stopping = EarlyStopping(output_path=args.model_path, patience=10, verbose=True)
    # writer = SummaryWriter(log_dir=args.tensorboard_path)

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
            for facts, articles, labels in [pos_sample, neg_sample]:
                facts, articles, labels = facts.to(device), articles.to(device), labels.to(device)
                batch_logits.append(model(facts, articles))
                batch_labels.append(labels)
                acc_denominator += facts.size(0)

            batch_logits = torch.cat(batch_logits, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            loss = criterion(batch_logits, batch_labels)
            acc = torch.eq(batch_logits.argmax(dim=-1), batch_labels).sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc

            # writer.add_scalars("Train", {
            #     "loss": loss.item(),
            #     "acc": acc / acc_denominator
            # }, global_step=epoch * len(pos_dataloader) + i + 1)

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
            for facts, articles, labels in valid_dataloader:
                facts, articles, labels = facts.to(device), articles.to(device), labels.to(device)
                preds = model(facts, articles)
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

        print('\nValid | Loss: {:.5f} | Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'
              .format(valid_loss, valid_acc, precision, recall, f1))
        # writer.add_scalars('Per Epoch', {
        #     'train loss': train_loss / len(pos_dataloader),
        #     'train acc': train_acc / (len(pos_dataset) + len(pos_dataloader) * neg_batch_size),
        #     'valid loss': valid_loss,
        #     'valid acc': valid_acc,
        #     'valid f1': f1
        # }, global_step=epoch + 1)

        early_stopping(valid_loss, model)
        print('-----------------------------------------------')
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('\nTest model...')
    model.load_state_dict(torch.load(args.model_path))
    test_dataset = MyDataset(test_pos + test_neg, args.fact_len, args.article_len, article_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for facts, articles, labels in test_dataloader:
            facts, articles = facts.to(device), articles.to(device)
            logits = F.softmax(model(facts, articles), dim=-1)[:, 1]
            preds = torch.ge(logits, 0.6).int().cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print('Test | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'.format(precision, recall, f1))
    with open(args.log_path, 'a', encoding='utf-8') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
        f.write('Model: ' + model.get_name() + '\n')
        f.write('Test | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}\n'.format(precision, recall, f1))
        f.write('\n')


def train_model_with_element(model, args, train_data, valid_data, test_data):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_pos, train_neg = train_data
    valid_pos, valid_neg = valid_data
    test_pos, test_neg = test_data

    article_dict = load_pkl(args.article_dict_with_element_path)
    pos_dataset = MyDatasetWithElement(train_pos, args.fact_len, args.article_len, article_dict)
    neg_dataset = MyDatasetWithElement(train_neg, args.fact_len, args.article_len, article_dict)
    valid_dataset = MyDatasetWithElement(valid_pos + valid_neg, args.fact_len, args.article_len, article_dict)

    pos_batch_size = math.ceil(args.batch_size / (args.negtive_multiple + 1))
    neg_batch_size = args.batch_size - pos_batch_size

    pos_dataloader = DataLoader(pos_dataset, batch_size=pos_batch_size, shuffle=True)
    neg_dataloader = DataLoader(neg_dataset, batch_size=neg_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.get_lr(), weight_decay=model.get_weight_decay())
    early_stopping = EarlyStopping(output_path=args.model_path, patience=10, verbose=True)
    # writer = SummaryWriter(log_dir=args.tensorboard_path)

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
            for facts, articles, article_labels, labels in [pos_sample, neg_sample]:
                facts, articles, article_labels, labels = \
                    facts.to(device), articles.to(device), article_labels.to(device), labels.to(device)
                batch_logits.append(model(facts, articles, article_labels))
                batch_labels.append(labels)
                acc_denominator += facts.size(0)

            batch_logits = torch.cat(batch_logits, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            loss = criterion(batch_logits, batch_labels)
            acc = torch.eq(batch_logits.argmax(dim=-1), batch_labels).sum().item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc

            # writer.add_scalars("Train", {
            #     "loss": loss.item(),
            #     "acc": acc / acc_denominator
            # }, global_step=epoch * len(pos_dataloader) + i + 1)

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
            for facts, articles, article_labels, labels in valid_dataloader:
                facts, articles, article_labels, labels = \
                    facts.to(device), articles.to(device), article_labels.to(device), labels.to(device)
                preds = model(facts, articles, article_labels)
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

        print('\nValid | Loss: {:.5f} | Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'
              .format(valid_loss, valid_acc, precision, recall, f1))
        # writer.add_scalars('Per Epoch', {
        #     'train loss': train_loss / len(pos_dataloader),
        #     'train acc': train_acc / (len(pos_dataset) + len(pos_dataloader) * neg_batch_size),
        #     'valid loss': valid_loss,
        #     'valid acc': valid_acc,
        #     'valid f1': f1
        # }, global_step=epoch + 1)

        early_stopping(valid_loss, model)
        print('-----------------------------------------------')
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('\nTest model...')
    model.load_state_dict(torch.load(args.model_path))
    test_dataset = MyDatasetWithElement(test_pos + test_neg, args.fact_len, args.article_len, article_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for facts, articles, article_labels, labels in test_dataloader:
            facts, articles, article_labels = facts.to(device), articles.to(device), article_labels.to(device)
            logits = F.softmax(model(facts, articles, article_labels), dim=-1)[:, 1]
            preds = torch.ge(logits, 0.6).int().cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print('Test | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f} | Accuracy: {:.3f} '
          .format(precision, recall, f1, acc))
    with open(args.log_path, 'a', encoding='utf-8') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
        f.write('Model: ' + model.get_name() + '\n')
        f.write('Test | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f} | Accuracy: {:.3f}\n'
                .format(precision, recall, f1, acc))
        f.write('\n')
