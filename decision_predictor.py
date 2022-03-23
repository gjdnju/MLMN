# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:decision_predictor.py
@time:2022/03/23
"""

import os
import time
import jieba
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from data_process import load_txt, load_pkl, load_json
from train_tools import EarlyStopping
from parameters import parse

warnings.filterwarnings('ignore')


class MyDataSet(Dataset):
    def __init__(self, corpus):
        stop_words = [line.strip() for line in load_txt(args.data_dir + args.stopwords_name)]
        vocab = load_pkl(args.data_dir + args.vocab_name)
        articles = [item[1].split(':')[1]
                    for item in [line.strip().split('|') for line in
                                 load_txt(args.data_dir + args.crime + '/' + args.law_article_content_name)]]
        articles = [[vocab[word] if word in vocab else vocab['<UNK>']
                     for word in article if word not in stop_words] for article in articles]
        sorted_articles = sorted(articles, key=lambda x: len(x), reverse=True)
        idx_reflect = {articles.index(article): sorted_articles.index(article) for article in articles}
        sorted_articles = [torch.tensor(article) for article in sorted_articles]
        self.articles_lens = [len(article) for article in sorted_articles]
        self.articles = torch.nn.utils.rnn.pad_sequence(sorted_articles, batch_first=True)

        self.corpus = []
        self.labels = []
        self.max_fact_len = 0

        for data in corpus:
            label = data['decision']
            if label is None:
                continue

            tokenized_data = []

            for sen in data['sentences']:
                fact = [vocab[word] if word in vocab else vocab['<UNK>']
                        for word in jieba.lcut(sen) if word not in stop_words]
                if len(fact) == 0:
                    continue
                self.max_fact_len = max(self.max_fact_len, len(fact))
                article = [idx_reflect[idx] for idx in data['sentences'][sen]]
                if len(article) == 0:
                    continue
                tokenized_data.append((fact, article))
            if len(tokenized_data) == 0:
                continue
            self.corpus.append(tokenized_data)
            self.labels.append(label_reflect(label))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item], self.labels[item], self.max_fact_len, self.articles, self.articles_lens


def collate_fn_fine(batch):
    max_fact_len, articles, article_lens = batch[0][2], batch[0][3], batch[0][4]
    facts = []
    fact_lens = []
    fact_indices = []
    article_indices = []
    labels = []

    for item in batch:
        para = item[0]
        labels.append(item[1])
        _fact_indices = []
        _article_indices = []
        for i, (fact, a_indices) in enumerate(para):
            facts.append(torch.tensor(pad_sequence(fact, max_fact_len)))
            fact_lens.append(len(fact))
            _fact_indices.append(len(facts) - 1)
            _article_indices.append(torch.tensor(a_indices))

        fact_indices.append(torch.tensor(_fact_indices))
        article_indices.append(_article_indices)  # 有对应的情况

    return (torch.stack(facts).long(), torch.tensor(fact_lens), articles.long(), torch.tensor(article_lens),
            fact_indices, article_indices, torch.tensor(labels))


def collate_fn_coarse(batch):
    max_fact_len, articles, article_lens = batch[0][2], batch[0][3], batch[0][4]
    facts = []
    fact_lens = []
    fact_indices = []
    article_indices = []
    labels = []

    for item in batch:
        para = item[0]
        labels.append(item[1])
        _fact_indices = []
        _article_indices = []
        for i, (fact, a_indices) in enumerate(para):
            facts.append(torch.tensor(pad_sequence(fact, max_fact_len)))
            fact_lens.append(len(fact))
            _fact_indices.append(len(facts) - 1)
            for index in a_indices:
                if index not in _article_indices:
                    _article_indices.append(index)
        fact_indices.append(torch.tensor(_fact_indices))
        article_indices.append(torch.tensor(_article_indices))  # 无对应的情况

    return (torch.stack(facts).long(), torch.tensor(fact_lens), articles.long(), torch.tensor(article_lens),
            fact_indices, article_indices, torch.tensor(labels))


class FactEncoder(nn.Module):
    def __init__(self, arg):
        super(FactEncoder, self).__init__()
        self.fact_encoder = nn.LSTM(arg.d_model, arg.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, facts, fact_lens):
        """

        :param facts: (batch_size, batch_max_len, embedding_size)
        :param fact_lens: (batch_size)
        :return:
        """
        sorted_lens, indices = torch.sort(fact_lens, descending=True, dim=-1)
        unsorted_indices = torch.argsort(indices, dim=-1)
        sorted_facts = torch.index_select(facts, 0, indices)
        packed_facts = pack_padded_sequence(sorted_facts, sorted_lens.cpu(), batch_first=True)
        encoded_facts = self.fact_encoder(packed_facts)[0]
        encoded_facts = pad_packed_sequence(encoded_facts, batch_first=True)[0]
        encoded_facts = torch.index_select(encoded_facts, 0, unsorted_indices)

        return torch.sum(encoded_facts, dim=1)


class ArticleEncoder(nn.Module):
    def __init__(self, arg):
        super(ArticleEncoder, self).__init__()
        self.article_encoder = nn.LSTM(arg.d_model, arg.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, articles, article_lens):
        """

        :param articles: (article_num, max_article_len, embedding_size)
        :param article_lens: (article_num)
        :return:
        """
        packed_articles = pack_padded_sequence(articles, article_lens.cpu(), batch_first=True)
        encoded_articles = self.article_encoder(packed_articles)[0]
        encoded_articles = pad_packed_sequence(encoded_articles, batch_first=True)[0]

        return torch.sum(encoded_articles, dim=1)


class DecisionPredictor(nn.Module):
    def __init__(self, arg):
        super(DecisionPredictor, self).__init__()
        self.fine_grained = arg.fine_grained_penalty_predictor
        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.data_dir + args.embedding_matrix_name), freeze=True)
        self.fact_encoder = FactEncoder(arg)
        self.article_encoder = ArticleEncoder(arg)

        self.interaction = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(arg.hidden_size * 4, 64)
        )
        self.predict = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, arg.num_decisions)
        )

    def forward(self, facts, fact_lens, artis, arti_lens, fact_indices, arti_indices):
        """

        :param facts: (fact_num, fact_padding_len)
        :param fact_lens: (fact_num)
        :param artis: (article_num, max_article_len)
        :param arti_lens: (article_num)
        :param fact_indices: (batch_size, *)
        :param arti_indices: (batch_size, *, *)
        :return:
        """
        embeded_facts = self.embedding(facts)
        embeded_artis = self.embedding(artis)

        encoded_facts = self.fact_encoder(embeded_facts, fact_lens)
        encoded_artis = self.article_encoder(embeded_artis, arti_lens)

        inter_res = []
        if self.fine_grained:
            for f_indices, a_indices in zip(fact_indices, arti_indices):
                f_indices = f_indices.to(encoded_facts.device)
                select_facts = torch.index_select(encoded_facts, 0, f_indices)
                select_articles = []
                for index in a_indices:
                    index = index.to(encoded_artis.device)
                    select_articles.append(torch.sum(torch.index_select(encoded_artis, 0, index), dim=0).unsqueeze(dim=0))
                select_articles = torch.cat(select_articles, dim=0)
                inter_info = self.interaction(torch.tanh(torch.cat([select_facts, select_articles], dim=-1)))
                inter_info = torch.sum(inter_info, dim=0)
                inter_res.append(inter_info.unsqueeze(0))
        else:
            for f_indices, a_indices in zip(fact_indices, arti_indices):
                f_indices, a_indices = f_indices.to(encoded_facts.device), a_indices.to(encoded_artis.device)
                select_facts = torch.index_select(encoded_facts, 0, f_indices)
                select_articles = torch.index_select(encoded_artis, 0, a_indices)

                select_facts = torch.sum(select_facts, dim=0)
                select_articles = torch.sum(select_articles, dim=0)
                inter_info = self.interaction(torch.tanh(torch.cat([select_facts, select_articles], dim=-1)))
                inter_res.append(inter_info.unsqueeze(0))

        output = self.predict(torch.tanh(torch.cat(inter_res, dim=0)))

        return output


def train_model(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.checkpoint_dir + args.crime + '/'):
        os.makedirs(args.checkpoint_dir + args.crime + '/')

    print('Load data...')
    all_data = load_json(args.data_dir + args.crime + '/' + args.penalty_predict_corpus_name)
    random.shuffle(all_data)
    train_data, test_data = train_test_split(all_data, test_size=0.2)
    test_data, valid_data = train_test_split(test_data, test_size=0.5)

    train_dataset, valid_dataset, test_dataset = \
        MyDataSet(train_data), MyDataSet(valid_data), MyDataSet(test_data)
    if args.fine_grained_penalty_predictor:
        train_dataloader, valid_dataloader, test_dataloader = \
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_fine), \
            DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn_fine), \
            DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_fine)
    else:
        train_dataloader, valid_dataloader, test_dataloader = \
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_coarse), \
            DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn_coarse), \
            DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_coarse)

    print('Initialize model...')
    model = DecisionPredictor(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.fine_grained_penalty_predictor:
        model_path = args.checkpoint_dir + args.crime + '/fine_grained_penalty_predictor'
    else:
        model_path = args.checkpoint_dir + args.crime + '/coarse_grained_penalty_predictor'
    early_stopping = EarlyStopping(output_path=model_path, patience=10, verbose=True)

    print('Train model...')
    for epoch in range(args.epochs):
        print('Epoch ' + str(epoch + 1))
        model.train()
        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        for i, (facts, fact_lens, artis, arti_lens, fact_indices, arti_indices, labels) in enumerate(train_dataloader):
            facts, fact_lens, artis, arti_lens, labels = \
                facts.to(device), fact_lens.to(device), artis.to(device), arti_lens.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(facts, fact_lens, artis, arti_lens, fact_indices, arti_indices)
            loss = criterion(output, labels)
            acc = accuracy_score(labels.cpu(), torch.argmax(output.cpu(), dim=-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc
            if (i + 1) % 10 == 0:
                print('[ {}/{} ] loss: {:.3f} acc: {:.3f}'.format(i + 1, len(train_dataloader), loss.item(), acc))

        secs = int(time.time() - start_time)
        print('\nTrain | Loss: {:.5f} | Accuracy: {:.3f} | Time: {:.1f}s'
              .format(train_loss / len(train_dataloader), train_acc / len(train_dataloader), secs))

        model.eval()
        valid_loss = 0.0
        valid_preds = []
        valid_labels = []
        with torch.no_grad():
            for facts, fact_lens, artis, arti_lens, fact_indices, arti_indices, labels in valid_dataloader:
                facts, fact_lens, artis, arti_lens, labels = \
                    facts.to(device), fact_lens.to(device), artis.to(device), arti_lens.to(device), labels.to(device)
                preds = model(facts, fact_lens, artis, arti_lens, fact_indices, arti_indices)
                loss = criterion(preds, labels)
                valid_loss += loss.item()
                valid_preds.append(torch.argmax(preds.cpu(), dim=-1))
                valid_labels.append(labels.cpu())

        valid_loss = valid_loss / len(valid_dataloader)
        valid_preds = torch.cat(valid_preds, dim=0)
        valid_labels = torch.cat(valid_labels, dim=0)
        valid_acc = accuracy_score(valid_labels, valid_preds)
        valid_precision = precision_score(valid_labels, valid_preds, average='macro')
        valid_recall = recall_score(valid_labels, valid_preds, average='macro')
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')

        print('Valid | Loss: {:.5f} | Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'
              .format(valid_loss, valid_acc, valid_precision, valid_recall, valid_f1))
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('-----------------------------------------------\n')

    print('-----------------------------------------------\n')
    print('Test model...')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for facts, fact_lens, artis, arti_lens, fact_indices, arti_indices, labels in test_dataloader:
            facts, fact_lens, artis, arti_lens, labels = \
                facts.to(device), fact_lens.to(device), artis.to(device), arti_lens.to(device), labels.to(device)
            preds = model(facts, fact_lens, artis, arti_lens, fact_indices, arti_indices)
            loss = criterion(preds, labels)
            test_loss += loss.item()
            test_preds.append(torch.argmax(preds.cpu(), dim=-1))
            test_labels.append(labels.cpu())

    test_loss = test_loss / len(test_dataloader)
    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_acc = accuracy_score(test_labels, test_preds)
    for way in ['micro', 'macro', 'weighted']:
        print(way + ' metrics: ')
        test_precision = precision_score(test_labels, test_preds, average=way)
        test_recall = recall_score(test_labels, test_preds, average=way)
        test_f1 = f1_score(test_labels, test_preds, average=way)

        print('Test | Loss: {:.5f} | Accuracy: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}'
              .format(test_loss, test_acc, test_precision, test_recall, test_f1))


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def label_reflect(label):
    if label == 0:  # 免予刑事处罚
        return 0
    elif label == -1:  # 拘役
        return 1
    elif 0 < label < 12:  # 1年以下有期徒刑
        return 2
    elif 12 <= label < 36:  # 1年及1年以上，3年以下有期徒刑
        return 3
    else:  # 3年及3年以上，10年以下有期徒刑
        return 4


def pad_sequence(seq, max_len):
    padded_seq = np.zeros(max_len)
    for i, c in enumerate(seq):
        padded_seq[i] = c
    return padded_seq


if __name__ == '__main__':
    args = parse()
    seed_torch(args.seed)
    train_model(args)
