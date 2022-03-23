# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:data_process.py
@time:2022/03/22
"""
import os
import re
import json
import pickle
import jieba
import jieba.posseg as pos
from random import shuffle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def generate_dataset_for_recommendation(args):
    all_data = load_txt(args.data_dir + args.crime + '/' + args.recommendation_corpus_name)
    cases = set()
    for line in all_data:
        cases.add(line.split('|')[0])

    # 按案件划分训练集、测试集
    train_cases, test_cases = train_test_split(list(cases), shuffle=True, test_size=0.2)
    valid_cases, test_cases = train_test_split(test_cases, shuffle=True, test_size=0.5)

    train_data, test_data, valid_data = [], [], []
    for line in all_data:
        case = line.split('|')[0]
        if case in train_cases:
            train_data.append(line)
        elif case in test_cases:
            test_data.append(line)
        else:
            valid_data.append(line)

    if not os.path.exists(args.data_dir + args.crime + args.split_data_dir):
        os.makedirs(args.data_dir + args.crime + args.split_data_dir)

    write_txt(train_data, args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name)
    write_txt(valid_data, args.data_dir + args.crime + args.split_data_dir + args.valid_recommendation_corpus_name)
    write_txt(test_data, args.data_dir + args.crime + args.split_data_dir + args.test_recommendation_corpus_name)


def load_json(path):
    f = open(path, "r", encoding="utf8")
    data =json.load(f)
    f.close()

    return data


def load_txt(path):
    f = open(path, "r", encoding="utf8")
    data = f.readlines()
    f.close()

    return data


def write_txt(all_data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(data.strip() + '\n')


def load_pkl(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()

    return obj


def get_process_data(data, stop_words, vocab, use_pretrain_model):
    pos_res = []
    neg_res = []
    for line in data:
        items = line.strip().split('|')
        assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
        fact = items[1] if use_pretrain_model else text2id(items[1], stop_words, vocab)
        positive_samples = set([int(num) for num in items[2].split(',')])
        negative_samples = set([int(num) for num in items[3].split(',')])

        for sample in positive_samples:
            pos_res.append([fact, sample, 1])

        for sample in negative_samples:
            neg_res.append([fact, sample, 0])

    shuffle(pos_res)
    shuffle(neg_res)
    return pos_res, neg_res


def get_article_dict(args, stop_words, vocab):
    article_dict = load_txt(args.data_dir + args.crime + '/' + args.law_article_content_name)
    article_dict = {int(item[0]): item[1] for item in [line.split('|') for line in article_dict]}

    for idx in article_dict:
        article_content = article_dict[idx].split(':')[1].replace('\n', '').replace('\t', '').replace(' ', '')
        if not args.use_pretrain_model:
            article_content = text2id_without_filter(article_content, stop_words, vocab)
        article_dict[idx] = article_content

    return article_dict


def get_article_dict_with_element(args, stop_words, vocab):
    article_dict = load_txt(args.data_dir + args.crime + '/' + args.law_article_content_name)
    article_dict = {int(item[0]): item[1] for item in [line.split('|') for line in article_dict]}
    article_qhj_dict = load_json(args.data_dir + args.crime + '/' + args.law_article_qhj_dict_name)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name) if args.use_pretrain_model else None

    for idx in article_dict:
        article_units = article_dict[idx].split(':')
        article_name = article_units[0]
        article_content = article_units[1]

        content_split = re.split(r'[，；。：]', article_content)
        content_split = list(filter(lambda x: x != "", list(map(lambda x: x.lstrip().rstrip(), content_split))))
        article_qj = [s.lstrip().rstrip() for s in article_qhj_dict[article_name]['qj'].split('。')]
        article_hj = [s.lstrip().rstrip() for s in article_qhj_dict[article_name]['hj'].split('。')]

        if args.use_pretrain_model:
            article_dict[idx] = [article_content, None]
            split_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(article_content, truncation=True, max_length=args.article_len + 2)['input_ids'][1:-1])
            article_elements = []
            for content in content_split:
                if content in article_qj:
                    article_elements.append([1, 0])
                elif content in article_hj:
                    article_elements.append([0, 1])
                else:
                    article_elements.append([0, 0])

            article_label = []
            cnt = 0
            for token in split_tokens:
                article_label.append(article_elements[cnt])
                if token in [',', '。', ';', ':']:
                    cnt += 1
            article_dict[idx][1] = article_label
        else:
            article = []
            article_label = []
            for content in content_split:
                if content == '但是' or content == '其中':
                    continue
                content_vector = text2id_without_filter(content, stop_words, vocab)
                article.extend(content_vector)
                this_article_label = [1, 0] if content in article_qj else [0, 1]
                article_label += [this_article_label for _ in range(len(content_vector))]
            article_dict[idx] = (article, article_label)
    return article_dict


def text2id(text, stopwords, vocab):
    # filter 'nr', 'ns', 'p', 'u', 'm'
    words = pos.cut(text)
    filter_words = []
    for w in words:
        word = w.word.strip()
        if word == '' or word in stopwords or w.flag in ['nr', 'ns', 'p', 'u', 'm']:
            continue
        filter_words.append(word)

    ids = []
    for word in filter_words:
        if word not in vocab.keys():
            ids.append(vocab['<UNK>'])
        else:
            ids.append(vocab[word])

    return ids


def text2id_without_filter(text, stopwords, vocab):
    words = jieba.lcut(text)

    ids = []
    for word in words:
        if word in stopwords:
            continue
        if word not in vocab.keys():
            ids.append(vocab['<UNK>'])
        else:
            ids.append(vocab[word])

    return ids
