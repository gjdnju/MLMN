# -*- coding: UTF-8 -*-
import re
import pickle
import jieba
import jieba.posseg as pos
from sklearn.model_selection import train_test_split

from utils import load_txt, load_pkl, load_json
from parameters import parse


def generate_dataset_for_recommendation(args):
    all_data = load_txt(args.corpus_path)
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
        elif case in valid_cases:
            valid_data.append(line)
        else:
            print('Error!')

    with open(args.train_corpus_for_recommendation, 'w', encoding='utf8') as f:
        for data in train_data:
            f.write(data)

    with open(args.valid_corpus_for_recommendation, 'w', encoding='utf8') as f:
        for data in valid_data:
            f.write(data)

    with open(args.test_corpus_for_recommendation, 'w', encoding='utf8') as f:
        for data in test_data:
            f.write(data)


def generate_article_dict_for_train(args):
    stop_words = [line.strip() for line in load_txt(args.stopwords_path)]
    vocab = load_pkl(args.vocab_path)
    article_dict = load_txt('./data/mixed/article_dict.txt')
    article_dict = {int(item[0]): item[1] for item in [line.split('|') for line in article_dict]}

    for idx in article_dict:
        article_content = article_dict[idx].split(':')[1].replace('\n', '').replace('\t', '').replace(' ', '')
        article_content = text2id_without_filter(article_content, stop_words, vocab)
        article_dict[idx] = article_content

    with open('./data/mixed/article_dict_for_train.pkl', 'wb') as f:
        pickle.dump(article_dict, f)


def generate_article_dict_for_train_with_element(args):
    stop_words = [line.strip() for line in load_txt(args.stopwords_path)]
    vocab = load_pkl(args.vocab_path)
    article_dict = load_txt('./data/mixed/article_dict.txt')
    article_dict = {int(item[0]): item[1] for item in [line.split('|') for line in article_dict]}
    article_qhj_dict = load_json(args.article_qhj_dict_path)

    for idx in article_dict:
        article_units = article_dict[idx].split(':')
        article_name = article_units[0]
        article_content = article_units[1]

        content_split = re.split(r'[，；。：]', article_content)
        content_split = list(filter(lambda x: x != "", list(map(lambda x: x.lstrip().rstrip(), content_split))))
        article_qj = [s.lstrip().rstrip() for s in article_qhj_dict[article_name]['qj'].split('。')]

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

    with open('./data/mixed/article_dict_for_train_with_element.pkl', 'wb') as f:
        pickle.dump(article_dict, f)


def text2id(text, stopwords, vocab):
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


if __name__ == '__main__':
    args = parse()
    generate_dataset_for_recommendation(args)
    generate_article_dict_for_train(args)
    generate_article_dict_for_train_with_element(args)
