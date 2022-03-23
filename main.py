# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:main.py
@time:2022/03/22
"""
from models import *
import os
import random
import warnings

import numpy as np

from parameters import parse
from train_tools import train_model
from data_process import generate_dataset_for_recommendation, load_txt, get_process_data, get_article_dict, get_article_dict_with_element

warnings.filterwarnings('ignore')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = parse()
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.checkpoint_dir + args.crime + '/'):
        os.makedirs(args.checkpoint_dir + args.crime + '/')

    stop_words = [line.strip() for line in load_txt(args.data_dir + args.stopwords_name)]
    vocab = load_pkl(args.data_dir + args.vocab_name)

    article_dict = get_article_dict_with_element(args, stop_words, vocab) if 'WithElement' in args.model_name \
        else get_article_dict(args, stop_words, vocab)

    if not os.path.exists(args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name):
        generate_dataset_for_recommendation(args)

    train_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name)
    train_data = get_process_data(train_data, stop_words, vocab, args.use_pretrain_model)

    valid_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.valid_recommendation_corpus_name)
    valid_data = get_process_data(valid_data, stop_words, vocab, args.use_pretrain_model)

    test_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.test_recommendation_corpus_name)
    test_data = get_process_data(test_data, stop_words, vocab, args.use_pretrain_model)

    assert args.use_pretrain_model ^ ('Pretrain' in args.model_name) is False, \
        "Use pretrain models should use command 'python --use_pretrain_model --... main.py'"

    model = globals()[args.model_name](args)

    train_model(model, args, train_data, valid_data, test_data, article_dict)
