# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:parameters.py
@time:2022/03/22
"""

import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', dest='seed', type=int, default=1024)
    parser.add_argument('--crime', type=str, default='traffic', choices=['traffic', 'hurt'])
    parser.add_argument('--fact_len', dest='fact_len', type=int, default=50)
    parser.add_argument('--article_len', dest='article_len', type=int, default=50)
    parser.add_argument('--use_pretrain_model', action='store_true')
    parser.add_argument('--model_name', type=str, default='ThreeLayers',
                        choices=['ThreeLayers', 'ThreeLayersWithElement',
                                 'ThreeLayersPretrain', 'ThreeLayersWithElementPretrain',
                                 'ArcI', 'ArcII', 'MatchPyramid', 'MVLSTM'])
    parser.add_argument('--pretrain_model_name', type=str, default='xlm-roberta-base')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--recommendation_corpus_name', type=str, default='labeled_corpus.txt')
    parser.add_argument('--stopwords_name', type=str, default='stopwords.txt')
    parser.add_argument('--vocab_name', type=str, default='vocab.pkl')
    parser.add_argument('--embedding_matrix_name', type=str, default='embedding_matrix.pkl')
    parser.add_argument('--split_data_dir', type=str, default='/split_data/')
    parser.add_argument('--train_recommendation_corpus_name', type=str, default='train_data.txt')
    parser.add_argument('--valid_recommendation_corpus_name', type=str, default='valid_data.txt')
    parser.add_argument('--test_recommendation_corpus_name', type=str, default='test_data.txt')
    parser.add_argument('--law_article_content_name', type=str, default='article_dict.txt')
    parser.add_argument('--law_article_qhj_dict_name', type=str, default='law_qhj_dict.json')

    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128)
    parser.add_argument('--filters_num', dest='filters_num', type=int, default=128)
    parser.add_argument('--kernel_size_1', dest='kernel_size_1', type=int, default=2)
    parser.add_argument('--kernel_size_2', dest='kernel_size_2', type=int, default=4)
    parser.add_argument('--linear_output', dest='linear_output', type=int, default=128)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    parser.add_argument('--negtive_multiple', dest='negtive_multiple', type=int, default=12)
    parser.add_argument('--earlystop_patience', dest='earlystop_patience', type=int, default=10)

    # For penalty predictor
    parser.add_argument('--num_decisions', dest='num_decisions', type=int, default=5)
    parser.add_argument('--fine_grained_penalty_predictor', action='store_true')
    parser.add_argument('--penalty_predict_corpus_name', dest='penalty_predict_corpus_name', type=str,
                        default='penalty_corpus.json')
    parser.add_argument('--d_model', dest='d_model', type=int, default=128)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)

    args = parser.parse_args()

    return args
