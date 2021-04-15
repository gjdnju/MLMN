# -*- coding: utf-8 -*-
import re
import pickle
import json
import numpy as np
from utils import load_txt
import article_rules


def generate_article_qhj_dict(article_path, common_words_path, model_path, output_path):
    # 根据模型预测法条前后件
    all_articles = load_txt(article_path)
    all_articles = [items[1] for items in [line.strip().split('|') for line in all_articles]]
    all_articles = [article.split(':') for article in all_articles]
    all_articles = {article[0]: 'S' + article[1] + 'E' for article in all_articles}

    f = open(common_words_path, "r", encoding="utf8")
    common_words = [line.strip() for line in f.readlines()]
    f.close()

    f = open(model_path, 'rb')
    model = pickle.load(f)
    f.close()

    article_qhj_dict = {}

    for article in all_articles:
        content = all_articles[article].strip().replace(' ', '').replace('\n', '')
        test_X, qj, hj = [], [], []
        split_pos = re.finditer(r'，|；|。|：', content)
        pos_idx = []
        for i in split_pos:
            pos_idx.append(i.span()[0])
        if len(pos_idx) == 1:
            test_X.append(process_multilaw_text2id(content, common_words))
        else:
            test_X.append(process_multilaw_text2id(content[:pos_idx[1]+1], common_words))
            if len(pos_idx) > 2:
                test_X.append(process_multilaw_text2id(content[1:pos_idx[2]+1], common_words))
                for i in range(len(pos_idx)-3):
                    test_X.append(process_multilaw_text2id(content[pos_idx[i]+1: pos_idx[i+3]+1], common_words))
                test_X.append(process_multilaw_text2id(content[pos_idx[-3]+1:], common_words))
            else:
                test_X.append(process_multilaw_text2id(content[1:], common_words))

        test_X = np.array(test_X)
        pred_Y = model.predict(test_X)
        content = re.split(r'，|；|。|：', content[1:-1])
        for i in range(test_X.shape[0]):
            if pred_Y[i] == 0:
                qj.append(content[i])
            else:
                hj.append(content[i])

        article_qhj_dict[article] = {
            'whole': all_articles[article][1:-1],
            'qj': '。'.join(qj),
            'hj': '。'.join(hj)
        }

    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(article_qhj_dict, f, ensure_ascii=False)


def process_multilaw_text2id(line, dictionary):
    pattern = '，|。|；|：'
    regx = re.compile(pattern)
    array = regx.split(line)
    pre, cText, next = '', '', ''
    v1, v2, v3 = [], [], []
    puncVectors = []
    punction = {"，": 1, '。': 2, '：': 3, "；": 4}

    assert len(array) in [2, 3, 4], ValueError(
        "Contain wrong number of sub items:{0} with {1} sub items".format(input, len(array)))
    if len(array) == 2:  # 独立的一个句子
        v1 = [0 for _ in range(len(dictionary) - 2)] + [1, 0]
        v2 = process_law_text2id(array[0][1:], dictionary)
        v3 = [0 for _ in range(len(dictionary) - 1)] + [1]

        # 前后都是空
        cText = array[0][1:]

        # 标点符号
        puncVectors = [0, punction[line[-2]], 0]
    elif len(array) == 3:  #
        if line[0] == 'S':
            v1 = [0 for _ in range(len(dictionary) - 2)] + [1, 0]
            v2 = process_law_text2id(array[0][1:], dictionary)
            v3 = process_law_text2id(array[1], dictionary)

            # 前面是空
            cText = array[0][1:]
            next = array[1]

            # 标点符号
            puncVectors = [0, punction[line[len(array[0])]], punction[line[-1]]]
        elif line[-1] == 'E':
            v1 = process_law_text2id(array[0], dictionary)
            v2 = process_law_text2id(array[1], dictionary)
            v3 = [0 for _ in range(len(dictionary) - 1)] + [1]

            # 后面是空
            pre = array[0]
            cText = array[1]

            # 标点符号
            puncVectors = [punction[line[len(array[0])]], punction[line[-2]], 0]

    else:
        v1 = process_law_text2id(array[0], dictionary)
        v2 = process_law_text2id(array[1], dictionary)
        v3 = process_law_text2id(array[2], dictionary)

        # 没有是空
        pre = array[0]
        cText = array[1]
        next = array[2]

        # 标点符号
        # print("line len:{0}, array[0]:{1}, array[1]:{2}".format(len(line),len(array[0]),len(array[1])))
        puncVectors = [punction[line[len(array[0])]], punction[line[len(array[0]) + len(array[1]) + 1]], \
                       punction[line[-1]]]

    assert len(v1) == len(v2) == len(v3), ValueError("Wrong vector size:" + line)
    rulefeatures = rule_features(pre, cText, next)
    return v2 + rulefeatures + puncVectors


def process_law_text2id(line, dictionary):
    init_content = line.strip()
    vector = []
    if init_content != "":
        for word in dictionary:
            times = 1 if str(init_content).count(word) > 0 else 0
            vector.append(times)
    return vector


def rule_features(pre, cText, next):
    pre_QRule = article_rules.QRules()
    pre_HRule = article_rules.HRules()
    cText_QRule = article_rules.QRules()
    cText_HRule = article_rules.HRules()
    next_QRule = article_rules.QRules()
    next_HRule = article_rules.HRules()

    # 判断当前和之后的句子是否以连词开头
    # cText_conRule = rules.Rules(cText)
    # next_conRule = rules.Rules(next)
    # con_words = ["并且","或者","但是","并","而且"]
    return pre_QRule.inter(pre) + pre_HRule.inter(pre) + cText_QRule.inter(cText) + cText_HRule.inter(cText) \
           + next_QRule.inter(next) + next_HRule.inter(next)
    # + [int(cText_conRule.rule1(con_words)),int(next_conRule.rule1(con_words))]


if __name__ == '__main__':
    generate_article_qhj_dict('./data/hurt/article_dict.txt', './data/law common words.txt',
                              './models/article_condition_classifier', './data/hurt/article_qhj_dict.json')
