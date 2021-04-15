# -*- coding: UTF-8 -*-
# 法条前后件分类模型
import json
import re
import article_rules
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import numpy as np


def load_dataset_for_rf(data_path, common_words_path):
    f = open(data_path, "r", encoding='utf8')
    dataset = json.load(f)
    f.close()

    f = open(common_words_path, "r", encoding="utf8")
    common_words = [line.strip() for line in f.readlines()]
    f.close()

    train_X, train_Y, test_X, test_Y = [], [], [], []
    trainSet, valSet, testSet = dataset['trainSet'], dataset['valSet'], dataset['testSet']

    for sample in trainSet + valSet:
        text, label = sample[0].strip().replace(' ', '').replace('\n', ''), int(sample[1])

        if label != 0 and label != 1:
            continue

        tokenized_article = process_multilaw_text2id(text, common_words)
        train_X.append(tokenized_article)
        train_Y.append(label)

    for sample in testSet:
        text, label = sample[0].strip().replace(' ', '').replace('\n', ''), int(sample[1])

        if label != 0 and label != 1:
            continue

        tokenized_article = process_multilaw_text2id(text, common_words)
        test_X.append(tokenized_article)
        test_Y.append(label)

    return train_X, train_Y, test_X, test_Y


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


def train_rf(dataset_path, common_words_path, model_path):
    print("Load Data...")
    train_X, train_Y, test_X, test_Y = load_dataset_for_rf(dataset_path, common_words_path)

    # print("Train Model...")
    # model = RandomForestClassifier(n_estimators=100)
    # train_X, train_Y = np.array(train_X), np.array(train_Y)
    # model.fit(train_X, train_Y)
    #
    # print("Save Model...")
    # with open(model_path, 'wb') as f:
    #     pickle.dump(model, f)

    print("Load Model...")
    f = open(model_path, 'rb')
    model = pickle.load(f)
    f.close()

    print("Test Model...")
    test_X, test_Y = np.array(test_X), np.array(test_Y)
    pred_y = model.predict(test_X)
    print(metrics.classification_report(test_Y, pred_y, digits=4))
    print(metrics.accuracy_score(test_Y, pred_y))


if __name__ == "__main__":
    train_rf("./data/law_dataset.json", "./data/law common words.txt", "./output/article_condition_classifier")
