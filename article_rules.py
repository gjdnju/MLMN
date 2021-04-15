# -*- coding: UTF-8 -*-
# 该文件是为了通过正则化方式抽取特征
import re


class Rules:
    def __init__(self, x):
        self.x = x

    # 以某些词开头的
    def rule1(self, words):
        for w in words:
            if str(self.x).startswith(w):
                return True
        return False

    # 以某些词结尾
    def rule2(self, words):
        for w in words:
            if str(self.x).endswith(w):
                return True
        return False

    # "XXX"这种名词性规则
    def rule3(self):
        pattern = r"^\"\S+\"$"
        if re.search(pattern, self.x, re.M | re.I):
            return True
        return False

    # 包含某些词的
    def rule4(self, words):
        for w in words:
            if str(self.x).count(w) > 0:
                return True
        return False


class QRules:
    # 以某些词开头
    def rule1(self, input):
        words = ["如果", "根据", "对于", "是指"]
        rule = Rules(input)
        return rule.rule1(words)

    # 以某些词结尾
    def rule2(self, input):
        words = ["除外", "时", "的"]
        rule = Rules(input)
        return rule.rule2(words)

    # 包含某些词
    def rule3(self, input):
        words = ["下列情形"]
        rule = Rules(input)
        return rule.rule4(words)

    def inter(self, input):
        return [int(self.rule1(input)), int(self.rule2(input)), int(self.rule3(input))]


class HRules:
    # 以某些词开头
    def rule1(self, input):
        words = ["应该", "可以", "也可以", "不得"]
        rule = Rules(input)
        return rule.rule1(words)

    # 包含某些词
    def rule2(self, input):
        words = ["是指", "应当按规定", "应当"]
        rule = Rules(input)
        return rule.rule4(words)

    # 名词解释
    def rule3(self, input):
        rule = Rules(input)
        return rule.rule3()

    def inter(self, input):
        return [int(self.rule1(input)), int(self.rule2(input)), int(self.rule3(input))]


class QRulesEx(QRules):
    # 首先使用父类最基础的来判断看效果
    def predict(self, input):
        return self.rule1(input) or self.rule2(input) or self.rule3(input)

    def predict2(self, input, line):
        if self.predict(input):
            return True

        # 判断前一句是否是已经结束，如果已经结束，则很有可能是前件。或者当前句子是第一句
        if line[0] == 'S':
            return True

        index = str(line).index(input)
        if line[index - 1] in ['。', '；']:
            return True
        return False

    def predict3(self, input, line):
        if self.predict2(input, line):
            return True

        # 增加前一句如果是前件，且二者是并列关系的话：或者、并且，但是，则当前也是前件
        index = str(line).index(input)
        pre = line[:index]
        rule = Rules(input)
        if self.predict(pre) and rule.rule1(["或者", "并且", "但是"]):
            return True
        return False

    def predict_s(self, input, line):
        # 对规则一~规则三覆盖效果进行检查
        # return self.rule1(input)

        # #对规则六覆盖效果进行检查
        # if line[0] == 'S':
        #     return True
        #
        # index = str(line).index(input)
        # if line[index - 1] in ['。', '；']:
        #     return True

        # 对规则八覆盖效果进行检查
        index = str(line).index(input)
        pre = line[:index]
        rule = Rules(input)
        if self.predict(pre) and rule.rule1(["或者", "并且", "但是"]):
            return True

        return False

    def predict_merge(self, input, line):
        return self.rule1(input) or self.rule2(input) or self.rule3(input)


class HRulesEx(HRules):
    def predict(self, input):
        return self.rule1(input) or self.rule2(input) or self.rule3(input)

    def predict2(self, input, line):
        if self.predict(input):
            return True

        # 判断当前句子是否是最后一句
        if line[-1] == 'E' and not str(input).endswith("除外"):
            return True
        return False

    def predict3(self, input, line):
        if self.predict2(input, line):
            return True

        # 增加前一句如果是后件，且二者是并列关系的话：或者、并且，但是，则当前也是后件
        index = str(line).index(input)
        pre = line[:index]
        rule = Rules(input)
        if self.predict(pre) and rule.rule1(["或者", "并且", "但是"]):
            return True
        return False

    def predict_s(self, input, line):
        # 检查规则四五
        # return self.rule2(input)

        # 检查规则七
        # if line[-1] == 'E' and not str(input).endswith("除外"):
        #     return True
        # return False

        # 检查规则八
        index = str(line).index(input)
        pre = line[:index]
        rule = Rules(input)
        if rule.rule1(["或者", "并且", "但是"]):
            print(input)
        if self.predict(pre) and rule.rule1(["或者", "并且", "但是"]):
            return True
        return False

    def predict_merge(self, input, line):
        return self.rule1(input) or self.rule2(input)
