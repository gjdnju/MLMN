import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import load_pkl


class ThreeLayersWithElement(nn.Module):

    def __init__(self, args):
        super(ThreeLayersWithElement, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)
        self.fact_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2)
        ])
        self.article_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2)
        ])

        self.conv_paddings = nn.ModuleList([
            nn.ConstantPad1d((args.kernel_size_1 // 2 - 1, args.kernel_size_1 // 2), 0.),
            nn.ConstantPad1d((args.kernel_size_2 // 2 - 1, args.kernel_size_2 // 2), 0.)
        ])

        self.ffs = nn.ModuleList(
            [nn.Linear(args.filters_num + 2, args.linear_output) for _ in range(2)] +
            [nn.Linear(args.article_len * 2, args.linear_output)]
        )

        self.dropout = nn.Dropout(p=args.dropout)

        self.predict = nn.Linear(args.linear_output, 2)

    def forward(self, fact, article, article_label):
        """

        :param fact: (batch_size, fact_len, embedding_dim)
        :param article: (batch_size, article_len, embedding_dim)
        :return:
        """
        fact, article = self.embedding(fact), self.embedding(article)

        # fisrt-layer
        input_fact_1 = self.conv_paddings[0](fact.permute(0, 2, 1))
        input_article_1 = self.conv_paddings[0](article.permute(0, 2, 1))

        fact_conv_1 = self.fact_convs[0](input_fact_1).permute(0, 2, 1)
        article_conv_1 = self.article_convs[0](input_article_1).permute(0, 2, 1)

        inter_1 = self.interaction(fact_conv_1, article_conv_1).unsqueeze(-1)
        inter_repeat_1 = inter_1.repeat(1, 1, article_conv_1.size(-1))
        info_article_1 = inter_repeat_1.mul(article_conv_1)

        fusion_output_1 = self.ffs[0](torch.cat([info_article_1, article_label], dim=-1))
        fusion_output_max_1 = torch.max(fusion_output_1, dim=-1)[0]

        # second_layer
        input_fact_2 = self.conv_paddings[1](fact_conv_1.permute(0, 2, 1))
        input_article_2 = self.conv_paddings[1](article_conv_1.permute(0, 2, 1))

        fact_conv_2 = self.fact_convs[1](input_fact_2).permute(0, 2, 1)
        article_conv_2 = self.article_convs[1](input_article_2).permute(0, 2, 1)

        inter_2 = self.interaction(fact_conv_2, article_conv_2).unsqueeze(-1)
        inter_repeat_2 = inter_2.repeat(1, 1, article_conv_2.size(-1))
        info_article_2 = inter_repeat_2.mul(article_conv_2)

        fusion_output_2 = self.ffs[1](torch.cat([info_article_2, article_label], dim=-1))
        fusion_output_max_2 = torch.max(fusion_output_2, dim=-1)[0]

        fusion_output = torch.cat([fusion_output_max_1, fusion_output_max_2],
                                  dim=-1)
        fusion_output = self.dropout(F.relu(self.ffs[-1](fusion_output)))

        output = self.predict(fusion_output)

        return output

    @staticmethod
    def interaction(x1, x2):
        """

        :param x1: (batch_size, x1_seq_len, feature_size)
        :param x2: (batch_size, x2_seq_len, feature_size)
        :return: (batch_size, x2_seq_len)
        """
        dot_matrix = torch.matmul(x1, x2.permute(0, 2, 1))  # (batch_size, x1_seq_len, x2_seq_len)

        x1_2_x2 = F.softmax(dot_matrix, dim=2)
        x2_2_x1 = F.softmax(dot_matrix, dim=1)

        x1_weight = torch.sum(x2_2_x1, dim=2).unsqueeze(dim=1)  # (batch_size, 1, x1_seq_len)
        x2_weight = torch.matmul(x1_weight, x1_2_x2).squeeze(dim=1)  # (batch_size, x2_seq_len)

        return x2_weight

    @staticmethod
    def get_name():
        return 'Add Element'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0.01


class ThreeLayers(nn.Module):
    def __init__(self, args):
        super(ThreeLayers, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)
        self.fact_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2)
        ])
        self.article_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2),
        ])

        self.conv_paddings = nn.ModuleList([
            nn.ConstantPad1d((args.kernel_size_1 // 2 - 1, args.kernel_size_1 // 2), 0.),
            nn.ConstantPad1d((args.kernel_size_2 // 2 - 1, args.kernel_size_2 // 2), 0.)
        ])

        self.ffs = nn.ModuleList(
            [nn.Linear(args.embedding_dim, args.linear_output)] +
            [nn.Linear(args.filters_num, args.linear_output) for _ in range(2)] +
            [nn.Linear(args.article_len * 3, args.linear_output)]
            # [nn.Linear(args.article_len, args.linear_output)]
        )

        # self.dropout = nn.Dropout(p=args.dropout)

        self.predict = nn.Linear(args.linear_output, 2)

    def forward(self, fact, article):
        """

        :param fact: (batch_size, fact_len, embedding_dim)
        :param article: (batch_size, article_len, embedding_dim)
        :return:
        """
        fact, article = self.embedding(fact), self.embedding(article)

        # zero-layer
        inter_0 = self.interaction(fact, article).unsqueeze(-1)  # (batch_size, article_len, 1)
        inter_repeat_0 = inter_0.repeat(1, 1, fact.size(-1))  # (batch_size, article_len, embedding_dim)
        info_article_0 = inter_repeat_0.mul(article)  # (batch_size, article_len, embedding_dim)

        fusion_output_0 = self.ffs[0](info_article_0)
        fusion_output_max_0 = torch.max(fusion_output_0, dim=-1)[0]

        # fisrt-layer
        input_fact_1 = self.conv_paddings[0](fact.permute(0, 2, 1))
        input_article_1 = self.conv_paddings[0](article.permute(0, 2, 1))

        fact_conv_1 = self.fact_convs[0](input_fact_1).permute(0, 2, 1)
        article_conv_1 = self.article_convs[0](input_article_1).permute(0, 2, 1)

        inter_1 = self.interaction(fact_conv_1, article_conv_1).unsqueeze(-1)
        inter_repeat_1 = inter_1.repeat(1, 1, article_conv_1.size(-1))
        info_article_1 = inter_repeat_1.mul(article_conv_1)

        fusion_output_1 = self.ffs[1](info_article_1)
        fusion_output_max_1 = torch.max(fusion_output_1, dim=-1)[0]

        # second_layer
        input_fact_2 = self.conv_paddings[1](fact_conv_1.permute(0, 2, 1))
        input_article_2 = self.conv_paddings[1](article_conv_1.permute(0, 2, 1))

        fact_conv_2 = self.fact_convs[1](input_fact_2).permute(0, 2, 1)
        article_conv_2 = self.article_convs[1](input_article_2).permute(0, 2, 1)

        inter_2 = self.interaction(fact_conv_2, article_conv_2).unsqueeze(-1)
        inter_repeat_2 = inter_2.repeat(1, 1, article_conv_2.size(-1))
        info_article_2 = inter_repeat_2.mul(article_conv_2)

        fusion_output_2 = self.ffs[2](info_article_2)
        fusion_output_max_2 = torch.max(fusion_output_2, dim=-1)[0]

        fusion_output = torch.cat([fusion_output_max_0, fusion_output_max_1, fusion_output_max_2],
                                  dim=-1)
        # fusion_output = torch.cat([fusion_output_max_0, fusion_output_max_2],
        #                           dim=-1)
        fusion_output = F.dropout(F.relu(self.ffs[-1](fusion_output)))
        # fusion_output = F.dropout(F.relu(self.ffs[-1](fusion_output_max_2)))

        output = self.predict(fusion_output)

        return output

    @staticmethod
    def interaction(x1, x2):
        """

        :param x1: (batch_size, x1_seq_len, feature_size)
        :param x2: (batch_size, x2_seq_len, feature_size)
        :return: (batch_size, x2_seq_len)
        """
        dot_matrix = torch.matmul(x1, x2.permute(0, 2, 1))  # (batch_size, x1_seq_len, x2_seq_len)

        x1_2_x2 = F.softmax(dot_matrix, dim=2)
        x2_2_x1 = F.softmax(dot_matrix, dim=1)

        x1_weight = torch.sum(x2_2_x1, dim=2).unsqueeze(dim=1)  # (batch_size, 1, x1_seq_len)
        x2_weight = torch.matmul(x1_weight, x1_2_x2).squeeze(dim=1)  # (batch_size, x2_seq_len)

        return x2_weight

    @staticmethod
    def get_name():
        # return 'Three Layers Only Final'
        return 'Three Layers'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0.01


class LSTMModel(nn.Module):

    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)
        self.fact_lstms = nn.ModuleList([
            nn.GRU(args.embedding_dim, args.hidden_size, batch_first=True, bidirectional=True)
        ])
        self.article_lstms = nn.ModuleList([
            nn.GRU(args.embedding_dim, args.hidden_size, batch_first=True, bidirectional=True)
        ])
        self.ffs = nn.ModuleList([
            nn.Linear(args.embedding_dim, args.linear_output),
            nn.Linear(args.hidden_size * 2, args.linear_output),
            nn.Linear(args.article_len, args.linear_output)
        ])
        self.dropout = nn.Dropout(p=args.dropout)
        self.predict = nn.Linear(args.linear_output, 2)

    def forward(self, fact, article):
        # zero-layer
        fact_0, article_0 = self.embedding(fact), self.embedding(article)

        # first-layer
        fact_1, _ = self.fact_lstms[0](fact_0)
        article_1, _ = self.article_lstms[0](article_0)
        inter_1 = self.interaction(fact_1, article_1).unsqueeze(-1).repeat(1, 1, fact_1.size(-1))
        info_article_1 = inter_1.mul(article_1)
        fusion_output_1 = self.ffs[1](info_article_1)
        fusion_output_max_1 = torch.max(fusion_output_1, dim=-1)[0]

        fusion_output = self.dropout(F.relu(self.ffs[-1](fusion_output_max_1)))

        output = self.predict(fusion_output)

        return output

    @staticmethod
    def interaction(x1, x2):
        """

        :param x1: (batch_size, x1_seq_len, feature_size)
        :param x2: (batch_size, x2_seq_len, feature_size)
        :return: (batch_size, x2_seq_len)
        """
        dot_matrix = torch.matmul(x1, x2.permute(0, 2, 1))  # (batch_size, x1_seq_len, x2_seq_len)

        x1_2_x2 = F.softmax(dot_matrix, dim=2)
        x2_2_x1 = F.softmax(dot_matrix, dim=1)

        x1_weight = torch.sum(x2_2_x1, dim=2).unsqueeze(dim=1)  # (batch_size, 1, x1_seq_len)
        x2_weight = torch.matmul(x1_weight, x1_2_x2).squeeze(dim=1)  # (batch_size, x2_seq_len)

        return x2_weight

    @staticmethod
    def get_name():
        return 'GRU-based model'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0


class TransformerModel(nn.Module):

    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(97506, 128)
        self.fact_mask = None
        self.article_mask = None
        self.fact_pos_encoder = PositionalEncoding(args.d_model, 0.1)
        self.article_pos_encoder = PositionalEncoding(args.d_model, 0.1)
        self.fact_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(args.d_model, args.nhead, args.nhid, 0.1) for _ in range(3)
        ])
        self.article_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(args.d_model, args.nhead, args.nhid, 0.1) for _ in range(3)
        ])
        self.ffs = nn.ModuleList([
            nn.Linear(args.embedding_dim * 4, args.linear_output),
            nn.Linear(args.d_model * 4, args.linear_output),
            nn.Linear(args.d_model * 4, args.linear_output),
            nn.Linear(args.d_model * 4, args.linear_output),
            nn.Linear(args.article_len * 4, args.linear_output)
        ])
        self.dropout = nn.Dropout(p=args.dropout)
        self.predict = nn.Linear(args.linear_output, 2)

        self.d_model = args.d_model
        self.fact_len = args.fact_len
        self.article_len = args.article_len

    def forward(self, fact, article):
        if self.fact_mask is None or self.fact_mask.size(0) != len(fact):
            device = fact.device
            mask = (fact == 0)
            self.fact_mask = mask.to(device)

        if self.article_mask is None or self.article_mask.size(0) != len(article):
            device = article.device
            mask = (article == 0)
            self.article_mask = mask.to(device)

        # zero-layer
        fact_0 = self.fact_pos_encoder(self.embedding(fact) * math.sqrt(self.d_model))
        article_0 = self.article_pos_encoder(self.embedding(article) * math.sqrt(self.d_model))

        inter_0 = self.interaction(fact_0, article_0).unsqueeze(-1).repeat(1, 1, fact_0.size(-1))
        info_article_0 = inter_0.mul(article_0)  # (batch_size, article_len, embedding_dim)

        fusion_output_0 = torch.cat([article_0, info_article_0,
                                     article_0.sub(info_article_0),
                                     article_0.mul(info_article_0)], dim=-1)
        fusion_output_0 = self.ffs[0](fusion_output_0)
        fusion_output_max_0 = torch.max(fusion_output_0, dim=-1)[0]

        # first-layer
        fact_1 = self.fact_encoder_layers[0](fact_0.permute(1, 0, 2),
                                             src_key_padding_mask=self.fact_mask).permute(1, 0, 2)
        article_1 = self.article_encoder_layers[0](article_0.permute(1, 0, 2),
                                                   src_key_padding_mask=self.article_mask).permute(1, 0, 2)

        inter_1 = self.interaction(fact_1, article_1).unsqueeze(-1).repeat(1, 1, fact_1.size(-1))
        info_article_1 = inter_1.mul(article_1)
        fusion_output_1 = torch.cat([article_1, info_article_1,
                                     article_1.sub(info_article_1),
                                     article_1.mul(info_article_1)], dim=-1)
        fusion_output_1 = self.ffs[1](fusion_output_1)
        fusion_output_max_1 = torch.max(fusion_output_1, dim=-1)[0]

        # second-layer
        fact_2 = self.fact_encoder_layers[1](fact_1.permute(1, 0, 2),
                                             src_key_padding_mask=self.fact_mask).permute(1, 0, 2)
        article_2 = self.article_encoder_layers[1](article_1.permute(1, 0, 2),
                                                   src_key_padding_mask=self.article_mask).permute(1, 0, 2)
        inter_2 = self.interaction(fact_2, article_2).unsqueeze(-1).repeat(1, 1, fact_2.size(-1))
        info_article_2 = inter_2.mul(article_2)
        fusion_output_2 = torch.cat([article_2, info_article_2,
                                     article_2.sub(info_article_2),
                                     article_2.mul(info_article_2)], dim=-1)
        fusion_output_2 = self.ffs[2](fusion_output_2)
        fusion_output_max_2 = torch.max(fusion_output_2, dim=-1)[0]

        # third-layer
        fact_3 = self.fact_encoder_layers[2](fact_2.permute(1, 0, 2),
                                             src_key_padding_mask=self.fact_mask).permute(1, 0, 2)
        article_3 = self.article_encoder_layers[2](article_2.permute(1, 0, 2),
                                                   src_key_padding_mask=self.article_mask).permute(1, 0, 2)
        inter_3 = self.interaction(fact_3, article_3).unsqueeze(-1).repeat(1, 1, fact_3.size(-1))
        info_article_3 = inter_3.mul(article_3)
        fusion_output_3 = torch.cat([article_3, info_article_3,
                                     article_3.sub(info_article_3),
                                     article_3.mul(info_article_3)], dim=-1)
        fusion_output_3 = self.ffs[3](fusion_output_3)
        fusion_output_max_3 = torch.max(fusion_output_3, dim=-1)[0]

        fusion_output = torch.cat([fusion_output_max_0, fusion_output_max_1, fusion_output_max_2, fusion_output_max_3],
                                  dim=-1)
        fusion_output = self.dropout(F.relu(self.ffs[-1](fusion_output)))

        output = self.predict(fusion_output)

        return output

    @staticmethod
    def interaction(x1, x2):
        """

        :param x1: (batch_size, x1_seq_len, feature_size)
        :param x2: (batch_size, x2_seq_len, feature_size)
        :return: (batch_size, x2_seq_len)
        """
        dot_matrix = torch.matmul(x1, x2.permute(0, 2, 1))  # (batch_size, x1_seq_len, x2_seq_len)

        x1_2_x2 = F.softmax(dot_matrix, dim=2)
        x2_2_x1 = F.softmax(dot_matrix, dim=1)

        x1_weight = torch.sum(x2_2_x1, dim=2).unsqueeze(dim=1)  # (batch_size, 1, x1_seq_len)
        x2_weight = torch.matmul(x1_weight, x1_2_x2).squeeze(dim=1)  # (batch_size, x2_seq_len)

        return x2_weight

    @staticmethod
    def get_name():
        return 'Transformer-based model'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
