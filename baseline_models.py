import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_pkl


class ArcI(nn.Module):

    def __init__(self, args):
        super(ArcI, self).__init__()

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)

        # two for convolution, two for pooling, and two for MLP
        self.conv_left = nn.ModuleList([
            self.make_conv1d_pool_block(128, 128, 3, nn.ReLU(), 2),
            self.make_conv1d_pool_block(128, 128, 5, nn.ReLU(), 2),
            self.make_conv1d_pool_block(128, 128, 7, nn.ReLU(), 2)
        ])
        self.conv_right = nn.ModuleList([
            self.make_conv1d_pool_block(128, 128, 3, nn.ReLU(), 2),
            self.make_conv1d_pool_block(128, 128, 5, nn.ReLU(), 2),
            self.make_conv1d_pool_block(128, 128, 7, nn.ReLU(), 2)
        ])

        self.dropout = nn.Dropout(p=0.5)
        self.mlp1 = nn.Sequential(nn.Linear(128 * 12, 128), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        self.predict = nn.Linear(64, 2)

    def forward(self, left, right):
        left, right = self.embedding(left), self.embedding(right)
        left = left.transpose(1, 2)  # (batch_size, embedding_size, left_len)
        right = right.transpose(1, 2)  # (batch_size, embedding_size, right_len)

        conv_left = self.conv_left[2](self.conv_left[1](self.conv_left[0](left)))
        conv_right = self.conv_right[2](self.conv_right[1](self.conv_right[0](right)))

        rep_left = torch.flatten(conv_left, start_dim=1)
        rep_right = torch.flatten(conv_right, start_dim=1)
        concat = self.dropout(torch.cat((rep_left, rep_right), dim=1))

        mlp_output = self.mlp1(concat)
        mlp_output = self.mlp2(mlp_output)
        output = self.predict(mlp_output)
        return output

    @classmethod
    def make_conv1d_pool_block(cls, in_channels, out_channels, kernel_size, activation, pool_size):
        return nn.Sequential(
            nn.ConstantPad1d((0, kernel_size - 1), 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        )

    @staticmethod
    def get_name():
        return 'Baseline: Arc-I'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0


class ArcII(nn.Module):

    def __init__(self, args):
        super(ArcII, self).__init__()

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)

        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)
        # three for convolution, three for pooling, and two for MLP
        self.conv1d_left = nn.Sequential(
            nn.ConstantPad1d((0, 3 - 1), 0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        )
        self.conv1d_right = nn.Sequential(
            nn.ConstantPad1d((0, 3 - 1), 0),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        )
        self.layer2_activation = nn.ReLU()
        self.layer2_pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2ds = nn.ModuleList([
            self.make_conv2d_pool_block(128, 128, (3, 3), nn.ReLU(), (2, 2)),
            self.make_conv2d_pool_block(128, 128, (5, 5), nn.ReLU(), (2, 2))
        ])
        self.dropout = nn.Dropout(p=0.2)
        self.mlp = nn.Sequential(nn.Linear(128 * 6 * 6, 128), nn.ReLU())
        self.predict = nn.Linear(128, 2)

    def forward(self, left, right):
        left, right = self.embedding(left), self.embedding(right)

        conv_left = self.conv1d_left(left.transpose(1, 2)).transpose(1, 2)
        conv_right = self.conv1d_right(right.transpose(1, 2)).transpose(1, 2)

        # (batch_size, left_len, right_len, output_channels)
        match_signals = conv_left.unsqueeze(dim=2).repeat(1, 1, right.size(1), 1) \
                        + conv_right.unsqueeze(dim=1).repeat(1, left.size(1), 1, 1)

        # (batch_size, output_channels, left_len//2, right_len//2)
        pooled_match_signals = self.layer2_pooling(self.layer2_activation(match_signals.permute(0, 3, 1, 2)))
        conv = self.conv2ds[1](self.conv2ds[0](pooled_match_signals))

        # conv = self.conv2ds[1](self.conv2ds[0](match_signals.permute(0, 3, 1, 2)))

        flat = self.dropout(torch.flatten(conv, start_dim=1))
        mlp_output = self.mlp(flat)
        output = self.predict(mlp_output)
        return output

    @classmethod
    def make_conv2d_pool_block(cls, in_channels, out_channels, kernel_size, activation, pool_size):
        return nn.Sequential(
            # Same padding
            nn.ConstantPad2d((0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool2d(kernel_size=pool_size)
        )

    @staticmethod
    def get_name():
        return 'Baseline: Arc-II'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0


class MatchPyramid(nn.Module):

    def __init__(self, args):
        super(MatchPyramid, self).__init__()

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)

        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)

        # two convolutional layers
        # two max-pooling layers (one of which is a dynamic pooling layer for variable length)
        # two full connection layers
        self.conv_1 = nn.Sequential(
            nn.ConstantPad2d((0, 5 - 1, 0, 5 - 1), 0),
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(5, 5)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_2 = nn.Sequential(
            nn.ConstantPad2d((0, 3 - 1, 0, 3 - 1), 0),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3)
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 10))
        )
        self.dropout = nn.Dropout(p=0.2)
        self.mlp = nn.Linear(128 * 5 * 10, 128)
        self.predict = nn.Linear(128, 2)

    def forward(self, left, right):
        left, right = self.embedding(left), self.embedding(right)

        match_scores = torch.einsum('bld,brd->blr', left, right).unsqueeze(dim=1)
        conv = self.conv_2(self.conv_1(match_scores))
        flat = self.dropout(torch.flatten(conv, start_dim=1))
        mlp_output = self.mlp(flat)
        output = self.predict(mlp_output)
        return output

    @staticmethod
    def get_name():
        return 'Baseline: Match-Pyramid'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0


class MVLSTM(nn.Module):

    def __init__(self, args):
        super(MVLSTM, self).__init__()

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

        self.embedding = nn.Embedding.from_pretrained(load_pkl(args.embedding_matrix_path), freeze=True)
        self.left_lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.right_lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.mlp = nn.Linear(64, 32)
        self.predict = nn.Linear(32, 2)

    def forward(self, left, right):
        left, right = self.embedding(left), self.embedding(right)
        rep_left, _ = self.left_lstm(left)
        rep_right, _ = self.right_lstm(right)

        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(rep_left, p=2, dim=-1),
            F.normalize(rep_right, p=2, dim=-1)
        )

        matching_signals = torch.flatten(matching_matrix, start_dim=1)

        matching_topk = torch.topk(
            matching_signals,
            k=64,
            dim=-1,
            sorted=True
        )[0]

        mlp_output = self.dropout(self.mlp(matching_topk))
        output = self.predict(mlp_output)
        return output

    def get_name(self):
        return 'Baseline: MV-LSTM'

    @staticmethod
    def get_lr():
        return 1e-4

    @staticmethod
    def get_weight_decay():
        return 0.01
