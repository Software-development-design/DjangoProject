#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):

    # 在`__init__`方法中，初始化了一些模型的组件。
    def __init__(self, num_vocab):
        super(BiLSTMModel, self).__init__()
        # `embedding`是一个`nn.Embedding`层，用于将输入的词索引转换为词向量表示，其中`num_vocab`表示词汇表的大小，`embedding_dim`表示词向量的维度为128。
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=128)
        # `lstm`是一个`nn.LSTM`层，用于对输入的词向量进行双向LSTM处理，其中`input_size`表示输入的特征维度为128，`hidden_size`表示LSTM隐藏状态的维度为256，
        # `bidirectional=True`表示使用双向LSTM，`batch_first=True`表示输入的数据格式为(batch_size, sequence_length, input_size)，
        # `num_layers=2`表示LSTM层的层数为2。
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True, batch_first=True, num_layers=2)
        # `fc1`是一个包含一个线性层和ReLU激活函数的序列，用于进行非线性变换，其中线性层的输入维度为512，输出维度为512。
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )
        # `fc2`是一个线性层，用于将非线性变换的结果映射为输出类别的分数，其中输入维度为512，输出维度为2（假设有2个类别）。
        self.fc2 = nn.Linear(512, 2)

    # 在`forward`方法中，定义了模型的前向计算过程。
    def forward(self, x):
        # 将输入`x`通过嵌入层`embedding`转换为词向量表示，得到`out`。
        out = self.embedding(x)
        # 将词向量输入到LSTM层`lstm`中，得到输出`outputs`和最后一个时刻的隐藏状态`h`和细胞状态`c`。
        outputs, (h, c) = self.lstm(out)
        # 将正向和反向的最后一个时刻的隐藏状态`h`拼接起来，形成一个512维的表示，通过`torch.cat`函数和`dim=-1`实现，得到`out`。
        # 将`out`输入到非线性变换层`fc1`中，经过ReLU激活函数进行非线性变换。
        out = torch.cat([h[-1, :, :], h[-2, :, :]], dim=-1)
        # 将非线性变换后的结果`out`输入到线性层`fc2`中，得到最终的输出结果。
        out = self.fc1(out)

        # 返回最终的输出结果`fc2(out)`。
        return self.fc2(out)
# 综上所述，该代码定义了一个包含嵌入层、双向LSTM层和两个线性层的双向LSTM模型。该模型接收词索引作为输入，经过嵌入层和双向LSTM层的处理，最终输出分类结果的分数。

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
