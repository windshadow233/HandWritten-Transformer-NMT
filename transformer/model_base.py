from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import re
from .utils import *


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


activation_fcns = {'relu': F.relu, 'gelu': gelu, 'leaky_relu': lambda x: F.leaky_relu(x, negative_slope=0.2)}


# 模型的一些参数设置
class Config(object):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if value.isdecimal():
                self.__setattr__(key, int(value))
            elif re.fullmatch('[0-9]+.[0-9]+', value):
                self.__setattr__(key, float(value))
            else:
                self.__setattr__(key, value)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('positional_encoding', self.get_positional_enc(config.max_seq_len, config.embedding_dim))

    @staticmethod
    def get_positional_enc(max_seq_len, embedding_dim):
        position_code = np.array([
            [pos / np.power(10000, i / embedding_dim) for i in range(embedding_dim)]
            if pos != 0 else np.zeros(embedding_dim) for pos in range(max_seq_len)
        ])
        position_code[1:, 0::2] = np.sin(position_code[1:, 0::2])
        position_code[1:, 1::2] = np.cos(position_code[1:, 1::2])
        # 每一行做标准化
        s = position_code.sum(axis=-1, keepdims=True)
        position_code = position_code / np.sqrt(s + 1e-5)
        position_code = torch.from_numpy(position_code).float()
        return position_code

    def forward(self, x):
        """
        :param x: (B, L, D) or (B, L)
        """
        length = x.shape[1]
        return self.positional_encoding[:length]


# 词嵌入(包含位置嵌入)
class Embedding(nn.Module):
    def __init__(self, config: Config, vocab_size):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)  # 指定0为padding符号
        self.pos_enc = PositionalEncoding(config)

    def forward(self, input_ids):
        """

        Args:
            input_ids: (B, L)

        Returns: embeddings: (B, L, D)

        """
        word_embeddings = self.word_embeddings(input_ids)  # (B, L, D)
        position_encode = self.pos_enc(word_embeddings)
        embeddings = word_embeddings + position_encode  # position_encode将广播成(B, L, D)维度
        return embeddings


# 多头注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadSelfAttention, self).__init__()
        assert config.embedding_dim % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.embedding_dim // config.num_attention_heads
        self.embedding_dim = config.embedding_dim
        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.K = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, q, k, v, attention_mask, get_attention_mat=False):
        """

        Args:
            q: (B, L1, D)
            k: (B, L2, D)
            v: (B, L2, D)
            attention_mask: (B, L1, L2)
            get_attention_mat: 是否获取注意力矩阵

        Returns: hidden_layer: (B, L1, D), attention_probs: (B, H, L1, L2)

        """
        attention_mask = attention_mask * -10000
        query = self.Q(q)  # (B, L1, D)
        key = self.K(k)  # (B, L2, D)
        value = self.V(v)  # (B, L2, D)
        multi_heads_q = self.get_multi_heads(query)  # (B, H, L1, D // H)
        multi_heads_k = self.get_multi_heads(key)  # (B, H, L2, D // H)
        multi_heads_v = self.get_multi_heads(value)  # (B, H, L2, D // H)
        attention_mat = multi_heads_q.matmul(multi_heads_k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)  # (B, H, L1, L2)
        attention_mat = attention_mat + attention_mask[:, None]  # (B, H, L1, L2) + (B, 1, L1, L2) => (B, H, L1, L2)
        attention_probs = nn.functional.softmax(attention_mat, dim=-1)  # (B, H, L1, L2)
        attention_probs = self.dropout(attention_probs)
        hidden_layer = attention_probs.matmul(multi_heads_v).transpose(1, 2)  # (B, L1, H, D // h)
        new_output_layer_shape = hidden_layer.shape[:-2] + (self.embedding_dim,)
        hidden_layer = hidden_layer.reshape(new_output_layer_shape)  # (B, L1, D)
        if get_attention_mat:
            return hidden_layer, attention_probs
        return hidden_layer, None

    def get_multi_heads(self, x):
        """

        Args:
            x: (B, L, D)

        Returns: x: (B, H, L, D // H)

        """
        new_shape_x = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape_x).transpose(1, 2)
        return x


# 层标准化
class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, embedding_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        """

        Args:
            x: (B, L, D)

        Returns: (B, L, D)

        """
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


# 注意力层的输出
class SelfOutput(nn.Module):
    """
    Add + Norm for self-attention-layer
    """
    def __init__(self, config: Config):
        super(SelfOutput, self).__init__()
        self.linear = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.LayerNorm = LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, output_layer, input_tensor):
        """

        Args:
            output_layer: (B, L, D)
            input_tensor: (B, L, D)

        Returns: output_layer: (B, L, D)

        """
        output_layer = self.linear(output_layer)
        output_layer = self.dropout(output_layer)
        output_layer = self.LayerNorm(output_layer + input_tensor)
        return output_layer


# 注意力层(包含自注意力层+输出层)
class Attention(nn.Module):
    def __init__(self, config: Config):
        super(Attention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(config)
        self.self_output = SelfOutput(config)

    def forward(self, q, k, v, attention_mask, get_attention_mat=False):
        """

        Args:
            q: (B, L, D)
            k: (B, L, D)
            v: (B, L, D)
            attention_mask: (B, L, L)
            get_attention_mat: True or False

        Returns: attention_output: (B, L, D), attention_mat: (B, H, L, L) or None

        """
        hidden_layer, attention_mat = self.self_attention(q, k, v, attention_mask, get_attention_mat)
        attention_output = self.self_output(hidden_layer, q)
        return attention_output, attention_mat


# FeedForward层
class Intermediate(nn.Module):
    def __init__(self, config: Config):
        super(Intermediate, self).__init__()
        self.feed_forward = nn.Linear(config.embedding_dim, config.intermediate_size)
        self.activation_fcn = activation_fcns.get(config.hidden_act)

    def forward(self, hidden_layer):
        """

        Args:
            hidden_layer: (B, L, D)

        Returns: (B, L, I)

        """
        hidden_layer = self.feed_forward(hidden_layer)
        hidden_layer = self.activation_fcn(hidden_layer)
        return hidden_layer


# FeedForward层的输出(包含线性变换与LayerNorm)
class IntermediateOutput(nn.Module):
    """
    Add + Norm for Intermediate
    """
    def __init__(self, config: Config):
        super(IntermediateOutput, self).__init__()
        self.linear = nn.Linear(config.intermediate_size, config.embedding_dim)
        self.layernorm = LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_layer, input_tensor):
        """

        Args:
            hidden_layer: (B, L, I)
            input_tensor: (B, L, D)

        Returns: hidden_layer: (B, L, D)

        """
        hidden_layer = self.linear(hidden_layer)  # (B, L, D)
        hidden_layer = self.dropout(hidden_layer)
        hidden_layer = self.layernorm(hidden_layer + input_tensor)
        return hidden_layer
