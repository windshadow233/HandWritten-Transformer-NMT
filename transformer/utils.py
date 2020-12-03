import torch
from configparser import ConfigParser


parser = ConfigParser()
parser.read('./config.ini', encoding='utf-8')
model_config = parser['model_config']
pad_idx = model_config.getint('pad_idx')


def get_pad_mask(sequences, pad_idx=pad_idx):
    """
    :param sequences: (B, L)
    :param pad_idx: 0
    """
    pad_mask = torch.ones_like(sequences, dtype=torch.float32)
    pad_mask[sequences == pad_idx] = 0
    return pad_mask


def get_attention_mask(seq_k, seq_q, pad_idx=pad_idx):
    seq_len = seq_q.shape[1]
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1).float()
    return padding_mask


def get_subsequent_mask(sequences):
    batch_size, seq_len = sequences.shape
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=sequences.device, dtype=torch.float32), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return subsequent_mask
