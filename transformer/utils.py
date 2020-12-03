import torch


def get_pad_mask(padded_seq, pad_idx):
    """
    输入序列矩阵
    数值为pad_idx的置为0,其余置为1
    """
    pad_mask = torch.ones_like(padded_seq, dtype=torch.float32)
    pad_mask[padded_seq == pad_idx] = 0
    return pad_mask


def get_attention_mask(seq_k, seq_q, pad_idx):
    seq_len = seq_q.shape[1]
    pad_mask = seq_k.eq(pad_idx)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1).float()
    return pad_mask


def get_subsequent_mask(sequences):
    batch_size, seq_len = sequences.shape
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=sequences.device, dtype=torch.float32), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return subsequent_mask
