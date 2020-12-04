import torch
from torch import nn
from model_base import Config
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.pad_idx = config.pad_idx
        self.sos_idx = config.sos_idx
        self.eos_idx = config.eos_idx
        self.max_seq_len = config.max_seq_len

    def forward(self, padded_src, padded_target):
        encoder_output, _ = self.encoder(padded_src)
        pred_logits, target = self.decoder(padded_target, padded_src, encoder_output)
        return pred_logits, target

    def predict_next(self, current_pred_sens, src_seq, encoder_output, beam_size=5):
        """
        :param current_pred_sens: 当前预测到的几个概率最大的序列(一个列表,每个元素为(current_len,))
        :param src_seq: 源句子
        :param encoder_output: encoder层编码结果
        :param beam_size: 搜索空间
        :return: 概率最大的beam_size个下一个预测序列
        """
        next_possible_sentences = []
        for sen, score in current_pred_sens:
            # sen:(1, current_len)
            # score:(1,)
            if sen.squeeze(0)[-1].item() == self.eos_idx:
                next_possible_sentences.append((sen, score.item()))
                continue
            sen = torch.cat([sen, sen.new_full((1, 1), self.eos_idx)], 1)
            pred_prob = self.decoder(sen, src_seq, encoder_output, softmax_at_end=True)[0]  # (1, current_len - 1, vocab_size)
            max_prob, max_prob_word = pred_prob.sort(-1, descending=True)  # (1, current_len - 1, vocab_size)
            max_prob, max_prob_word = max_prob[0, -1, :beam_size], max_prob_word[0, -1, :beam_size]  # (beam_size,)
            next_possible_sentences.extend([(torch.cat([sen[:, :-1], word[None, None]], 1), (score + prob).item()) for word, prob in zip(max_prob_word, max_prob)])
        next_possible_sentences.sort(key=lambda x: x[1], reverse=True)
        return next_possible_sentences[:beam_size]

    def beam_search(self, input_seq, max_len=100, beam_size=5):
        """
        对一个句子进行解码
        :param input_seq: (L,)
        :param max_len: 最大句长
        :param beam_size: 单步搜索空间
        :return:
        """
        input_seq = input_seq[None]  # (1, L)
        encoder_output = self.encoder(input_seq)[0]  # (1, L, D)
        sos = input_seq.new_full(size=(1, 1), fill_value=self.sos_idx)
        pred = [(torch.cat([sos], dim=1), input_seq.new_zeros(size=(1,), dtype=torch.float32))]
        for _ in range(max_len):
            pred = self.predict_next(pred, input_seq, encoder_output, beam_size)
        return pred[0]


