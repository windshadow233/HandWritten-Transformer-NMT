from model_base import *


# Decoder的其中一层
class DecodeLayer(nn.Module):
    def __init__(self, config: Config):
        super(DecodeLayer, self).__init__()
        self.masked_attention = Attention(config)
        self.enc_dec_attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = IntermediateOutput(config)

    def forward(self, dec_input, enc_output, pad_mask, self_attn_mask, dec_enc_attn_mask):
        dec_output, dec_attention_mat = self.masked_attention(
            dec_input, dec_input, dec_input, self_attn_mask)
        dec_output *= pad_mask[..., None]
        dec_output, dec_enc_attn = self.enc_dec_attention(
            dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output *= pad_mask[..., None]
        intermediate_output = self.intermediate(dec_output)
        output = self.output(intermediate_output, dec_output)
        output *= pad_mask[..., None]
        return output


# 解码器
class Decoder(nn.Module):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()
        self.pad_idx = config.pad_idx
        self.sos_idx = config.sos_idx
        self.eos_idx = config.eos_idx
        self.unk_idx = config.unk_idx
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_dim = config.embedding_dim
        self.max_len = config.max_seq_len

        self.word_embedding = Embedding(config)
        self.layers = nn.ModuleList([DecodeLayer(config) for _ in range(self.num_hidden_layers)])
        self.tgt_word_prj = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)

    def pre_process(self, padded_target: torch.Tensor):
        # batch_size, seq_len = padded_target.shape
        # sos = padded_target.new_ones(size=(batch_size, 1)) * self.sos_idx
        # eos = padded_target.new_ones(size=(batch_size, 1)) * self.eos_idx
        # return torch.cat([sos, padded_target], dim=1), torch.cat([padded_target, eos], dim=1)
        ys_in_pad = padded_target.clone()
        ys_in_pad[ys_in_pad == self.eos_idx] = 0
        ys_in_pad = ys_in_pad[:, :-1]
        ys_out_pad = padded_target[:, 1:]
        return ys_in_pad, ys_out_pad

    def forward(self, padded_target, padded_src, encoder_output):
        ys_in_pad, ys_out_pad = self.pre_process(padded_target)
        pad_mask = get_pad_mask(ys_in_pad, self.pad_idx)
        self_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        self_attn_mask = get_attention_mask(ys_in_pad, ys_in_pad, pad_idx=self.pad_idx)
        self_attn_mask = ((self_attn_mask + self_attn_mask_subseq) > 0).float()
        dec_enc_attn_mask = get_attention_mask(padded_src, ys_in_pad, pad_idx=self.pad_idx)

        decode_output = self.word_embedding(ys_in_pad)
        for layer in self.layers:
            decode_output = layer(decode_output, encoder_output, pad_mask, self_attn_mask, dec_enc_attn_mask)
        decode_output = self.tgt_word_prj(decode_output)
        return decode_output, ys_out_pad

