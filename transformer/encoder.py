from .model_base import *


# Encoder的其中一层(包含注意力层与feedforward)
class EncodeLayer(nn.Module):
    def __init__(self, config):
        super(EncodeLayer, self).__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = IntermediateOutput(config)

    def forward(self, hidden_layer, pad_mask, attention_mask, get_attention_mat=False):
        """

        Args:
            hidden_layer: (B, L, D)
            pad_mask: (B, L)
            attention_mask: (B, L, L)
            get_attention_mat: True or False

        Returns: output: (B, L, D), attention_mat: (B, H, L, L) or None

        """
        attention_output, attention_mat = self.attention(hidden_layer, hidden_layer, hidden_layer,
                                                         attention_mask, get_attention_mat)
        attention_output *= pad_mask[..., None]
        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output)
        output *= pad_mask[..., None]
        return output, attention_mat


# 编码器
class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        self.pad_idx = config.pad_idx
        self.word_embedding = Embedding(config, config.src_vocab_size)
        self.layers = nn.ModuleList([EncodeLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, output_all_encoded_layers=False, get_attention_mat=False):
        """

        Args:
            x: (B, L)
            output_all_encoded_layers: True or False
            get_attention_mat: True or False

        Returns: encoded_layers: [(B, L, D) * n] or (B, L, D) attention_mats: [(B, H, D, D) * n] or None

        """
        pad_mask = get_pad_mask(x, pad_idx=self.pad_idx)
        attention_mask = get_attention_key_mask(x, x, pad_idx=self.pad_idx)
        attention_mats = [] if get_attention_mat else None
        encoded_layers = []
        hidden_layer = self.word_embedding(x)
        for layer in self.layers:
            hidden_layer, attention_mat = layer(hidden_layer, pad_mask, attention_mask, get_attention_mat)
            if get_attention_mat:
                attention_mats.append(attention_mat)
            encoded_layers.append(hidden_layer)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, attention_mats
