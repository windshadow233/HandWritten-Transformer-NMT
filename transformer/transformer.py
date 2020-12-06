from torch import nn
from model_base import Config
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, padded_src, padded_target):
        encoder_output, _ = self.encoder(padded_src)
        pred_logits, target = self.decoder(padded_target, padded_src, encoder_output)
        return pred_logits, target
