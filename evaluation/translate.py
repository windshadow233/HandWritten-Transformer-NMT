import re
from evaluation.beam_search import *
from data.utils import full_width2half_width


@torch.no_grad()
def translate_batch(model, converter, src, tgt=None):
    """
    :param model: Transformer
    :param converter: TokenSentenceConverter
    :param src: 一个英文句子列表
    :param tgt: target
    :return: predicts: 中文句子列表
    """
    src = [full_width2half_width(s) for s in src]
    src = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in src]
    token = torch.nn.utils.rnn.pad_sequence([converter.sentence2token(sen, 'en').to(device)
                                             for sen in src],
                                            batch_first=True)
    decode = beam_search(model, token, num_beams=5)
    predicts = [converter.token2sentence(sen, 'cn').replace('<sos>', '').replace('<eos>', '') for sen in decode]
    if tgt is not None:
        for s, p, t in zip(src, predicts, tgt):
            print(f'Src  | {s}\nTgt  | {t}\nPred | {p}\n')
    else:
        for s, p in zip(src, predicts):
            print(f'Src  | {s}\nPred | {p}\n')
    predicts = dict(zip(src, predicts))
    return predicts
