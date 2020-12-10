import re
from evaluation.beam_search import *
from data.utils import full_width2half_width
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
fcn = SmoothingFunction()


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
    tgt = [full_width2half_width(t) for t in tgt]
    src = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in src]
    token = torch.nn.utils.rnn.pad_sequence([converter.sentence2token(sen, 'en').to(device)
                                             for sen in src],
                                            batch_first=True)
    decode = beam_search(model, token, num_beams=5)
    predicts = [converter.token2sentence(sen, 'cn').replace('<sos>', '').replace('<eos>', '') for sen in decode]
    if tgt is not None:
        bleus = [sentence_bleu([list(t)], list(p), weights=(0.25,) * 4, smoothing_function=fcn.method1)
                 for t, p in zip(tgt, predicts)]
        for s, p, t, b in zip(src, predicts, tgt, bleus):
            print(f'Src  | {s}\nTgt  | {t}\nPred | {p}\nBLEU | {b}\n')
        predicts = zip(src, tgt, predicts, bleus)
    else:
        for s, p in zip(src, predicts):
            print(f'Src  | {s}\nPred | {p}\n')
        predicts = zip(src, predicts)
    return predicts
