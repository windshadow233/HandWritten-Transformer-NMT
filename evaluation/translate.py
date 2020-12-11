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
    :return: predicts: 一个zip
    """
    src = [full_width2half_width(s) for s in src]
    if tgt is not None:
        tgt = [full_width2half_width(t) for t in tgt]
    src = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in src]
    token = torch.nn.utils.rnn.pad_sequence([converter.sentence2token(sen, 'en').to(device)
                                             for sen in src],
                                            batch_first=True)
    decode = beam_search(model, token, num_beams=5)
    predicts = [re.sub('(<sos>)|(<eos>)', '', converter.token2sentence(sen, 'cn')) for sen in decode]
    translate_str = ''
    if tgt is not None:
        bleus = [sentence_bleu([list(t)], list(p), weights=(0.25,) * 4, smoothing_function=fcn.method1)
                 for t, p in zip(tgt, predicts)]
        for s, t, p, b in zip(src, tgt, predicts, bleus):
            translate_str += f'Src  | {s}\nTgt  | {t}\nPred | {p}\nBLEU | {b}\n\n'
        predicts = zip(src, tgt, predicts, bleus)
    else:
        for s, p in zip(src, predicts):
            translate_str += f'Src  | {s}\nPred | {p}\n\n'
        predicts = zip(src, predicts)
    return predicts, translate_str
