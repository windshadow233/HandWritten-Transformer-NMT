from config import *


class BeamHypotheses(object):
    """
    每个样本绑定一个该容器,该容器将维护不大于num_beams个最优序列,当往该容器中添加
    一个新的序列且序列数大于num_beams时,将删除分数最低的序列
    """
    def __init__(self, num_beams, max_length):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []  # 存储最优的几个序列
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, sentence, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / sentence.shape[0]
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, sentence))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """

        if len(self) < self.num_beams:
            return False
        if cur_len is None:
            cur_len = self.max_length
        cur_score = best_sum_logprobs / cur_len
        # 是否最高分比当前保存的最低分还差
        ret = self.worst_score >= cur_score
        return ret


def beam_search(model, src_sens: torch.Tensor, num_beams=3):
    """
    :param model: Transformer
    :param src_sens: (B, L)
    :param num_beams: 单步搜索空间
    """
    batch_size, length = src_sens.shape
    max_length = min(max_seq_len, 2 * length + 2)
    encoder_output = model.encoder(src_sens)[0]
    encoder_output = encoder_output.repeat_interleave(num_beams, 0)
    src_sens = src_sens.repeat_interleave(num_beams, 0)
    # 为每个句子构造一个存储beam_size个可能序列及它们的得分的容器
    generated_sens = [
        BeamHypotheses(num_beams, max_length)
        for _ in range(batch_size)
    ]
    # 为每个句子初始化beam_size个得分
    beam_scores = src_sens.new_zeros((batch_size, num_beams), dtype=torch.float)
    beam_scores = beam_scores.flatten()  # (batch_size * num_beams,)
    # 每个样本是否完成生成，共batch_size个
    done = [False] * batch_size
    # 一次生成batch_size * num_beams个序列
    sos = src_sens.new_full(
        (batch_size * num_beams, 1),
        fill_value=sos_idx
    )
    eos = src_sens.new_full(
        (batch_size * num_beams, 1),
        fill_value=eos_idx
    )
    input_ids = torch.cat([sos, eos], dim=1)
    # 当前长度为1
    cur_len = 1
    while cur_len < max_length:
        output, _ = model.decoder(input_ids, src_sens, encoder_output, True)
        # output: (batch_size * num_beams, current_len, vocab_size)
        # 计算logprobs
        scores = next_token_probs = output[:, -1].log()  # (batch_size * num_beams, vocab_size)
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(batch_size, -1)  # (batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

        next_batch_beam = []
        for batch_idx in range(batch_size):
            # 若该样本已经结束
            if done[batch_idx]:
                next_batch_beam.extend([(0, pad_idx, 0)] * num_beams)
                continue
            next_sen_beam = []
            # 对于还未结束的样本需要找到分数最高的num_beams个扩展
            # 注意，next_scores和next_tokens是对应的
            # 而且已经按照next_scores排好顺序
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effect_beam_id = batch_idx * num_beams + beam_id
                if token_id.item() == eos_idx:
                    # 该预测为eos
                    generated_sens[batch_idx].add(
                        input_ids[effect_beam_id].clone(),
                        beam_token_score.item()
                    )
                    next_batch_beam.append((beam_token_score, eos_idx, effect_beam_id))
                else:
                    next_sen_beam.append((beam_token_score, token_id, effect_beam_id))
                if len(next_sen_beam) == num_beams:
                    break
            done[batch_idx] = done[batch_idx] or generated_sens[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )
            next_batch_beam.extend(next_sen_beam)
        if all(done):
            break
        beam_scores = beam_scores.new_tensor([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new_tensor([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new_tensor([x[2] for x in next_batch_beam])

        input_ids = input_ids[beam_idx]
        input_ids = torch.cat([input_ids[:, :-1], beam_tokens.unsqueeze(-1), eos], dim=1)
        cur_len = cur_len + 1
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            effect_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effect_beam_id].item()
            final_tokens = input_ids[effect_beam_id]
            generated_sens[batch_idx].add(final_tokens, final_score)
    # select the best hypotheses，最终输出
    # 每个样本返回几个句子
    output_num_return_sequences_per_batch = 1
    # 记录每个返回句子的长度，用于后面pad
    sent_lengths = input_ids.new_zeros((batch_size,))
    best = []
    # 对每个样本取出最好的output_num_return_sequences_per_batch个句子
    for i, hypotheses in enumerate(generated_sens):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
    # 如果长短不一则pad句子，使得最后返回结果的长度一样
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        # 先把输出矩阵填满PAD token
        decoded = input_ids.new_full(size=(batch_size, sent_max_len + 1), fill_value=pad_idx)
        # 填入真正的内容
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
    else:
        decoded = torch.stack(best).type(torch.long).to(device)
    return decoded
