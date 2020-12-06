from config import embedding_dim


class WarmUpLr(object):
    def __init__(self, optimizer, warmup_step):
        self.optimizer = optimizer
        self.warmup_step = warmup_step
        self.step_num = 0

    def update_lr(self):
        new_lr = embedding_dim ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_step ** -1.5)
        self.optimizer.defaults['lr'] = new_lr

    def step(self):
        self.step_num += 1
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()