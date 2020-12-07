from config import embedding_dim


class WarmUpLr(object):
    def __init__(self, optimizer, init_lr, warmup_step, step_num=0):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_step = warmup_step
        self.step_num = step_num

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d)

    def state_dict(self):
        return self.optimizer.state_dict()

    def update_lr(self):
        new_lr = self.init_lr * embedding_dim ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_step ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def step(self):
        self.step_num += 1
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
