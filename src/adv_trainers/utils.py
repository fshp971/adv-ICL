import torch


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


__optimizer_zoo__ = {
    "sgd"   : torch.optim.SGD,
    "adam"  : torch.optim.Adam,
    "adamw" : torch.optim.AdamW,
}

def build_optimizer(name, parameters, **kwargs):
    return __optimizer_zoo__[name](parameters, **kwargs)


__scheduler_zoo__ = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
}

def build_scheduler(name, optimizer, **kwargs):
    return __scheduler_zoo__[name](optimizer, **kwargs)
