import torch


class EMA:
    def __init__(self, mu: float):
        self.mu = mu
        self.average = torch.tensor(1.0).float().cuda()

    def apply(self, x: torch.Tensor):
        _x = x.abs().mean(0)
        self.average = self.mu * _x + (1-self.mu) * self.average
