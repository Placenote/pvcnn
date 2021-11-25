import torch

__all__ = ['MeterLeresPCM']


class MeterLeresPCM:
    def __init__(self):
        super().__init__()

    def reset(self):
        self.batch_avg_loss = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        # outputs: B x scalar, targets: B x scalar
        self.batch_avg_loss = 0
        for b in range(outputs.size(0)):
            prediction = outputs[b]
            target = targets[b]
            self.batch_avg_loss = self.batch_avg_loss + torch.square(prediction - target)
        self.batch_avg_loss = self.batch_avg_loss / outputs.size(0)


    def compute(self):
        return self.batch_avg_loss
