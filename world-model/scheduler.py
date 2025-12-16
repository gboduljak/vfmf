from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupScheduler(LRScheduler):
  def __init__(
          self,
          optimizer: Optimizer,
          warmup_steps: int,
          lr_max: float,
          last_epoch: int = -1):
    self.warmup_steps = warmup_steps
    self.lr_max = lr_max
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    step = self.last_epoch + 1
    if step <= self.warmup_steps:
      scale = step / float(self.warmup_steps)
    else:
      scale = 1.0
    return [self.lr_max * scale for _ in self.base_lrs]
