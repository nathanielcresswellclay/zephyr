import torch
from torch.nn import functional as F

class LossOnStep(torch.nn.Module):
    """
    Allows computation of an existing loss function on only one step of a sequence of outputs.
    """
    def __init__(self, loss: torch.nn.Module, time_dim: int, step: int):
        super().__init__()
        self.loss = loss
        self.time_slice = slice(step * time_dim, (step + 1) * time_dim)

    def forward(self, inputs, targets):
        return self.loss(inputs[:, self.time_slice], targets[:, self.time_slice])


class GeneratorLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, disc_score_weight: float):
        super().__init__()
        self.loss = loss
        self.disc_score_weight = torch.tensor(disc_score_weight, dtype=torch.float32)

    def forward(self, inputs, targets, disc_score):
        if disc_score is not None:
            return self.loss(inputs, targets) + self.disc_score_weight * disc_score
        else:
            return self.loss(inputs, targets)


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = F.relu(1.0 - score_real)
    loss = torch.mean(l1)
    l2 = F.relu(1.0 + score_generated)
    loss += torch.mean(l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -torch.mean(score_generated)
    return loss
