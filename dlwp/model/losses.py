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
        return self.loss(inputs, targets)


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    loss_1 = F.relu(1.0 - score_real)
    loss = torch.mean(loss_1)
    loss_2 = F.relu(1.0 + score_generated)
    loss += torch.mean(loss_2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -torch.mean(score_generated)
    return loss


def loss_wass_disc(score_generated, score_real):
    """
    Discriminator Wasserstein loss.

    :param score_generated: 1-d sequence of scores on generated samples
    :param score_real: 1-d sequence of scores on real samples
    :return: Tensor: loss
    """
    gen_samples, real_samples = len(score_generated), len(score_real)
    loss_1 = (torch.mean(score_real) * real_samples)
    loss_2 = (torch.mean(score_generated) * -1 * gen_samples)
    return (loss_1 + loss_2) / (gen_samples + real_samples)


def loss_wass_gen(score_generated):
    """
    Generator Wasserstein loss.

    :param score_generated: 1-d sequence of scores on generated samples
    :return: Tensor: loss
    """
    loss = torch.mean(score_generated)
    return loss


def loss_wass_sig_disc(score_generated, score_real):
    """
    Discriminator Wasserstein loss with sigmoid compression.

    :param score_generated: 1-d sequence of scores on generated samples
    :param score_real: 1-d sequence of scores on real samples
    :return: Tensor: loss
    """
    gen_samples, real_samples = len(score_generated), len(score_real)
    total_samples = gen_samples + real_samples
    loss_1 = torch.sigmoid(torch.mean(score_real) * 2 * real_samples / total_samples)
    loss_2 = torch.sigmoid(torch.mean(score_generated) * -2 * gen_samples / total_samples)
    return loss_1 + loss_2


def loss_wass_sig_gen(score_generated):
    """
    Generator Wasserstein loss with sigmoid compression.

    :param score_generated: 1-d sequence of scores on generated samples
    :return: Tensor: loss
    """
    loss = torch.sigmoid(torch.mean(score_generated))
    return loss
