import torch
import torch.autograd as ag
import torch.nn.functional as func

from plasma.training.GAN.losses.root_class import Loss


class NonSaturating(Loss):
    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        real_labels = torch.ones_like(real_scores)
        loss1 = func.binary_cross_entropy_with_logits(real_scores, real_labels).mean()

        fake_labels = torch.zeros_like(fake_scores)
        loss2 = func.binary_cross_entropy_with_logits(fake_scores, fake_labels).mean()

        return loss1 + loss2

    def generator_loss(self, fakes, fake_scores):
        real_labels = torch.ones_like(fake_scores)
        loss = func.binary_cross_entropy_with_logits(fake_scores, real_labels).mean()

        return loss


class R1(Loss):

    def __init__(self):
        super().__init__()

        self.requires_real_grads = True

    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        grads = ag.grad(real_scores.sum(), reals, retain_graph=True, create_graph=True)

        loss = 0
        for g in grads:
            loss = loss + g.pow(2).sum(dim=[1, 2, 3])
        loss = loss.mean()

        return loss

    def generator_loss(self, fakes, fake_scores):
        return 0


class LSGAN(Loss):

    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        losses = (real_scores - 1).pow(2) + fake_scores.pow(2)

        return losses.mean()

    def generator_loss(self, fakes, fake_scores):
        return (fake_scores - 1).pow(2).mean()


class WGAN(Loss):

    def __init__(self, epsilon=1e-3):
        super().__init__()

        self.epsilon = epsilon

    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        losses = fake_scores - real_scores

        if self.epsilon > 0:
            losses = losses + self.epsilon * real_scores.pow(2)

        return losses.mean()

    def generator_loss(self, fakes, fake_scores):
        return (-fake_scores).mean()
