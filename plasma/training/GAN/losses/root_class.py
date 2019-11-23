from abc import abstractmethod


class Loss:

    def __init__(self):
        self.requires_real_grads = False

        self.trainer = None
        self.discriminator = None
        self.generator = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.discriminator = trainer.discriminator
        self.generator = trainer.generator

    @abstractmethod
    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        pass

    @abstractmethod
    def generator_loss(self, fakes, fake_scores):
        pass

    def __add__(self, other):
        return AddedLoss(self, other)

    def __mul__(self, other):
        return MulLoss(other, self)

    def __rmul__(self, other):
        return MulLoss(other, self)


class AddedLoss(Loss):
    def __init__(self, loss1, loss2):
        super().__init__()

        self.loss1 = loss1
        self.loss2 = loss2

        self.requires_real_grads = bool(sum([loss1.requires_real_grads, loss2.requires_real_grads]))

    def set_trainer(self, trainer):
        self.loss1.set_trainer(trainer)
        self.loss2.set_trainer(trainer)

    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        loss1 = self.loss1.discriminator_loss(reals, real_scores, fakes, fake_scores)
        loss2 = self.loss2.discriminator_loss(reals, real_scores, fakes, fake_scores)

        return loss1 + loss2

    def generator_loss(self, fakes, fake_scores):
        loss1 = self.loss1.generator_loss(fakes, fake_scores)
        loss2 = self.loss2.generator_loss(fakes, fake_scores)

        return loss1 + loss2


class MulLoss(Loss):

    def __init__(self, alpha, loss):
        super().__init__()

        self.alpha = alpha
        self.loss = loss

        self.requires_real_grads = loss.requires_real_grads

    def set_trainer(self, trainer):
        self.loss.set_trainer(trainer)

    def discriminator_loss(self, reals, real_scores, fakes, fake_scores):
        return self.alpha * self.loss.discriminator_loss(reals, real_scores, fakes, fake_scores)

    def generator_loss(self, fakes, fake_scores):
        return self.alpha * self.loss.generator_loss(fakes, fake_scores)
