from plasma.modules import *


class Classifier(nn.Sequential):

    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)

        self.preprocess = Normalization()

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]) # 14 x 14

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ])

        self.classifier = nn.Sequential(*[
            GlobalAverage(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ])


def model(**kwargs):
    return Classifier(**kwargs)
