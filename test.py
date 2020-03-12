from plasma.modules import *


class Conv_BN_ReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, attention=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels * groups, out_channels * groups, kernel, stride, padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_channels * groups)

        if attention:
            self.att = CBAM(out_channels * groups)

        self.act = nn.ReLU(inplace=True)


class ResCap(nn.Module):

    def __init__(self, capsules=1, iters=1):
        super().__init__()
        encoder = dynamic_routing_next50()

        self.layer0 = nn.Sequential(*[
            ImagenetNormalization(from_gray=False),
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool
        ])

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.classifier = nn.Sequential(*[
            Conv_BN_ReLU(2048, 2048),
            GlobalAverage(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        ])

        if iters > 1:
            self.encoder.apply(apply_iters)

        self.up2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.decoder4 = Conv_BN_ReLU(2048, 64, attention=True)  # 16 x 16
        self.decoder3 = Conv_BN_ReLU(1024 + 64, 64, attention=True)   # 32 x 32
        self.decoder2 = Conv_BN_ReLU(512 + 64, 64, attention=True)   # 64 x 64
        self.decoder1 = Conv_BN_ReLU(256 + 64, 64, attention=True)    # 128 x 128
        self.mask = nn.Sequential(*[
            nn.Conv2d(64 * 4, 1, kernel_size=3),
            nn.Sigmoid()
        ])

    def forward(self, x):
        prob, cons = self.forward_classification(x)
        mask = self.forward_segmentation(*cons)

        return prob, mask

    def forward_classification(self, x):
        con0 = self.layer0(x)
        con1 = self.layer1(con0)
        con2 = self.layer2(con1)
        con3 = self.layer3(con2)
        con4 = self.layer4(con3)

        prob = self.classifier(con4)
        return prob, [con1, con2, con3, con4]

    def forward_segmentation(self, skip1, skip2, skip3, skip4):
        up4 = self.up2x(skip4)
        con4 = self.decoder4(up4)  # 16 x16

        up3 = torch.cat([con4, skip3], dim=1)
        up3 = self.up2x(up3)
        con3 = self.decoder3(up3)  # 32 x 32

        up2 = torch.cat([con3, skip2], dim=1)
        up2 = self.up2x(up2)
        con2 = self.decoder2(up2)  # 64 x 64

        up1 = torch.cat([con2, skip1], dim=1)
        up1 = self.up2x(up1)
        con1 = self.decoder1(up1)  # 128 x 128

        all_skip = torch.cat([self.up2x(con1), self.up4x(con2), self.up8x(con3), self.up16x(con4)], dim=1)
        mask = self.mask(all_skip)

        return mask
