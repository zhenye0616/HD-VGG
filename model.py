import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onlinehd import OnlineHD as HD

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name='VGG11',
        num_classes=10,
        use_hd_classifier=False,
        hd_dim=4000,
        hd_normalize=True,
    ):
        super(VGG, self).__init__()
        self.use_hd_classifier = use_hd_classifier
        self.hd_normalize = hd_normalize
        self.features = self._make_layers(cfg[vgg_name])

        shared_layers = [
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        ]

        if use_hd_classifier:
            self.classifier = nn.Sequential(*shared_layers)
            self.hd_head = HD(
                classes=num_classes,
                features=4096,
                dim=hd_dim,
            )
        else:
            shared_layers.append(nn.Linear(4096, num_classes))
            self.classifier = nn.Sequential(*shared_layers)
            self.hd_head = None

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.use_hd_classifier:
            if self.hd_normalize:
                out = F.normalize(out, p=2, dim=1)
            out = self.hd_head.scores(out, encoded=False)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
