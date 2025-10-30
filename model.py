import torch
import torch.nn as nn
import torch.nn.functional as F

from onlinehd import OnlineHD as HD
from quantization import GaussianActivationNoise

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
        activation_noise=False,
        activation_noise_bits=5,
        activation_noise_sigma=1.0,
        activation_noise_eval=False,
        activation_noise_clamp=False,
    ):
        super(VGG, self).__init__()
        self.use_hd_classifier = use_hd_classifier
        self.hd_normalize = hd_normalize
        self.activation_noise = activation_noise
        self.activation_noise_bits = activation_noise_bits
        self.activation_noise_sigma = activation_noise_sigma
        self.activation_noise_eval = activation_noise_eval
        self.activation_noise_clamp = activation_noise_clamp
        self.feature_noise = nn.ModuleList()
        self.classifier_noise = nn.ModuleList()
        self._feature_noise_indices = []
        self._classifier_noise_indices = []

        self.features = self._make_layers(cfg[vgg_name])

        classifier_layers = [
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        ]

        for idx, layer in enumerate(classifier_layers):
            if isinstance(layer, nn.ReLU):
                self._classifier_noise_indices.append(idx)
                self.classifier_noise.append(self._make_noise_layer())

        if use_hd_classifier:
            self.classifier = nn.Sequential(*classifier_layers)
            self.hd_head = HD(
                classes=num_classes,
                features=4096,
                dim=hd_dim,
            )
        else:
            classifier_layers.append(nn.Linear(4096, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)
            self.hd_head = None
        self._sync_noise_state()

    def forward(self, x):
        out = self._apply_noise_after_layers(
            x, self.features, self._feature_noise_indices, self.feature_noise
        )
        out = out.view(out.size(0), -1)
        out = self._apply_noise_after_layers(
            out, self.classifier, self._classifier_noise_indices, self.classifier_noise
        )
        if self.use_hd_classifier:
            if self.hd_normalize:
                out = F.normalize(out, p=2, dim=1)
            out = self.hd_head.scores(out, encoded=False)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        noise_indices = []
        for x in cfg:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
                noise_indices.append(len(layers) - 1)
                self.feature_noise.append(self._make_noise_layer())
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                noise_indices.append(len(layers) - 1)
                self.feature_noise.append(self._make_noise_layer())
                in_channels = x
        layers += [
            nn.AvgPool2d(kernel_size=1, stride=1),
        ]
        noise_indices.append(len(layers) - 1)
        self.feature_noise.append(self._make_noise_layer())
        self._feature_noise_indices = noise_indices
        return nn.Sequential(*layers)

    def _make_noise_layer(self):
        return GaussianActivationNoise(
            num_bits=self.activation_noise_bits,
            sigma_factor=self.activation_noise_sigma,
            enabled=self.activation_noise,
            apply_during_eval=self.activation_noise_eval,
            clamp=self.activation_noise_clamp,
        )

    def set_activation_noise(self, enabled: bool):
        """Enable/disable Gaussian activation noise at runtime."""
        self.activation_noise = enabled
        self._sync_noise_state()

    def _sync_noise_state(self):
        for module in self.feature_noise:
            module.enabled = self.activation_noise
        for module in self.classifier_noise:
            module.enabled = self.activation_noise

    def _apply_noise_after_layers(self, x, layers, noise_indices, noise_modules):
        noise_ptr = 0
        for idx, layer in enumerate(layers):
            x = layer(x)
            if noise_ptr < len(noise_indices) and idx == noise_indices[noise_ptr]:
                x = noise_modules[noise_ptr](x)
                noise_ptr += 1
        return x
