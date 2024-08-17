import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels_x=1, in_channels_y=3, features=[64, 128, 256, 512]):
        super().__init__()

        # downconv just for thorlabs image
        self.prep_x = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels_x,
                out_channels=int(in_channels_y/2),
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=int(in_channels_y/2),
                out_channels=in_channels_y,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(42,42), mode='bilinear'),
        )

        # therefore
        in_channels_x = 106

        # conv to first feature amount
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels_x + in_channels_y,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # more and more features
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        # conv to just one feature, the probality of realness
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers, nn.Upsample((1,1), mode='bilinear'))

    def forward(self, x, y):
        x = self.prep_x(x)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 1, 900, 900))
    y = torch.randn((1, 106, 42, 42))
    model = Discriminator(in_channels_x=1, in_channels_y=106)
    preds = model(x, y)
    print("\nModel:\n", model)
    print("\nShape of prediction:\n", preds.shape)


if __name__ == "__main__":
    test()
