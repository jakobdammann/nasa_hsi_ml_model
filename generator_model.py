import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()

        self.pad = nn.ZeroPad2d(int((1024-900)/2))

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up4 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.Conv2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.downsample = nn.Upsample(size=(42,42), mode='bilinear')

    def forward(self, x):
        #print("")
        p = self.pad(x)
        #print(p.shape)
        d1 = self.initial_down(p)
        #print(d1.shape)
        d2 = self.down1(d1)
        #print(d2.shape)
        d3 = self.down2(d2)
        #print(d3.shape)
        d4 = self.down3(d3)
        #print(d4.shape)
        d5 = self.down4(d4)
        #print(d5.shape)
        d6 = self.down5(d5)
        #print(d6.shape)
        bottleneck = self.bottleneck(d6)
        #print("bottleneck:", bottleneck.shape)
        u1 = self.up1(bottleneck)
        #print(u1.shape)
        u2 = self.up2(torch.cat([u1, d6], 1))
        #print(u2.shape)
        u3 = self.up3(torch.cat([u2, d5], 1))
        #print(u3.shape)
        u4 = self.up4(torch.cat([u3, d4], 1))
        #print(u4.shape)
        u5 = self.up5(torch.cat([u4, d3], 1))
        #print(u5.shape)
        u6 = self.up6(torch.cat([u5, d2], 1))
        #print(u6.shape)
        result = self.final_up(torch.cat([u6, d1], 1))
        #print(result.shape)
        result_ds = self.downsample(result)
        return result_ds


def test():
    x = torch.randn((1, 1, 900, 900))
    model = Generator(in_channels=1, out_channels=106, features=64)
    preds = model(x)
    print("\nShape of prediction:\n", preds.shape)


if __name__ == "__main__":
    test()
