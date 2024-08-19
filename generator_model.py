import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import print_info
import time

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

        self.up3_v2 = Block(features * 8 + features * 4, features * 8, down=False, act="relu", use_dropout=False)
        self.up4_v2 = Block(features * 8 + features * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up5_v2 = Block(features * 4 + features * 1, features * 2, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.Conv2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.downsample = nn.Upsample(size=(42,42), mode='bilinear')

    def forward(self, x):
        return self.forward_v2(x)
    
    def forward_v2(self, x):
        p = self.pad(x)
        d1 = self.initial_down(p)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        bottleneck = self.bottleneck(d6)

        u1 = self.up1(bottleneck)
        print("u1:", u1.shape, "d4:", d4.shape)

        d4_ip = func.interpolate(d4, u1.shape[2:], mode='bilinear')
        u2 = self.up2(torch.cat([u1, d4_ip], 1))
        print("u2:", u2.shape)

        d3_ip = func.interpolate(d3, u2.shape[2:], mode='bilinear')
        u3 = self.up3_v2(torch.cat([u2, d3_ip], 1))
        print("u3:", u3.shape)

        d2_ip = func.interpolate(d2, u3.shape[2:], mode='bilinear')
        u4 = self.up4_v2(torch.cat([u3, d2_ip], 1))
        print("u4:", u4.shape)

        d1_ip = func.interpolate(d1, u4.shape[2:], mode='bilinear')
        u5 = self.up5_v2(torch.cat([u4, d1_ip], 1))
        print("u5:", u5.shape)

        ds = self.downsample(u5)
        print("ds:", ds.shape)
        return ds

    def forward_v1(self, x):
        p = self.pad(x)
        d1 = self.initial_down(p)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        bottleneck = self.bottleneck(d6)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        result = self.final_up(torch.cat([u6, d1], 1))
        result_ds = self.downsample(result)
        return result_ds


def test():
    start=time.time()
    x = torch.randn((1, 1, 900, 900))
    model = Generator(in_channels=1, out_channels=106, features=64)
    preds = model(x)
    end=time.time()
    print("\nShape of prediction:\n", preds.shape)
    print_info(preds, "Preds")
    print("Time (ms):", (end-start)*1000)


if __name__ == "__main__":
    test()
