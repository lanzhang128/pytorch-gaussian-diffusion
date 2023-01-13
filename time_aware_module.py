import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAwareResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.emb = nn.Linear(t_channels, out_channels)
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = None

    def forward(self, x, time_emb):
        temp = self.layer1(x) + self.emb(time_emb)[:, :, None, None]
        residual = self.proj(x) if self.proj else x
        return self.layer2(temp) + residual


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.block1 = TimeAwareResBlock(in_channels, out_channels, t_channels)
        self.block2 = TimeAwareResBlock(out_channels, out_channels, t_channels)

    def forward(self, x, time_emb):
        x = self.down(x)
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.block1 = TimeAwareResBlock(2 * out_channels, out_channels, t_channels)
        self.block2 = TimeAwareResBlock(out_channels, out_channels, t_channels)

    def forward(self, x1, x2, time_emb):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class TimeAwareUNet(nn.Module):
    def __init__(self, n_channels, base_channels=128, t_channels=64, ch_mult=[1, 2, 4, 8]):
        super().__init__()
        self.n_channels = n_channels
        self.down = nn.ModuleList([
            TimeAwareResBlock(n_channels, 64, t_channels)
        ])
        self.up = nn.ModuleList([nn.Conv2d(64, n_channels, kernel_size=1)])
        
        channels = [64] + [base_channels * i for i in ch_mult]
        for i in range(len(channels)-1):
            self.down.append(Down(channels[i], channels[i+1], t_channels))
            self.up.insert(0, Up(channels[i+1], channels[i], t_channels))
            
        emb = torch.log(torch.tensor(10000)) / (32 - 1)
        self.emb = torch.exp(torch.arange(32) * -emb)

    def forward(self, x, t):
        time_emb = (t[:, None] * self.emb[None, :]).to(x.device)
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=1)
        temp = []
        for i in range(len(self.down)):
            x = self.down[i](x, time_emb)
            temp.append(x)
        for i in range(len(self.up)-1):
            x = self.up[i](x, temp[-i-2], time_emb) 
        x = self.up[-1](x)
        return x
