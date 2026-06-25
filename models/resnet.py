import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, inchannels, outchannels, stride=1):

        super().__init__()
        self.short_cut = nn.Identity()
        if stride != 1 or inchannels != outchannels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=outchannels)
            )

        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=outchannels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=outchannels)

    def forward(self, x):
        identity = self.short_cut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.layer1 = nn.Sequential(
            BasicBlock(inchannels=64, outchannels=64, stride=1),
            BasicBlock(inchannels=64, outchannels=64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(inchannels=64, outchannels=128, stride=2),
            BasicBlock(inchannels=128, outchannels=128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(inchannels=128, outchannels=256, stride=2),
            BasicBlock(inchannels=256, outchannels=256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(inchannels=256, outchannels=512, stride=2),
            BasicBlock(inchannels=512, outchannels=512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
if __name__ == "__main__":
    net = ResNet18(num_classes=10)

    x = torch.randn(
        8,
        3,
        32,
        32
    )

    y = net(x)

    print(y.shape)
    print(
        sum(
            p.numel()
            for p in net.parameters()
        )
    )
