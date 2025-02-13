import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
            self.num_inchannels[branch_index] = num_channels[branch_index]
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif i < j:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                else:
                    convs = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.num_inchannels[i]
                            convs.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = self.num_inchannels[j]
                            convs.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if len(x) != self.num_branches:
            raise ValueError("Input size does not match the number of branches")
        
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fused = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fused.append(self.relu(y))
        
        return x_fused


class HRNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(HRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(1, 64, [4, 4], [32, 64])
        self.stage2 = self._make_stage(2, 64, [4, 4], [32, 64])

        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_stage(self, num_branches, num_inchannels, num_blocks, num_channels):
        blocks = BasicBlock
        return HighResolutionModule(num_branches, blocks, num_blocks, num_inchannels, num_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = [x, x]  # Example with 2 branches for simplicity
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.final_layer(x[0])
        return F.log_softmax(x, dim=1)


# Example usage:
if __name__ == '__main__':
    model = HRNet(num_classes=1000)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
