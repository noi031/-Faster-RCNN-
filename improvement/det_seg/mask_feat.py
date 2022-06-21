from torch import nn
import torch

class MaskFeatHead(nn.Module):
    def __init__(self, num_channels, num_levels):
        """
        list[Tensor] -> Tensor(batch_size, num_channels, H, W)
        After the feature pyramid network, level p2, p3, p4, p5 are all convolved and
        upsampled to the size that each edge is a quater of the original one. This module
        outputs a unified feature map for all levels.
        Parameters:
            num_channels: number of channels without coordinate channels.
            num_levels: number of feature levels.
        """
        super(MaskFeatHead, self).__init__()
        self.convs_all_levels, self.num_levels = nn.ModuleList(), num_levels
        for i in range(num_levels):
            convs_per_level = nn.Sequential()
            if not i:
                convs_per_level.add_module('conv' + str(i), nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, 3, padding = 1),
                    nn.ReLU()
                ))
                self.convs_all_levels.append(convs_per_level)
                continue
            for j in range(i):
                if not j:
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(
                        nn.Conv2d(num_channels, num_channels, 3, padding = 1),
                        nn.ReLU()
                    ))
                    convs_per_level.add_module('upsample' + str(j),
                                               nn.Upsample(None, 2, 'bilinear', True))
                    continue
                convs_per_level.add_module('conv' + str(j), nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, 3, padding = 1),
                    nn.ReLU()
                ))
                convs_per_level.add_module('upsample' + str(j),
                                           nn.Upsample(None, 2, 'bilinear', True))
            self.convs_all_levels.append(convs_per_level)
        self.conv_pred = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1, padding = 1),
            nn.ReLU()
        )
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        return self.conv_pred(sum([conv(x) for x, conv in zip(inputs, self.convs_all_levels)]))