import torch
import torchvision
import torch.nn as nn

efficientnet_b1 = torchvision.models.efficientnet_b1(num_classes=2)


# class EfficientNetB1(nn.Module):
#     def __init__(self):
#         super(EfficientNetB1, self).__init__()
#
#         # inverted_residual_setting = Sequence[Union[MBConvConfig, FusedMBConvConfig]],
#         # dropout = float
#         # last_channel = Optional[int],
#         # weights = Optional[WeightsEnum],
#         # progress = bool,
#
#         self.model = torchvision.models.efficientnet_b1(num_classes=2)
#
#         #torchvision.models.EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, num_classes=1)
#
#     def forward(self, x):
#         x = self.model.forward(x)
#
#         return x

#
# if __name__ == "__main__":
#
#     nn = Classifier()
#
#
#     nn.train()
