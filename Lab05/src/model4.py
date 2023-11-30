from torch import nn
import dsntnn
from torchvision.models import resnet34
class CoordinateRegression_b3(nn.Module):
    def __init__(self, n_coordinates=1):
        super().__init__()

        # The previous resnet34 only had a heatmap of size (8,8) which makes it hard to accuratly locate the nose pixel
        # Switch all strides of 2 in Resnet to strides of 1 to create a larger output image
        # The heatmap will now be of size (64, 64)

        res34_model = resnet34()
        #Reduce the stride to 1 in the second layer
        res34_model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        res34_model.layer2[0].downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Reduce the stride to 1 in the third layer
        res34_model.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        res34_model.layer3[0].downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Reduce the stride to 1 in the fourth layer
        res34_model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        res34_model.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Remove the fully connected layer
        self.backend_layers = nn.Sequential(*list(res34_model.children())[:-2])

        self.heatmap_conv = nn.Conv2d(512, n_coordinates, kernel_size=1, bias=False)

    def forward(self, imgs):
        imgs = self.backend_layers(imgs)

        # Use a 1x1 conv to get one unnormalized heatmap per location
        imgs = self.heatmap_conv(imgs)

        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(imgs)

        # Calculate the coordinates
        coordinates = dsntnn.dsnt(heatmaps)

        return coordinates, heatmaps
