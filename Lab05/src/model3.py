from torch import nn
import dsntnn
from torchvision.models import resnet34
class CoordinateRegression_b2(nn.Module):
    def __init__(self, n_coordinates=1):
        super().__init__()

        # try the resnet34 conv layers rather than vgg-19

        # FOR NEXT MODEL, KEEP OUTPUTTED HEATMAX LARGER
        # PAPER USER RESNET-32 BUT REDUCES ALL STRIDES OF 2 TO BE STRIDES OF 1
        # ALSO MAKES USE OF DILATED CONVOLUTIONS IN SUBSEQUENT LAYERS (Add holes inbetweed kernel elements)

        res34_model = resnet34()
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
