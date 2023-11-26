from torch import nn
import dsntnn

class CoordinateRegression(nn.Module):
    def __init__(self, n_coordinates=1):
        super().__init__()

        # the vgg model from lab2 but 1/4 the channel depth
        # Has to be shallower since the 1d conv at the end brings it back down to a depth of 1
        # Going from 512 to 1 would lose too much information, 64 down to 1 is better (I think)
        self.backend_layers = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 16, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
        )

        self.heatmap_conv = nn.Conv2d(64, n_coordinates, kernel_size=1, bias=False)

    def forward(self, imgs):
        imgs = self.backend_layers(imgs)

        # Use a 1x1 conv to get one unnormalized heatmap per location
        imgs = self.heatmap_conv(imgs)

        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(imgs)

        # Calculate the coordinates
        coordinates = dsntnn.dsnt(heatmaps)

        return coordinates, heatmaps
