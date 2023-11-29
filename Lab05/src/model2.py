from torch import nn
import dsntnn

class CoordinateRegression_b1(nn.Module):
    def __init__(self, n_coordinates=1):
        super().__init__()

        # the vgg model from lab2 but with 1/2 the channel depth
        # Added dropout of 0.25 to reduce overfitting. From research, dropout should occur after MaxPool2d
        self.backend_layers = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Dropout(p=0.25),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Dropout(p=0.25),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Dropout(p=0.25),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
        )
        self.heatmap_conv = nn.Conv2d(128, n_coordinates, kernel_size=1, bias=False)

    def forward(self, imgs):
        imgs = self.backend_layers(imgs)

        # Use a 1x1 conv to get one unnormalized heatmap per location
        imgs = self.heatmap_conv(imgs)

        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(imgs)

        # Calculate the coordinates
        coordinates = dsntnn.dsnt(heatmaps)

        return coordinates, heatmaps
