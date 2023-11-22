from torch import nn
import Lab05.dsntnn as dsntnn

def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)

class CoordinateRegression(nn.Module):
    def __init__(self, n_coordinates=1):
        super().__init__()

        self.backend_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

        self.heatmap_conv = nn.Conv2d(16, n_coordinates, kernel_size=1, bias=False)

    def forward(self, imgs):
        imgs = self.layers(imgs)

        # Use a 1x1 conv to get one unnormalized heatmap per location
        imgs = self.heatmap_conv(imgs)

        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(imgs)

        # Calculate the coordinates
        coordinates = dsntnn.dsnt(heatmaps)

        return coordinates, heatmaps
