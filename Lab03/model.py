import torch
import torch.nn as nn

output_classes = 10

# The backend of the network
encoder_vanilla = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
)



class ResidualBlock_2(nn.Module):
    def __init__(self, c):
        super(ResidualBlock_2, self).__init__()

        # When kernal size is 3x3, to maintain spatial dimension, we need padding of 2x2
        # Reminder:
        # If n is the input size, k is the kernal size, s is stride, and p is the padding,
        # then the output dimension will be:
        #       ((n_h - k_h + 2*p_h)/s) + 1) x ((n_w - k_w + 2*p_w)/s + 1)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(c),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(c))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + residual
        output = self.relu(output)

        return output


class ResidualBlock_3(nn.Module):
    def __init__(self, c):
        super(ResidualBlock_3, self).__init__()

        # When kernal size is 3x3, to maintain spatial dimension, we need padding of 2x2
        # Reminder:
        # If n is the input size, k is the kernal size, s is stride, and p is the padding,
        # then the output dimension will be:
        #       ((n_h - k_h + 2*p_h)/s) + 1) x ((n_w - k_w + 2*p_w)/s + 1)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(c),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(c),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(c))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output + residual
        output = self.relu(output)

        return output



# The backend of the network
encoder_mod = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        # nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.Conv2d(64, 64, (3, 3)),
        # nn.ReLU(),  # relu1-2
        ResidualBlock_2(64),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        # nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.Conv2d(128, 128, (3, 3)),
        # nn.ReLU(),  # relu2-2
        ResidualBlock_2(128),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        # nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.Conv2d(256, 256, (3, 3)),
        # nn.ReLU(),  # relu3-2
        # nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.Conv2d(256, 256, (3, 3)),
        # nn.ReLU(),  # relu3-3
        # nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.Conv2d(256, 256, (3, 3)),
        # nn.ReLU(),  # relu3-4
        ResidualBlock_3(256),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
)


# The frontend of the network
class classifier(nn.Module):

    def __init__(self, N_classes, encoder, is_encoder_trained):
        super(classifier, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.encoder = encoder


        # encoder_network = encoder
        # encoder_network.load_state_dict(torch.load("./encoder.pth"))
        # test_data = torch.randn((3, 32, 32))
        # encoder_output_size = encoder_network.forward(test_data).size()
        # c_in = encoder_output_size[0]
        # h_in = encoder_output_size[1]
        # w_in = encoder_output_size[2]
        #
        # c_out = 4096/(h_in * w_in)


        if is_encoder_trained:
            # freeze encoder weights
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.one_d_conv = nn.Conv2d(512, 256, (1, 1))
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, N_classes)

    def forward(self, X):
        # Convert to feature space using vgg network
        X = self.encoder(X)
        # Reduce the channel depth
        X = self.one_d_conv(X)
        # Flatten the tensor so it can fit in the FC layers
        X = torch.flatten(X, start_dim=1)

        # FC layers for classification
        X = self.fc1(X)
        X = torch.relu(X)
        X = self.fc2(X)
        X = torch.relu(X)
        X = self.fc3(X)
        X = torch.softmax(X, dim=0)

        return X

# FOR TESTING PURPOSE OF MODEL FILE DIRECTLY
if __name__ == "__main__":

    # test = classifier(output_classes)
    encoder_network = encoder_vanilla
    encoder_network.load_state_dict(torch.load("./encoder.pth"))
    myModel = classifier(10, encoder_network)
    test_data = torch.randn((3, 32, 32))
    print(myModel.forward(test_data))

    # encoder_network = encoder
    # encoder_network.load_state_dict(torch.load("./encoder.pth"))
    # test_data = torch.randn((3, 32, 32))
    # print(encoder_network.forward(test_data).size())

