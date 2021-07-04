import torch.nn as nn


class ConvNet(nn.Module):

    """For convenience, we assume that input images have shape (3, 84, 84)"""

    def __init__(self, input_depth, embedding_dim):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_depth, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 35 * 35, embedding_dim),  # 35 x 35 is the shape of 84 x 84 images after processing
            nn.LayerNorm(embedding_dim),
            nn.Tanh()  # just following drq's choice here, I don't know why tanh should be used here
        )

    def forward(self, obs):
        return self.layers(obs)
