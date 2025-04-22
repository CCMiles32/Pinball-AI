import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        
        # Image processing branch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # After flattening, compute size
        dummy_input = torch.zeros(1, 1, *IMAGE_SIZE)
        cnn_output_size = self.cnn(dummy_input).shape[1]

        # Ball position processing branch
        self.fc_ball = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )

        # Combined fully connected layers
        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_output_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, image_input, ball_input):
        img_features = self.cnn(image_input)
        ball_features = self.fc_ball(ball_input)
        combined = torch.cat((img_features, ball_features), dim=1)
        return self.fc_combined(combined)