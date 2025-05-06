# pinball_ai/dqn_model.py

import torch
import torch.nn as nn

# Import necessary config values
from config import IMAGE_SIZE, NUM_AGENT_ACTIONS, NUM_FRAMES_STACKED

class DQN(nn.Module):
    # Use NUM_AGENT_ACTIONS for the output layer size
    def __init__(self, num_actions=NUM_AGENT_ACTIONS):
        super(DQN, self).__init__()

        # --- Use NUM_FRAMES_STACKED from config for input channels ---
        input_channels = NUM_FRAMES_STACKED 

        # Image processing branch (CNN)
        self.cnn = nn.Sequential(
            # --- Use input_channels (now 4) here ---
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size dynamically
        # --- Use input_channels (now 4) for dummy input shape ---
        dummy_input_img = torch.zeros(1, input_channels, *IMAGE_SIZE) 
        cnn_output_size = self._get_cnn_output_size(dummy_input_img)

        # Ball coordinate processing branch - unchanged
        ball_input_size = 2 # (x, y) coordinates
        ball_feature_size = 32
        self.fc_ball = nn.Sequential(
            nn.Linear(ball_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, ball_feature_size),
            nn.ReLU()
        )

        # Combined fully connected layers - input size calculation is dynamic
        combined_input_size = cnn_output_size + ball_feature_size

        self.fc_combined = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    # Helper function to compute CNN output size
    def _get_cnn_output_size(self, shape_input):
        with torch.no_grad():
             output = self.cnn(shape_input)
        return output.shape[1]

    # Forward pass - takes stacked image input and ball input
    def forward(self, stacked_image_input, ball_input):
        # Process stacked image (expects [batch, NUM_FRAMES_STACKED, H, W])
        img_features = self.cnn(stacked_image_input) # <-- Pass stacked input

        # Process ball coordinates
        ball_features = self.fc_ball(ball_input)

        # Concatenate features
        combined_features = torch.cat((img_features, ball_features), dim=1)

        # Pass through combined layers
        return self.fc_combined(combined_features)