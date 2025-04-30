# pinball_ai/dqn_model.py

import torch
import torch.nn as nn
# Import IMAGE_SIZE and NUM_AGENT_ACTIONS.

from config import IMAGE_SIZE, NUM_AGENT_ACTIONS 

class DQN(nn.Module):
    # Use NUM_AGENT_ACTIONS for the output layer size
    def __init__(self, num_actions=NUM_AGENT_ACTIONS):
        super(DQN, self).__init__()

        # --- Define number of input channels ---
        input_channels = 1

        # Image processing branch (CNN)
        self.cnn = nn.Sequential(
            # --- FIX: Use input_channels (which is 1) here ---
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output size dynamically
        # The dummy input should also have the same number of channels
        # --- FIX: Use input_channels (which is 1) for dummy input shape ---
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

        # Combined fully connected layers - unchanged input calculation logic
        combined_input_size = cnn_output_size + ball_feature_size

        self.fc_combined = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    # Helper function to compute CNN output size - unchanged
    def _get_cnn_output_size(self, shape_input):
        with torch.no_grad():
             output = self.cnn(shape_input)
        return output.shape[1]

    # Forward pass - unchanged (still takes image and ball inputs)
    def forward(self, image_input, ball_input):
        # Process image (now expects [batch, 1, H, W])
        img_features = self.cnn(image_input)

        # Process ball coordinates
        ball_features = self.fc_ball(ball_input)

        # Concatenate features
        combined_features = torch.cat((img_features, ball_features), dim=1)

        # Pass through combined layers
        return self.fc_combined(combined_features)