import cv2
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from camera_system import get_camera_stream, capture_frame
from ball_tracker import detect_ball
from preprocessing import preprocess_frame
from dqn_model import DQN
from servo_controller import setup_serial_connection, send_action
from environment import choose_action, calculate_reward
from utils import ReplayMemory
from config import *

def create_gui(state_color, run_number, points):
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    if state_color == 'green':
        frame[:] = (0, 255, 0)
    elif state_color == 'red':
        frame[:] = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Run: {run_number}', (30, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Points: {points}', (30, 200), font, 1, (255, 255, 255), 2)
    cv2.imshow('Pinball AI Status', frame)
    cv2.waitKey(1)

def optimize_model(model, target_model, optimizer, batch, device):
    state_images, state_balls, actions, rewards, next_state_images, next_state_balls, dones = zip(*batch)

    state_images = torch.cat(state_images)
    state_balls = torch.cat(state_balls)
    next_state_images = torch.cat(next_state_images)
    next_state_balls = torch.cat(next_state_balls)

    actions = torch.tensor(actions, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, device=device).unsqueeze(1)
    dones = torch.tensor(dones, device=device).unsqueeze(1)

    q_values = model(state_images, state_balls).gather(1, actions)
    next_q_values = target_model(next_state_images, next_state_balls).max(1)[0].detach().unsqueeze(1)

    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(len(ACTION_SPACE)).to(device)
    target_model = DQN(len(ACTION_SPACE)).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    cap = get_camera_stream()
    ser = setup_serial_connection()

    epsilon = EPSILON_START
    steps = 0
    run_number = 1
    total_points = 0

    while True:
        create_gui('green', run_number, total_points)

        frame = capture_frame(cap)
        ball_position = detect_ball(frame)
        ball_x, ball_y = (ball_position if ball_position else (0, 0))

        frame_height, frame_width = frame.shape[:2]
        ball_x_norm = ball_x / frame_width
        ball_y_norm = ball_y / frame_height

        # Preprocess current state
        state_image = preprocess_frame(frame)
        state_image = torch.tensor(state_image, dtype=torch.float32).unsqueeze(0).to(device)
        state_ball = torch.tensor([ball_x_norm, ball_y_norm], dtype=torch.float32).unsqueeze(0).to(device)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        done = False
        if key == ord(' '):
            done = True

        # Choose and perform action
        action = choose_action(model, state_image, state_ball, epsilon)
        ser.send_action(action)

        # Capture next state
        next_frame = capture_frame(cap)
        next_ball_position = detect_ball(next_frame)
        next_ball_x, next_ball_y = (next_ball_position if next_ball_position else (0, 0))

        next_frame_height, next_frame_width = next_frame.shape[:2]
        next_ball_x_norm = next_ball_x / next_frame_width
        next_ball_y_norm = next_ball_y / next_frame_height

        next_state_image = preprocess_frame(next_frame)
        next_state_image = torch.tensor(next_state_image, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_ball = torch.tensor([next_ball_x_norm, next_ball_y_norm], dtype=torch.float32).unsqueeze(0).to(device)

        # Calculate reward
        events = {}  # You can improve this later with real event detection
        reward = calculate_reward(events)
        total_points += reward

        # Save transition
        memory.push((state_image, state_ball, action, reward, next_state_image, next_state_ball, done))

        # Optimize model
        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            optimize_model(model, target_model, optimizer, batch, device)

        # Update target network
        if steps % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        # Update epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        steps += 1

        if done:
            print(f"[INFO] Space pressed. Ending Run {run_number}. Points earned: {total_points}")
            create_gui('red', run_number, total_points)
            cv2.waitKey(1)
            time.sleep(2)

            run_number += 1
            total_points = 0
            continue